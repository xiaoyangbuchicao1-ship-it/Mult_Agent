from __future__ import annotations

import copy
import itertools
import math
import random
from langgraph.graph import END, START, StateGraph

from shared import prompts
from shared.metrics import pair_key, parse_score, semantic_metrics, voting_metrics
from shared.runtime import ConsensusState, Context, event, parallel_map, stance_changes, stop_reason, validate_review

HISTORICAL_ROUTE = [("a1", "a5"), ("a2", "a5"), ("a2", "a3"), ("a3", "a4")]


def choose_pair(metrics, exp, round_number):
    scores = metrics["pair_scores"]
    lowest = min(scores.values())
    tied = sorted(k for k, v in scores.items() if math.isclose(v, lowest, abs_tol=1e-12))
    rng = random.Random(exp["tie_seed"] + round_number - 1)
    if exp["route_mode"] == "historical_manual":
        selected = pair_key(*HISTORICAL_ROUTE[round_number - 1])
    elif exp["route_mode"] == "random_pair":
        selected = rng.choice(sorted(scores))
    else:
        selected = rng.choice(tied)
    return {"pair": selected.split("__"), "score_before": scores[selected],
            "minimum_score": lowest, "minimum_ties": [k.split("__") for k in tied],
            "is_low_similarity_edge": scores[selected] < exp["gamma"],
            "route_mode": exp["route_mode"], "tie_seed": exp["tie_seed"] + round_number - 1}


def run_bilateral_dialogue(ctx: Context, state):
    before = copy.deepcopy(state["reviews"])
    a, b = state["pair"]
    transcript = []
    shared_history = [{"pair": item["pair"], "round_after": item["round_after"], "transcript": item["transcript"]}
                      for item in state.get("interactions", [])]
    for turn in range(1, ctx.exp["bilateral_turns"] + 1):
        for agent, opponent in ((a, b), (b, a)):
            system, user = prompts.bilateral(ctx.case, ctx.agents[agent], opponent, before, transcript, shared_history,
                                             mode=ctx.exp.get("bilateral_prompt_mode", "baseline"))
            text = ctx.backend.text(f"semantic.dialogue.R{state['round']}.T{turn}.{agent}", system, user,
                                    max_tokens=ctx.exp["dialogue_max_tokens"])
            transcript.append({"turn": turn, "agent": agent, "text": text})
    return {"before_reviews": before, "transcript": transcript,
            "trace": [event("bilateral_dialogue", state["round"])]}


def run_pair_update(ctx: Context, state):
    def update(agent):
        system, user = prompts.revise(ctx.case, ctx.agents[agent], state["before_reviews"][agent],
                                      {"pair": state["pair"], "transcript": state["transcript"]}, "参与双边协商后的完整意见",
                                      writeback_mode=ctx.exp.get("pair_writeback_mode", "baseline"))
        return validate_review(ctx.backend.json(f"semantic.pair_update.R{state['round']}.{agent}", system, user), agent)
    revised = dict(zip(state["pair"], parallel_map(update, state["pair"], ctx.exp["max_workers"])))
    return {"pair_reviews": revised, "trace": [event("pair_state_update", state["round"])]}


def run_global_feedback(ctx: Context, state):
    before = state["before_reviews"]
    after = copy.deepcopy(before)
    after.update(copy.deepcopy(state["pair_reviews"]))
    observer_updates = {}
    if ctx.exp["feedback_mode"] == "all_agents":
        observers = [a for a in ctx.ids if a not in state["pair"]]
        evidence = {"pair": state["pair"], "transcript": state["transcript"], "pair_reviews": state["pair_reviews"]}
        def update(agent):
            system, user = prompts.revise(ctx.case, ctx.agents[agent], before[agent], evidence, "未参与者独立吸收双边成果")
            return validate_review(ctx.backend.json(f"semantic.observer.R{state['round']}.{agent}", system, user), agent)
        observer_updates = dict(zip(observers, parallel_map(update, observers, ctx.exp["max_workers"])))
        after.update(observer_updates)
    interaction = {"round_before": state["round"], "round_after": state["round"] + 1,
                   "pair": state["pair"], "selection": state["selection"], "transcript": state["transcript"],
                   "before_reviews": before, "pair_reviews": state["pair_reviews"],
                   "observer_updates": observer_updates, "after_reviews": after,
                   "stance_changes": stance_changes(before, after)}
    return {"reviews": after, "round": state["round"] + 1, "interactions": [interaction],
            "trace": [event("global_feedback", state["round"] + 1)]}


def build_graph(ctx: Context):
    def summarize(state):
        def summarize_one(agent):
            system, user = prompts.summary(ctx.case, agent, state["reviews"][agent])
            value = ctx.backend.json(f"semantic.summary.R{state['round']}.{agent}", system, user,
                                      max_tokens=ctx.exp["summary_max_tokens"])
            if not isinstance(value.get("summary"), str) or not value["summary"].strip():
                raise ValueError("Summary must be a nonempty string")
            return value["summary"]
        summaries = dict(zip(ctx.ids, parallel_map(summarize_one, ctx.ids, ctx.exp["max_workers"])))
        return {"summaries": summaries, "trace": [event("summarize", state["round"])]}

    def semantic_measure(state):
        def score(pair):
            a, b = pair
            system, user = prompts.similarity(state["summaries"][a], state["summaries"][b])
            raw = ctx.backend.text(f"semantic.score.R{state['round']}.{a}->{b}", system, user, role="judge")
            return f"{a}->{b}", parse_score(raw), raw
        rows = parallel_map(score, itertools.permutations(ctx.ids, 2), ctx.exp["max_workers"])
        directed = {k: value for k, value, _ in rows}
        raw_scores = {k: raw for k, _, raw in rows}
        metrics = semantic_metrics(ctx.ids, directed, ctx.exp["gamma"])
        snapshot = ctx.snapshot(state, metrics, summaries=copy.deepcopy(state["summaries"]),
                                directed_scores=directed, raw_scores=raw_scores,
                                interaction_pair=state.get("pair"),
                                stance_metrics=voting_metrics(state["reviews"], ctx.case["issues"]))
        return {"metrics": metrics, "directed_scores": directed, "raw_scores": raw_scores,
                "snapshots": [snapshot], "stop_reason": stop_reason({**state, "metrics": metrics}, ctx.exp),
                "trace": [event("semantic_measure", state["round"])]}

    def select_pair(state):
        selection = choose_pair(state["metrics"], ctx.exp, state["round"])
        return {"selection": selection, "pair": selection["pair"], "trace": [event("select_pair", state["round"])]}

    def bilateral_dialogue(state):
        return run_bilateral_dialogue(ctx, state)

    def pair_state_update(state):
        return run_pair_update(ctx, state)

    def global_feedback(state):
        return run_global_feedback(ctx, state)

    def finish(state):
        return ctx.finish(state, "semantic")

    graph = StateGraph(ConsensusState)
    for name, fn in (("prepare_case", ctx.prepare), ("initialize_agents", ctx.initialize),
                     ("summarize", summarize), ("semantic_measure", semantic_measure),
                     ("select_pair", select_pair), ("bilateral_dialogue", bilateral_dialogue),
                     ("pair_state_update", pair_state_update), ("global_feedback", global_feedback),
                     ("final_determination", finish)):
        graph.add_node(name, fn)
    graph.add_edge(START, "prepare_case")
    graph.add_edge("prepare_case", "initialize_agents")
    graph.add_edge("initialize_agents", "summarize")
    graph.add_edge("summarize", "semantic_measure")
    graph.add_conditional_edges("semantic_measure", lambda s: "finish" if s["stop_reason"] else "continue",
                                {"finish": "final_determination", "continue": "select_pair"})
    graph.add_edge("select_pair", "bilateral_dialogue")
    graph.add_edge("bilateral_dialogue", "pair_state_update")
    graph.add_edge("pair_state_update", "global_feedback")
    graph.add_edge("global_feedback", "summarize")
    graph.add_edge("final_determination", END)
    return graph.compile()
