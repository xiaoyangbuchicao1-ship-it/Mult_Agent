from __future__ import annotations

import copy
from langgraph.graph import END, START, StateGraph

from shared import prompts
from shared.metrics import voting_metrics
from shared.runtime import ConsensusState, Context, event, parallel_map, stance_changes, stop_reason, validate_review


def build_graph(ctx: Context):
    def vote_measure(state):
        metrics = voting_metrics(state["reviews"], ctx.case["issues"])
        return {"metrics": metrics,
                "snapshots": [ctx.snapshot(state, metrics)],
                "stop_reason": stop_reason({**state, "metrics": metrics}, ctx.exp),
                "trace": [event("vote_measure", state["round"])]}

    def identify_disagreement(state):
        disagreement = {
            "agents": state["metrics"]["outliers"],
            "issues": sorted(
                [{"id": int(q), "consensus": v} for q, v in state["metrics"]["per_issue_consensus"].items() if v < 1],
                key=lambda q: (q["consensus"], q["id"]),
            ),
        }
        return {"disagreement": disagreement, "trace": [event("identify_disagreement", state["round"])]}

    def debate(state):
        before = copy.deepcopy(state["reviews"])
        transcript = []
        for agent in ctx.ids:
            system, user = prompts.multilateral(ctx.case, ctx.agents[agent], before, state["disagreement"], transcript)
            speech = ctx.backend.text(f"voting.debate.R{state['round']}.{agent}", system, user,
                                      max_tokens=ctx.exp["dialogue_max_tokens"])
            transcript.append({"agent": agent, "text": speech})
        return {"transcript": transcript, "before_reviews": before,
                "trace": [event("debate", state["round"])]}

    def mediation(state):
        system, user = prompts.mediation(ctx.case, state["before_reviews"], state["transcript"], state["disagreement"])
        memo = ctx.backend.json(f"voting.mediation.R{state['round']}", system, user)
        if (not isinstance(memo.get("diagnosis"), (str, dict)) or not memo["diagnosis"]
                or not isinstance(memo.get("proposed_agreements"), list)
                or not isinstance(memo.get("unresolved"), list)):
            raise ValueError("Mediation must contain diagnosis, proposed_agreements and unresolved")
        return {"memo": memo, "trace": [event("mediation", state["round"])]}

    def state_update(state):
        before = state["before_reviews"]
        evidence = {"speeches": state["transcript"], "nonbinding_mediation": state["memo"]}

        def update(agent):
            system, user = prompts.revise(ctx.case, ctx.agents[agent], before[agent], evidence, "多边讨论与候选调解后的独立更新")
            return validate_review(ctx.backend.json(f"voting.update.R{state['round']}.{agent}", system, user), agent)

        after = dict(zip(ctx.ids, parallel_map(update, ctx.ids, ctx.exp["max_workers"])))
        interaction = {"round_before": state["round"], "round_after": state["round"] + 1,
                       "before_reviews": before, "speeches": state["transcript"], "mediation": state["memo"],
                       "after_reviews": after, "stance_changes": stance_changes(before, after)}
        return {"reviews": after, "round": state["round"] + 1, "interactions": [interaction],
                "trace": [event("state_update", state["round"] + 1)]}

    def finish(state):
        return ctx.finish(state, "voting")

    graph = StateGraph(ConsensusState)
    for name, fn in (("prepare_case", ctx.prepare), ("initialize_agents", ctx.initialize),
                     ("vote_measure", vote_measure), ("identify_disagreement", identify_disagreement),
                     ("debate", debate), ("mediation", mediation), ("state_update", state_update),
                     ("final_determination", finish)):
        graph.add_node(name, fn)
    graph.add_edge(START, "prepare_case")
    graph.add_edge("prepare_case", "initialize_agents")
    graph.add_edge("initialize_agents", "vote_measure")
    graph.add_conditional_edges("vote_measure", lambda s: "finish" if s["stop_reason"] else "continue",
                                {"finish": "final_determination", "continue": "identify_disagreement"})
    graph.add_edge("identify_disagreement", "debate")
    graph.add_edge("debate", "mediation")
    graph.add_edge("mediation", "state_update")
    graph.add_edge("state_update", "vote_measure")
    graph.add_edge("final_determination", END)
    return graph.compile()
