from __future__ import annotations

import itertools
import math
import re
from statistics import mean, stdev


def parse_score(text: str) -> float:
    match = re.fullmatch(r"\s*<score>\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*</score>\s*", text)
    if not match:
        raise ValueError("Expected exactly one <score> value in [0, 1]")
    return float(match[1])


def pair_key(a: str, b: str) -> str:
    return "__".join(sorted((a, b)))


def voting_metrics(reviews: dict, issues: list[dict]) -> dict:
    agents = sorted(reviews)
    n = len(agents)
    weights = {q["id"]: q["weight"] for q in issues}
    if n < 2 or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Need at least two agents and weights summing to 1")
    positions = {a: {q["id"]: q["stance"] for q in reviews[a]["issues"]} for a in agents}
    matrices, per_issue, counts = {}, {}, {}
    opposition = dict.fromkeys(agents, 0)
    per_agent = dict.fromkeys(agents, 0.0)
    for qid, weight in weights.items():
        matrix = [[None if a == b else int(positions[a][qid] == positions[b][qid])
                   for b in agents] for a in agents]
        matrices[str(qid)] = matrix
        per_issue[str(qid)] = sum(v for row in matrix for v in row if v is not None) / (n * (n - 1))
        counts[str(qid)] = {"ones": sum(positions[a][qid] for a in agents), "n": n}
        for i, a in enumerate(agents):
            agreements = sum(v for v in matrix[i] if v is not None)
            opposition[a] += n - 1 - agreements
            per_agent[a] += weight * agreements / (n - 1)
    total = sum(opposition.values())
    return {
        "definition": "binary stance equality; diagonal excluded",
        "matrices": matrices,
        "per_issue_consensus": per_issue,
        "stance_counts": counts,
        "per_agent_consensus": per_agent,
        "group_consensus": sum(weights[q] * per_issue[str(q)] for q in weights),
        "opposition": opposition,
        "inconsistency_index": {a: opposition[a] / total if total else 0.0 for a in agents},
        "outliers": [a for a in agents if total and opposition[a] == max(opposition.values())],
    }


def semantic_metrics(agents: list[str], directed: dict[str, float], gamma: float) -> dict:
    expected = {f"{a}->{b}" for a, b in itertools.permutations(agents, 2)}
    if set(directed) != expected or any(not math.isfinite(v) or not 0 <= v <= 1 for v in directed.values()):
        raise ValueError("Need every directed non-diagonal score, each in [0,1]")
    pairs = {pair_key(a, b): mean([directed[f"{a}->{b}"], directed[f"{b}->{a}"]])
             for a, b in itertools.combinations(sorted(agents), 2)}
    per_agent = {a: mean([pairs[pair_key(a, b)] for b in agents if a != b]) for a in agents}
    low_edges = [{"pair": key.split("__"), "score": value} for key, value in pairs.items() if value < gamma]
    degrees = {a: sum(a in edge["pair"] for edge in low_edges) for a in agents}
    group = mean(pairs.values())
    if not math.isclose(group, mean(per_agent.values()), abs_tol=1e-12):
        raise AssertionError("Agent mean and pair mean must agree")
    return {
        "pair_scores": pairs,
        "per_agent_consensus": per_agent,
        "group_consensus": group,
        "gamma": gamma,
        "low_similarity_edges": low_edges,
        "low_similarity_degree": degrees,
        "outliers": [a for a in agents if low_edges and degrees[a] == max(degrees.values())],
    }


def summarize_results(results: list[dict]) -> dict:
    rows = sorted(results[0]["snapshots"][0]["metrics"]["per_agent_consensus"]) + ["group"]
    output = {"runs": len(results), "sd_definition": "sample, n-1; null when n<2", "rows": {}}
    rounds = max(len(r["snapshots"]) for r in results)
    for row in rows:
        output["rows"][row] = []
        for idx in range(rounds):
            values = []
            for result in results:
                if idx < len(result["snapshots"]):
                    metric = result["snapshots"][idx]["metrics"]
                    values.append(metric["group_consensus"] if row == "group" else metric["per_agent_consensus"][row])
            output["rows"][row].append({"round": idx + 1, "n": len(values), "values": values,
                                        "mean": mean(values), "sd": stdev(values) if len(values) > 1 else None})
    return output
