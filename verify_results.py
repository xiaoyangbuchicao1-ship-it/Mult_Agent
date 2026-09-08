from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_consensus import choose_pair
from shared.metrics import parse_score, semantic_metrics, summarize_results, voting_metrics
from shared.runtime import fingerprint, validate_case, validate_review
from run_experiment import load_initial_reviews, table_text


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_batch(folder):
    cfg, case, manifest = read(folder / "config.json"), read(folder / "case.json"), read(folder / "manifest.json")
    validate_case(case)
    assert manifest["status"] == "complete", "An incomplete batch is not a validated experiment"
    assert fingerprint(cfg) == manifest["config_sha256"] and fingerprint(case) == manifest["case_sha256"]
    runs = sorted(folder.glob("run_*"))
    assert len(runs) == manifest["runs_requested"]
    results = {m: [] for m in manifest["mechanisms"]}
    snapshots_checked = 0
    for run in runs:
        initial = load_initial_reviews(run / "initial_reviews.json", case)
        assert fingerprint(initial) == read(run / "run_metadata.json")["initial_reviews_sha256"]
        records = [json.loads(line) for line in (run / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
        calls = {row["task"]: row for row in records}
        if case.get("role_profile") == "paper_appendix_v1":
            agents = {a["id"]: a for a in case["agents"]}
            own_tasks = ("initial.", "voting.debate.", "voting.update.", "semantic.dialogue.",
                         "semantic.pair_update.", "semantic.observer.")
            for row in records:
                if row["task"].startswith(own_tasks):
                    agent = agents[row["task"].rsplit(".", 1)[-1]]
                    assert agent["role_prompt"] in row["user"], "Agent role missing from actual call"
                    if agent["faction"] == "neutral" and row["task"].startswith("semantic.dialogue."):
                        assert "代表指定一方" not in row.get("system", "")
        for mechanism in results:
            result = read(run / mechanism / "result.json")
            assert result["snapshots"][0]["reviews"] == initial
            assert len(result["snapshots"]) == len(result.get("interactions", [])) + 1
            if cfg["experiment"]["window"] == "fixed":
                assert len(result["snapshots"]) == cfg["experiment"]["max_interactions"] + 1
            for number, snap in enumerate(result["snapshots"], start=1):
                assert snap["round"] == number
                for agent, opinion in snap["reviews"].items():
                    validate_review(opinion, agent)
                if mechanism == "voting":
                    calculated = voting_metrics(snap["reviews"], case["issues"])
                else:
                    directed = {k: parse_score(v) for k, v in snap["raw_scores"].items()}
                    assert directed == snap["directed_scores"]
                    for key, raw in snap["raw_scores"].items():
                        assert calls[f"semantic.score.R{number}.{key}"]["output"] == raw
                    calculated = semantic_metrics(sorted(initial), directed, cfg["experiment"]["gamma"])
                    assert snap["stance_metrics"] == voting_metrics(snap["reviews"], case["issues"])
                assert calculated == snap["metrics"], f"Metric mismatch: {run.name} {mechanism} R{number}"
                snapshots_checked += 1
            for idx, interaction in enumerate(result.get("interactions", [])):
                assert interaction["before_reviews"] == result["snapshots"][idx]["reviews"]
                assert interaction["after_reviews"] == result["snapshots"][idx + 1]["reviews"]
                if mechanism == "semantic":
                    assert interaction["selection"] == choose_pair(result["snapshots"][idx]["metrics"], cfg["experiment"], idx + 1)
            assert read(run / mechanism / "table6.json") == summarize_results([result])
            results[mechanism].append(result)
    for mechanism, values in results.items():
        summary = summarize_results(values)
        assert read(folder / f"{mechanism}_summary.json") == summary
        assert (folder / f"{mechanism}_summary.md").read_text(encoding="utf-8") == table_text(summary, f"{mechanism}：{cfg['model_role']}")
    return {"verified": True, "runs": len(runs), "snapshots": snapshots_checked, "mechanisms": list(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute every exported number from saved opinions and raw score strings.")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_batch(args.folder.resolve()), ensure_ascii=False))
