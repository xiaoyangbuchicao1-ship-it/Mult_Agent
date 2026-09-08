from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import semantic_consensus
import voting_consensus
from shared import prompts
from shared.metrics import summarize_results
from shared.runtime import Context, ModelBackend, fingerprint, load_config, read_key_file, validate_case, validate_config, write_json

ROOT = Path(__file__).resolve().parent
BUILDERS = {"voting": voting_consensus.build_graph, "semantic": semantic_consensus.build_graph}


def table_text(summary, title):
    rounds = len(summary["rows"]["group"])
    lines = [f"# {title}", "", "每格为均值 ± 样本标准差（n−1）；单次运行不报告标准差。", "",
             "| 成员 | " + " | ".join(f"R{i}" for i in range(1, rounds + 1)) + " |",
             "|---|" + "---|" * rounds]
    for agent, values in summary["rows"].items():
        cells = [f"{v['mean']:.4f}" + (f" ± {v['sd']:.4f}" if v["sd"] is not None else "（单次）") for v in values]
        lines.append(f"| {agent} | " + " | ".join(cells) + " |")
    lines += ["", "各列实际样本数：" + "，".join(f"R{v['round']} n={v['n']}" for v in summary["rows"]["group"]), "",
              "R1 为初始状态，R2–R5 为对应干预后的状态。完整轨迹独立重复，不对某一轮脱离上下文反复采样。",
              "语义分支：成员值为其与另外四名成员的对称相似度均值；group 为十对相似度的均值，等于五名成员值的均值。",
              "投票分支：成员值为争点加权的相同立场比例；group 为争点加权的全体两两相同立场比例。",
              "语义值与投票值定义不同，不能直接用数值大小判断哪种方法更好。", ""]
    return "\n".join(lines)


def save_branch(folder, mechanism, result):
    write_json(folder / "result.json", result)
    summary = summarize_results([result])
    write_json(folder / "table6.json", summary)
    title = "表6格式：语义共识（重建结果）" if mechanism == "semantic" else "投票共识轨迹（同型表格，并非论文表6）"
    (folder / "table6.md").write_text(table_text(summary, title), encoding="utf-8")
    if mechanism == "semantic":
        snapshots = result["snapshots"]
        lines = ["# 表5格式：十对对称语义相似度", "", "每对取两种输入顺序的评分均值，不包含自身相似度。", "",
                 "| 代理对 | " + " | ".join(f"R{s['round']}" for s in snapshots) + " |",
                 "|---|" + "---|" * len(snapshots)]
        for pair in snapshots[0]["metrics"]["pair_scores"]:
            lines.append(f"| {pair} | " + " | ".join(f"{s['metrics']['pair_scores'][pair]:.4f}" for s in snapshots) + " |")
        (folder / "table5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_manifest():
    paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "shared").glob("*.py")) + [ROOT / "requirements.txt"]
    lock = ROOT / "requirements.lock.txt"
    if lock.exists():
        paths.append(lock)
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def execute_graph(ctx, mechanism, initial, folder):
    graph = BUILDERS[mechanism](ctx)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    start = {} if initial is None else {"reviews": copy.deepcopy(initial)}
    result = None
    seen_rounds = 0
    for step, state in enumerate(graph.stream(start, {"recursion_limit": max(100, 20 * ctx.exp["max_interactions"] + 30)}, stream_mode="values")):
        result = state
        node = (state.get("trace") or [{}])[-1].get("node", "start").split(":", 1)[0]
        write_json(folder / "checkpoints" / f"{step:03d}_{node}.json", state)
        if len(state.get("snapshots", [])) > seen_rounds:
            seen_rounds = len(state["snapshots"])
            snap = state["snapshots"][-1]
            print(f"{mechanism} R{snap['round']}: group={snap['metrics']['group_consensus']:.4f}", flush=True)
    if result is None or "final_determination" not in result:
        raise RuntimeError("Graph did not reach final determination")
    save_branch(folder, mechanism, result)
    return result


def load_initial_reviews(path, case):
    initial = json.loads(path.read_text(encoding="utf-8"))
    metadata_path = path.with_suffix(".meta.json")
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (meta.get("case_sha256") != fingerprint(case)
                or meta.get("initial_reviews_sha256") != fingerprint(initial)):
            raise ValueError("Initial snapshot case/role provenance or content hash does not match")
    else:
        source_case_path = path.parent.parent / "case.json"
        if source_case_path.exists():
            source_case = json.loads(source_case_path.read_text(encoding="utf-8"))
            if fingerprint(source_case) != fingerprint(case):
                raise ValueError("Initial snapshot belongs to a different case or role profile")
        elif case.get("role_profile") == "paper_appendix_v1":
            raise ValueError("Paper-role initial snapshot requires matching case provenance; generate new opinions")
    return initial


def save_initial(folder, initial, case):
    write_json(folder / "initial_reviews.json", initial)
    write_json(folder / "initial_reviews.meta.json", {
        "case_sha256": fingerprint(case), "initial_reviews_sha256": fingerprint(initial),
        "role_profile": case.get("role_profile", "legacy_three_two"),
    })


def run_batch(cfg, case, mechanisms, runs, output, key_override=None, initial=None, *, initial_case_sha256=None):
    validate_case(case)
    validate_config(cfg)
    if initial is not None and (case.get("role_profile") == "paper_appendix_v1" or initial_case_sha256 is not None):
        if initial_case_sha256 != fingerprint(case):
            raise ValueError("External initial opinions require matching case/role provenance")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_role": cfg["model_role"], "runs_requested": runs, "mechanisms": mechanisms,
        "role_profile": case.get("role_profile", "legacy_three_two"),
        "prompt_version": prompts.VERSION, "config_sha256": fingerprint(cfg), "case_sha256": fingerprint(case),
        "sources_sha256": source_manifest(),
        "packages": {p: importlib.metadata.version(p) for p in ("langgraph", "langchain-openai", "langchain-core", "openai")},
        "initialization": "externally fixed initial opinions; conditional variability" if initial is not None else "new LLM initialization per repetition; shared only within a repetition",
        "limitations": ["Reconstructed implementation, not recovered original source", "Same-model judge is not independent validation",
                        "Routing seed controls tie-breaking only, not LLM response determinism"],
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.json", cfg)
    write_json(output / "case.json", case)
    by_mechanism = {m: [] for m in mechanisms}
    try:
        for index in range(1, runs + 1):
            print(f"Repetition {index}/{runs}", flush=True)
            run_folder = output / f"run_{index:02d}"
            backend = ModelBackend(cfg, run_folder / "llm_calls.jsonl", key_override)
            shared_initial = copy.deepcopy(initial)
            for mechanism in mechanisms:
                ctx = Context(cfg, case, backend)
                result = execute_graph(ctx, mechanism, shared_initial, run_folder / mechanism)
                observed_initial = result["snapshots"][0]["reviews"]
                if shared_initial is None:
                    shared_initial = copy.deepcopy(observed_initial)
                    save_initial(run_folder, shared_initial, case)
                elif observed_initial != shared_initial:
                    raise AssertionError("The branches must share only the exact initial opinions")
                by_mechanism[mechanism].append(result)
            save_initial(run_folder, shared_initial, case)
            write_json(run_folder / "run_metadata.json", {
                "initial_reviews_sha256": fingerprint(shared_initial), "application_calls": len(backend.calls),
                "application_errors": sum(r["status"] != "ok" for r in backend.calls),
                "note": "SDK transport retries may occur inside one application call; all application outputs are logged",
            })
        for mechanism, results in by_mechanism.items():
            summary = summarize_results(results)
            write_json(output / f"{mechanism}_summary.json", summary)
            (output / f"{mechanism}_summary.md").write_text(table_text(summary, f"{mechanism}：{cfg['model_role']}"), encoding="utf-8")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__,
                         "note": "No values fabricated or imputed. Inspect retained raw outputs/checkpoints; failed batches are not silently excluded."})
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description="Run the two independent LangGraph mechanisms and save reproducible evidence.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "lawllm_vllm.json")
    parser.add_argument("--mechanism", choices=["both", "voting", "semantic"], default="both")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--initial-reviews-file", type=Path)
    parser.add_argument("--max-interactions", type=int)
    parser.add_argument("--window", choices=["fixed", "adaptive"])
    parser.add_argument("--group-threshold", type=float)
    parser.add_argument("--route-mode", choices=["lowest_pair", "random_pair", "historical_manual"])
    parser.add_argument("--feedback-mode", choices=["all_agents", "pair_only"])
    parser.add_argument("--plan", action="store_true", help="Compile and inspect graphs without any model calls")
    args = parser.parse_args()
    cfg, case = load_config(args.config.resolve(), ROOT)
    for name in ("max_interactions", "window", "group_threshold", "route_mode", "feedback_mode"):
        value = getattr(args, name)
        if value is not None:
            cfg["experiment"][name] = value
    validate_config(cfg)
    if args.runs < 1:
        parser.error("--runs must be positive")
    mechanisms = list(BUILDERS) if args.mechanism == "both" else [args.mechanism]
    if args.plan:
        for mechanism in mechanisms:
            graph = BUILDERS[mechanism](Context(cfg, case, None))
            print(json.dumps({"mechanism": mechanism, "nodes": list(graph.get_graph().nodes),
                              "model": cfg["generation"]["model"], "experiment": cfg["experiment"],
                              "agents": [{k: a[k] for k in ("id", "name", "faction")} for a in case["agents"]]}, ensure_ascii=False))
        return
    try:
        initial = load_initial_reviews(args.initial_reviews_file.resolve(), case) if args.initial_reviews_file else None
    except ValueError as exc:
        parser.error(str(exc))
    key = read_key_file(args.api_key_file)
    output = ROOT / "outputs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    print(f"Output: {output}", flush=True)
    try:
        run_batch(cfg, case, mechanisms, args.runs, output, key, initial,
                  initial_case_sha256=fingerprint(case) if initial is not None else None)
    except Exception as exc:
        print(f"FAILED ({type(exc).__name__}); evidence retained in {output}", flush=True)
        raise SystemExit(1) from None
    print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
