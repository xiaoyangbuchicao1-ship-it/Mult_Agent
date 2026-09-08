from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import DefaultHttpxClient

from . import prompts


FINAL_DETERMINATION_SCHEMA = {
    "type": "object",
    "properties": {
        "disposition": {"type": "string", "minLength": 1},
        "reasoning": {"type": "string", "minLength": 1},
        "issue_findings": {
            "type": "array", "minItems": 10, "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer", "minimum": 1, "maximum": 10},
                               "finding": {"type": "string", "minLength": 1}},
                "required": ["id", "finding"], "additionalProperties": False,
            },
        },
        "unresolved_disagreements": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["disposition", "reasoning", "issue_findings", "unresolved_disagreements"],
    "additionalProperties": False,
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_config(path: Path, root: Path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    case = json.loads((root / cfg["case_file"]).read_text(encoding="utf-8"))
    validate_case(case)
    validate_config(cfg)
    return cfg, case


def validate_config(cfg):
    exp = cfg["experiment"]
    for key in ("max_interactions", "bilateral_turns", "max_workers", "summary_max_tokens", "dialogue_max_tokens"):
        minimum = 0 if key == "max_interactions" else 1
        if type(exp[key]) is not int or exp[key] < minimum:
            raise ValueError(f"Invalid experiment.{key}")
    if exp["window"] not in {"fixed", "adaptive"}:
        raise ValueError("window must be fixed or adaptive")
    if exp["window"] == "adaptive" and not isinstance(exp["group_threshold"], (float, int)):
        raise ValueError("Adaptive runs require an explicit group_threshold")
    for key in ("gamma", "group_threshold"):
        value = exp[key]
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{key} must be in [0,1]")
    if exp["route_mode"] not in {"lowest_pair", "random_pair", "historical_manual"}:
        raise ValueError("Unknown route mode")
    if exp["route_mode"] == "historical_manual" and exp["max_interactions"] > 4:
        raise ValueError("Historical manual route has four interactions")
    if exp["feedback_mode"] not in {"all_agents", "pair_only"}:
        raise ValueError("Unknown feedback mode")
    if exp.get("pair_writeback_mode", "baseline") not in {"baseline", "agreement_aware_v1"}:
        raise ValueError("Unknown pair writeback mode")
    if exp.get("bilateral_prompt_mode", "baseline") not in {"baseline", "appendix_consensus_v1"}:
        raise ValueError("Unknown bilateral prompt mode")
    for role in ("generation", "judge"):
        value = cfg[role]
        if any(k in value for k in ("api_key", "token", "password")):
            raise ValueError("Credentials belong in environment variables, not configuration")
        if not value["model"] or not value["base_url"].startswith(("http://", "https://")):
            raise ValueError("Invalid model or endpoint")
        if not 0 <= value["temperature"] <= 2 or not 0 < value["top_p"] <= 1:
            raise ValueError("Invalid sampling parameters")


def validate_case(case):
    agents = case["agents"]
    if {a["id"] for a in agents} != {f"a{i}" for i in range(1, 6)} or len(agents) != 5:
        raise ValueError("This reconstruction requires exactly a1-a5")
    profile = case.get("role_profile", "legacy_three_two")
    expected = {
        "legacy_three_two": ["claimant", "claimant", "claimant", "respondent", "respondent"],
        "paper_appendix_v1": ["neutral", "claimant", "claimant", "respondent", "respondent"],
    }
    if profile not in expected:
        raise ValueError("Unknown role profile")
    actual = {a["id"]: a["faction"] for a in agents}
    if actual != dict(zip((f"a{i}" for i in range(1, 6)), expected[profile])):
        raise ValueError(f"Agent identities do not match role profile: {profile}")
    if any(not isinstance(a.get("role_prompt"), str) or not a["role_prompt"].strip() for a in agents):
        raise ValueError("Every agent needs an explicit role prompt")
    if [q["id"] for q in case["issues"]] != list(range(1, 11)):
        raise ValueError("Expected the frozen Q1-Q10 definitions")
    weights = [q["weight"] for q in case["issues"]]
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("Issue weights must be nonnegative and sum to 1")


def validate_review(value, agent_id):
    if value.get("agent_id") != agent_id or not isinstance(value.get("overall_opinion"), str) or not value["overall_opinion"].strip():
        raise ValueError(f"Invalid review identity or overall opinion: {agent_id}")
    items = value.get("issues")
    if not isinstance(items, list) or len(items) != 10:
        raise ValueError(f"{agent_id} must return ten issues")
    for item in items:
        if (type(item.get("id")) is not int or type(item.get("stance")) is not int
                or item["stance"] not in {0, 1}):
            raise ValueError(f"{agent_id} returned an invalid binary stance")
        confidence = item.get("confidence")
        if (type(confidence) not in {int, float} or not math.isfinite(confidence)
                or not 0 <= confidence <= 1 or not isinstance(item.get("reason"), str) or not item["reason"].strip()):
            raise ValueError(f"{agent_id} returned an invalid confidence/reason")
    if sorted(q["id"] for q in items) != list(range(1, 11)):
        raise ValueError(f"{agent_id} returned missing or duplicate issues")
    result = copy.deepcopy(value)
    result["issues"] = sorted(items, key=lambda q: q["id"])
    return result


def parallel_map(fn, values, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, values))


def read_key_file(path: Path | None):
    if path is None:
        return None
    key = path.read_text(encoding="utf-8-sig").strip()
    if key.startswith(("DEEPSEEK_API_KEY=", "OPENAI_API_KEY=")):
        key = key.split("=", 1)[1].strip().strip("\"'")
    if not key or "\n" in key:
        raise ValueError("Key file must contain one key, not a script")
    return key


class ModelBackend:
    def __init__(self, cfg, log_path: Path, key_override=None):
        self.cfg = cfg
        self.models = {}
        self.log_path = log_path
        self.lock = threading.Lock()
        self.counter = 0
        self.calls = []
        log_path.parent.mkdir(parents=True, exist_ok=True)
        for role in ("generation", "judge"):
            options = cfg[role]
            shared_credentials = (options["base_url"] == cfg["generation"]["base_url"]
                                  and options["api_key_env"] == cfg["generation"]["api_key_env"])
            key = os.environ.get(options["api_key_env"])
            if not key and (role == "generation" or shared_credentials):
                key = key_override
            if not key and options.get("allow_empty_key"):
                key = "EMPTY"
            if not key:
                raise ValueError(f"Set {options['api_key_env']} or provide the matching key file")
            local_endpoint = urlparse(options["base_url"]).hostname in {"localhost", "127.0.0.1", "::1"}
            transport = {"http_client": DefaultHttpxClient(trust_env=False)} if local_endpoint else {}
            self.models[role] = ChatOpenAI(
                model=options["model"], base_url=options["base_url"], api_key=key,
                temperature=options["temperature"], top_p=options["top_p"],
                max_tokens=options["max_tokens"], timeout=180, max_retries=2,
                use_responses_api=False, extra_body=options.get("extra_body", {}),
                **transport,
            )
        self.template = ChatPromptTemplate.from_messages([("system", "{system_text}"), ("human", "{user_text}")])

    def _log(self, record):
        with self.lock:
            self.calls.append(record)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def text(self, task, system, user, *, role="generation", max_tokens=None, json_mode=False):
        with self.lock:
            self.counter += 1
            call_id = self.counter
        options = self.cfg[role]
        limit = max_tokens or options["max_tokens"]
        bound = {"max_tokens": limit}
        if json_mode:
            bound["response_format"] = {"type": "json_object"}
            if task.endswith(".final") and options.get("structured_final", False):
                bound["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "final_determination", "schema": FINAL_DETERMINATION_SCHEMA},
                }
        record = {"call_id": call_id, "task": task, "role": role,
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "model_requested": options["model"], "base_url": options["base_url"],
                  "temperature": options["temperature"], "top_p": options["top_p"],
                  "max_tokens": limit, "extra_body": options.get("extra_body", {}),
                  "response_format": bound.get("response_format"),
                  "json_mode": json_mode, "system": system, "user": user}
        started = time.monotonic()
        try:
            message = (self.template | self.models[role].bind(**bound)).invoke({"system_text": system, "user_text": user})
            if not isinstance(message.content, str) or not message.content.strip():
                raise ValueError("Expected nonempty textual model output")
            record.update({"output": message.content, "response_metadata": message.response_metadata,
                           "usage": message.usage_metadata, "message_id": message.id})
            if message.response_metadata.get("finish_reason") == "length":
                raise ValueError("Truncated output; increase the token limit before rerunning")
            record["status"] = "ok"
            return message.content
        except Exception as exc:
            record.update({"status": "error", "error_type": type(exc).__name__})
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self._log(record)

    def json(self, task, system, user, **kwargs):
        raw = self.text(task, system, user, json_mode=True, **kwargs).strip()
        if raw.startswith("```json") and raw.endswith("```"):
            raw = raw[7:-3].strip()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value


class ConsensusState(TypedDict, total=False):
    reviews: dict
    round: int
    metrics: dict
    disagreement: dict
    summaries: dict
    directed_scores: dict
    raw_scores: dict
    pair: list[str]
    selection: dict
    transcript: list[dict]
    before_reviews: dict
    pair_reviews: dict
    memo: dict
    stop_reason: str | None
    final_determination: dict
    snapshots: Annotated[list[dict], operator.add]
    interactions: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]


def event(node, round_number):
    return {"node": node, "round": round_number, "utc": datetime.now(timezone.utc).isoformat()}


def stop_reason(state, exp):
    if exp["window"] == "adaptive" and state["metrics"]["group_consensus"] >= exp["group_threshold"]:
        return "group_threshold"
    if state["round"] - 1 >= exp["max_interactions"]:
        return "max_interactions"
    return None


def stance_changes(before, after):
    changes = []
    for agent in sorted(before):
        previous = {q["id"]: q for q in before[agent]["issues"]}
        for item in after[agent]["issues"]:
            old = previous[item["id"]]
            if old["stance"] != item["stance"]:
                changes.append({"agent": agent, "issue": item["id"], "before": old["stance"],
                                "after": item["stance"], "old_reason": old["reason"], "new_reason": item["reason"]})
    return changes


class Context:
    def __init__(self, cfg, case, backend):
        self.cfg, self.case, self.backend = cfg, copy.deepcopy(case), backend
        self.exp = cfg["experiment"]
        self.agents = {a["id"]: a for a in case["agents"]}
        self.ids = sorted(self.agents)

    def prepare(self, state):
        validate_case(self.case)
        return {"trace": [event("prepare_case:provided_structured_input", 1)]}

    def initialize(self, state):
        reviews = state.get("reviews")
        if reviews is None:
            def generate(agent):
                system, user = prompts.initial(self.case, self.agents[agent])
                return validate_review(self.backend.json(f"initial.{agent}", system, user), agent)
            reviews = dict(zip(self.ids, parallel_map(generate, self.ids, self.exp["max_workers"])))
        if set(reviews) != set(self.ids):
            raise ValueError("Initial snapshot must contain exactly a1-a5")
        reviews = {a: validate_review(reviews[a], a) for a in self.ids}
        return {"reviews": copy.deepcopy(reviews), "round": 1, "stop_reason": None,
                "trace": [event("initialize_agents", 1)]}

    def finish(self, state, mechanism):
        system, user = prompts.determination(self.case, mechanism, state["reviews"])
        value = self.backend.json(f"{mechanism}.final", system, user)
        findings = value.get("issue_findings", [])
        if (not isinstance(findings, list) or len(findings) != 10
                or any(not isinstance(q, dict) or type(q.get("id")) is not int
                       or not isinstance(q.get("finding"), str) or not q["finding"].strip() for q in findings)
                or {q["id"] for q in findings} != set(range(1, 11))
                or not isinstance(value.get("disposition"), str) or not value["disposition"].strip()
                or not isinstance(value.get("reasoning"), str) or not value["reasoning"].strip()
                or not isinstance(value.get("unresolved_disagreements"), list)
                or any(not isinstance(item, str) for item in value["unresolved_disagreements"])):
            raise ValueError("Final determination must retain all Q1-Q10 and unresolved disagreements")
        return {"final_determination": value, "trace": [event("final_determination", state["round"])]}

    def snapshot(self, state, metrics, **extra):
        return {"round": state["round"], "reviews": copy.deepcopy(state["reviews"]),
                "metrics": metrics, **extra}
