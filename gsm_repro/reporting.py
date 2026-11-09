from __future__ import annotations
import csv
import json
from pathlib import Path
from .common import atomic_json
from .engine import load_run
from .scoring import summary, failed_scores


def export_report(project,directory):
    directory = Path(directory)
    profile,manifest,store = load_run(project,directory,executing=False)
    rows_by_id = {r["qid"]:r for r in manifest["records"]}
    results = []
    for task in store.tasks():
        if task["result_json"]:
            result = json.loads(task["result_json"])
        else:
            row = rows_by_id[task["qid"]]
            # Unstarted items stay in the denominator; a partial report is labelled partial.
            result = dict(qid=task["qid"],method=task["method"],repeat=task["repeat"],status=task["status"],
                question=row["question"],gold=row["answer"],outputs=[],
                scores={**failed_scores(),"author_faithful":dict(correct=None,status="not_yet_scored")})
        results.append(result)
    complete = all(t["status"] in ("completed","failed","interrupted") for t in store.tasks())
    calls = store.calls()
    report = dict(schema_version=1,backend=profile["config"]["backend"],complete=complete,
        scoring_version=profile.get("scoring_version", "gsm8k-fixed-v1-string-vote"),
        profile_sha256=profile["profile_sha256"],manifest_sha256=profile["manifest_sha256"],
        methods=summary(results),request_attempts=len(calls),
        actual_model_requests=len(calls) if profile["config"]["backend"] == "openai_compatible" else 0,
        unknown_requests=sum(c["status"]=="unknown" for c in calls),
        usage={key:sum(c["response"]["usage"][key] for c in calls) if calls and all(
            c["response"] and isinstance(c["response"].get("usage",{}).get(key),int) for c in calls) else None
            for key in ("prompt_tokens","completion_tokens","total_tokens")},
        note="Offline fake results are execution checks, not model-quality evidence.")
    for method, metrics in report["methods"].items():
        own_calls = [c for c in calls if c["task_key"].startswith(method+":")]
        durations = [r["duration_ms"] for r in results if r["method"]==method and r.get("duration_ms") is not None]
        metrics["request_attempts"] = len(own_calls)
        metrics["request_time_ms"] = sum(c["duration_ms"] or 0 for c in own_calls)
        metrics["mean_task_duration_ms"] = sum(durations)/len(durations) if durations else None
        metrics["duration_samples"] = len(durations)
        metrics["truncated_generations"] = sum(c["response"] is not None and c["response"].get("finish_reason")=="length" for c in own_calls)
        metrics["usage"] = {key:sum(c["response"]["usage"][key] for c in own_calls)
            if own_calls and all(c["response"] and isinstance(c["response"].get("usage",{}).get(key),int) for c in own_calls)
            else None for key in ("prompt_tokens","completion_tokens","total_tokens")}
    with (directory/"calls.jsonl").open("w",encoding="utf-8",newline="\n") as handle:
        for call in calls:
            handle.write(json.dumps({k:v for k,v in call.items() if not k.endswith("_json")},
                                    ensure_ascii=False,allow_nan=False)+"\n")
    atomic_json(directory/"report.json",report)
    with (directory/"results.jsonl").open("w",encoding="utf-8",newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    with (directory/"results.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid","method","repeat","status","fixed_correct","faithful_correct","tie_error_correct","exact_correct","question","gold","outputs"])
        for row in results:
            data = [row["qid"],row["method"],row["repeat"],row["status"],
                *[row["scores"][v]["correct"] for v in ("fixed","author_faithful","sensitivity.tie_error","sensitivity.exact")],
                row["question"],row["gold"],json.dumps(row["outputs"],ensure_ascii=False)]
            writer.writerow(["'"+v if isinstance(v,str) and v.startswith(("=","+","-","@","\t","\r")) else v for v in data])
    return report
