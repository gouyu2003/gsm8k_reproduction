from __future__ import annotations
import copy
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from uuid import uuid4
from . import prompts
from .backends import FakeBackend, OpenAICompatibleBackend, CallFailure
from .budget import Budget, BudgetStop
from .common import atomic_json, digest, file_hash, execution_lock, utcnow
from .config import METHODS, validate, request
from .manifests import load_manifest
from .scoring import score_all, failed_scores
from .storage import Store


def identity(project):
    project = Path(project)
    files = list((project/"gsm_repro").glob("*.py")) + list((project/"author_reference").glob("*.py"))
    lock = project/"requirements-repro.txt"
    if not lock.is_file():
        raise ValueError("Dependency lock is missing")
    files.append(lock)
    return dict(python=platform.python_version(),
                packages=dict(sorted((d.metadata["Name"],d.version) for d in importlib.metadata.distributions())),
                sources={p.relative_to(project).as_posix():file_hash(p) for p in sorted(files)})


def create_run(project, directory, config, manifest_path, *, limit=None):
    project, directory = Path(project).resolve(), Path(directory).resolve()
    validate(config)
    manifest = load_manifest(manifest_path)
    if limit is not None and (not isinstance(limit,int) or limit < 1 or limit > len(manifest["records"])):
        raise ValueError("Limit outside manifest")
    if manifest["split"] == "test":
        if config["stage"] != "formal" or not config["parameters_verified"] or not re.fullmatch(r"[0-9a-f]{64}",config["conditions_sha256"] or ""):
            raise ValueError("Test execution requires a formally frozen, verified configuration")
        if limit is not None:
            raise ValueError("Formal test manifest cannot be silently truncated")
    selected = manifest["records"][:limit] if limit else manifest["records"]
    profile = dict(schema_version=1,config=copy.deepcopy(config),manifest_sha256=manifest["manifest_sha256"],
        selection=[r["qid"] for r in selected],identity=identity(project),prompt_sha256=prompts.fingerprint(),
        methods={k:dict(agents=METHODS[k][0],rounds=METHODS[k][1]) for k in config["methods"]})
    seeds = {}
    for repeat in range(config["repeats"]):
        for row in selected:
            for method in config["methods"]:
                agents, rounds = METHODS[method]
                for round_ in range(rounds):
                    for agent in range(agents):
                        key = f"{method}:{repeat}:{row['qid']}:{round_}:{agent}"
                        seeds[key] = request(config,[],key).get("seed")
    if config["seed_mode"] == "per_call" and len(set(seeds.values())) != len(seeds):
        raise ValueError("Seed collision; choose and record a different base before execution")
    profile["seed_plan"] = seeds
    profile["profile_sha256"] = digest(profile)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Run directory is not empty; use resume or a new directory")
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/"manifest.json",manifest)
    atomic_json(directory/"profile.json",profile)
    store = Store(directory,create=True)
    store.initialize(selected,config["methods"],config["repeats"])
    planned = len(selected)*config["repeats"]*sum(a*r for a,r in (METHODS[m] for m in config["methods"]))
    return dict(run_directory=str(directory),profile_sha256=profile["profile_sha256"],samples=len(selected),
                tasks=len(store.tasks()),normal_generations=planned,backend=config["backend"],model_calls_started=0)


def load_run(project, directory, *, executing=True):
    directory = Path(directory)
    profile = json.loads((directory/"profile.json").read_text(encoding="utf-8"))
    if profile["profile_sha256"] != digest({k:v for k,v in profile.items() if k != "profile_sha256"}):
        raise ValueError("Profile fingerprint mismatch")
    validate(profile["config"])
    manifest = load_manifest(directory/"manifest.json")
    if profile["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Run manifest changed")
    if executing and (profile["identity"] != identity(project) or profile["prompt_sha256"] != prompts.fingerprint()):
        raise ValueError("Runtime changed after planning; keep evidence and create a new run")
    store = Store(directory)
    selected = [r for qid in profile["selection"] for r in manifest["records"] if r["qid"] == qid]
    if len(selected) != len(profile["selection"]) or len(set(profile["selection"])) != len(selected):
        raise ValueError("Run selection is invalid")
    expected = [(f"{method}:{repeat}:{r['qid']}",r["qid"],method,repeat)
                for repeat in range(profile["config"]["repeats"]) for r in selected
                for method in profile["config"]["methods"]]
    actual = [(t["task_key"],t["qid"],t["method"],t["repeat"]) for t in store.tasks()]
    if expected != actual or any(t["status"] not in ("pending","running","completed","failed","interrupted") for t in store.tasks()):
        raise ValueError("Task ledger differs from the frozen run; refusing reconstruction")
    return profile,manifest,store


def complete_task(store,task,row,project,*,duration_ms=None):
    agents, rounds = METHODS[task["method"]]
    calls = [c for c in store.calls(task["task_key"]) if c["status"] == "completed"]
    logical = {(c["round"],c["agent"]):c for c in calls}
    if len(logical) != agents*rounds:
        return False
    outputs = [logical[(rounds-1,a)]["response"]["content"] for a in range(agents)]
    result = dict(qid=row["qid"],method=task["method"],repeat=task["repeat"],status="completed",
        question=row["question"],gold=row["answer"],outputs=outputs,duration_ms=duration_ms,
        scores=score_all(row["answer"],outputs,Path(project)/"author_reference/eval_gsm.py"))
    store.status(task["task_key"],"completed",result=result)
    return True


def fail_task(store,task,row,status,error,*,duration_ms=None):
    result = dict(qid=row["qid"],method=task["method"],repeat=task["repeat"],status=status,
                  question=row["question"],gold=row["answer"],outputs=[],duration_ms=duration_ms,scores=failed_scores(),error=error)
    store.status(task["task_key"],status,result=result,error=error)


def reconcile(store,rows,project):
    # The caller owns the OS execution lock: no living writer can be mistaken for an orphan.
    store.mark_unknown_calls()
    for task in store.tasks():
        if task["status"] == "running":
            if not complete_task(store,task,rows[task["qid"]],project):
                fail_task(store,task,rows[task["qid"]],"interrupted","Unconfirmed execution; not automatically replayed")


def execute(project,directory,*,allow_model_calls=False,allow_test_evaluation=False,backend=None,max_tasks=None):
    project,directory = Path(project),Path(directory)
    profile,manifest,store = load_run(project,directory)
    config = profile["config"]
    real = config["backend"] == "openai_compatible"
    if real and (not allow_model_calls or os.getenv("GSM_REPRO_DISABLE_MODEL_CALLS") == "1"):
        raise ValueError("Explicit --allow-model-calls is required; process guard must also allow it")
    if real and manifest["split"] == "test" and not allow_test_evaluation:
        raise ValueError("Explicit --allow-test-evaluation is required for formal test generation")
    if backend is not None and backend.kind != config["backend"]:
        raise ValueError("Injected backend must match frozen profile")
    with execution_lock(directory):
        rows = {r["qid"]:r for r in manifest["records"]}
        reconcile(store,rows,project)
        budget = Budget(project,config) if real else None
        backend = backend or (OpenAICompatibleBackend(config,allow_model_calls=True) if real else FakeBackend())
        if real:
            endpoint = directory/"endpoint.json"
            marker = dict(endpoint_sha256=backend.endpoint_sha256)
            if endpoint.exists() and json.loads(endpoint.read_text()) != marker:
                raise ValueError("Endpoint changed since first execution")
            if not endpoint.exists():
                atomic_json(endpoint,marker)
        processed,blocked = 0,None
        for task in store.tasks():
            if task["status"] != "pending":
                continue
            if max_tasks is not None and processed >= max_tasks:
                break
            processed += 1
            row = rows[task["qid"]]
            agents,rounds = METHODS[task["method"]]
            contexts = [[prompts.initial(row["question"])] for _ in range(agents)]
            task_started = time.perf_counter()
            store.status(task["task_key"],"running")
            try:
                for round_ in range(rounds):
                    previous = [ctx[2*round_-1]["content"] for ctx in contexts] if round_ else None
                    for agent,context in enumerate(contexts):
                        if round_:
                            peers = [previous[i] for i in range(agents) if i != agent]
                            context.append(prompts.update(peers,row["question"]))
                        logical = f"{task['task_key']}:{round_}:{agent}"
                        body = request(config,copy.deepcopy(context),logical)
                        if body.get("seed") != profile["seed_plan"][logical]:
                            raise ValueError("Seed differs from frozen plan")
                        for attempt in range(1,config["max_retries"]+2):
                            request_id = uuid4().hex
                            if budget:
                                budget.reserve(request_id,body)
                            store.begin_call(request_id,task["task_key"],logical,agent,round_,attempt,body)
                            started = time.perf_counter()
                            try:
                                response = backend.complete(copy.deepcopy(body))
                            except BaseException as exc:
                                elapsed = round((time.perf_counter()-started)*1000)
                                error = exc.document if isinstance(exc,CallFailure) else dict(category=type(exc).__name__,unknown=True,retryable=False)
                                store.end_call(request_id,"unknown" if error.get("unknown") else "failed",elapsed,error=error)
                                known_cost = budget.settle(request_id,None) if budget else True
                                if isinstance(exc,CallFailure) and error["retryable"] and not error["unknown"] and attempt <= config["max_retries"] and known_cost:
                                    time.sleep(config["retry_wait_seconds"])
                                    continue
                                raise
                            store.end_call(request_id,"completed",round((time.perf_counter()-started)*1000),response=response)
                            context.append(dict(role="assistant",content=response["content"]))
                            if budget and not budget.settle(request_id,response):
                                raise BudgetStop("Usage/cost unconfirmed; no further model calls")
                            if response.get("reasoning_state") == "nonempty":
                                raise CallFailure("unexpected_reasoning_mode")
                            break
                if not complete_task(store,task,row,project,duration_ms=round((time.perf_counter()-task_started)*1000)):
                    raise ValueError("Required generation set incomplete")
            except BudgetStop as exc:
                calls = store.calls(task["task_key"])
                if not calls:
                    store.status(task["task_key"],"pending",error=str(exc))
                else:
                    fail_task(store,task,row,"interrupted",str(exc),duration_ms=round((time.perf_counter()-task_started)*1000))
                blocked = str(exc)
                break
            except (KeyboardInterrupt,SystemExit):
                fail_task(store,task,row,"interrupted","Execution interrupted; saved requests are not replayed",duration_ms=round((time.perf_counter()-task_started)*1000))
                raise
            except Exception as exc:
                label = exc.document["category"] if isinstance(exc,CallFailure) else type(exc).__name__
                fail_task(store,task,row,"failed",label,duration_ms=round((time.perf_counter()-task_started)*1000))
                if isinstance(exc,CallFailure) and (exc.document.get("unknown") or
                        exc.document["category"] in ("unexpected_reasoning_mode","configuration_error","quota_error")):
                    blocked = "Unconfirmed request or incompatible model conditions; inspect records before continuing"
                    break
        tasks = store.tasks()
        return dict(processed=processed,blocked=blocked,
                    counts={status:sum(t["status"]==status for t in tasks)
                        for status in ("pending","running","completed","failed","interrupted")},
                    backend=config["backend"],actual_model_requests=sum(1 for _ in store.calls()) if real else 0)
