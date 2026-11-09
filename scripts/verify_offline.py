"""Run step 3 acceptance with fake responses and a network guard in every child."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsm_repro.common import atomic_json, file_hash
from gsm_repro.scoring import SCORING_VERSION


GUARD = '''import os,socket
def blocked(*args,**kwargs):
    with open(os.environ["GSM_REPRO_NETWORK_GUARD_LOG"],"a",encoding="utf-8") as log:
        log.write("blocked network attempt\\n")
    raise AssertionError("Network is forbidden in offline acceptance")
socket.socket.connect=blocked
socket.socket.connect_ex=blocked
socket.getaddrinfo=blocked
socket.create_connection=blocked
'''

UNIT_RUNNER = '''import io,json,sys,tempfile,unittest
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);tmp=Path(sys.argv[3])
tempfile.tempdir=str(tmp)
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(root/"tests")))
out.with_suffix(".log").write_text(stream.getvalue(),encoding="utf-8")
out.write_text(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    skipped=len(result.skipped),passed=result.wasSuccessful()),indent=2)+"\\n",encoding="utf-8")
print(out.read_text())
raise SystemExit(0 if result.wasSuccessful() else 1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / ("step3-offline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--data", type=Path, default=ROOT.parent / "llm-multiagent_debate_needData")
    parser.add_argument("--author", type=Path, default=ROOT.parent / "llm_multiagent_debate")
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.author = args.author.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty acceptance output directory")
    output.mkdir(parents=True, exist_ok=True)
    guard = output / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(GUARD, encoding="utf-8")
    temporary = output / "tmp"
    temporary.mkdir()
    network_log = output / "network-attempts.log"
    network_log.write_text("", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(GSM_REPRO_DISABLE_MODEL_CALLS="1", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=os.pathsep.join((str(guard), str(ROOT))),
                       GSM_REPRO_NETWORK_GUARD_LOG=str(network_log))

    def child(arguments, label):
        completed = subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT, env=environment,
                                   capture_output=True, text=True, encoding="utf-8", timeout=120)
        (output / (label + ".stdout.log")).write_text(completed.stdout, encoding="utf-8")
        (output / (label + ".stderr.log")).write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError("Offline acceptance failed at " + label + "; inspect its saved logs")
        return json.loads(completed.stdout)

    def cli(label, *arguments):
        return child(["-m", "gsm_repro", *arguments], label)

    child(["-c", "import json,socket,os; assert socket.socket.connect.__module__=='sitecustomize'; "
           "assert socket.getaddrinfo.__module__=='sitecustomize'; "
           "assert os.environ['GSM_REPRO_DISABLE_MODEL_CALLS']=='1'; print(json.dumps({'network_guard':'active'}))"], "guard-check")
    references = [args.data / name for name in ("train.jsonl", "test.jsonl")]
    references += [args.author / name for name in ("gsm/gen_gsm.py", "gsm/eval_gsm.py", "requirements.txt", "README.md")]
    before = {str(path.resolve()): file_hash(path) for path in references}
    prepared = cli("prepare", "prepare", "--data", str(args.data), "--author", str(args.author))
    tests = child(["-c", UNIT_RUNNER, str(ROOT), str(output / "unit-tests.json"), str(temporary)], "unit-tests")

    run = output / "cli-run"
    plan = cli("plan", "plan", "--run-dir", str(run), "--config", str(ROOT / "configs/development.fake.json"))
    if plan["backend"] != "fake" or plan["model_calls_started"] != 0:
        raise AssertionError("Acceptance profile must be fake-only")
    first = cli("run-first", "run", "--run-dir", str(run), "--max-tasks", "2")
    resumed = cli("resume", "resume", "--run-dir", str(run))
    saved_calls = (run / "calls.jsonl").read_bytes()
    repeated = cli("resume-again", "resume", "--run-dir", str(run))
    if saved_calls != (run / "calls.jsonl").read_bytes() or repeated["processed"] != 0:
        raise AssertionError("Repeated resume must not add generations")
    report = cli("report", "report", "--run-dir", str(run))
    status = cli("status", "status", "--run-dir", str(run))
    cli("score", "score", "--input", str(run / "results.jsonl"), "--output", str(run / "rescored.json"))
    calls = [json.loads(line) for line in saved_calls.decode("utf-8").splitlines()]
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    rescored = json.loads((run / "rescored.json").read_text(encoding="utf-8"))["results"]
    samples = plan["samples"]
    expected = dict(single=samples, reflection=samples * 2, majority=samples * 6, debate=samples * 6)
    actual = {method: sum(call["task_key"].startswith(method + ":") for call in calls) for method in expected}
    if actual != expected or len(rows) != samples * 4 or not all(row["status"] == "completed" for row in rows):
        raise AssertionError("Four-method fake generation structure is incorrect")
    if any(not all(key in call for key in ("request_call_id", "task_key", "logical_key", "agent", "round", "attempt", "request", "response")) for call in calls):
        raise AssertionError("Per-agent/per-round trace fields are missing")
    if any("####" in json.dumps(call["request"]) for call in calls):
        raise AssertionError("Gold labels entered model inputs")
    if [row["scores"] for row in rows] != [row["scores"] for row in rescored]:
        raise AssertionError("Offline rescoring differs from execution scoring")
    if report["actual_model_requests"] != 0 or not report["complete"] or report["scoring_version"] != SCORING_VERSION:
        raise AssertionError("Report identity or fake-only accounting mismatch")
    after = {str(path.resolve()): file_hash(path) for path in references}
    if before != after:
        raise AssertionError("Author source or original GSM8K data changed")
    network_attempts = len(network_log.read_text(encoding="utf-8").splitlines())
    if network_attempts:
        raise AssertionError("An offline child attempted a network connection")
    packages = {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()}
    sources = {path.relative_to(ROOT).as_posix(): file_hash(path)
               for folder in ("gsm_repro", "tests", "scripts") for path in sorted((ROOT / folder).glob("*.py"))}
    sources["requirements-repro.txt"] = file_hash(ROOT / "requirements-repro.txt")
    verification = dict(schema_version=1, step=3, status="passed", recorded_at=datetime.now(timezone.utc).isoformat(),
        python=sys.version, sdk=packages.get("openai"), environment_packages=packages,
        unit_tests=tests, scoring_version=SCORING_VERSION, prepared_manifests=prepared,
        cli=dict(samples=samples, tasks=len(rows), fake_generations=len(calls), generations_by_method=actual,
                 first_processed=first["processed"], resumed_processed=resumed["processed"],
                 repeated_resume_new_generations=0, persisted_tasks=len(status["tasks"]), offline_rescoring_matches=True),
        reference_integrity=dict(checked=len(before), changed=0, sha256=after),
        network=dict(blocked_at_socket_and_dns=True, attempted_connections=network_attempts, real_model_requests=0),
        sources=sources, artifacts_directory=str(output),
        later_stages=dict(step4_real_validation="not_started", step5_platform_integration="not_started", formal_freeze="not_done"),
        note="Fake/HTTP-mocked responses verify software behavior; no model quality or billable request evidence.")
    atomic_json(output / "verification-step3.json", verification)
    print(json.dumps(verification, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
