"""Step 3 behavioral acceptance; every response is fake or HTTP MockTransport."""
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gsm_repro import prompts
from gsm_repro.backends import FakeBackend, OpenAICompatibleBackend
from gsm_repro.common import execution_lock
from gsm_repro.config import DEFAULT
from gsm_repro.engine import create_run, execute, load_run
from gsm_repro.reporting import export_report
from gsm_repro.scoring import fixed_score, score_all


ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    global guards
    os.environ["GSM_REPRO_DISABLE_MODEL_CALLS"] = "1"
    guards = [patch.object(socket.socket, "connect", side_effect=AssertionError("External connection in offline acceptance")),
              patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS in offline acceptance"))]
    for guard in guards:
        guard.start()


def tearDownModule():
    for guard in reversed(guards):
        guard.stop()


class SimulatedPowerLoss(BaseException):
    """Bypasses normal exception handlers, like termination before cleanup."""


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = copy.deepcopy(DEFAULT)
        self.config["retry_wait_seconds"] = 0

    def plan(self, methods=None, *, samples=1, project=ROOT):
        if methods is not None:
            self.config["methods"] = methods
        directory = self.root / "run"
        create_run(project, directory, self.config, ROOT / "manifests/train_debug_10.json", limit=samples)
        return directory

    def test_equivalent_numbers_share_votes(self):
        outputs = [r"\boxed{1}", r"\boxed{1.0}", r"\boxed{1.00}",
                   r"\boxed{2}", r"\boxed{2}", r"\boxed{3}"]
        result = fixed_score("#### 1", outputs)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["prediction"], "1")
        self.assertEqual(result["votes"], {"1": 3, "2": 2, "3": 1})

    def test_signed_zero_and_thousands_formats_share_votes(self):
        for target, values in [("0", ["-0.0", "+0", "0.00", "2", "2", "3"]),
                               ("1000", ["1,000", "1000.0", "+1000", "2", "2", "3"])]:
            with self.subTest(target=target):
                result = fixed_score("#### " + target, ["\\boxed{" + v + "}" for v in values])
                self.assertEqual(result["correct"], 1)
                self.assertEqual(result["votes"][target], 3)

    def test_nearby_values_are_not_merged_by_comparison_tolerance(self):
        values = [r"\boxed{1}", r"\boxed{1.0000005}", r"\boxed{2}", r"\boxed{2}"]
        result = fixed_score("#### 2", values)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(len(result["votes"]), 3)

    def test_all_unparseable_votes_cannot_be_correct(self):
        result = fixed_score("#### 42", ["", "unknown", r"\boxed{}", r"\boxed{NaN}",
                                            r"\boxed{1/2}", "no answer"])
        self.assertEqual(result["correct"], 0)
        self.assertIsNone(result["prediction"])
        self.assertTrue(result["parse_failure"])
        self.assertFalse(result["votes"])

    def test_invalid_votes_do_not_outvote_a_numeric_answer(self):
        result = fixed_score("#### -5", ["unknown"] * 5 + [r"\boxed{-5}"])
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["votes"], {"-5": 1})

    def test_conflicting_boxes_are_invalid_under_fixed_rule(self):
        result = fixed_score("#### 5", [r"First \boxed{2}; final \boxed{5}"])
        self.assertEqual(result["correct"], 0)
        self.assertTrue(result["parse_failure"])

    def test_exactly_equal_decimal_spellings_do_not_create_a_tie(self):
        result = fixed_score("#### 1", [r"\boxed{1.0}", r"\boxed{1}"], tie_error=True)
        self.assertEqual(result["correct"], 1)
        self.assertFalse(result["tie"])

    def test_author_score_is_preserved_while_fixed_votes_are_corrected(self):
        outputs = [r"\boxed{1}", r"\boxed{1.0}", r"\boxed{1.00}",
                   r"\boxed{2}", r"\boxed{2}", r"\boxed{3}"]
        scores = score_all("#### 1", outputs, ROOT / "author_reference/eval_gsm.py")
        self.assertEqual(scores["author_faithful"]["correct"], 0)
        self.assertEqual(scores["fixed"]["correct"], 1)

    def test_reflection_retains_only_its_own_history(self):
        directory = self.plan(["reflection"])
        backend = FakeBackend()
        execute(ROOT, directory, backend=backend)
        first, second = backend.requests
        self.assertEqual(len(first["messages"]), 1)
        self.assertEqual(len(second["messages"]), 3)
        self.assertEqual(second["messages"][0], first["messages"][0])
        self.assertEqual(second["messages"][-1], prompts.update([], "unused"))
        self.assertNotIn("One agent solution", second["messages"][-1]["content"])

    def test_questions_and_repeats_have_separate_histories_and_seeds(self):
        self.config["repeats"] = 2
        directory = self.plan(samples=2)
        backend = FakeBackend()
        execute(ROOT, directory, backend=backend)
        _, manifest, store = load_run(ROOT, directory)
        questions = {r["qid"]: r["question"] for r in manifest["records"]}
        self.assertEqual(len(store.tasks()), 16)
        self.assertEqual(len(store.calls()), 60)
        self.assertEqual(len({c["request"]["seed"] for c in store.calls()}), 60)
        for call in store.calls():
            task = next(t for t in store.tasks() if t["task_key"] == call["task_key"])
            self.assertEqual(call["request"]["messages"][0], prompts.initial(questions[task["qid"]]))
            self.assertNotIn("####", json.dumps(call["request"]))
            if call["round"] == 0:
                self.assertEqual(len(call["request"]["messages"]), 1)

    def test_backend_mutation_cannot_rewrite_saved_requests(self):
        directory = self.plan(["debate"])
        class Mutating(FakeBackend):
            def complete(self, body):
                result = super().complete(body)
                body["messages"].append({"role": "user", "content": "MUTATED_BACKEND_ONLY"})
                return result
        execute(ROOT, directory, backend=Mutating())
        calls = load_run(ROOT, directory)[2].calls()
        self.assertTrue(all("MUTATED_BACKEND_ONLY" not in json.dumps(c["request"]) for c in calls))

    def test_unparseable_replies_complete_execution_but_score_zero(self):
        directory = self.plan()
        class Invalid(FakeBackend):
            def complete(self, body):
                reply = super().complete(body)
                reply["content"] = "There is no numeric answer."
                return reply
        result = execute(ROOT, directory, backend=Invalid())
        self.assertEqual(result["counts"]["completed"], 4)
        for task in load_run(ROOT, directory)[2].tasks():
            scores = json.loads(task["result_json"])["scores"]
            self.assertEqual(scores["fixed"]["correct"], 0)
            self.assertTrue(scores["fixed"]["parse_failure"])

    def test_power_loss_after_saved_replies_recovers_without_new_generations(self):
        directory = self.plan(["debate"])
        with patch("gsm_repro.engine.score_all", side_effect=SimulatedPowerLoss):
            with self.assertRaises(SimulatedPowerLoss):
                execute(ROOT, directory)
        backend = FakeBackend()
        result = execute(ROOT, directory, backend=backend)
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertFalse(backend.requests)
        self.assertEqual(len(load_run(ROOT, directory)[2].calls()), 6)

    def test_power_loss_cannot_promote_reasoning_mismatch_to_completed(self):
        directory = self.plan()
        class Reasoning(FakeBackend):
            def complete(self, body):
                reply = super().complete(body)
                reply["reasoning_state"] = "nonempty"
                return reply
        with patch("gsm_repro.engine.fail_task", side_effect=SimulatedPowerLoss):
            with self.assertRaises(SimulatedPowerLoss):
                execute(ROOT, directory, backend=Reasoning())
        backend = FakeBackend()
        result = execute(ROOT, directory, backend=backend)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["completed"], 0)
        self.assertEqual(result["counts"]["pending"], 3)
        self.assertTrue(result["blocked"])
        self.assertFalse(backend.requests)

    def test_corrupt_call_slot_is_rejected_without_resampling(self):
        directory = self.plan(["single"])
        execute(ROOT, directory)
        store = load_run(ROOT, directory)[2]
        with store.connection() as db:
            db.execute("UPDATE tasks SET status='running',result_json=NULL")
            db.execute("UPDATE calls SET agent=99")
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            execute(ROOT, directory, backend=backend)
        self.assertFalse(backend.requests)

    def test_missing_database_is_not_recreated(self):
        directory = self.plan()
        database = (directory / "results.sqlite3").resolve()
        self.assertTrue(database.is_relative_to(self.root.resolve()))
        database.unlink()
        with self.assertRaises(ValueError):
            execute(ROOT, directory)
        self.assertFalse(database.exists())

    def test_runtime_drift_is_rejected_before_backend_call(self):
        directory = self.plan()
        backend = FakeBackend()
        with patch("gsm_repro.engine.identity", return_value={"changed": True}):
            with self.assertRaises(ValueError):
                execute(ROOT, directory, backend=backend)
        self.assertFalse(backend.requests)

    def test_two_processes_cannot_own_the_same_run(self):
        directory = self.plan()
        code = ("import sys;from gsm_repro.common import execution_lock\n"
                "try:\n with execution_lock(sys.argv[1]): print('ACQUIRED')\n"
                "except ValueError: print('LOCKED')\n")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), environment.get("PYTHONPATH", "")))
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with execution_lock(directory):
            child = subprocess.run([sys.executable, "-B", "-c", code, str(directory)],
                                   env=environment, capture_output=True, text=True, timeout=20)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(child.stdout.strip(), "LOCKED")
        with execution_lock(directory):
            pass

    def test_failed_tasks_stay_in_fixed_score_denominator(self):
        directory = self.plan(["single", "reflection"])
        class FirstFails(FakeBackend):
            def complete(self, body):
                super().complete(body)
                if len(self.requests) == 1:
                    raise RuntimeError("offline injected failure")
                return dict(content=r"\boxed{42}", usage={}, reasoning_state="missing")
        execute(ROOT, directory, backend=FirstFails())
        report = export_report(ROOT, directory)
        self.assertEqual(report["methods"]["single"]["metrics"]["fixed"]["n"], 1)
        self.assertEqual(report["methods"]["single"]["metrics"]["fixed"]["accuracy"], 0)
        self.assertEqual(report["actual_model_requests"], 0)

    def test_sdk_mock_runs_all_four_methods_without_network(self):
        import httpx
        import openai
        project = self.root / "project"
        project.mkdir()
        for name in ("gsm_repro", "author_reference"):
            shutil.copytree(ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "requirements-repro.txt", project / "requirements-repro.txt")
        self.config["backend"] = "openai_compatible"
        self.config["pricing"] = dict(confirmed=True, input_per_million_cny="1", output_per_million_cny="4",
                                      input_token_reserve=16384, source="OFFLINE TEST ONLY", verified_at="2026-11-09")
        directory = self.plan(project=project)
        captured = []
        def respond(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, headers={"x-request-id": "offline-mock"}, json={
                "id": "offline", "object": "chat.completion", "created": 0, "model": "offline-sdk-mock",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": r"\boxed{42}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})
        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            client = openai.OpenAI(api_key="offline-not-a-key", base_url="https://offline.invalid/v1",
                                   timeout=120, max_retries=0, http_client=transport)
            with patch.dict(os.environ, {"GSM_REPRO_DISABLE_MODEL_CALLS": "0"}):
                backend = OpenAICompatibleBackend(self.config, allow_model_calls=True, client=client)
                result = execute(project, directory, backend=backend, allow_model_calls=True)
        self.assertEqual(len(captured), 15)
        self.assertEqual(result["counts"]["completed"], 4)
        self.assertTrue(all("response_format" not in r and len(r["messages"]) in (1, 3) for r in captured))
        self.assertTrue(all("####" not in json.dumps(r) for r in captured))


if __name__ == "__main__":
    unittest.main()
