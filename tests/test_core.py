import ast
import copy
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from gsm_repro import prompts
from gsm_repro.backends import FakeBackend,OpenAICompatibleBackend,CallFailure
from gsm_repro.budget import Budget,BudgetStop
from gsm_repro.common import atomic_json,digest
from gsm_repro.config import DEFAULT,request,validate
from gsm_repro.engine import create_run,execute,load_run
from gsm_repro.manifests import load_manifest,TEST_QUESTIONS_HASH
from gsm_repro.reporting import export_report
from gsm_repro.scoring import parse_prediction,parse_gold,fixed_score,author_score

ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    os.environ["GSM_REPRO_DISABLE_MODEL_CALLS"] = "1"
    global network_guard, dns_guard
    network_guard = patch.object(socket.socket,"connect",side_effect=AssertionError("Test attempted network"))
    dns_guard = patch.object(socket,"getaddrinfo",side_effect=AssertionError("Test attempted DNS"))
    network_guard.start()
    dns_guard.start()


def tearDownModule():
    network_guard.stop()
    dns_guard.stop()


class Tagged(FakeBackend):
    def complete(self,body):
        result = super().complete(body)
        result["content"] = f"reply_{len(self.requests)} \\boxed{{42}}"
        return result


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(DEFAULT)

    def plan(self,methods=None):
        if methods:
            self.config["methods"] = methods
        directory = self.root/"run"
        create_run(ROOT,directory,self.config,ROOT/"manifests/train_debug_10.json",limit=1)
        return directory

    def test_author_prompt_strings_are_exact(self):
        tree = ast.parse((ROOT/"author_reference/gen_gsm.py").read_text(encoding="utf-8"))
        func = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="construct_message")
        env = {}
        exec(compile(ast.Module(body=[func],type_ignores=[]),"author", "exec"),env)
        initial = next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)
                       and isinstance(n.value,str) and n.value.startswith("Can you solve the following math problem?"))
        self.assertEqual(prompts.initial("Q")["content"],initial.format("Q"))
        self.assertEqual(prompts.update([],"Q"),env["construct_message"]([],"Q",1))
        contexts = [[{},{"content":"peer_1"}],[{},{"content":"peer_2"}]]
        self.assertEqual(prompts.update(["peer_1","peer_2"],"Q"),
                         env["construct_message"](contexts,"Q",1))

    def test_prepared_manifests_match_and_are_disjoint(self):
        manifests = [load_manifest(ROOT/"manifests"/name) for name in
                     ("train_debug_10.json","train_calibration_30.json","questions_100.json")]
        self.assertEqual([len(m["records"]) for m in manifests],[10,30,100])
        self.assertEqual(manifests[2]["questions_sha256"],TEST_QUESTIONS_HASH)
        sets = [set(r["question"] for r in m["records"]) for m in manifests]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        copied = copy.deepcopy(manifests[0])
        copied["records"][0]["question"] += "changed"
        atomic_json(self.root/"bad.json",copied)
        with self.assertRaises(ValueError):
            load_manifest(self.root/"bad.json")

    def test_four_methods_have_one_two_six_six_generations(self):
        directory = self.plan()
        backend = Tagged()
        result = execute(ROOT,directory,backend=backend)
        _,_,store = load_run(ROOT,directory)
        counts = {m:sum(c["task_key"].startswith(m+":") for c in store.calls()) for m in self.config["methods"]}
        self.assertEqual(counts,dict(single=1,reflection=2,majority=6,debate=6))
        self.assertEqual(result["actual_model_requests"],0)
        self.assertTrue(all("####" not in json.dumps(c["request"]) for c in store.calls()))
        self.assertTrue(all("response_format" not in c["request"] and c["request"]["n"]==1 for c in store.calls()))

    def test_debate_reads_only_previous_round_and_retains_own_history(self):
        directory = self.plan(["debate"])
        backend = Tagged()
        execute(ROOT,directory,backend=backend)
        calls = backend.requests
        for i in range(3):
            self.assertEqual(len(calls[3+i]["messages"]),3)
            self.assertEqual(calls[3+i]["messages"][1]["content"],f"reply_{i+1} \\boxed{{42}}")
            update = calls[3+i]["messages"][2]["content"]
            self.assertTrue(all(f"reply_{j+1}" in update for j in range(3) if j != i))
            self.assertNotIn(f"reply_{i+1}",update)
            self.assertTrue(all(f"reply_{j}" not in update for j in (4,5,6)))

    def test_resume_only_executes_pending_tasks(self):
        directory = self.plan()
        first = Tagged()
        execute(ROOT,directory,backend=first,max_tasks=2)
        self.assertEqual(len(first.requests),3)
        second = Tagged()
        execute(ROOT,directory,backend=second)
        self.assertEqual(len(second.requests),12)
        third = Tagged()
        execute(ROOT,directory,backend=third)
        self.assertFalse(third.requests)

    def test_interrupted_request_is_not_replayed(self):
        directory = self.plan()
        class Interrupted(FakeBackend):
            def complete(self,body):
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            execute(ROOT,directory,backend=Interrupted())
        backend = Tagged()
        result = execute(ROOT,directory,backend=backend)
        self.assertEqual(result["counts"]["interrupted"],1)
        self.assertEqual(len(backend.requests),14)
        _,_,store = load_run(ROOT,directory)
        self.assertEqual(store.calls()[0]["status"],"unknown")

    def test_completed_generations_recover_without_resampling(self):
        directory = self.plan(["debate"])
        execute(ROOT,directory)
        _,_,store = load_run(ROOT,directory)
        with store.connection() as db:
            db.execute("UPDATE tasks SET status='running',result_json=NULL")
        backend = Tagged()
        result = execute(ROOT,directory,backend=backend)
        self.assertFalse(backend.requests)
        self.assertEqual(result["counts"]["completed"],1)

    def test_retry_is_bounded_and_never_chooses_best_answer(self):
        self.config["retry_wait_seconds"] = 0
        directory = self.plan(["single"])
        class Retry(FakeBackend):
            def complete(self,body):
                self.requests.append(body)
                if len(self.requests)==1:
                    raise CallFailure("http_error",retryable=True,status_code=429)
                return dict(content="\\boxed{42}",usage={},reasoning_state="missing")
        backend = Retry()
        execute(ROOT,directory,backend=backend)
        _,_,store = load_run(ROOT,directory)
        self.assertEqual([c["attempt"] for c in store.calls()],[1,2])
        self.assertEqual(len(backend.requests),2)

    def test_reasoning_mismatch_halts_before_next_task(self):
        directory = self.plan()
        class Reasoning(FakeBackend):
            def complete(self,body):
                result = super().complete(body)
                result["reasoning_state"] = "nonempty"
                return result
        backend = Reasoning()
        result = execute(ROOT,directory,backend=backend)
        self.assertTrue(result["blocked"])
        self.assertEqual(result["counts"]["failed"],1)
        self.assertEqual(result["counts"]["pending"],3)
        self.assertEqual(len(backend.requests),1)

    def test_content_rejection_is_not_retried(self):
        self.config["retry_wait_seconds"] = 0
        directory = self.plan(["single"])
        class Rejected(FakeBackend):
            def complete(self,body):
                self.requests.append(body)
                raise CallFailure("provider_rejected",status_code=400)
        backend = Rejected()
        execute(ROOT,directory,backend=backend)
        self.assertEqual(len(backend.requests),1)

    def test_real_calls_are_blocked_before_client_or_credentials(self):
        self.config["backend"] = "openai_compatible"
        directory = self.plan(["single"])
        with self.assertRaises(ValueError):
            execute(ROOT,directory)
        with self.assertRaises(ValueError):
            execute(ROOT,directory,allow_model_calls=True)
        self.assertFalse(load_run(ROOT,directory)[2].calls())

    def test_truncated_task_ledger_is_rejected(self):
        directory = self.plan()
        _,_,store = load_run(ROOT,directory)
        with store.connection() as db:
            db.execute("DELETE FROM tasks WHERE method='debate'")
        with self.assertRaises(ValueError):
            execute(ROOT,directory)

    def test_failed_task_also_consumes_max_tasks_limit(self):
        directory = self.plan()
        class Rejected(FakeBackend):
            def complete(self,body):
                self.requests.append(body)
                raise CallFailure("provider_rejected",status_code=400)
        backend = Rejected()
        result = execute(ROOT,directory,backend=backend,max_tasks=1)
        self.assertEqual(result["processed"],1)
        self.assertEqual(result["counts"]["failed"],1)
        self.assertEqual(result["counts"]["pending"],3)
        self.assertEqual(len(backend.requests),1)

    def test_unknown_request_is_not_retried(self):
        self.config["retry_wait_seconds"] = 0
        directory = self.plan(["single"])
        class Unknown(FakeBackend):
            def complete(self,body):
                self.requests.append(body)
                raise CallFailure("connection_error",retryable=True,unknown=True)
        backend = Unknown()
        execute(ROOT,directory,backend=backend)
        self.assertEqual(len(backend.requests),1)

    def test_formal_test_cannot_be_used_as_development(self):
        with self.assertRaises(ValueError):
            create_run(ROOT,self.root/"test",self.config,ROOT/"manifests/questions_100.json")

    def test_profile_tamper_is_rejected(self):
        directory = self.plan()
        profile = json.loads((directory/"profile.json").read_text())
        profile["config"]["temperature"] = 0
        atomic_json(directory/"profile.json",profile)
        with self.assertRaises(ValueError):
            execute(ROOT,directory)

    def test_report_includes_failures_and_fixed_denominator(self):
        directory = self.plan()
        execute(ROOT,directory,max_tasks=1)
        report = export_report(ROOT,directory)
        self.assertFalse(report["complete"])
        self.assertEqual(report["methods"]["debate"]["metrics"]["fixed"]["n"],1)
        self.assertIsNone(report["methods"]["debate"]["metrics"]["author_faithful"]["accuracy"])
        self.assertEqual(report["actual_model_requests"],0)
        self.assertTrue((directory/"results.csv").exists())

    def test_parser_boundaries_and_author_failures(self):
        for text,expected in [(r"\\boxed{-72}","-72"),(r"\\boxed{1,450,000}","1450000"),
                              (r"\\boxed{{72}}","72"),(r"\\boxed{72.5}","72.5"),
                              (r"Before 72, final \\boxed{}",None),(r"\\boxed{3\\frac{1}{2}}",None)]:
            with self.subTest(text=text):
                self.assertEqual(parse_prediction(text),expected)
        ref = ROOT/"author_reference/eval_gsm.py"
        self.assertEqual(author_score("#### 72",[r"\\boxed{ }"],ref)["correct"],1)
        self.assertEqual(author_score("#### 72",[r"\\boxed{}"],ref)["status"],"author_scoring_error")
        self.assertEqual(fixed_score("#### 72",[r"\\boxed{ }"])["correct"],0)

    def test_tie_is_distinct_from_strict_majority(self):
        values = [r"\\boxed{2}",r"\\boxed{2}",r"\\boxed{3}",r"\\boxed{4}",r"\\boxed{5}",r"\\boxed{6}"]
        self.assertEqual(fixed_score("#### 2",values,tie_error=True)["correct"],1)
        values = [r"\\boxed{2}",r"\\boxed{3}"]
        self.assertEqual(fixed_score("#### 2",values)["correct"],1)
        self.assertEqual(fixed_score("#### 2",values,tie_error=True)["correct"],0)

    def test_tolerance_changes_comparison_only(self):
        self.assertEqual(fixed_score("#### 1",[r"\\boxed{1.0000005}"])["correct"],1)
        self.assertEqual(fixed_score("#### 1",[r"\\boxed{1.0000005}"],exact=True)["correct"],0)
        self.assertEqual(parse_gold("work\n#### -1,234.50"),"-1234.50")
        with self.assertRaises(ValueError):
            parse_gold("#### NaN")

    def test_budget_is_campaign_wide_and_unknown_cost_stops(self):
        self.config["pricing"] = dict(confirmed=True,input_per_million_cny="1",output_per_million_cny="4",
            input_token_reserve=16384,source="offline test price",verified_at="2026-11-09")
        budget = Budget(self.root,self.config)
        body = request(self.config,[prompts.initial("Q")],"test")
        budget.reserve("first",body)
        self.assertTrue(budget.settle("first",dict(usage=dict(prompt_tokens=10,completion_tokens=5))))
        same = Budget(self.root,self.config)
        same.reserve("second",body)
        self.assertFalse(same.settle("second",None))
        with self.assertRaises(BudgetStop):
            budget.reserve("third",body)

    def test_sdk_request_is_validated_with_mock_transport_only(self):
        import httpx
        import openai
        captured = []
        def respond(req):
            captured.append(json.loads(req.content))
            return httpx.Response(200,headers={"x-request-id":"offline-id"},json={
                "id":"fake","object":"chat.completion","created":0,"model":"offline",
                "choices":[{"index":0,"message":{"role":"assistant","content":r"\\boxed{72}","reasoning_content":None},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}})
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            sdk = openai.OpenAI(api_key="offline-not-a-real-key",base_url="https://offline.invalid/v1",
                                max_retries=0,timeout=120,http_client=http)
            with patch.dict(os.environ,{"GSM_REPRO_DISABLE_MODEL_CALLS":"0"}):
                backend = OpenAICompatibleBackend(self.config,allow_model_calls=True,client=sdk)
                reply = backend.complete(request(self.config,[prompts.initial("Q")],"mock"))
        self.assertEqual(reply["provider_request_id"],"offline-id")
        self.assertEqual(reply["reasoning_state"],"null")
        self.assertNotIn("response_format",captured[0])
        self.assertEqual(captured[0]["thinking"],{"type":"disabled"})
        self.assertEqual(captured[0]["n"],1)


if __name__ == "__main__":
    unittest.main()
