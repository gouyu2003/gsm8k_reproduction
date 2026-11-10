"""Provider errors and manual billing reconciliation; all HTTP responses are mocked."""
import copy
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from unittest.mock import patch

import httpx
import openai

from gsm_repro.__main__ import main
from gsm_repro.backends import CallFailure, OpenAICompatibleBackend
from gsm_repro.budget import Budget, BudgetStop, budget_status, reconcile_budget
from gsm_repro.common import file_hash
from gsm_repro.config import DEFAULT, request
from gsm_repro.engine import create_run, execute, load_run
from gsm_repro.reporting import export_report

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-offline-test-credential"


def setUpModule():
    global guards
    guards = [patch.object(socket.socket, name, side_effect=AssertionError("Offline test attempted network"))
              for name in ("connect", "connect_ex")]
    guards += [patch.object(socket, name, side_effect=AssertionError("Offline test attempted DNS/network"))
               for name in ("getaddrinfo", "create_connection")]
    for guard in guards:
        guard.start()


def tearDownModule():
    for guard in reversed(guards):
        guard.stop()


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = copy.deepcopy(DEFAULT)
        self.config["pricing"] = dict(confirmed=True, input_per_million_cny="2", output_per_million_cny="8",
                                      input_token_reserve=16384, source="OFFLINE TEST ONLY", verified_at="2026-11-10")
        self.body = request(self.config, [{"role": "user", "content": "offline question"}], "test")

    def failed_call(self, response):
        def respond(req):
            return response
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            client = openai.OpenAI(api_key=KEY, base_url="https://offline.invalid/v1",
                                   timeout=1, max_retries=0, http_client=http)
            with patch.dict(os.environ, {"GSM_REPRO_DISABLE_MODEL_CALLS": "0"}):
                backend = OpenAICompatibleBackend(self.config, allow_model_calls=True, client=client)
                with self.assertRaises(CallFailure) as caught:
                    backend.complete(self.body)
        return caught.exception.document

    def unknown_charge(self):
        budget = Budget(self.root, self.config)
        budget.reserve("failed-request", self.body)
        budget.settle("failed-request", None)
        return budget

    def reconcile(self, actual="0.001", evidence="Verified offline bill reference", **kwargs):
        return reconcile_budget(self.root, self.config["campaign_id"], "failed-request", actual, evidence, **kwargs)

    def test_nested_error_has_code_message_type_param_and_header_request_id(self):
        error = self.failed_call(httpx.Response(400, headers={"x-dashscope-request-id": "ali-request"}, json={
            "error": {"code": "InvalidParameter", "message": "seed is unsupported", "type": "invalid_request_error", "param": "seed"}}))
        self.assertEqual(error["provider_request_id"], "ali-request")
        self.assertEqual(error["provider_error_code"], "InvalidParameter")
        self.assertEqual(error["provider_error_message"], "seed is unsupported")
        self.assertEqual(error["provider_error_type"], "invalid_request_error")
        self.assertEqual(error["provider_error_param"], "seed")
        self.assertEqual(error["category"], "configuration_error")
        self.assertFalse(error["retryable"])

    def test_flat_error_uses_body_request_id(self):
        error = self.failed_call(httpx.Response(400, json={"code": "InvalidApiKey", "message": "Key rejected", "request_id": "body-request"}))
        self.assertEqual(error["provider_request_id"], "body-request")
        self.assertEqual(error["provider_error_code"], "InvalidApiKey")
        self.assertEqual(error["provider_error_message"], "Key rejected")

    def test_sdk_request_id_takes_precedence(self):
        error = self.failed_call(httpx.Response(400, headers={"x-request-id": "sdk-request"}, json={
            "request_id": "body-request", "error": {"message": "Bad request"}}))
        self.assertEqual(error["provider_request_id"], "sdk-request")

    def test_nested_error_retains_top_level_request_id_unwrapped_by_sdk(self):
        error = self.failed_call(httpx.Response(400, json={"request_id": "outer-request", "error": {
            "code": "InvalidParameter", "message": "Bad parameter"}}))
        self.assertEqual(error["provider_request_id"], "outer-request")
        self.assertEqual(error["provider_error_message"], "Bad parameter")

    def test_non_json_and_missing_fields_do_not_hide_original_status(self):
        error = self.failed_call(httpx.Response(400, text="Gateway rejected request"))
        self.assertEqual(error["provider_error_message"], "Gateway rejected request")
        self.assertEqual(error["status_code"], 400)
        error = self.failed_call(httpx.Response(400, json={}))
        self.assertIsNone(error["provider_error_code"])
        self.assertIsNone(error["provider_error_message"])

    def test_diagnostics_redact_credentials_and_bound_message_size(self):
        error = self.failed_call(httpx.Response(400, json={"error": {
            "code": "InvalidParameter", "message": f"key={KEY}; Bearer opaque-secret; sk-another-secret " + "x" * 5000,
            "param": KEY}, "request_id": KEY}))
        saved = json.dumps(error)
        for secret in (KEY, "opaque-secret", "sk-another-secret"):
            self.assertNotIn(secret, saved)
        self.assertIn("[REDACTED]", saved)
        self.assertLessEqual(len(error["provider_error_message"]), 2014)

    def test_existing_failure_classification_and_retry_policy_are_preserved(self):
        for status, code, category, retryable in [(429, "RateLimit", "http_error", True),
                (503, "Unavailable", "http_error", True), (400, "QuotaExceeded", "quota_error", False),
                (400, "Content_Filter", "provider_rejected", False)]:
            with self.subTest(status=status, code=code):
                error = self.failed_call(httpx.Response(status, json={"error": {"code": code, "message": "offline"}}))
                self.assertEqual(error["category"], category)
                self.assertEqual(error["retryable"], retryable)

    def test_incomplete_key_is_rejected_before_sdk_construction(self):
        for key in ("x", "sk-key with-space"):
            with self.subTest(key=key), patch.dict(os.environ, {"GSM_REPRO_DISABLE_MODEL_CALLS": "0",
                    self.config["api_key_env"]: key, self.config["base_url_env"]: "https://offline.invalid/v1"}), \
                    patch.object(openai, "OpenAI") as constructor:
                with self.assertRaisesRegex(ValueError, "incomplete or contains whitespace"):
                    OpenAICompatibleBackend(self.config, allow_model_calls=True)
                constructor.assert_not_called()

    def test_error_survives_sqlite_json_export_and_terminal_summary(self):
        project = self.root / "project"
        project.mkdir()
        for folder in ("gsm_repro", "author_reference"):
            shutil.copytree(ROOT / folder, project / folder, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "requirements-repro.txt", project / "requirements-repro.txt")
        self.config.update(backend="openai_compatible", methods=["single"])
        directory = project / "run"
        create_run(project, directory, self.config, ROOT / "manifests/train_debug_10.json", limit=1)
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(400, json={
                "code": "InvalidParameter", "message": "seed is unsupported", "request_id": "persisted-request"}))) as http:
            client = openai.OpenAI(api_key=KEY, base_url="https://offline.invalid/v1", max_retries=0, http_client=http)
            with patch.dict(os.environ, {"GSM_REPRO_DISABLE_MODEL_CALLS": "0"}):
                backend = OpenAICompatibleBackend(self.config, allow_model_calls=True, client=client)
                result = execute(project, directory, backend=backend, allow_model_calls=True)
        export_report(project, directory)
        call = load_run(project, directory)[2].calls()[0]
        exported = json.loads((directory / "calls.jsonl").read_text())
        self.assertEqual(call["error"], exported["error"])
        self.assertEqual(result["last_failure"]["provider_error_message"], "seed is unsupported")
        self.assertEqual(result["last_failure"]["request_call_id"], call["request_call_id"])
        self.assertEqual(result["last_failure"]["provider_request_id"], "persisted-request")
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(budget_status(project, self.config["campaign_id"])["unresolved_count"], 1)

    def test_status_and_reconciliation_preview_are_read_only(self):
        budget = self.unknown_charge()
        before = file_hash(budget.path)
        status = budget_status(self.root, self.config["campaign_id"])
        preview = self.reconcile()
        self.assertEqual(file_hash(budget.path), before)
        self.assertEqual(status["unresolved_count"], 1)
        self.assertTrue(status["blocked"])
        self.assertEqual(status["settled_cny"], "0")
        self.assertIsNone(preview["before"]["actual"])
        self.assertFalse(preview["applied"])
        self.assertFalse((budget.path.parent / "backups").exists())
        with self.assertRaises(BudgetStop):
            budget.reserve("still-blocked", self.body)

    def test_confirmed_reconciliation_backs_up_and_audits_without_resetting_cap(self):
        budget = self.unknown_charge()
        result = self.reconcile(confirm_billing=True)
        with closing(sqlite3.connect(result["backup_path"])) as db:
            self.assertEqual(db.execute("SELECT actual,status FROM charges").fetchone(), (None, "unknown"))
        status = budget_status(self.root, self.config["campaign_id"])
        self.assertEqual(status["cap_cny"], "20")
        self.assertEqual(status["settled_cny"], "0.001")
        self.assertEqual(status["unresolved_count"], 0)
        audit = status["reconciliations"][0]
        self.assertEqual(json.loads(audit["before_json"]), result["before"])
        self.assertEqual(json.loads(audit["after_json"]), result["after"])
        self.assertEqual(audit["evidence"], "Verified offline bill reference")
        self.assertEqual(status["charges"][0]["reserve"], str(budget.maximum))
        budget.reserve("next-request", self.body)

    def test_zero_requires_explicit_confirmation_and_keeps_evidence(self):
        self.unknown_charge()
        self.assertFalse(self.reconcile("0", evidence="Offline support confirmed no charge")["applied"])
        self.assertEqual(budget_status(self.root, self.config["campaign_id"])["unresolved_count"], 1)
        self.reconcile("0", evidence="Offline support confirmed no charge", confirm_billing=True)
        self.assertEqual(budget_status(self.root, self.config["campaign_id"])["settled_cny"], "0")

    def test_invalid_amount_and_missing_evidence_do_not_modify_ledger(self):
        budget = self.unknown_charge()
        before = file_hash(budget.path)
        for amount in ("NaN", "Infinity", "-Infinity", "-0.01", "bad", True):
            with self.subTest(amount=amount), self.assertRaises(BudgetStop):
                self.reconcile(amount, confirm_billing=True)
        with self.assertRaises(BudgetStop):
            self.reconcile(evidence="  ", confirm_billing=True)
        self.assertEqual(file_hash(budget.path), before)

    def test_wrong_campaign_request_active_and_already_settled_are_rejected(self):
        budget = self.unknown_charge()
        with self.assertRaises(BudgetStop):
            reconcile_budget(self.root, "other-campaign", "failed-request", "0", "offline", confirm_billing=True)
        with self.assertRaises(BudgetStop):
            reconcile_budget(self.root, self.config["campaign_id"], "missing", "0", "offline", confirm_billing=True)
        self.reconcile(confirm_billing=True)
        with self.assertRaises(BudgetStop):
            self.reconcile(confirm_billing=True)
        budget.reserve("active-request", self.body)
        with self.assertRaises(BudgetStop):
            reconcile_budget(self.root, self.config["campaign_id"], "active-request", "0", "offline", confirm_billing=True)

    def test_actual_cost_over_reservation_is_recorded_and_cap_still_blocks(self):
        budget = self.unknown_charge()
        self.reconcile("20.01", confirm_billing=True)
        with self.assertRaisesRegex(BudgetStop, "cannot reserve"):
            budget.reserve("over-cap", self.body)

    def test_backup_failure_leaves_unknown_charge_unchanged(self):
        budget = self.unknown_charge()
        (budget.path.parent / "backups").write_text("offline path obstruction")
        before = file_hash(budget.path)
        with self.assertRaises(OSError):
            self.reconcile(confirm_billing=True)
        self.assertEqual(file_hash(budget.path), before)

    def test_missing_ledger_is_not_reconstructed(self):
        with self.assertRaisesRegex(BudgetStop, "missing"):
            budget_status(self.root, self.config["campaign_id"])
        self.assertFalse((self.root / "budgets").exists())

    def test_failed_update_rolls_back_the_audit_and_charge_together(self):
        budget = self.unknown_charge()
        with closing(sqlite3.connect(budget.path)) as db, db:
            db.execute("""CREATE TRIGGER reject_reconciliation BEFORE UPDATE ON charges
                WHEN NEW.status='settled' BEGIN SELECT RAISE(ABORT,'offline injected failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.reconcile(confirm_billing=True)
        status = budget_status(self.root, self.config["campaign_id"])
        self.assertEqual(status["unresolved_count"], 1)
        self.assertIsNone(status["charges"][0]["actual"])
        self.assertEqual(status["reconciliations"], [])

    def test_cli_status_preview_and_confirmation_do_not_call_models(self):
        self.unknown_charge()
        arguments = ["budget-reconcile", "--campaign-id", self.config["campaign_id"],
                     "--request-id", "failed-request", "--actual-cny", "0.001", "--evidence", "Offline bill"]
        with patch("gsm_repro.__main__.ROOT", self.root), patch.object(OpenAICompatibleBackend, "complete") as model:
            for args, applied in [(arguments, False), (arguments + ["--confirm-billing"], True)]:
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(args), 0)
                self.assertEqual(json.loads(output.getvalue())["applied"], applied)
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["budget-status", "--campaign-id", self.config["campaign_id"]]), 0)
            self.assertEqual(json.loads(output.getvalue())["unresolved_count"], 0)
            model.assert_not_called()
