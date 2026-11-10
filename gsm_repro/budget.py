from __future__ import annotations
import sqlite3
from contextlib import closing
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4
from .common import canonical, utcnow

class BudgetStop(RuntimeError):
    pass


class Budget:
    """A campaign is shared by every real run/model; unknown charges stop new requests."""
    def __init__(self, root, config):
        pricing = config["pricing"]
        if not pricing.get("confirmed") or not pricing.get("source") or not pricing.get("verified_at"):
            raise BudgetStop("Real requests require verified prices and their source/date")
        self.input_price = Decimal(str(pricing["input_per_million_cny"]))
        self.output_price = Decimal(str(pricing["output_per_million_cny"]))
        self.cap = Decimal(str(config["budget_cny"]))
        if not all(p.is_finite() and p >= 0 for p in (self.input_price,self.output_price)):
            raise BudgetStop("Invalid price")
        self.input_reserve = pricing["input_token_reserve"]
        if not isinstance(self.input_reserve, int) or self.input_reserve <= 0:
            raise BudgetStop("Invalid input token reservation")
        self.output_reserve = config["max_tokens"]
        self.maximum = (self.input_reserve*self.input_price+self.output_reserve*self.output_price)/1000000
        self.campaign = config["campaign_id"]
        path = Path(root) / "budgets"
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / "campaigns.sqlite3"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY,cap TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS charges (request_id TEXT PRIMARY KEY,campaign TEXT NOT NULL,
                    reserve TEXT NOT NULL,actual TEXT,status TEXT NOT NULL,updated_at TEXT NOT NULL);
            """)
            db.execute("INSERT OR IGNORE INTO campaigns VALUES (?,?)", (self.campaign,str(self.cap)))
            if Decimal(db.execute("SELECT cap FROM campaigns WHERE id=?", (self.campaign,)).fetchone()[0]) != self.cap:
                raise BudgetStop("Campaign cap changed; cannot silently reset budget")

    def reserve(self, request_id, request):
        # UTF-8 byte bound plus per-message overhead is conservative for the intended BPE endpoint,
        # not a proof about unknown vendor tokenizers or hidden billing.
        import json
        input_bound = len(json.dumps(request["messages"],ensure_ascii=False).encode()) + 512*len(request["messages"])
        if input_bound > self.input_reserve:
            raise BudgetStop("Input reservation is too small for this request")
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT reserve,actual,status FROM charges WHERE campaign=?", (self.campaign,)).fetchall()
            if any(r[2] != "settled" for r in rows):
                raise BudgetStop("A previous request has unconfirmed cost; use budget-status, verify billing, "
                                 "then preview budget-reconcile before continuing")
            spent = sum((Decimal(r[1]) for r in rows), Decimal(0))
            if spent + self.maximum > self.cap:
                raise BudgetStop("Campaign budget cannot reserve the next request")
            db.execute("INSERT INTO charges VALUES (?,?,?,NULL,'reserved',?)",
                       (request_id,self.campaign,str(self.maximum),utcnow()))

    def settle(self, request_id, response):
        usage = (response or {}).get("usage") or {}
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        known = all(isinstance(v,int) and not isinstance(v,bool) and v >= 0 for v in (prompt,completion))
        actual = (prompt*self.input_price+completion*self.output_price)/1000000 if known else None
        with closing(sqlite3.connect(self.path)) as db, db:
            row = db.execute("SELECT reserve FROM charges WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                raise BudgetStop("Missing cost reservation")
            if known and actual > Decimal(row[0]):
                known = False  # Reservation was not conservative; stop for reconciliation.
            db.execute("UPDATE charges SET actual=?,status=?,updated_at=? WHERE request_id=?",
                       (str(actual) if actual is not None else None,"settled" if known else "unknown",utcnow(),request_id))
        return known


def _ledger_path(root):
    path = (Path(root) / "budgets" / "campaigns.sqlite3").resolve()
    if not path.is_file():
        raise BudgetStop("Budget ledger is missing; no ledger was created")
    return path


def _read_connection(path):
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def budget_status(root, campaign):
    """Read-only inspection, including unresolved reservations; never infer a zero charge."""
    with closing(_read_connection(_ledger_path(root))) as db:
        row = db.execute("SELECT cap FROM campaigns WHERE id=?", (campaign,)).fetchone()
        if row is None:
            raise BudgetStop("Unknown campaign; no campaign was created")
        charges = [dict(r) for r in db.execute(
            "SELECT * FROM charges WHERE campaign=? ORDER BY rowid", (campaign,))]
        has_audit = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                               "AND name='budget_reconciliations'").fetchone()
        audits = [dict(r) for r in db.execute(
            "SELECT * FROM budget_reconciliations WHERE campaign=? ORDER BY id", (campaign,))] if has_audit else []
    unresolved = [r for r in charges if r["status"] != "settled"]
    spent = sum((Decimal(r["actual"]) for r in charges if r["status"] == "settled"), Decimal(0))
    return dict(campaign_id=campaign, cap_cny=row["cap"], settled_cny=str(spent),
                unconfirmed_reserved_cny=str(sum((Decimal(r["reserve"]) for r in unresolved), Decimal(0))),
                unresolved_count=len(unresolved), blocked=bool(unresolved) or spent >= Decimal(row["cap"]),
                charges=charges, reconciliations=audits, model_calls=0)


def reconcile_budget(root, campaign, request_id, actual_cny, evidence, *, confirm_billing=False):
    """Preview by default. Apply only a user-confirmed bill, with a backup and atomic audit."""
    try:
        amount = Decimal(str(actual_cny))
    except InvalidOperation as exc:
        raise BudgetStop("Actual charge must be a finite nonnegative CNY amount") from exc
    if not amount.is_finite() or amount < 0:
        raise BudgetStop("Actual charge must be a finite nonnegative CNY amount")
    if not isinstance(evidence,str) or not evidence.strip():
        raise BudgetStop("Billing evidence/reference is required; absence of a bill is not proof of zero cost")
    evidence = evidence.strip()
    if len(evidence) > 4000:
        raise BudgetStop("Billing evidence/reference is too long")
    path = _ledger_path(root)

    def preview(db):
        row = db.execute("SELECT * FROM charges WHERE request_id=? AND campaign=?",
                         (request_id,campaign)).fetchone()
        if row is None:
            raise BudgetStop("Request does not belong to this campaign")
        if row["status"] != "unknown":
            raise BudgetStop("Only an unknown charge may be reconciled; active and settled requests are unchanged")
        before = dict(row)
        after = dict(before,actual=str(amount),status="settled",updated_at=utcnow())
        return dict(applied=False,campaign_id=campaign,request_id=request_id,before=before,after=after,
                    evidence=evidence,model_calls=0)

    if not confirm_billing:
        with closing(_read_connection(path)) as db:
            return preview(db)

    with closing(sqlite3.connect(path,timeout=10)) as db, db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        result = preview(db)
        # Back up through a separate read connection before any database writes.
        # BEGIN IMMEDIATE prevents another writer changing the inspected charge.
        backup = path.parent / "backups" / ("before-reconcile-" + uuid4().hex + ".sqlite3")
        backup.parent.mkdir(parents=True,exist_ok=True)
        with closing(_read_connection(path)) as source, closing(sqlite3.connect(backup)) as target:
            source.backup(target)
        db.execute("""CREATE TABLE IF NOT EXISTS budget_reconciliations (
            id INTEGER PRIMARY KEY, request_id TEXT NOT NULL UNIQUE, campaign TEXT NOT NULL,
            before_json TEXT NOT NULL, after_json TEXT NOT NULL, evidence TEXT NOT NULL,
            confirmed_at TEXT NOT NULL, backup_path TEXT NOT NULL)""")
        db.execute("INSERT INTO budget_reconciliations "
                   "(request_id,campaign,before_json,after_json,evidence,confirmed_at,backup_path) "
                   "VALUES (?,?,?,?,?,?,?)", (request_id,campaign,canonical(result["before"]),
                   canonical(result["after"]),evidence,result["after"]["updated_at"],str(backup)))
        db.execute("UPDATE charges SET actual=?,status='settled',updated_at=? "
                   "WHERE request_id=? AND campaign=? AND status='unknown'",
                   (str(amount),result["after"]["updated_at"],request_id,campaign))
        result.update(applied=True,backup_path=str(backup))
    return result
