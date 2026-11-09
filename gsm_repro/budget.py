from __future__ import annotations
import sqlite3
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from .common import utcnow

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
                raise BudgetStop("A previous request has unconfirmed cost; inspect billing before continuing")
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
