from __future__ import annotations
import json
from pathlib import Path
from .common import digest

METHODS = {"single": (1, 1), "reflection": (1, 2), "majority": (6, 1), "debate": (3, 2)}
DEFAULT = dict(schema_version=1, backend="fake", stage="development",
    methods=list(METHODS), model="deepseek-v4.1-flash-1017",
    temperature=1.0, top_p=1.0, max_tokens=2048, timeout=120.0,
    seed_mode="per_call", seed_base=1234, extra_body={"thinking": {"type": "disabled"}},
    repeats=1, max_retries=1, retry_wait_seconds=1.0, budget_cny="20",
    campaign_id="gsm8k-20261109-v1", api_key_env="GSM_REPRO_API_KEY", base_url_env="GSM_REPRO_BASE_URL",
    parameters_verified=False, conditions_sha256=None,
    pricing={"confirmed": False, "input_per_million_cny": None, "output_per_million_cny": None,
             "input_token_reserve": 16384, "source": None, "verified_at": None})


def validate(value):
    if set(value) != set(DEFAULT):
        raise ValueError("Config fields differ from contract: " + str(sorted(set(value) ^ set(DEFAULT))))
    if value["schema_version"] != 1 or value["backend"] not in ("fake", "openai_compatible"):
        raise ValueError("Unsupported schema/backend")
    if value["stage"] not in ("development", "formal"):
        raise ValueError("Unknown stage")
    if not value["methods"] or len(value["methods"]) != len(set(value["methods"])) or set(value["methods"]) - METHODS.keys():
        raise ValueError("Invalid method list")
    if value["seed_mode"] not in ("per_call", "none"):
        raise ValueError("Unsupported seed mode")
    if value["max_retries"] not in (0, 1) or value["max_tokens"] < 1 or value["timeout"] <= 0:
        raise ValueError("Invalid request limits")
    if not isinstance(value["repeats"], int) or not 1 <= value["repeats"] <= 100:
        raise ValueError("Invalid repeats")
    if not 0 <= value["temperature"] <= 2 or not 0 <= value["top_p"] <= 1:
        raise ValueError("Invalid sampling values")
    if set(value["extra_body"]) - {"thinking", "enable_thinking", "top_k", "repetition_penalty"}:
        raise ValueError("Unsupported extra_body fields; cannot override model/messages")
    from decimal import Decimal
    if not 0 < Decimal(str(value["budget_cny"])) <= 20:
        raise ValueError("Campaign budget must be >0 and <=20 CNY")
    if not value["campaign_id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value["campaign_id"]):
        raise ValueError("Invalid campaign ID")
    return value


def load(path):
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def request(config, messages, call_key):
    fields = dict(model=config["model"], messages=messages, temperature=config["temperature"],
                  top_p=config["top_p"], max_tokens=config["max_tokens"], n=1,
                  stream=False, extra_body=config["extra_body"])
    if config["seed_mode"] == "per_call":
        fields["seed"] = int(digest([config["seed_base"], call_key])[:15], 16) % (2**31-1)
    return fields
