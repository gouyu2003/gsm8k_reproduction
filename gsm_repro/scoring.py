from __future__ import annotations
import ast
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from .common import file_hash

NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)")
TOKEN = re.compile(r"(?<![\w.,])" + NUMBER.pattern + r"(?![\w.,])")
AUTHOR_FUNCTIONS = {"solve_math_problems", "parse_answer", "compute_accuracy", "most_frequent"}


def number(text):
    value = text.strip().replace("\u2212", "-")
    value = re.sub(r"^(?:\\\$|\$)\s*", "", value)
    while value.startswith("{") and value.endswith("}"):
        value = value[1:-1].strip()
    if not NUMBER.fullmatch(value) or len(value) > 1000:
        return None
    try:
        parsed = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    return value.replace(",", "") if parsed.is_finite() else None


def parse_gold(text):
    if "####" not in text:
        raise ValueError("Gold must have #### marker")
    value = number(text.rsplit("####", 1)[1])
    if value is None:
        raise ValueError("Gold must be one finite numeric answer")
    return value


def boxed_values(text):
    found = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, level, end = match.end(), 1, match.end()
        while end < len(text) and level:
            level += (text[end] == "{") - (text[end] == "}")
            end += 1
        found.append(text[start:end-1] if level == 0 else None)
    return found


def parse_prediction(text):
    if not isinstance(text, str):
        return None
    boxes = boxed_values(text)
    if boxes:
        values = [number(b) if b is not None else None for b in boxes]
        if any(v is None for v in values):
            return None
        if len({Decimal(v) for v in values}) > 1:
            return None
        return values[-1]
    # Unsupported fraction as a final expression must not become its denominator.
    if re.search(r"(?:\\(?:d?frac)\s*\{[^}]*\}\s*\{[^}]*\}|\d+\s*/\s*\d+)\s*[.!]?\s*$", text):
        return None
    for match in reversed(list(re.finditer(r"\{([^{}]*)\}", text))):
        candidate = number(match.group(1))
        if candidate is not None:
            return candidate
    matches = list(TOKEN.finditer(text))
    return number(matches[-1].group()) if matches else None


def fixed_score(gold, outputs, *, tie_error=False, exact=False):
    target = parse_gold(gold)
    parsed = [parse_prediction(output) for output in outputs]
    valid = [v for v in parsed if v is not None]
    counts = Counter(valid)
    if not valid:
        return dict(correct=0, prediction=None, parsed=parsed, parse_failure=True, tie=False, votes={})
    top = max(counts.values())
    winners = [v for v in valid if counts[v] == top]
    prediction = winners[0]  # Agent order, not sorted numeric order.
    tie = len(set(winners)) > 1
    a, b = Decimal(prediction), Decimal(target)
    with localcontext() as ctx:
        ctx.prec = max(50, len(a.as_tuple().digits) + abs(a.as_tuple().exponent) + 2,
                       len(b.as_tuple().digits) + abs(b.as_tuple().exponent) + 2)
        correct = a == b if exact else abs(a-b) <= Decimal("0.000001")
    return dict(correct=int(correct and not (tie_error and tie)), prediction=prediction,
                parsed=parsed, parse_failure=False, tie=tie, votes=dict(counts))


def author_score(gold, outputs, reference):
    reference = Path(reference)
    tree = ast.parse(reference.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in AUTHOR_FUNCTIONS]
    if {n.name for n in nodes} != AUTHOR_FUNCTIONS:
        raise ValueError("Incomplete author scorer")
    # New namespace on every item: the original global cannot leak across questions.
    namespace = {"re": re, "pred_solutions": list(outputs)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference), "exec"), namespace)
    try:
        value = namespace["compute_accuracy"](gold, list(outputs))
        if value is None:
            return dict(correct=None, status="author_scoring_error", error_type="MissingGold")
        return dict(correct=int(value), status="scored")
    except (ValueError, TypeError, IndexError, OverflowError, ZeroDivisionError) as exc:
        return dict(correct=None, status="author_scoring_error", error_type=type(exc).__name__)


def score_all(gold, outputs, reference):
    return {
        "author_faithful": author_score(gold, outputs, reference),
        "fixed": fixed_score(gold, outputs),
        "sensitivity.tie_error": fixed_score(gold, outputs, tie_error=True),
        "sensitivity.exact": fixed_score(gold, outputs, exact=True),
    }


def failed_scores():
    base = dict(correct=0, status="execution_failure", prediction=None, parse_failure=False)
    return {"author_faithful": dict(correct=None, status="not_scored_execution_failure"),
            **{name: dict(base) for name in ("fixed", "sensitivity.tie_error", "sensitivity.exact")}}


def summary(rows):
    result = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        own = [r for r in rows if r["method"] == method]
        metrics = {}
        for variant in ("author_faithful", "fixed", "sensitivity.tie_error", "sensitivity.exact"):
            values = [r["scores"][variant]["correct"] for r in own]
            n = len(values)
            available = sum(v is not None for v in values)
            accuracy = sum(values)/n if n and available == n else None
            metrics[variant] = dict(n=n, scored=available, correct=sum(v or 0 for v in values),
                                    accuracy=accuracy,
                                    standard_error=math.sqrt(accuracy*(1-accuracy)/n) if accuracy is not None else None)
        result[method] = dict(samples=len(own), completed=sum(r["status"] == "completed" for r in own),
                              failed=sum(r["status"] in ("failed", "interrupted") for r in own), metrics=metrics)
    return result
