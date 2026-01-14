"""Deterministic information degradation metrics.

Provides programmatic (non-LLM) metrics for measuring information loss through
multi-agent workflows.

Metrics implemented:
- JSON structure preservation: key overlap and value retention
- Entity extraction: names, dates, amounts (heuristic, deterministic)
- Constraint retention/violation: measure mentions and violations

Optional:
- semantic similarity via embeddings is designed as an interface but not enabled
  by default to keep the benchmark deterministic and dependency-light.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class JsonPreservationResult:
    key_precision: float
    key_recall: float
    key_f1: float
    value_exact_match_rate: float


@dataclass
class EntityExtractionResult:
    names: Set[str]
    dates: Set[str]
    amounts: Set[str]


@dataclass
class ConstraintResult:
    constraints_total: int
    constraints_mentioned: int
    constraints_violated: int


def _try_extract_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction.

    - If text contains a fenced json block, parse that.
    - Else try to parse the first {...} or [...] span.
    - Returns None if not parseable.
    """
    # fenced block
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            candidate = text[start:end].strip()
            try:
                return json.loads(candidate)
            except Exception:
                return None

    # naive bracket span
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(1)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            items.update(_flatten_json(v, p))
    elif isinstance(obj, list):
        # lists are stored as a value at the current path
        items[prefix] = obj
    else:
        items[prefix] = obj
    return items


def json_structure_preservation(expected: Any, observed_text: str) -> Optional[JsonPreservationResult]:
    observed = _try_extract_json(observed_text)
    if observed is None:
        return None

    exp_flat = _flatten_json(expected)
    obs_flat = _flatten_json(observed)

    exp_keys = set(exp_flat.keys())
    obs_keys = set(obs_flat.keys())

    if not exp_keys and not obs_keys:
        return JsonPreservationResult(1.0, 1.0, 1.0, 1.0)

    intersect = exp_keys & obs_keys
    precision = len(intersect) / len(obs_keys) if obs_keys else 0.0
    recall = len(intersect) / len(exp_keys) if exp_keys else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Exact match rate on overlapping keys
    if not intersect:
        value_rate = 0.0
    else:
        exact = 0
        for k in intersect:
            if exp_flat.get(k) == obs_flat.get(k):
                exact += 1
        value_rate = exact / len(intersect)

    return JsonPreservationResult(
        key_precision=precision,
        key_recall=recall,
        key_f1=f1,
        value_exact_match_rate=value_rate,
    )


_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")


def extract_entities(text: str) -> EntityExtractionResult:
    names = set(m.group(1) for m in _NAME_RE.finditer(text))
    dates = set(m.group(0) for m in _DATE_RE.finditer(text))
    amounts = set(m.group(0).replace(" ", "") for m in _AMOUNT_RE.finditer(text))
    return EntityExtractionResult(names=names, dates=dates, amounts=amounts)


def entity_retention(expected_text: str, observed_text: str) -> dict[str, float]:
    exp = extract_entities(expected_text)
    obs = extract_entities(observed_text)

    def rate(exp_set: Set[str], obs_set: Set[str]) -> float:
        if not exp_set:
            return 1.0
        return len(exp_set & obs_set) / len(exp_set)

    return {
        "names_retention": rate(exp.names, obs.names),
        "dates_retention": rate(exp.dates, obs.dates),
        "amounts_retention": rate(exp.amounts, obs.amounts),
    }


def constraint_metrics(constraints: List[str], observed_text: str) -> ConstraintResult:
    if not constraints:
        return ConstraintResult(0, 0, 0)

    mentioned = 0
    violated = 0

    lower = observed_text.lower()
    for c in constraints:
        c_lower = c.lower()
        if c_lower in lower:
            mentioned += 1

        # Very simple deterministic violation heuristics:
        # - If constraint includes "must not" and observed contains a negation flip like "you can" or "allowed"
        if "must not" in c_lower and ("you can" in lower or "allowed" in lower or "ok to" in lower):
            violated += 1
        if "do not" in c_lower and ("you can" in lower or "allowed" in lower or "ok to" in lower):
            violated += 1

    return ConstraintResult(
        constraints_total=len(constraints),
        constraints_mentioned=mentioned,
        constraints_violated=violated,
    )
