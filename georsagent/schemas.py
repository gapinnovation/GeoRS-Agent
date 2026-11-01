"""JSON schemas for every model output, plus deterministic validators.

Two layers of control, reported separately in the paper:
  1. the schema may be passed to Ollama as `format` (constrained decoding);
  2. validate_* always re-checks the parsed output in Python, independently.
Running with enforce_schema=False measures instruction-only compliance
(reviewer R3-4); enforce_schema=True measures the guardrail.
"""
from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from .vocab import ABSENT, ATTR_IDS, CLASSES, PRESENT, UNCERTAIN


def scene_schema(constrained_labels: bool) -> dict:
    label = {"type": "string", "enum": CLASSES} if constrained_labels else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "label": label,
            "alternatives": {"type": "array", "items": label, "maxItems": 2},
            "explanation": {"type": "string"},
        },
        "required": ["label", "alternatives", "explanation"],
        "additionalProperties": False,
    }


ATTRIBUTE_SCHEMA = {
    "type": "object",
    "properties": {
        a: {"type": "string", "enum": [PRESENT, ABSENT, UNCERTAIN]} for a in ATTR_IDS
    },
    "required": ATTR_IDS,
    "additionalProperties": False,
}

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "cited_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {"type": "string", "enum": ATTR_IDS},
                    "state": {"type": "string", "enum": [PRESENT, ABSENT]},
                },
                "required": ["attribute", "state"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["explanation", "cited_evidence"],
    "additionalProperties": False,
}


def parse_json(text: str):
    """Strict parse; returns (obj, error). No repair heuristics: a malformed
    output is a format failure and is counted as such."""
    try:
        return json.loads(text), None
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"json_decode: {e}"


def schema_errors(obj, schema) -> list[str]:
    return [e.message for e in Draft202012Validator(schema).iter_errors(obj)]


def taxonomy_check(obj: dict) -> dict:
    """Case-sensitive membership in the 19 admissible labels."""
    primary_ok = obj.get("label") in CLASSES
    alts = obj.get("alternatives") or []
    alts_ok = all(a in CLASSES for a in alts)
    return {"primary_in_taxonomy": primary_ok, "alternatives_in_taxonomy": alts_ok,
            "full_compliance": primary_ok and alts_ok}


def explanation_faithfulness(cited: list[dict], evidence: dict[str, str]) -> dict:
    """Each cited fact must match the evidence the scorer actually used.

    - citing an attribute as present/absent when the evidence says the
      opposite -> contradiction
    - citing an attribute that was uncertain / not provided -> unsupported
      (this is exactly the 'omitted is not absent' error, reviewer C1.8)
    """
    n = len(cited)
    contradicted, unsupported = [], []
    for c in cited:
        state = evidence.get(c["attribute"], UNCERTAIN)
        if state == UNCERTAIN:
            unsupported.append(c["attribute"])
        elif state != c["state"]:
            contradicted.append(c["attribute"])
    ok = n - len(contradicted) - len(unsupported)
    return {"n_cited": n, "n_consistent": ok,
            "contradicted": contradicted, "unsupported": unsupported,
            "faithful": n > 0 and ok == n}
