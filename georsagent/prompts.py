"""All prompts, verbatim. Their sha256 is written into every log record."""
from __future__ import annotations

import hashlib

from .vocab import ATTR_DEF, ATTR_IDS, CLASSES

SYSTEM_SCENE = (
    "You are an expert in remote sensing image interpretation. "
    "Answer only with a JSON object, no other text."
)

# B0: direct prompting, no class list
USER_B0 = (
    "Classify the scene shown in this aerial/satellite image.\n"
    'Return JSON: {"label": <scene class>, "alternatives": [<up to 2 other plausible classes>], '
    '"explanation": <one or two sentences citing visible evidence>}.'
)

# B1: same model + the 19 admissible classes
USER_B1 = (
    "Classify the scene shown in this aerial/satellite image.\n"
    "You MUST choose the label and the alternatives only from this list, written exactly as shown:\n"
    + ", ".join(CLASSES) + "\n"
    'Return JSON: {"label": <one class from the list>, "alternatives": [<up to 2 other classes from the list>], '
    '"explanation": <one or two sentences citing visible evidence>}.'
)

SYSTEM_ATTR = (
    "You are an expert in remote sensing image interpretation. You report only what is "
    "visible in the image. Answer only with a JSON object, no other text."
)

USER_ATTR = (
    "For each attribute below, state whether it is visible in this aerial/satellite image.\n"
    'Use "present" if clearly visible, "absent" if clearly not in the scene, '
    '"uncertain" if you cannot tell.\n'
    "Attributes (id: definition):\n"
    + "\n".join(f"- {a}: {ATTR_DEF[a]}" for a in ATTR_IDS)
    + "\nReturn a JSON object with exactly these keys and one of the three values for each."
)

SYSTEM_EXPLAIN = (
    "You write short, faithful explanations of a remote sensing scene classification. "
    "You may cite ONLY the evidence listed in the input, with the state given there. "
    "Never mention attributes that are not listed. Answer only with JSON."
)


def user_explain(label: str, runner_up: str, supporting: list[tuple[str, str, float]]) -> str:
    lines = [f"- {a} = {state} (weight {w:+.2f})" for a, state, w in supporting]
    return (
        f"The deterministic classifier selected '{label}' over '{runner_up}'.\n"
        "Evidence that most supported this decision:\n" + "\n".join(lines) + "\n"
        'Return JSON: {"explanation": <2-3 sentences>, "cited_evidence": '
        '[{"attribute": <id>, "state": "present"|"absent"}, ...]}. '
        "Cite at least one item and only items from the list above."
    )


def prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\n\x1e\n".join(parts).encode()).hexdigest()[:16]
