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


# B1_KB / B1_VLMKB (added after the primary analysis, at reviewer request):
# same as B1 + the WHOLE class-attribute knowledge base as text in the prompt,
# so that the SAME model can use the same knowledge the deterministic scorer uses.
# Values = percentage of training images of the class in which the attribute is
# present (p_present of the KB, rounded), listed for all 38 attributes, sorted by
# decreasing value. Pre-specified before running; do not edit afterwards.
def kb_text(kb: dict) -> str:
    lines = []
    for c in kb["classes"]:
        p = kb["profiles"][c]["p_present"]
        items = sorted(ATTR_IDS, key=lambda a: (-p[a], ATTR_IDS.index(a)))
        lines.append(f"{c}: " + ", ".join(f"{a} {round(100 * p[a])}" for a in items))
    return "\n".join(lines)


def user_b1_kb(kb: dict) -> str:
    return (
        "Classify the scene shown in this aerial/satellite image.\n"
        "You MUST choose the label and the alternatives only from this list, written exactly as shown:\n"
        + ", ".join(CLASSES) + "\n"
        "Reference knowledge base. Attribute definitions (id: definition):\n"
        + "\n".join(f"- {a}: {ATTR_DEF[a]}" for a in ATTR_IDS) + "\n"
        "For each class, the percentage of training images of that class in which each "
        "attribute is present (all 38 attributes, highest first):\n"
        + kb_text(kb) + "\n"
        "Use the knowledge base together with what is visible in the image.\n"
        'Return JSON: {"label": <one class from the list>, "alternatives": [<up to 2 other classes from the list>], '
        '"explanation": <one or two sentences citing visible evidence>}.'
    )


def prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\n\x1e\n".join(parts).encode()).hexdigest()[:16]
