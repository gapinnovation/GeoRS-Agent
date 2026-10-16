"""Deterministic post-hoc audit of agent responses.

The prototype evaluated in the manuscript relies on prompt-level constraints
only. This module implements the deterministic output validation proposed as
future work (manuscript Sec. 6-7) as an *audit* tool: it reports problems and
never rewrites or relabels a response. A response can pass every check here
and still misdescribe the image; expert review remains necessary.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

from .metrics import primary_labels_from_response
from .taxonomy import ATTRIBUTE_GROUPS, REPO_ROOT, Taxonomy, normalize_label

SCHEMA_PATH = REPO_ROOT / "agent" / "output_schema.json"
STRICT_SCHEMA_PATH = REPO_ROOT / "agent" / "output_schema.strict.json"
KNOWN_TOP_LEVEL = {
    "scene_interpretation",
    "attribute_analysis",
    "alternative_classes",
    "explanation",
    "uncertainty_note",
}
_NUMERIC = re.compile(r"\d")


@dataclass
class Issue:
    level: str  # "error" | "warning" | "info"
    code: str
    path: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_schema(path: Path | str = SCHEMA_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_strict_schema(taxonomy: Taxonomy, base_schema: dict | None = None) -> dict:
    """Derive the strict-taxonomy schema from the structural schema and the knowledge base."""
    schema = copy.deepcopy(base_schema if base_schema is not None else load_schema())
    schema["title"] = "GeoRS-Agent structured interpretation (strict taxonomy)"
    schema["description"] = (
        "Strict variant of output_schema.json for deterministic output validation: every class field must be "
        "one of the 19 canonical WHU-RS19 labels and every attribute a canonical WHU-RS19 ABZSL attribute "
        "(exact lowercase strings). Proposed extension (manuscript Sec. 6-7); NOT enforced in the prototype "
        "evaluated in the manuscript. Generated from knowledge_base/ by `georsagent export-strict-schema`; "
        "do not edit by hand."
    )
    schema["$defs"] = {
        "canonicalClass": {"enum": list(taxonomy.classes)},
        "object": {"enum": list(taxonomy.attributes["objects"])},
        "dominantColour": {"enum": list(taxonomy.attributes["dominant_colours"])},
        "geometricPattern": {"enum": list(taxonomy.attributes["geometric_patterns"])},
    }
    props = schema["properties"]
    props["scene_interpretation"]["properties"]["predicted_class"]["oneOf"] = [
        {"$ref": "#/$defs/canonicalClass"},
        {"type": "array", "items": {"$ref": "#/$defs/canonicalClass"}, "minItems": 1, "uniqueItems": True},
    ]
    attrs = props["attribute_analysis"]["properties"]
    attrs["objects"]["items"] = {"$ref": "#/$defs/object"}
    attrs["dominant_colours"]["items"] = {"$ref": "#/$defs/dominantColour"}
    attrs["geometric_patterns"]["items"] = {"$ref": "#/$defs/geometricPattern"}
    props["alternative_classes"]["items"]["properties"]["class"] = {"$ref": "#/$defs/canonicalClass"}
    return schema


def _schema_issues(response: dict, schema: dict) -> list[Issue]:
    validator = Draft202012Validator(schema)
    issues = []
    for err in sorted(validator.iter_errors(response), key=lambda e: list(e.path)):
        path = "/".join(str(p) for p in err.path) or "(root)"
        issues.append(Issue("error", "schema", path, err.message))
    return issues


def validate_response(response: dict, taxonomy: Taxonomy, schema: dict | None = None) -> list[Issue]:
    """Audit one structured response. Returns a list of issues (empty = no problems found)."""
    issues = _schema_issues(response, schema if schema is not None else load_schema())
    if any(i.path in {"(root)", "scene_interpretation"} for i in issues):
        return issues  # structure too broken for content checks

    for key in response:
        if key not in KNOWN_TOP_LEVEL:
            issues.append(Issue("warning", "unexpected_field", key, f"Field '{key}' is not part of the output schema"))

    try:
        primaries = primary_labels_from_response(response)
    except (KeyError, TypeError):
        primaries = []
    for label in primaries:
        if not taxonomy.is_canonical_class(label):
            issues.append(
                Issue("error", "non_canonical_primary", "scene_interpretation/predicted_class",
                      f"'{label}' is not a WHU-RS19 class label")
            )

    raw_alternatives = response.get("alternative_classes")
    alternatives = [
        alt["class"]
        for alt in (raw_alternatives if isinstance(raw_alternatives, list) else [])
        if isinstance(alt, dict) and isinstance(alt.get("class"), str)
    ]
    primary_norm = {normalize_label(p) for p in primaries}
    seen = set()
    for idx, label in enumerate(alternatives):
        path = f"alternative_classes/{idx}/class"
        norm = normalize_label(label)
        if not taxonomy.is_canonical_class(label):
            issues.append(Issue("error", "non_canonical_alternative", path, f"'{label}' is not a WHU-RS19 class label"))
        if norm in primary_norm:
            issues.append(Issue("warning", "alternative_repeats_primary", path, f"'{label}' is also the predicted class"))
        if norm in seen:
            issues.append(Issue("warning", "duplicate_alternative", path, f"'{label}' is listed more than once"))
        seen.add(norm)

    analysis = response.get("attribute_analysis", {})
    for group in ATTRIBUTE_GROUPS:
        for idx, term in enumerate(analysis.get(group, []) if isinstance(analysis, dict) else []):
            path = f"attribute_analysis/{group}/{idx}"
            found = taxonomy.lookup_attribute(term)
            if found.status == "unknown":
                issues.append(Issue("warning", "out_of_vocabulary_attribute", path,
                                    f"'{term}' is not a WHU-RS19 ABZSL attribute"))
                continue
            if found.status == "alias":
                issues.append(Issue("info", "attribute_alias", path,
                                    f"'{term}' is an accepted alias of '{found.canonical}'"))
            if found.group != group:
                issues.append(Issue("warning", "attribute_wrong_group", path,
                                    f"'{term}' belongs to '{found.group}', not '{group}'"))

    confidence = response.get("scene_interpretation", {}).get("confidence")
    if isinstance(confidence, (int, float)) or (isinstance(confidence, str) and _NUMERIC.search(confidence)):
        issues.append(Issue("warning", "numeric_confidence", "scene_interpretation/confidence",
                            "Confidence should be a verbal descriptor; the prototype computes no numeric score"))
    return issues


def summarize_issues(issues: list[Issue]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for issue in issues:
        counts[issue.level] += 1
    return counts
