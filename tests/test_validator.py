import json
from pathlib import Path

from georsagent.taxonomy import REPO_ROOT, load_taxonomy
from georsagent.validator import STRICT_SCHEMA_PATH, build_strict_schema, validate_response

EXAMPLES = REPO_ROOT / "examples" / "controlled"


def _codes(issues, level=None):
    return [i.code for i in issues if level is None or i.level == level]


def _load(name):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))["response"]


def test_case_a_port_passes_with_alias_note():
    issues = validate_response(_load("case_a_port.json"), load_taxonomy())
    assert _codes(issues, "error") == []
    assert _codes(issues, "warning") == []
    assert "attribute_alias" in _codes(issues, "info")  # "water vehicles" -> water transport


def test_case_b_multiple_primary_labels_are_valid():
    issues = validate_response(_load("case_b_river_pond.json"), load_taxonomy())
    assert _codes(issues, "error") == []


def test_non_compliant_response_is_flagged_not_relabelled():
    response = {
        "scene_interpretation": {"predicted_class": "overpass", "confidence": "87%"},
        "attribute_analysis": {
            "objects": ["road", "blue"],
            "dominant_colours": ["turquoise"],
            "geometric_patterns": ["line"],
        },
        "alternative_classes": [
            {"class": "bridge", "reason": "-"},
            {"class": "island", "reason": "-"},
            {"class": "bridge", "reason": "-"},
        ],
        "explanation": "-",
        "uncertainty_note": "-",
        "semantic_score": 87,
    }
    original = json.dumps(response, sort_keys=True)
    codes = _codes(validate_response(response, load_taxonomy()))
    assert "non_canonical_primary" in codes
    assert "non_canonical_alternative" in codes
    assert "duplicate_alternative" in codes
    assert "attribute_wrong_group" in codes  # "blue" reported as an object
    assert "out_of_vocabulary_attribute" in codes  # "turquoise"
    assert "numeric_confidence" in codes
    assert "unexpected_field" in codes  # semantic_score
    assert json.dumps(response, sort_keys=True) == original  # audit never modifies the response


def test_missing_required_fields_fail_schema():
    issues = validate_response({"explanation": "x"}, load_taxonomy())
    assert "schema" in _codes(issues, "error")


def test_committed_strict_schema_is_in_sync_with_knowledge_base():
    committed = json.loads(Path(STRICT_SCHEMA_PATH).read_text(encoding="utf-8"))
    assert committed == build_strict_schema(load_taxonomy())
