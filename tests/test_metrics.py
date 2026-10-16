import json

import pytest

from georsagent.metrics import (
    EvalRecord,
    check_table_consistency,
    evaluate,
    exact_match,
    load_equivalences,
    load_manuscript_table,
    load_raw_outputs,
    semantic_agreement,
    wilson_interval,
)
from georsagent.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture(scope="module")
def eq():
    return load_equivalences()


def test_manuscript_table_reproduces_reported_values(tax, eq):
    """Abstract, Sec. 5.5 and Sec. 7: 8/10 exact, 9/10 semantic, 3/10 fully compliant."""
    records = load_manuscript_table()
    result = evaluate(records, tax, eq)
    assert result["n_tests"] == 10
    assert result["exact_match"]["k"] == 8
    assert result["semantic_agreement"]["k"] == 9
    assert result["full_taxonomy_compliance"]["k"] == 3
    assert result["primary_label_canonical"]["k"] == 8


def test_manuscript_table_is_internally_consistent(tax):
    assert check_table_consistency(load_manuscript_table(), tax) == []


def test_exact_match_does_not_apply_synonyms(eq):
    assert exact_match(["Railway Station"], "railway station")
    assert not exact_match(["parking lot"], "parking")
    assert semantic_agreement(["parking lot"], "parking", eq)
    assert not semantic_agreement(["overpass"], "viaduct", eq)


def test_multiple_primary_labels_do_not_count_as_exact_match():
    assert not exact_match(["river", "pond"], "river")


def test_consistency_check_catches_contradictions(tax):
    bad = EvalRecord("X", "viaduct", ["overpass"], reported_fully_compliant=True)
    assert check_table_consistency([bad], tax)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_compliance_is_computed_from_raw_outputs(tmp_path, tax, eq):
    def write(test_id, ref, primary, alts):
        doc = {
            "test_id": test_id,
            "reference_class": ref,
            "response": {
                "scene_interpretation": {"predicted_class": primary, "confidence": "high"},
                "attribute_analysis": {"objects": [], "dominant_colours": [], "geometric_patterns": []},
                "alternative_classes": [{"class": a, "reason": "-"} for a in alts],
                "explanation": "-",
                "uncertainty_note": "-",
            },
        }
        (tmp_path / f"{test_id}.json").write_text(json.dumps(doc))

    write("T1", "beach", "beach", ["river", "island"])
    write("T2", "forest", "forest", ["park", "meadow"])
    write("T3", "parking", "parking lot", ["commercial"])
    result = evaluate(load_raw_outputs(tmp_path), tax, eq)
    assert [r["fully_compliant"] for r in result["per_test"]] == [False, True, False]
    assert result["exact_match"]["k"] == 2
    assert result["semantic_agreement"]["k"] == 3
