import csv
import json

import pytest

from georsagent.profiles import build_profiles, write_prevalence_csv, write_profiles_markdown
from georsagent.taxonomy import KB_DIR, load_taxonomy


def _original_columns():
    doc = json.loads((KB_DIR / "abzsl_attributes.json").read_text(encoding="utf-8"))
    return [a["annotation_column"] for group in doc["groups"].values() for a in group]


def _write_annotations(path, rows, tax):
    """Synthetic file mimicking the ABZSL layout: Filepath, label, 38 original attribute columns, split."""
    columns = _original_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["Filepath", "Label", *columns, "Split"])
        for filepath, split, present in rows:
            w.writerow([filepath, "x", *(int(tax.annotation_columns[c] in present) for c in columns), split])


@pytest.fixture
def annotations(tmp_path):
    tax = load_taxonomy()
    rows = [
        ("Train\\Port\\port_01.jpg", "train", {"water", "dock", "water transport"}),
        ("Train\\Port\\port_02.jpg", "train", {"water", "dock"}),
        ("Train\\Port\\port_03.jpg", "test", {"water"}),
        ("Train\\railwayStation\\rs_01.jpg", "train", {"rails", "trains"}),
        ("Train\\railwayStation\\rs_02.jpg", "train", {"rails"}),
    ]
    path = tmp_path / "annotations.csv"
    _write_annotations(path, rows, tax)
    return path, tax


def test_prevalence_uses_training_rows_only(annotations):
    path, tax = annotations
    table = build_profiles(path, tax, split_column="Split", train_value="train")
    assert table.classes == ["port", "railway station"]
    assert table.counts == {"port": 2, "railway station": 2}
    assert table.prevalence["port"]["dock"] == 1.0
    assert table.prevalence["port"]["water transport"] == 0.5
    assert table.prevalence["railway station"]["trains"] == 0.5


def test_exclusion_list_removes_evaluation_images(annotations):
    path, tax = annotations
    table = build_profiles(path, tax, exclude={"port_01.jpg"})
    assert table.counts["port"] == 2  # port_02 + port_03
    assert table.prevalence["port"]["water transport"] == 0.0


def test_outputs_are_written(annotations, tmp_path):
    path, tax = annotations
    table = build_profiles(path, tax, split_column="Split")
    write_prevalence_csv(table, tmp_path / "prev.csv")
    write_profiles_markdown(table, tax, tmp_path / "profiles.md")
    text = (tmp_path / "profiles.md").read_text(encoding="utf-8")
    assert "## port (n = 2)" in text
    assert "dock (100%" in text
