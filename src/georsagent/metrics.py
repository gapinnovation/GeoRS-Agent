"""Outcome measures for the image-based evaluation (manuscript Sec. 4.7).

Three measures are reported in the manuscript:

* exact-match class accuracy — primary label equals the reference class after
  case normalisation, without synonym replacement;
* semantic class agreement — exact match, or a manually defined equivalence
  (``evaluation/label_equivalences.json``), reported separately;
* full-response taxonomy compliance — the primary label and every alternative
  are canonical WHU-RS19 class names.

Compliance measures vocabulary adherence, not factual correctness.
Wilson score intervals are added here as a descriptive indication of sampling
uncertainty with n = 10; they are not reported in the manuscript.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .taxonomy import REPO_ROOT, Taxonomy, normalize_label

DEFAULT_TABLE = REPO_ROOT / "evaluation" / "image_tests.csv"
DEFAULT_EQUIVALENCES = REPO_ROOT / "evaluation" / "label_equivalences.json"
Z_95 = 1.959963984540054


@dataclass
class EvalRecord:
    test_id: str
    reference_class: str
    primary_labels: list[str]
    alternatives: list[str] | None = None  # None = full list unknown
    reported_fully_compliant: bool | None = None
    noncanonical_alternatives_reported: list[str] = field(default_factory=list)
    observation: str = ""


def load_equivalences(path: Path | str = DEFAULT_EQUIVALENCES) -> set[tuple[str, str]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        (normalize_label(e["label"]), normalize_label(e["canonical_class"]))
        for e in doc["equivalences"]
    }


def exact_match(primary_labels: list[str], reference: str) -> bool:
    """A response matches only if it gives a single primary label equal to the reference.

    Responses with several co-equal primary labels are counted as non-matching,
    so that hedging cannot inflate accuracy. (All ten image responses in the
    manuscript have a single primary label.)
    """
    return len(primary_labels) == 1 and normalize_label(primary_labels[0]) == normalize_label(reference)


def semantic_agreement(primary_labels: list[str], reference: str, equivalences: set[tuple[str, str]]) -> bool:
    if exact_match(primary_labels, reference):
        return True
    if len(primary_labels) != 1:
        return False
    return (normalize_label(primary_labels[0]), normalize_label(reference)) in equivalences


def fully_compliant(primary_labels: list[str], alternatives: list[str], taxonomy: Taxonomy) -> bool:
    return all(taxonomy.is_canonical_class(label) for label in [*primary_labels, *alternatives])


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must lie in [0, n]")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def _parse_bool(value: str) -> bool | None:
    v = value.strip().casefold()
    if v in {"true", "yes", "1"}:
        return True
    if v in {"false", "no", "0"}:
        return False
    if v in {"", "na", "unknown"}:
        return None
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def load_manuscript_table(path: Path | str = DEFAULT_TABLE) -> list[EvalRecord]:
    """Load the per-image results as reported in the manuscript (Table 3)."""
    records = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            records.append(
                EvalRecord(
                    test_id=row["test_id"],
                    reference_class=row["reference_class"],
                    primary_labels=[row["predicted_label"]],
                    alternatives=None,
                    reported_fully_compliant=_parse_bool(row["fully_compliant"]),
                    noncanonical_alternatives_reported=_split_list(row["noncanonical_alternatives_reported"]),
                    observation=row.get("observation", ""),
                )
            )
    return records


def primary_labels_from_response(response: dict) -> list[str]:
    predicted = response["scene_interpretation"]["predicted_class"]
    return [predicted] if isinstance(predicted, str) else list(predicted)


def alternatives_from_response(response: dict) -> list[str]:
    return [alt["class"] for alt in response.get("alternative_classes", [])]


def load_raw_outputs(directory: Path | str) -> list[EvalRecord]:
    """Load original agent responses (one JSON file per test)."""
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        response = doc["response"]
        records.append(
            EvalRecord(
                test_id=doc["test_id"],
                reference_class=doc["reference_class"],
                primary_labels=primary_labels_from_response(response),
                alternatives=alternatives_from_response(response),
            )
        )
    return records


def check_table_consistency(records: list[EvalRecord], taxonomy: Taxonomy) -> list[str]:
    """Return contradictions between reported compliance and the labels listed.

    A row cannot be fully compliant if its primary label or any listed
    alternative is non-canonical.
    """
    problems = []
    for r in records:
        visible_noncanonical = [
            label
            for label in [*r.primary_labels, *r.noncanonical_alternatives_reported]
            if not taxonomy.is_canonical_class(label)
        ]
        if r.reported_fully_compliant and visible_noncanonical:
            problems.append(
                f"{r.test_id}: reported fully compliant but lists non-canonical label(s) {visible_noncanonical}"
            )
        for label in r.noncanonical_alternatives_reported:
            if taxonomy.is_canonical_class(label):
                problems.append(f"{r.test_id}: '{label}' is listed as non-canonical but is a canonical class")
    return problems


def evaluate(records: list[EvalRecord], taxonomy: Taxonomy, equivalences: set[tuple[str, str]]) -> dict:
    """Compute per-test outcomes and the three summary measures."""
    rows = []
    for r in records:
        if r.alternatives is not None:
            compliant = fully_compliant(r.primary_labels, r.alternatives, taxonomy)
            compliance_source = "computed"
        elif r.reported_fully_compliant is not None:
            compliant = r.reported_fully_compliant
            compliance_source = "reported"
        else:
            compliant = None
            compliance_source = "unknown"
        rows.append(
            {
                "test_id": r.test_id,
                "reference_class": r.reference_class,
                "primary": "; ".join(r.primary_labels),
                "primary_canonical": all(taxonomy.is_canonical_class(p) for p in r.primary_labels),
                "exact_match": exact_match(r.primary_labels, r.reference_class),
                "semantic_agreement": semantic_agreement(r.primary_labels, r.reference_class, equivalences),
                "fully_compliant": compliant,
                "compliance_source": compliance_source,
            }
        )

    n = len(rows)

    def summary(key: str) -> dict:
        values = [row[key] for row in rows if row[key] is not None]
        k, m = sum(bool(v) for v in values), len(values)
        lo, hi = wilson_interval(k, m) if m else (float("nan"), float("nan"))
        return {"k": k, "n": m, "proportion": k / m if m else float("nan"), "wilson95": [lo, hi]}

    return {
        "n_tests": n,
        "exact_match": summary("exact_match"),
        "semantic_agreement": summary("semantic_agreement"),
        "full_taxonomy_compliance": summary("fully_compliant"),
        "primary_label_canonical": summary("primary_canonical"),
        "per_test": rows,
    }
