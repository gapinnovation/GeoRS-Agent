"""Derive class-attribute profiles from WHU-RS19 ABZSL annotations.

The profiles summarise which attributes are commonly present in each class.
They are descriptive reference context for the agent, not decision rules.

To avoid leakage, build them from training images only, or exclude the images
used in the evaluation (manuscript Sec. 4.6 notes that the original record does
not establish whether the test images were excluded).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from .taxonomy import Taxonomy, normalize_label, split_camel_case

_TRUE = {"1", "1.0", "true", "yes", "y", "x"}
_FALSE = {"0", "0.0", "false", "no", "n", ""}


@dataclass
class ProfileTable:
    classes: list[str]
    attributes: list[str]  # canonical attribute names, ABZSL order
    counts: dict[str, int]  # images per class
    prevalence: dict[str, dict[str, float]]  # class -> attribute -> fraction of images


def _presence(value: str, column: str, row_no: int) -> int:
    v = value.strip().casefold()
    if v in _TRUE:
        return 1
    if v in _FALSE:
        return 0
    raise ValueError(f"Row {row_no}, column '{column}': cannot read {value!r} as presence/absence")


def _class_from_path(path_value: str, taxonomy: Taxonomy) -> str:
    parent = PureWindowsPath(path_value.replace("/", "\\")).parent.name
    label = split_camel_case(parent)
    if not taxonomy.is_canonical_class(label):
        raise ValueError(f"Cannot map folder '{parent}' (from '{path_value}') to a WHU-RS19 class")
    return label


def _sniff_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        return ","


def build_profiles(
    annotations: Path | str,
    taxonomy: Taxonomy,
    class_column: str | None = None,
    path_column: str = "Filepath",
    split_column: str | None = None,
    train_value: str | None = None,
    exclude: set[str] | None = None,
    delimiter: str | None = None,
) -> ProfileTable:
    """Compute per-class attribute prevalence.

    Class labels come from ``class_column`` if given, otherwise from the parent
    folder of ``path_column`` (WHU-RS19 layout, e.g. ``railwayStation/x.jpg``).
    Attribute columns are recognised by their original (Italian) or English
    names as listed in ``knowledge_base/abzsl_attributes.json``.
    """
    annotations = Path(annotations)
    delimiter = delimiter or _sniff_delimiter(annotations)
    exclude_names = {Path(e).name for e in (exclude or set())}
    canonical_order = [a for names in taxonomy.attributes.values() for a in names]

    with open(annotations, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        header = reader.fieldnames or []
        column_map = {col: taxonomy.annotation_columns[col.strip()] for col in header
                      if col.strip() in taxonomy.annotation_columns}
        missing = set(canonical_order) - set(column_map.values())
        if missing:
            raise ValueError(f"Attribute columns not found in {annotations.name}: {sorted(missing)}")
        if split_column and split_column not in header:
            raise ValueError(f"Split column '{split_column}' not found")
        if class_column and class_column not in header:
            raise ValueError(f"Class column '{class_column}' not found")

        totals: dict[str, dict[str, int]] = {}
        counts: dict[str, int] = {}
        for row_no, row in enumerate(reader, start=2):
            if split_column and normalize_label(row[split_column]) != normalize_label(train_value or "train"):
                continue
            if exclude_names and path_column in row:
                if PureWindowsPath(row[path_column].replace("/", "\\")).name in exclude_names:
                    continue
            if class_column:
                label = normalize_label(row[class_column])
                if not taxonomy.is_canonical_class(label):
                    label = split_camel_case(row[class_column])
                if not taxonomy.is_canonical_class(label):
                    raise ValueError(f"Row {row_no}: unknown class '{row[class_column]}'")
            else:
                label = _class_from_path(row[path_column], taxonomy)
            counts[label] = counts.get(label, 0) + 1
            bucket = totals.setdefault(label, {a: 0 for a in canonical_order})
            for col, attr in column_map.items():
                bucket[attr] += _presence(row[col], col, row_no)

    classes = [c for c in taxonomy.classes if c in counts]
    prevalence = {c: {a: totals[c][a] / counts[c] for a in canonical_order} for c in classes}
    return ProfileTable(classes=classes, attributes=canonical_order, counts=counts, prevalence=prevalence)


def write_prevalence_csv(table: ProfileTable, path: Path | str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "n_images", *table.attributes])
        for c in table.classes:
            writer.writerow([c, table.counts[c], *(f"{table.prevalence[c][a]:.3f}" for a in table.attributes)])


def write_profiles_markdown(
    table: ProfileTable,
    taxonomy: Taxonomy,
    path: Path | str,
    typical_threshold: float = 0.5,
    distinctive_margin: float = 0.3,
    source_note: str = "",
) -> None:
    """Readable per-class summary for use as agent reference context.

    *Typical*: present in at least ``typical_threshold`` of the class's images.
    *Distinctive*: prevalence in the class exceeds the mean prevalence over the
    other classes by at least ``distinctive_margin``.
    """
    lines = [
        "# WHU-RS19 class-attribute profiles",
        "",
        "Descriptive summaries derived from WHU-RS19 ABZSL annotations. They are reference context, not "
        "classification rules: a class can be plausible without some typical cues, and cues are shared across classes.",
        "",
        f"- Typical: attribute present in at least {typical_threshold:.0%} of the class's images.",
        f"- Distinctive: prevalence at least {distinctive_margin * 100:.0f} percentage points above the mean over the other classes.",
    ]
    if source_note:
        lines.append(f"- Source: {source_note}")
    lines.append("")
    for c in table.classes:
        others = [o for o in table.classes if o != c]
        lines.append(f"## {c} (n = {table.counts[c]})")
        for group in taxonomy.attributes:
            typical = [a for a in taxonomy.attributes[group] if table.prevalence[c][a] >= typical_threshold]
            label = group.replace("_", " ")
            lines.append(f"- Typical {label}: " + (", ".join(
                f"{a} ({table.prevalence[c][a]:.0%})" for a in typical) or "none"))
        distinctive = []
        for a in table.attributes:
            if not others:
                break
            mean_other = sum(table.prevalence[o][a] for o in others) / len(others)
            if table.prevalence[c][a] - mean_other >= distinctive_margin:
                distinctive.append(f"{a} ({table.prevalence[c][a]:.0%} vs {mean_other:.0%})")
        lines.append("- Distinctive cues: " + (", ".join(distinctive) or "none"))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
