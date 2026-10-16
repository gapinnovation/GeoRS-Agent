"""WHU-RS19 class vocabulary and WHU-RS19 ABZSL attribute vocabulary.

The two vocabularies are loaded from ``knowledge_base/``. They play different
roles: the class vocabulary defines the labels permitted in the closed-set
evaluation; the attribute vocabulary defines the semantic evidence the agent
may report (manuscript Sec. 3.2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KB_DIR = REPO_ROOT / "knowledge_base"

ATTRIBUTE_GROUPS = ("objects", "dominant_colours", "geometric_patterns")


def normalize_label(label: object) -> str:
    """Normalise a label for comparison.

    Trims, case-folds and collapses internal whitespace. This is the only
    normalisation applied for exact matching (manuscript Sec. 4.7): no synonym
    replacement, no punctuation stripping, no camelCase splitting.
    """
    return " ".join(str(label).split()).casefold()


def split_camel_case(name: str) -> str:
    """Turn dataset folder names such as ``railwayStation`` into ``railway station``.

    Used only when deriving class labels from WHU-RS19 file paths, never when
    scoring agent outputs.
    """
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return normalize_label(spaced.replace("_", " ").replace("-", " "))


@dataclass(frozen=True)
class AttributeLookup:
    canonical: str | None
    group: str | None
    status: str  # "canonical" | "alias" | "unknown"


@dataclass(frozen=True)
class Taxonomy:
    classes: tuple[str, ...]
    attributes: dict[str, tuple[str, ...]]
    aliases: dict[str, str] = field(default_factory=dict)
    annotation_columns: dict[str, str] = field(default_factory=dict)

    @property
    def class_set(self) -> frozenset[str]:
        return frozenset(self.classes)

    @property
    def attribute_count(self) -> int:
        return sum(len(v) for v in self.attributes.values())

    def is_canonical_class(self, label: object) -> bool:
        return normalize_label(label) in self.class_set

    def group_of(self, canonical_attribute: str) -> str | None:
        for group, names in self.attributes.items():
            if canonical_attribute in names:
                return group
        return None

    def lookup_attribute(self, term: object) -> AttributeLookup:
        norm = normalize_label(term)
        group = self.group_of(norm)
        if group is not None:
            return AttributeLookup(norm, group, "canonical")
        if norm in self.aliases:
            canonical = self.aliases[norm]
            return AttributeLookup(canonical, self.group_of(canonical), "alias")
        return AttributeLookup(None, None, "unknown")


def load_taxonomy(kb_dir: Path | str = KB_DIR) -> Taxonomy:
    kb_dir = Path(kb_dir)
    class_doc = json.loads((kb_dir / "whu_rs19_classes.json").read_text(encoding="utf-8"))
    attr_doc = json.loads((kb_dir / "abzsl_attributes.json").read_text(encoding="utf-8"))

    classes = tuple(normalize_label(c["canonical"]) for c in class_doc["classes"])
    attributes: dict[str, tuple[str, ...]] = {}
    aliases: dict[str, str] = {}
    columns: dict[str, str] = {}
    for group in ATTRIBUTE_GROUPS:
        entries = attr_doc["groups"][group]
        attributes[group] = tuple(normalize_label(a["canonical"]) for a in entries)
        for a in entries:
            canonical = normalize_label(a["canonical"])
            for alias in a.get("aliases", []):
                aliases[normalize_label(alias["term"])] = canonical
            columns[a["annotation_column"]] = canonical
            columns[a["display_name"]] = canonical

    if len(set(classes)) != len(classes):
        raise ValueError("Duplicate class labels in whu_rs19_classes.json")
    all_attrs = [a for names in attributes.values() for a in names]
    if len(set(all_attrs)) != len(all_attrs):
        raise ValueError("Duplicate attribute names in abzsl_attributes.json")
    return Taxonomy(classes=classes, attributes=attributes, aliases=aliases, annotation_columns=columns)
