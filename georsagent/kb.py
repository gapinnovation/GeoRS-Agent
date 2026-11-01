"""Class-attribute knowledge base, built from the TRAIN split only.

Each profile stores p(attribute present | class) with Laplace smoothing.
Test images never contribute to the profiles (no leakage, reviewer C1.4).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .vocab import ATTR_GROUP, ATTR_IDS, CLASSES


def build_kb(tidy: pd.DataFrame, alpha: float = 1.0) -> dict:
    train = tidy[tidy["split"] == "train"]
    if train.empty:
        raise ValueError("empty train split")
    profiles = {}
    for c in CLASSES:
        sub = train[train["label"] == c]
        n = len(sub)
        if n == 0:
            raise ValueError(f"class {c} has no training images")
        profiles[c] = {
            "n_images": int(n),
            "p_present": {a: (float(sub[a].sum()) + alpha) / (n + 2 * alpha) for a in ATTR_IDS},
        }
    kb = {
        "version": "1.0",
        "source": "WHU-RS19 ABZSL annotations, sheet '40 classi REF', split=train",
        "smoothing_alpha": alpha,
        "train_image_ids": sorted(train["image_id"].tolist()),
        "classes": CLASSES,
        "attributes": ATTR_IDS,
        "attribute_groups": ATTR_GROUP,
        "profiles": profiles,
    }
    kb["sha256"] = kb_hash(kb)
    return kb


def kb_hash(kb: dict) -> str:
    body = {k: v for k, v in kb.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def save_kb(kb: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(kb, indent=1, sort_keys=True))


def load_kb(path: str | Path) -> dict:
    kb = json.loads(Path(path).read_text())
    if kb.get("sha256") != kb_hash(kb):
        raise ValueError("knowledge base hash mismatch: file was modified")
    return kb


def assert_no_leakage(kb: dict, test_ids) -> None:
    overlap = set(kb["train_image_ids"]) & set(test_ids)
    if overlap:
        raise AssertionError(f"{len(overlap)} test images used to build the KB")
