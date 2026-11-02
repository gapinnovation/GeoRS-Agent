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


def build_kb_from_evidence(evidence_by_image: dict, labels: dict, alpha: float = 1.0,
                           source: str = "") -> dict:
    """Experiment A: class-attribute profiles estimated from attributes that the
    VLM itself extracted on TRAIN images (same prompt and model as B2), instead
    of the human annotations. 'uncertain' answers are treated as missing for
    that attribute (excluded from numerator and denominator).

    evidence_by_image: {image_id: {attr: present|absent|uncertain}}
    labels:            {image_id: class}
    """
    profiles = {}
    for c in CLASSES:
        ids = [i for i in evidence_by_image if labels[i] == c]
        if not ids:
            raise ValueError(f"class {c} has no training images with VLM evidence")
        p = {}
        for a in ATTR_IDS:
            obs = [evidence_by_image[i].get(a) for i in ids]
            k = sum(s == "present" for s in obs)
            n = sum(s in ("present", "absent") for s in obs)
            p[a] = (k + alpha) / (n + 2 * alpha)
        profiles[c] = {"n_images": len(ids), "p_present": p}
    kb = {"version": "1.0-vlm", "source": source, "smoothing_alpha": alpha,
          "train_image_ids": sorted(evidence_by_image), "classes": CLASSES,
          "attributes": ATTR_IDS, "attribute_groups": ATTR_GROUP, "profiles": profiles}
    kb["sha256"] = kb_hash(kb)
    return kb
