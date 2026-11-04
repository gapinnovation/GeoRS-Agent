"""Data ingestion: workbook (raw, never modified) -> tidy CSV (processed)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .vocab import ATTR_IDS, CLASS_MAP, IT2ATTR

SHEET = "40 classi REF"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_workbook(xlsx: str | Path) -> pd.DataFrame:
    """Read the reference sheet and return one row per image with canonical ids.

    Columns: image_id, filepath, label, split, <38 attribute ids as 0/1 int>.
    Raises on any inconsistency rather than silently fixing it.
    """
    df = pd.read_excel(xlsx, sheet_name=SHEET)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")

    missing = set(IT2ATTR) - set(df.columns)
    if missing:
        raise ValueError(f"attribute columns missing from workbook: {missing}")
    unknown = set(df["Classe"]) - set(CLASS_MAP)
    if unknown:
        raise ValueError(f"unknown class names: {unknown}")

    out = pd.DataFrame({
        "filepath": df["Filepath"].astype(str).str.replace("\\", "/", regex=False),
        "label": df["Classe"].map(CLASS_MAP),
        "split": df["Partition"].str.strip().str.lower(),
    })
    out["image_id"] = out["filepath"].map(lambda p: Path(p).stem)
    for it, en in IT2ATTR.items():
        out[en] = df[it].astype(int)

    # Sanity checks (§6 checkpoints): split must agree with path prefix,
    # ids unique, attributes strictly binary, folder name consistent with label.
    prefix = out["filepath"].str.split("/").str[0].str.lower()
    if not (prefix == out["split"]).all():
        raise ValueError("Partition column disagrees with Filepath prefix")
    if out["image_id"].duplicated().any():
        raise ValueError("duplicate image ids")
    if not out[ATTR_IDS].isin([0, 1]).all().all():
        raise ValueError("non-binary attribute values")
    folder = out["filepath"].str.split("/").str[1]
    if not (folder == out["label"]).all():
        raise ValueError("folder name disagrees with class label")
    return out[["image_id", "filepath", "label", "split", *ATTR_IDS]]


def load_tidy(csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv)
