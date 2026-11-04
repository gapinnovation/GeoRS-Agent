"""Step 1: workbook -> tidy CSV -> knowledge base (train split only).

    python scripts/prepare.py --config config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georsagent.data import load_workbook, sha256_file  # noqa: E402
from georsagent.kb import assert_no_leakage, build_kb, save_kb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    p = cfg["paths"]

    tidy = load_workbook(p["annotations_xlsx"])
    Path(p["tidy_csv"]).parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(p["tidy_csv"], index=False)

    kb = build_kb(tidy, alpha=cfg["kb"]["smoothing_alpha"])
    assert_no_leakage(kb, tidy.loc[tidy.split == "test", "image_id"])
    save_kb(kb, p["kb_json"])

    summary = {
        "annotations_sha256": sha256_file(p["annotations_xlsx"]),
        "n_images": len(tidy),
        "split_counts": tidy.groupby(["label", "split"]).size().unstack().to_dict(),
        "kb_sha256": kb["sha256"],
    }
    Path(p["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(p["results_dir"], "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(tidy)} images; train={sum(tidy.split=='train')} test={sum(tidy.split=='test')}")
    print(f"KB sha256 {kb['sha256'][:16]}… written to {p['kb_json']}")


if __name__ == "__main__":
    main()
