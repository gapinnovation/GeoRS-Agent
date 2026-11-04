"""Experiment A (exploratory, designed after the primary analysis).

Hypothesis: B2 underperforms B1 because the knowledge base encodes the human
annotators' attribute conventions, while the scorer receives attributes as the
VLM perceives them. If the profiles are instead estimated from the VLM's own
attribute answers on the TRAIN images, the mismatch disappears.

Steps
  1. extract attributes with the SAME model, prompt, seed and temperature as the
     B2 primary run on all 704 TRAIN images  -> results/train_attributes.jsonl
     (resumable; the only step that calls the model)
  2. build the knowledge base from those answers -> kb/class_profiles_vlm.json
  3. re-score the ALREADY LOGGED B2 test evidence with the new KB (no new model
     calls on the test set) -> condition B2_VLMKB appended to eval_runs.jsonl

The test set is never used in steps 1-2.

    python scripts/run_experiment_a.py --config config.yaml
    python scripts/run_experiment_a.py --config config.yaml --skip-extraction   # steps 2-3 only
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georsagent.kb import assert_no_leakage, build_kb_from_evidence, load_kb, save_kb  # noqa: E402
from georsagent.ollama_client import client_from_config  # noqa: E402
from georsagent.pipeline import extract_attributes, read_image  # noqa: E402
from georsagent.prompts import SYSTEM_ATTR, USER_ATTR, prompt_hash  # noqa: E402
from georsagent.runlog import JsonlLog, now, read_jsonl, write_manifest  # noqa: E402
from georsagent.scorer import score_classes  # noqa: E402


def main(argv=None, client=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip-extraction", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    p, proto = cfg["paths"], cfg["protocol"]
    run0 = proto["runs"][0]
    out = Path(p["results_dir"])
    tidy = pd.read_csv(p["tidy_csv"])
    train = tidy[tidy.split == "train"].sort_values("image_id")
    test_ids = set(tidy.loc[tidy.split == "test", "image_id"])

    # 1. attribute extraction on TRAIN images
    log = JsonlLog(out / "train_attributes.jsonl", ["image_id"])
    if not args.skip_extraction:
        if client is None:
            client = client_from_config(cfg)
        write_manifest(out / f"manifest_expA_{now().replace(':', '')}.json", config=cfg,
                       model=client.model, model_digest=client.model_digest(),
                       ollama_version=client.server_version(), step="expA_train_extraction",
                       attr_prompt_hash=prompt_hash(SYSTEM_ATTR, USER_ATTR))
        for row in tqdm(train.to_dict("records"), desc="train attributes"):
            if log.has(image_id=row["image_id"]):
                continue
            img = read_image(p["images_root"], row["filepath"])
            ev, attempts = extract_attributes(client, img, proto, run0["seed"], run0["temperature"])
            log.append({"image_id": row["image_id"], "true_label": row["label"],
                        "image_sha256": hashlib.sha256(img).hexdigest(),
                        "evidence": ev, "attempts": attempts, "finished_utc": now()})

    recs = {r["image_id"]: r for r in read_jsonl(out / "train_attributes.jsonl")}
    missing = set(train.image_id) - set(recs)
    if missing:
        sys.exit(f"{len(missing)} train images still to extract: rerun without --skip-extraction")
    failed = [i for i, r in recs.items() if r["evidence"] is None]
    ev = {i: r["evidence"] for i, r in recs.items() if r["evidence"] is not None}
    if set(ev) & test_ids:
        raise AssertionError("test images found in the training evidence")

    # 2. knowledge base from VLM answers
    labels = dict(zip(train.image_id, train.label))
    kb_vlm = build_kb_from_evidence(ev, labels, cfg["kb"]["smoothing_alpha"],
                                    source=f"VLM-extracted attributes on train split; model {cfg['ollama']['model']}")
    assert_no_leakage(kb_vlm, test_ids)
    kb_path = Path(p["kb_json"]).with_name("class_profiles_vlm.json")
    save_kb(kb_vlm, kb_path)
    print(f"VLM-KB from {len(ev)} train images ({len(failed)} failed extractions) -> {kb_path}")
    kb_vlm = load_kb(kb_path)

    # 3. re-score logged B2 test evidence (every run in which B2 exists)
    elog = JsonlLog(out / "eval_runs.jsonl", ["image_id", "condition", "run"])
    for r in read_jsonl(out / "eval_runs.jsonl"):
        if r["condition"] != "B2" or not r.get("evidence"):
            continue
        if elog.has(image_id=r["image_id"], condition="B2_VLMKB", run=r["run"]):
            continue
        evidence = {a: v["state"] for a, v in r["evidence"].items()}
        s = score_classes(evidence, kb_vlm)
        elog.append({"image_id": r["image_id"], "true_label": r["true_label"],
                     "condition": "B2_VLMKB", "run": r["run"], "seed": r["seed"],
                     "temperature": r["temperature"], "source_condition": "B2",
                     "kb_sha256": kb_vlm["sha256"], "pred_label": s["label"],
                     "alternatives": [x["label"] for x in s["ranking"][1:]], "scoring": s,
                     "final_taxonomy": {"primary_in_taxonomy": True,
                                        "alternatives_in_taxonomy": True, "full_compliance": True},
                     "finished_utc": now()})


if __name__ == "__main__":
    main()
