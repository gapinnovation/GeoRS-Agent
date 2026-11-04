"""Step 2: run all conditions on the frozen test split; append to JSONL; resumable.

    python scripts/run_eval.py --config config.yaml
    python scripts/run_eval.py --config config.yaml --conditions ORACLE   # no Ollama needed
    python scripts/run_eval.py --config config.yaml --limit 5             # smoke test
"""
import argparse
import hashlib
import zlib
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georsagent.data import load_tidy  # noqa: E402
from georsagent.kb import assert_no_leakage, load_kb  # noqa: E402
from georsagent.ollama_client import client_from_config as make_client  # noqa: E402
from georsagent.pipeline import read_image, run_agent, run_scene_prompt  # noqa: E402
from georsagent.schemas import lenient_scene_parse  # noqa: E402
from georsagent.runlog import JsonlLog, file_sha, now, write_manifest  # noqa: E402
from georsagent.scorer import evidence_from_binary  # noqa: E402
from georsagent.vocab import ATTR_IDS  # noqa: E402


def main(argv=None, client=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--runs", nargs="*", help="run names from config (default: all)")
    ap.add_argument("--limit", type=int, help="first N test images per class (smoke test)")
    ap.add_argument("--stability-v2", action="store_true",
                    help="corrected stability runs stab1b-stab3b: per-image seeds (see below)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    p, proto = cfg["paths"], cfg["protocol"]
    conditions = args.conditions or proto["conditions"]
    runs = [r for r in proto["runs"] if not args.runs or r["name"] in args.runs]
    if args.stability_v2:
        # Correction (added after the first stability runs): in stab1-stab3 every image
        # of a run used the SAME sampling seed, so sampling noise was correlated across
        # images. stab1b-stab3b derive an independent seed per image:
        #   seed = run_seed + crc32(image_id) % 100000
        runs = [{**r, "name": r["name"] + "b", "per_image_seed": True}
                for r in proto["runs"][1:]]

    tidy = load_tidy(p["tidy_csv"])
    test = tidy[tidy.split == "test"].sort_values("image_id")
    full_test = test  # stability subset is drawn from the FULL test set, independent of --limit
    if args.limit:
        test = test.groupby("label", group_keys=False).head(args.limit)
    kb = load_kb(p["kb_json"])
    assert_no_leakage(kb, test["image_id"])
    _kbv = {}

    def kb_vlm():
        if "kb" not in _kbv:
            path = Path(p["kb_json"]).with_name("class_profiles_vlm.json")
            if not path.exists():
                sys.exit(f"{path} not found: run scripts/run_experiment_a.py --skip-extraction first")
            _kbv["kb"] = load_kb(path)
            assert_no_leakage(_kbv["kb"], full_test["image_id"])
        return _kbv["kb"]

    needs_model = any(c != "ORACLE" for c in conditions) or proto.get("explain_oracle")
    if needs_model and client is None:
        client = make_client(cfg)
    out = Path(p["results_dir"])
    manifest = {"config": cfg, "config_sha256": file_sha(args.config), "kb_sha256": kb["sha256"],
                "test_image_ids": test["image_id"].tolist(), "conditions": conditions,
                "runs": runs}
    if needs_model:
        manifest.update({"model": client.model, "model_digest": client.model_digest(),
                         "ollama_version": client.server_version()})

    # Stability (sampling) runs may use a stratified subset to fit a compute budget.
    primary = proto["runs"][0]["name"]
    k = proto.get("stability_images_per_class")
    stab_ids = set(test["image_id"]) if not k else set(
        full_test.groupby("label").sample(n=k, random_state=proto.get("stability_subset_seed", 0))["image_id"])
    manifest["stability_image_ids"] = sorted(stab_ids)
    write_manifest(out / f"manifest_{now().replace(':', '')}.json", **manifest)

    log = JsonlLog(out / "eval_runs.jsonl", ["image_id", "condition", "run"])
    jobs = [(r, c, row) for r in runs for c in conditions for row in test.to_dict("records")
            if not (c == "ORACLE" and r["name"] != primary)          # oracle is deterministic
            and (r["name"] == primary or row["image_id"] in stab_ids)]
    for run, cond, row in tqdm(jobs, desc="eval"):
        if log.has(image_id=row["image_id"], condition=cond, run=run["name"]):
            continue
        seed, temp = run["seed"], run["temperature"]
        if run.get("per_image_seed"):
            seed = run["seed"] + zlib.crc32(row["image_id"].encode()) % 100000
        base = {"image_id": row["image_id"], "true_label": row["label"], "condition": cond,
                "run": run["name"], "seed": seed, "temperature": temp, "started_utc": now()}
        if cond == "ORACLE":
            ev = evidence_from_binary(row, ATTR_IDS)
            rec = run_agent(client if proto.get("explain_oracle") else None, None, kb, proto,
                            seed, temp, evidence_override=ev, provenance="reference_annotation",
                            with_explanation=bool(proto.get("explain_oracle")))
        else:
            img = read_image(p["images_root"], row["filepath"])
            base["image_sha256"] = hashlib.sha256(img).hexdigest()
            if cond in ("B0", "B1"):
                rec = run_scene_prompt(cond, client, img, proto, seed, temp)
            elif cond == "B1_NOSCHEMA":
                # Experiment B (added after the primary analysis, exploratory):
                # same prompt as B1, NO constrained decoding. First attempt =
                # instruction-only compliance; final = with Python validation + retry.
                rec = run_scene_prompt("B1", client, img, {**proto, "enforce_schema": False}, seed, temp)
                rec["first_attempt_lenient"] = lenient_scene_parse(rec["attempts"][0]["raw"])
            elif cond in ("B1_KB", "B1_VLMKB"):
                # Added after the primary analysis (reviewer request): B1 + the full
                # knowledge base in the prompt. B1_KB: KB from reference annotations
                # (same KB as B2); B1_VLMKB: KB from VLM attributes on the train split
                # (same KB as B2_VLMKB; build it first with run_experiment_a.py).
                rec = run_scene_prompt("B1_KB", client, img, proto, seed, temp,
                                       kb=kb if cond == "B1_KB" else kb_vlm())
            elif cond == "B2":
                rec = run_agent(client, img, kb, proto, seed, temp)
            else:
                raise ValueError(cond)
        rec.update(base)
        rec["condition"] = cond
        rec["finished_utc"] = now()
        log.append(rec)


if __name__ == "__main__":
    main()
