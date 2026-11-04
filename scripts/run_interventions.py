"""Step 4: pre-specified single-attribute interventions (reviewer C1.7-C1.8).

Evidence = reference ABZSL attributes of every test image (all 19 classes), so
the intervention acts on known-correct evidence. For each image:

  a* = present attribute with the largest positive contribution to the margin
       between the predicted class and the runner-up
  OMIT       a* -> not provided            (must NOT be read as absence)
  NEGATE     a* -> explicitly absent
  CONTRADICT b* -> present, where b* is the absent attribute most indicative
                   of the runner-up class

Pre-specified expectations (written before running; do not edit afterwards):
  K1 (KB level)  margin drop under NEGATE >= margin drop under OMIT
  E1 (expl.)     after OMIT, the explanation never cites a*        (no unsupported citation)
  E2 (expl.)     after NEGATE, a* is never cited as present        (no contradiction)
  E3 (expl.)     after CONTRADICT, b* is never cited as absent     (no contradiction)
  M1 (B1 only)   irrelevant metadata in the prompt changes the B1 label of
                 an image -> counted; the agent's scorer is invariant by design

    python scripts/run_interventions.py --config config.yaml            # full
    python scripts/run_interventions.py --config config.yaml --no-model # K1 only
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georsagent.kb import load_kb  # noqa: E402
from georsagent.pipeline import explain, read_image, run_scene_prompt  # noqa: E402
from georsagent.runlog import JsonlLog, now, read_jsonl  # noqa: E402
from georsagent.scorer import evidence_from_binary, score_classes  # noqa: E402
from georsagent.vocab import ABSENT, ATTR_IDS, PRESENT  # noqa: E402


def margin(evidence, kb, label):
    ll = {r["label"]: r["log_likelihood"] for r in score_classes(evidence, kb, top_k=len(kb["classes"]))["ranking"]}
    other = max(v for c, v in ll.items() if c != label)
    return ll[label] - other


def variants(ev, kb):
    base = score_classes(ev, kb)
    top, second = base["label"], base["runner_up"]
    pos = [a for a, w in base["margin_contributions"].items() if w > 0 and ev[a] == PRESENT]
    if not pos:
        return base, {}
    a_star = pos[0]
    prof = kb["profiles"]
    b_cands = [(math.log(prof[second]["p_present"][b]) - math.log(prof[top]["p_present"][b]), b)
               for b in ATTR_IDS if ev[b] == ABSENT]
    b_star = max(b_cands)[1] if b_cands else None
    v = {"OMIT": ({k: s for k, s in ev.items() if k != a_star}, a_star),
         "NEGATE": ({**ev, a_star: ABSENT}, a_star)}
    if b_star:
        v["CONTRADICT"] = ({**ev, b_star: PRESENT}, b_star)
    return base, v


def main(argv=None, client=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    p, proto, icfg = cfg["paths"], cfg["protocol"], cfg["interventions"]
    tidy = pd.read_csv(p["tidy_csv"])
    test = tidy[tidy.split == "test"].sort_values("image_id")
    if icfg["n_images"] != "all":
        n_per = max(1, int(icfg["n_images"]) // test.label.nunique())
        test = test.groupby("label").sample(n=n_per, random_state=icfg["seed"])
    kb = load_kb(p["kb_json"])
    run0 = proto["runs"][0]
    if not args.no_model and client is None:
        from georsagent.ollama_client import client_from_config
        client = client_from_config(cfg)

    out = Path(p["results_dir"])
    log = JsonlLog(out / "interventions.jsonl", ["image_id", "variant"])
    for row in tqdm(test.to_dict("records"), desc="interventions"):
        ev = evidence_from_binary(row, ATTR_IDS)
        base, vs = variants(ev, kb)
        m0 = margin(ev, kb, base["label"])
        for name, (ev2, target) in [("BASE", (ev, None)), *vs.items()]:
            if log.has(image_id=row["image_id"], variant=name):
                continue
            s2 = score_classes(ev2, kb)
            rec = {"image_id": row["image_id"], "true_label": row["label"], "variant": name,
                   "target_attribute": target, "base_label": base["label"],
                   "new_label": s2["label"], "label_changed": s2["label"] != base["label"],
                   "margin_before": m0, "margin_after": margin(ev2, kb, base["label"]),
                   "started_utc": now()}
            if client is not None and not args.no_model:
                ex = explain(client, s2, ev2, proto, run0["seed"] + 100, run0["temperature"],
                             icfg["explanation_top_k"])
                rec["explanation"] = ex
                f = ex.get("first_faithfulness")
                cited = [] if not ex.get("attempts") or ex["attempts"][0]["schema_errors"] \
                    else ex["attempts"][0]["parsed"]["cited_evidence"]
                rec["first_faithful"] = f["faithful"] if f else None
                rec["cites_target_as"] = [c["state"] for c in cited if c["attribute"] == target]
            log.append(rec)

    # M1: irrelevant metadata on B1 (end-to-end VLM), compared with the primary B1 run
    if client is not None and not args.no_model:
        mlog = JsonlLog(out / "interventions_metadata.jsonl", ["image_id"])
        for row in tqdm(test.to_dict("records"), desc="metadata"):
            if mlog.has(image_id=row["image_id"]):
                continue
            img = read_image(p["images_root"], row["filepath"])
            rec = run_scene_prompt("B1", client, img, proto, run0["seed"], run0["temperature"],
                                   extra_text=icfg["irrelevant_metadata_text"])
            rec.update({"image_id": row["image_id"], "true_label": row["label"], "finished_utc": now()})
            mlog.append(rec)

    summarise(out)


def summarise(out: Path):
    df = pd.DataFrame(read_jsonl(out / "interventions.jsonl"))
    df["margin_drop"] = df["margin_before"] - df["margin_after"]
    w = df.pivot_table(index="image_id", columns="variant", values="margin_drop")
    res = {"n_images": int(df.image_id.nunique())}
    if {"OMIT", "NEGATE"} <= set(w):
        res["K1_negate_drop_ge_omit_drop"] = float((w["NEGATE"] >= w["OMIT"] - 1e-12).mean())
    for v, g in df.groupby("variant"):
        res[f"{v}_label_change_rate"] = float(g["label_changed"].mean())
        res[f"{v}_median_margin_drop"] = float(g["margin_drop"].median())
        if "first_faithful" in g and g["first_faithful"].notna().any():
            res[f"{v}_first_attempt_faithful"] = float(g["first_faithful"].dropna().mean())
            states = g["cites_target_as"].dropna()
            if v == "OMIT":
                res["E1_omitted_attr_cited_rate"] = float(states.map(bool).mean())
                res["E1_omitted_read_as_absent_rate"] = float(states.map(lambda s: "absent" in s).mean())
            if v == "NEGATE":
                res["E2_negated_cited_present_rate"] = float(states.map(lambda s: "present" in s).mean())
            if v == "CONTRADICT":
                res["E3_contradiction_cited_absent_rate"] = float(states.map(lambda s: "absent" in s).mean())
    meta = out / "interventions_metadata.jsonl"
    runs = out / "eval_runs.jsonl"
    if meta.exists() and runs.exists():
        b1 = {r["image_id"]: r["pred_label"] for r in read_jsonl(runs)
              if r["condition"] == "B1" and r["run"] == "primary"}
        mm = [(r["image_id"], r["pred_label"]) for r in read_jsonl(meta) if r["image_id"] in b1]
        if mm:
            res["M1_B1_label_change_with_metadata"] = sum(b1[i] != l for i, l in mm) / len(mm)
    (out / "interventions_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
