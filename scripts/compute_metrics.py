"""Step 3: results/eval_runs.jsonl -> tables (CSV) + summary (Markdown).

    python scripts/compute_metrics.py --config config.yaml
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georsagent.metrics import (accuracy, attribute_prf, fleiss_kappa,  # noqa: E402
                                macro_f1, mcnemar_exact)
from georsagent.runlog import read_jsonl  # noqa: E402
from georsagent.vocab import ATTR_GROUP, ATTR_IDS, CLASSES  # noqa: E402


def normalise_b0(label, synonyms):
    if not isinstance(label, str):
        return None
    l = label.strip().lower().replace("_", " ")
    for canon, syns in synonyms.items():
        if l == canon.lower() or l in syns:
            return canon
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["paths"]["results_dir"])
    recs = read_jsonl(out / "eval_runs.jsonl")
    tidy = pd.read_csv(cfg["paths"]["tidy_csv"]).set_index("image_id")
    primary = cfg["protocol"]["runs"][0]["name"]

    rows, unmatched = [], []
    for r in recs:
        pred = r.get("pred_label")
        scored = pred
        if r["condition"] == "B0":
            scored = normalise_b0(pred, cfg["b0_synonyms"])
            if scored is None:
                unmatched.append({"image_id": r["image_id"], "run": r["run"], "raw_label": pred})
        ft = r.get("final_taxonomy") or {}
        rows.append({
            "image_id": r["image_id"], "condition": r["condition"], "run": r["run"],
            "true": r["true_label"], "pred": scored, "raw_pred": pred,
            "correct": scored == r["true_label"],
            "first_valid_json": r.get("first_valid_json", r.get("attr_first_valid", True)),
            "final_valid_json": r.get("final_valid_json", pred is not None),
            "first_full_compliance": (r.get("first_taxonomy") or ft).get("full_compliance"),
            "final_full_compliance": ft.get("full_compliance"),
            "n_attempts": r.get("n_attempts", r.get("attr_n_attempts", 1)),
        })
    df = pd.DataFrame(rows)
    pd.DataFrame(unmatched).to_csv(out / "b0_unmatched.csv", index=False)

    # Table A: accuracy, macro-F1, format and taxonomy compliance
    tab = []
    for (cond, run), g in df.groupby(["condition", "run"]):
        a = accuracy(g["true"].tolist(), g["pred"].tolist())
        tab.append({"condition": cond, "run": run, **a,
                    "macro_f1": macro_f1(g["true"].tolist(), g["pred"].tolist(), CLASSES),
                    "first_attempt_valid_json": g["first_valid_json"].mean(),
                    "final_valid_json": g["final_valid_json"].mean(),
                    "first_attempt_taxonomy": g["first_full_compliance"].astype(float).mean(),
                    "final_taxonomy": g["final_full_compliance"].astype(float).mean(),
                    "mean_attempts": g["n_attempts"].mean()})
    tabA = pd.DataFrame(tab)
    tabA.to_csv(out / "table_accuracy.csv", index=False)

    # Table B: paired McNemar tests on the primary run
    prim = df[df.run.eq(primary) | df.condition.eq("ORACLE")]
    piv = prim.pivot_table(index="image_id", columns="condition", values="correct", aggfunc="first")
    tests = []
    for a, b in combinations([c for c in ["B0", "B1", "B2", "ORACLE"] if c in piv], 2):
        sub = piv[[a, b]].dropna()
        tests.append({"A": a, "B": b, "n": len(sub),
                      **mcnemar_exact(sub[a].astype(bool).tolist(), sub[b].astype(bool).tolist())})
    pd.DataFrame(tests).to_csv(out / "table_mcnemar.csv", index=False)

    # Table C: confusion matrices (primary run)
    for cond, g in prim.groupby("condition"):
        pd.crosstab(g["true"], g["pred"].fillna("INVALID")).to_csv(out / f"confusion_{cond}.csv")

    # Table D: attribute correctness of B2 vs reference annotations
    attr_rows = []
    for r in recs:
        if r["condition"] == "B2" and r.get("evidence"):
            attr_rows.append((r["run"], r["image_id"], {a: v["state"] for a, v in r["evidence"].items()}))
    tabD = []
    for run in sorted({x[0] for x in attr_rows}):
        sel = [x for x in attr_rows if x[0] == run]
        gt = [tidy.loc[i] for _, i, _ in sel]
        pr = [p for _, _, p in sel]
        for grp in ["all", "object", "color", "shape"]:
            attrs = ATTR_IDS if grp == "all" else [a for a in ATTR_IDS if ATTR_GROUP[a] == grp]
            tabD.append({"run": run, "group": grp, **attribute_prf(gt, pr, attrs)})
        if run == primary:
            per_attr = [{"attribute": a, "group": ATTR_GROUP[a], **attribute_prf(gt, pr, [a])}
                        for a in ATTR_IDS]
            pd.DataFrame(per_attr).to_csv(out / "table_attribute_per_attr.csv", index=False)
    pd.DataFrame(tabD).to_csv(out / "table_attributes.csv", index=False)

    # Table E: explanation faithfulness (B2)
    tabE = []
    for run in sorted({r["run"] for r in recs if r["condition"] == "B2"}):
        ex = [r["explanation"] for r in recs if r["condition"] == "B2" and r["run"] == run
              and isinstance(r.get("explanation"), dict) and r["explanation"].get("attempts")]
        first = [e["first_faithfulness"] for e in ex if e.get("first_faithfulness")]
        final = [e["final_faithfulness"] for e in ex if e.get("final_faithfulness")]
        tabE.append({"run": run, "n": len(ex),
                     "first_attempt_faithful": sum(f["faithful"] for f in first) / max(len(ex), 1),
                     "final_faithful": sum(f["faithful"] for f in final) / max(len(ex), 1),
                     "first_attempt_unsupported_citations": sum(len(f["unsupported"]) for f in first),
                     "first_attempt_contradicted_citations": sum(len(f["contradicted"]) for f in first)})
    pd.DataFrame(tabE).to_csv(out / "table_explanations.csv", index=False)

    # Table F: run-to-run stability (sampling runs only)
    stab_runs = [r["name"] for r in cfg["protocol"]["runs"][1:]]
    tabF = []
    for cond in ["B0", "B1", "B2"]:
        g = df[(df.condition == cond) & df.run.isin(stab_runs)]
        if g.empty:
            continue
        w = g.pivot_table(index="image_id", columns="run", values="raw_pred", aggfunc="first").dropna()
        if w.shape[1] < 2:
            continue
        ratings = w.astype(str).values.tolist()
        tabF.append({"condition": cond, "n_images": len(w), "n_runs": w.shape[1],
                     "identical_label_all_runs": float(sum(len(set(r)) == 1 for r in ratings) / len(ratings)),
                     "fleiss_kappa": fleiss_kappa(ratings)})
    pd.DataFrame(tabF).to_csv(out / "table_stability.csv", index=False)

    md = ["# GeoRS-Agent evaluation summary", "", "## Accuracy and compliance",
          tabA.round(3).to_markdown(index=False), "", "## Paired McNemar (primary run)",
          pd.DataFrame(tests).round(4).to_markdown(index=False)]
    for name in ["table_attributes", "table_explanations", "table_stability"]:
        t = pd.read_csv(out / f"{name}.csv") if (out / f"{name}.csv").stat().st_size > 1 else pd.DataFrame()
        if not t.empty:
            md += ["", f"## {name.replace('table_', '').title()}", t.round(3).to_markdown(index=False)]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
