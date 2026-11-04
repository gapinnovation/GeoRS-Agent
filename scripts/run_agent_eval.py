"""Interactive agent evaluation on the verifiable question benchmark.

    python scripts/run_agent_eval.py --config config.yaml            # run + metrics
    python scripts/run_agent_eval.py --config config.yaml --metrics  # metrics only

Questions are generated once and frozen in results/agent_questions.jsonl
(their hash goes in the manifest) BEFORE any agent call. Modes:
  AGENT    LLM chooses tools step by step (text-only; perception via extract_attributes)
  NOTOOLS  same model, image + question, single answer, no tools (baseline)
By default extract_attributes serves the B2 primary-run evidence of the image
(same model, prompt, seed and temperature 0), so the agent is evaluated on
tool use and reasoning, not on a second, identical perception call.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georsagent.agent import run_agent_episode, run_notools  # noqa: E402
from georsagent.agent_bench import make_questions, score_answer  # noqa: E402
from georsagent.kb import load_kb  # noqa: E402
from georsagent.metrics import mcnemar_exact, wilson_ci  # noqa: E402
from georsagent.ollama_client import client_from_config  # noqa: E402
from georsagent.pipeline import extract_attributes, read_image  # noqa: E402
from georsagent.runlog import JsonlLog, file_sha, now, read_jsonl, write_manifest  # noqa: E402

DEFAULTS = {"images_per_class": 5, "subset_seed": 11, "max_steps": 8,
            "cached_attributes": True, "modes": ["AGENT", "NOTOOLS"]}


def main(argv=None, client=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--modes", nargs="*", help="override modes, e.g. AGENT_V2")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    p, proto = cfg["paths"], cfg["protocol"]
    acfg = {**DEFAULTS, **(cfg.get("agent") or {})}
    run0 = proto["runs"][0]
    out = Path(p["results_dir"])
    kb = load_kb(p["kb_json"])
    tidy = pd.read_csv(p["tidy_csv"])
    test = tidy[tidy.split == "test"].sort_values("image_id")
    rows = {r["image_id"]: r for r in test.to_dict("records")}

    b2 = {r["image_id"]: {a: v["state"] for a, v in r["evidence"].items()}
          for r in read_jsonl(out / "eval_runs.jsonl")
          if r["condition"] == "B2" and r["run"] == run0["name"] and r.get("evidence")}

    qfile = out / "agent_questions.jsonl"
    if not qfile.exists():
        ids = test.groupby("label").sample(n=acfg["images_per_class"],
                                           random_state=acfg["subset_seed"])["image_id"]
        ids = sorted(i for i in ids if i in b2)
        with qfile.open("w") as f:
            for i in ids:
                for q in make_questions(i, rows[i]["label"], b2[i], kb):
                    f.write(json.dumps(q) + "\n")
    questions = read_jsonl(qfile)

    log_path = out / "agent_runs.jsonl"
    if not args.metrics:
        if client is None:
            client = client_from_config(cfg)
        write_manifest(out / f"manifest_agent_{now().replace(':', '')}.json", config=cfg, agent=acfg,
                       questions_sha256=file_sha(qfile), n_questions=len(questions),
                       model=client.model, model_digest=client.model_digest(),
                       ollama_version=client.server_version())
        log = JsonlLog(log_path, ["qid", "mode"])
        # AGENT_V2 (added after the first agent run): identical prompt and questions,
        # but tools reject arguments they do not accept and say which tool to use.
        for mode in (args.modes or acfg["modes"]):
            for q in tqdm(questions, desc=mode):
                if log.has(qid=q["qid"], mode=mode):
                    continue
                row = rows[q["image_id"]]
                if mode in ("AGENT", "AGENT_V2"):
                    if acfg["cached_attributes"]:
                        provider = lambda i=q["image_id"]: b2[i]
                    else:
                        def provider(r=row):
                            ev, _ = extract_attributes(client, read_image(p["images_root"], r["filepath"]),
                                                       proto, run0["seed"], run0["temperature"])
                            return ev or {}
                    res = run_agent_episode(client, kb, q["question"], provider, run0["seed"],
                                            run0["temperature"], acfg["max_steps"],
                                            strict_args=(mode == "AGENT_V2"))
                else:
                    img = read_image(p["images_root"], row["filepath"])
                    res = run_notools(client, q["question"], img, run0["seed"], run0["temperature"])
                    res["image_sha256"] = hashlib.sha256(img).hexdigest()
                log.append({"qid": q["qid"], "mode": mode, "qtype": q["qtype"],
                            "image_id": q["image_id"], **res, "finished_utc": now()})
    summarise(out, questions, kb, b2)


def summarise(out, questions, kb, b2):
    qs = {q["qid"]: q for q in questions}
    recs = read_jsonl(out / "agent_runs.jsonl") if (out / "agent_runs.jsonl").exists() else []
    rows = []
    for r in recs:
        q = qs[r["qid"]]
        s = score_answer(q, r.get("final"), b2[q["image_id"]], kb,
                         r.get("tools_used") if r["mode"].startswith("AGENT") else None)
        rows.append({"qid": r["qid"], "mode": r["mode"], "qtype": r["qtype"], **s,
                     "n_steps": r.get("n_steps", 1), "format_errors": r.get("n_format_errors", 0),
                     "hit_max_steps": r.get("hit_max_steps", False)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no agent runs yet")
        return
    df.to_csv(out / "agent_scored.csv", index=False)
    tab = []
    for (mode, qt), g in df.groupby(["mode", "qtype"]):
        k, n = int(g["correct"].sum()), len(g)
        lo, hi = wilson_ci(k, n)
        row = {"mode": mode, "qtype": qt, "n": n, "accuracy": k / n, "ci95_low": lo, "ci95_high": hi,
               "answered": g["answered"].mean(), "mean_steps": g["n_steps"].mean(),
               "format_errors_per_episode": g["format_errors"].mean(),
               "hit_max_steps": g["hit_max_steps"].mean()}
        if "required_tools_used" in g and g["required_tools_used"].notna().any():
            row["required_tools_used"] = g["required_tools_used"].mean()
            row["grounded_accuracy"] = g["grounded_correct"].mean()
        if "contradicting_citations" in g and g["contradicting_citations"].notna().any():
            row["answers_with_contradicting_citation"] = (g["contradicting_citations"].fillna(0) > 0).mean()
        for extra in ("agrees_with_pipeline", "citation_precision", "f1"):
            if extra in g and g[extra].notna().any():
                row[extra] = g[extra].mean()
        tab.append(row)
    t = pd.DataFrame(tab)
    t.to_csv(out / "table_agent.csv", index=False)
    tests = []
    for qt, g in df.groupby("qtype"):
        w = g.pivot_table(index="qid", columns="mode", values="correct", aggfunc="first")
        for a, b in [("NOTOOLS", "AGENT"), ("NOTOOLS", "AGENT_V2"), ("AGENT", "AGENT_V2")]:
            if {a, b} <= set(w):
                ww = w[[a, b]].dropna()
                tests.append({"qtype": qt, "A": a, "B": b, "n": len(ww),
                              **mcnemar_exact(ww[a].astype(bool).tolist(), ww[b].astype(bool).tolist())})
    pd.DataFrame(tests).to_csv(out / "table_agent_mcnemar.csv", index=False)
    md = ["# Agent evaluation", "", t.round(3).to_markdown(index=False), "",
          "## Paired McNemar (acc_diff = B - A)",
          pd.DataFrame(tests).round(4).to_markdown(index=False)]
    (out / "agent_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
