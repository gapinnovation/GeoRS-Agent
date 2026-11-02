"""End-to-end tests with a fake Ollama backend (no GPU/model needed).

The fake VLM looks up each image by its bytes and answers from the reference
annotations, with controllable corruption, so every code path (retry,
taxonomy failure, unfaithful explanation, omitted vs negated) is exercised.
"""
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from georsagent.ollama_client import ChatResult  # noqa: E402
from georsagent.pipeline import resolve_image  # noqa: E402
from georsagent.prompts import SYSTEM_ATTR, SYSTEM_EXPLAIN  # noqa: E402
from georsagent.vocab import ATTR_IDS  # noqa: E402


class FakeClient:
    model = "fake-vlm"

    def __init__(self, tidy, images_root, flip=0.1, bad_json_every=7, seed=0):
        self.by_bytes = {}
        for r in tidy.to_dict("records"):
            try:
                self.by_bytes[resolve_image(images_root, r["filepath"]).read_bytes()] = r
            except FileNotFoundError:
                pass
        self.flip, self.bad_every, self.n = flip, bad_json_every, 0
        self.rng = random.Random(seed)

    def model_digest(self):
        return "sha256:fake"

    def server_version(self):
        return "0.0-test"

    def chat(self, system, user, images=None, fmt=None, seed=None, temperature=None):
        self.n += 1
        meta = {"model": self.model, "model_digest": "sha256:fake", "options": {"seed": seed}}
        if self.n % self.bad_every == 0:
            return ChatResult("not json {", meta)
        if system == SYSTEM_EXPLAIN:
            lines = [l for l in user.splitlines() if l.startswith("- ")]
            cited = [{"attribute": l[2:].split(" = ")[0], "state": l.split(" = ")[1].split(" ")[0]}
                     for l in lines[:3]]
            if self.n % 5 == 0:  # occasional hallucinated citation
                cited.append({"attribute": "airplane", "state": "absent"})
            return ChatResult(json.dumps({"explanation": "x", "cited_evidence": cited}), meta)
        row = self.by_bytes[images[0]]
        if system == SYSTEM_ATTR:
            out = {}
            for a in ATTR_IDS:
                s = "present" if row[a] == 1 else "absent"
                u = self.rng.random()
                if u < self.flip:
                    s = "absent" if s == "present" else "present"
                elif u < self.flip * 1.5:
                    s = "uncertain"
                out[a] = s
            return ChatResult(json.dumps(out), meta)
        label = row["label"]
        if "MUST choose" not in user and self.rng.random() < 0.5:
            label = label.lower()  # free-text style for B0
        if self.rng.random() < 0.15:
            label = "Harbour area"  # non-canonical
        return ChatResult(json.dumps({"label": label, "alternatives": [], "explanation": "e"}), meta)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    wd = tmp_path_factory.mktemp("wd")
    shutil.copy(ROOT / "data" / "annotations_raw.xlsx", wd / "annotations_raw.xlsx")
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["protocol"]["stability_images_per_class"] = 3
    cfg["paths"] = {"annotations_xlsx": str(wd / "annotations_raw.xlsx"),
                    "tidy_csv": str(wd / "processed.csv"), "kb_json": str(wd / "kb.json"),
                    "images_root": str(wd / "img"), "results_dir": str(wd / "results")}
    (wd / "config.yaml").write_text(yaml.safe_dump(cfg))
    import prepare
    sys.argv = ["prepare", "--config", str(wd / "config.yaml")]
    prepare.main()
    tidy = pd.read_csv(wd / "processed.csv")
    for r in tidy.to_dict("records"):  # unique fake image bytes per id
        p = wd / "img" / r["filepath"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r["image_id"].encode())
    return wd, tidy


def test_no_leakage_and_kb(workdir):
    wd, tidy = workdir
    kb = json.loads((wd / "kb.json").read_text())
    assert len(kb["train_image_ids"]) == 704
    assert not set(kb["train_image_ids"]) & set(tidy[tidy.split == "test"].image_id)
    assert len(kb["profiles"]) == 19


def test_omitted_differs_from_negated():
    from georsagent.scorer import score_classes
    kb = json.loads((ROOT / "kb" / "class_profiles.json").read_text())
    s_omit = score_classes({"airplane": "present"}, kb)
    s_neg = score_classes({"airplane": "present", "water": "absent"}, kb)
    assert s_omit["n_evidence"] == 1 and s_neg["n_evidence"] == 2
    assert s_omit["label"] == "Airport"


def test_end_to_end(workdir):
    wd, tidy = workdir
    import compute_metrics
    import run_eval
    import run_interventions
    client = FakeClient(tidy, wd / "img")
    run_eval.main(["--config", str(wd / "config.yaml"), "--limit", "2"], client=client)
    n = sum(1 for _ in open(wd / "results" / "eval_runs.jsonl"))
    recs = [json.loads(l) for l in open(wd / "results" / "eval_runs.jsonl")]
    assert sum(r["run"] == "primary" and r["condition"] != "ORACLE" for r in recs) == 38 * 3
    assert sum(r["condition"] == "ORACLE" for r in recs) == 38
    stab = {r["run"]: set() for r in recs if r["run"] != "primary"}
    for r in recs:
        if r["run"] != "primary":
            stab[r["run"]].add(r["image_id"])
    assert len(stab) == 3 and len({frozenset(v) for v in stab.values()}) == 1  # same subset each run
    # resume: second call adds nothing
    run_eval.main(["--config", str(wd / "config.yaml"), "--limit", "2"], client=client)
    assert sum(1 for _ in open(wd / "results" / "eval_runs.jsonl")) == n
    compute_metrics.main(["--config", str(wd / "config.yaml")])
    acc = pd.read_csv(wd / "results" / "table_accuracy.csv")
    assert set(acc.condition) == {"B0", "B1", "B2", "ORACLE"}
    b1 = acc[(acc.condition == "B1") & (acc.run == "primary")].iloc[0]
    assert b1.final_taxonomy >= b1.first_attempt_taxonomy  # guardrail never hurts
    assert (wd / "results" / "table_attributes.csv").exists()

    cfg = yaml.safe_load(open(wd / "config.yaml"))
    cfg["interventions"]["n_images"] = 19
    (wd / "config.yaml").write_text(yaml.safe_dump(cfg))
    run_interventions.main(["--config", str(wd / "config.yaml")], client=client)
    summ = json.loads((wd / "results" / "interventions_summary.json").read_text())
    assert "E1_omitted_attr_cited_rate" in summ and "M1_B1_label_change_with_metadata" in summ


def test_experiments_a_and_b(workdir):
    wd, tidy = workdir
    import compute_metrics
    import run_eval
    import run_experiment_a
    client = FakeClient(tidy, wd / "img", bad_json_every=1000)
    cfgp = str(wd / "config.yaml")
    run_eval.main(["--config", cfgp, "--runs", "primary", "--conditions", "B1_NOSCHEMA", "--limit", "2"], client=client)
    # experiment A on a reduced train set: restrict tidy to 3 train images per class
    cfg = yaml.safe_load(open(cfgp))
    small = tidy[(tidy.split == "test") | tidy.index.isin(tidy[tidy.split == "train"].groupby("label").head(3).index)]
    small.to_csv(wd / "small.csv", index=False)
    cfg["paths"]["tidy_csv"] = str(wd / "small.csv")
    cfg["paths"]["kb_json"] = str(wd / "kb.json")
    (wd / "cfgA.yaml").write_text(yaml.safe_dump(cfg))
    run_experiment_a.main(["--config", str(wd / "cfgA.yaml")], client=client)
    assert (wd / "class_profiles_vlm.json").exists()
    recs = [json.loads(l) for l in open(wd / "results" / "eval_runs.jsonl")]
    nb2 = sum(r["condition"] == "B2" for r in recs)
    assert sum(r["condition"] == "B2_VLMKB" for r in recs) == nb2
    run_experiment_a.main(["--config", str(wd / "cfgA.yaml"), "--skip-extraction"], client=client)  # idempotent
    assert sum(json.loads(l)["condition"] == "B2_VLMKB" for l in open(wd / "results" / "eval_runs.jsonl")) == nb2
    compute_metrics.main(["--config", cfgp])
    assert (wd / "results" / "table_noschema_compliance.csv").exists()
    acc = pd.read_csv(wd / "results" / "table_accuracy.csv")
    assert {"B1_NOSCHEMA", "B2_VLMKB"} <= set(acc.condition)


class FakeAgentClient(FakeClient):
    """Scripted tool-using policy, with occasional format errors and one wrong citation."""

    def chat_messages(self, messages, fmt=None, seed=None, temperature=None):
        import re
        meta = {"model": self.model, "model_digest": "sha256:fake"}
        q = messages[1]["content"]
        results = [json.loads(m["content"][len("TOOL RESULT: "):]) for m in messages
                   if m["role"] == "user" and m["content"].startswith("TOOL RESULT: ")]
        n = len(results)
        self.n += 1
        if self.n % 11 == 0:
            return ChatResult("oops", meta)
        def act(tool, **args):
            return ChatResult(json.dumps({"thought": "t", "tool": tool, "args": args}), meta)
        if q.startswith("Which attributes"):
            c = re.search(r"class (\w+)\?", q).group(1)
            if not any("typical_present" in r for r in results):
                return act("get_class_profile", **{"class": c})
            prof = [r for r in results if "typical_present" in r][-1]
            return act("answer", text="x", label="none",
                       cited_attributes=[{"attribute": a, "state": "present"} for a in prof["typical_present"][:3]])
        if not any("attributes" in r and isinstance(r["attributes"], dict) for r in results):
            return act("extract_attributes")
        ev = [r for r in results if isinstance(r.get("attributes"), dict)][-1]["attributes"]
        if q.startswith("What type"):
            rk = [r for r in results if "ranking" in r]
            if not rk:
                return act("score_classes")
            return act("answer", text="x", label=rk[-1]["ranking"][0]["label"], cited_attributes=[])
        if q.startswith("Why"):
            a, b = re.search(r"as (\w+) rather than (\w+)\?", q).groups()
            cp = [r for r in results if "favouring" in r]
            if not cp:
                return act("compare_classes", class_a=a, class_b=b)
            cited = [{"attribute": x["attribute"], "state": x["state"]} for x in cp[-1]["attributes"][:2]]
            return act("answer", text="x", label=a, cited_attributes=cited)
        if q.startswith("If the attribute"):
            att = re.search(r"attribute (\w+) were", q).group(1)
            if not any(r.get("new_state") for r in results):
                return act("set_attribute", attribute=att, state="absent")
            rk = [r for r in results if "ranking" in r]
            if not rk:
                return act("score_classes")
            return act("answer", text="x", label=rk[-1]["ranking"][0]["label"], cited_attributes=[])
        return act("list_classes")

    def chat(self, system, user, images=None, fmt=None, seed=None, temperature=None):
        from georsagent.agent import SYSTEM_NOTOOLS
        if system == SYSTEM_NOTOOLS:
            row = self.by_bytes[images[0]]
            return ChatResult(json.dumps({"text": "x", "label": row["label"],
                                          "cited_attributes": [{"attribute": "road", "state": "present"}]}),
                              {"model": self.model})
        return super().chat(system, user, images, fmt, seed, temperature)


def test_agent_benchmark(workdir):
    wd, tidy = workdir
    import run_agent_eval
    import run_eval
    cfgp = str(wd / "config.yaml")
    client = FakeAgentClient(tidy, wd / "img", bad_json_every=1000)
    run_eval.main(["--config", cfgp, "--runs", "primary", "--conditions", "B2"], client=client)
    cfg = yaml.safe_load(open(cfgp))
    cfg["agent"] = {"images_per_class": 2, "subset_seed": 11, "max_steps": 8,
                    "cached_attributes": True, "modes": ["AGENT", "NOTOOLS"]}
    (wd / "cfg_agent.yaml").write_text(yaml.safe_dump(cfg))
    run_agent_eval.main(["--config", str(wd / "cfg_agent.yaml")], client=client)
    t = pd.read_csv(wd / "results" / "table_agent.csv")
    ag = t[t["mode"] == "AGENT"].set_index("qtype")
    assert set(ag.index) >= {"Q1", "Q2", "Q4"}
    assert ag.loc["Q4", "required_tools_used"] == 1.0
    # scripted policy reproduces the deterministic tools -> high agreement with gold
    assert ag.loc["Q2", "accuracy"] > 0.8 and ag.loc["Q1", "agrees_with_pipeline"] > 0.8
    n = sum(1 for _ in open(wd / "results" / "agent_runs.jsonl"))
    run_agent_eval.main(["--config", str(wd / "cfg_agent.yaml")], client=client)  # resume
    assert sum(1 for _ in open(wd / "results" / "agent_runs.jsonl")) == n


def test_stability_v2_per_image_seed(workdir):
    wd, tidy = workdir
    import compute_metrics
    import run_eval
    client = FakeClient(tidy, wd / "img", bad_json_every=1000)
    cfgp = str(wd / "config.yaml")
    run_eval.main(["--config", cfgp, "--stability-v2", "--conditions", "B0", "B1", "B2", "--limit", "2"], client=client)
    recs = [json.loads(l) for l in open(wd / "results" / "eval_runs.jsonl")]
    v2 = [r for r in recs if r["run"] in ("stab1b", "stab2b", "stab3b")]
    assert v2 and {r["run"] for r in v2} == {"stab1b", "stab2b", "stab3b"}
    seeds = {r["seed"] for r in v2 if r["run"] == "stab1b" and r["condition"] == "B0"}
    assert len(seeds) == len([r for r in v2 if r["run"] == "stab1b" and r["condition"] == "B0"])  # one per image
    compute_metrics.main(["--config", cfgp])
    st = pd.read_csv(wd / "results" / "table_stability.csv")
    assert {"v1_shared_seed", "v2_per_image_seed"} <= set(st.version)


def test_agent_v2_strict_args():
    from georsagent.agent import Episode
    kb = json.loads((ROOT / "kb" / "class_profiles.json").read_text())
    ev = {"airplane": "present"}
    loose, strict = Episode(kb, lambda: ev), Episode(kb, lambda: ev, strict_args=True)
    for e in (loose, strict):
        e.run_tool("extract_attributes", {})
    assert "ranking" in loose.run_tool("score_classes", {"attribute": "airplane", "state": "absent"})
    r = strict.run_tool("score_classes", {"attribute": "airplane", "state": "absent"})
    assert "error" in r and "set_attribute" in r["error"]
    assert "ranking" in strict.run_tool("score_classes", {})


def test_unanswered_counts_as_not_grounded():
    from georsagent.agent_bench import score_answer
    q = {"qtype": "Q3", "gold": {"attribute": "road", "label": "Port", "original_label": "Port"}}
    s = score_answer(q, None, {}, None, tools_used=["extract_attributes", "set_attribute", "score_classes"])
    assert s["correct"] is False and s["grounded_correct"] is False and s["required_tools_used"] is True
