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
