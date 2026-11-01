# GeoRS-Agent

Attribute-grounded, explainable remote sensing scene interpretation on
WHU-RS19 with the WHU-RS19 ABZSL attribute annotations, using an open-weight
vision–language model (VLM) served locally with [Ollama](https://ollama.com).

This repository contains everything needed to reproduce the evaluation in the
paper: code, prompts, the class–attribute knowledge base, the annotation file,
the exact configuration, and the raw output of every model call.

## How it works

```
image ──► VLM: 38 ABZSL attributes, each present / absent / uncertain (JSON schema)
                │   provenance = model_observed
                ▼
         deterministic scorer: Bernoulli naive Bayes on class–attribute profiles
         built from the TRAIN split only
                │   label, top-3, per-attribute contribution to the decision
                ▼
         LLM: explanation that may cite ONLY the evidence used by the scorer
                │
         Python validators: JSON schema, 19-class taxonomy, citation consistency,
         bounded retry (max 2)
```

* "uncertain" and "not provided" are never treated as evidence of absence;
  an explicitly negated attribute is.
* The normalised score is a softmax of log-likelihoods, **not** a calibrated
  probability.
* Every call is an independent single-turn request (no conversation history).

## Experimental conditions (same model throughout)

| id | input | label decided by |
|----|-------|------------------|
| B0 | image | VLM, free text (direct prompting) |
| B1 | image + list of the 19 admissible classes | VLM |
| B2 | image → VLM attributes | knowledge-base scorer (GeoRS-Agent) |
| ORACLE | reference ABZSL attributes | knowledge-base scorer (upper bound, no VLM) |

B1 vs B2 isolates the contribution of the knowledge base; B2 vs ORACLE
separates attribute-extraction error from reasoning error.

## Data

* `data/annotations_raw.xlsx`: WHU-RS19 ABZSL annotations (sheet
  `40 classi REF`: 1,005 images, 19 classes, 38 binary attributes and the
  Train/Test partition). This file is never modified.
* `data/processed/annotations.csv`: the same data with canonical class and
  attribute identifiers (`scripts/prepare.py`).
* `kb/class_profiles.json`: class–attribute knowledge base built from the
  704 training images only. It stores the list of training image IDs and a
  SHA-256 hash, which is checked at load time.
* **Images are not redistributed.** Obtain WHU-RS19 from its original source.
  Both layouts are supported: `<root>/<Class>/<file>` and
  `<root>/Train|Test/<Class>/<file>`. The split is always taken from the
  annotation file. In the original distribution the Desert images are named
  `Desert_XX.jpg` while the annotation file uses `desert_XX.jpg`; file matching
  is case-insensitive when unambiguous.

Split: 704 train / 301 test, all 19 classes (15–18 test images per class).

## Run

### Google Colab (no local GPU needed)

Open `notebooks/GeoRS_Agent_Colab.ipynb`, select a GPU runtime, put the
image archive on Google Drive and run the cells in order. Results are written
to Drive, and every step resumes where it stopped if the session drops.

### Local machine with a GPU

```bash
pip install -r requirements.txt
ollama pull qwen2.5vl:7b                  # must match ollama.model in config.yaml
# edit config.yaml: paths.images_root
python scripts/prepare.py                 # tidy CSV + knowledge base + leakage check
python scripts/run_eval.py --limit 1      # smoke test: 1 image per class
python scripts/run_eval.py                # primary + stability runs (resumable)
python scripts/run_interventions.py       # pre-specified attribute interventions
python scripts/compute_metrics.py         # tables + results/summary.md
pytest -q                                 # tests with a fake model backend
```

`python scripts/run_eval.py --conditions ORACLE` and
`python scripts/run_interventions.py --no-model` run without any model.

## Protocol

* **Primary run:** temperature 0, seed 1000, all 301 test images, B0/B1/B2/ORACLE.
* **Stability:** 3 runs at temperature 0.7 (seeds 2000/3000/4000) on a stratified
  subset of 5 test images per class (95 images, subset seed 11). Reported as
  share of images with identical label across runs and Fleiss' κ.
* **Interventions** (`scripts/run_interventions.py`), on reference attributes:
  single-attribute omission, explicit negation, contradictory evidence, plus
  irrelevant metadata added to the B1 prompt. Expected behaviours (K1, E1–E3,
  M1) are written in the script header and were fixed before running.
* **Free-text labels (B0)** are mapped to the 19 classes with the pre-specified
  synonym list in `config.yaml`; unmatched labels are counted as errors and
  listed in `results/b0_unmatched.csv`.
* Images are sent as the original file bytes; their SHA-256 is logged. Any
  resizing is the model's own preprocessing.

## Outputs (`results/`)

| file | content |
|------|---------|
| `eval_runs.jsonl`, `interventions*.jsonl` | every raw model response and attempt, with seed, temperature, model digest, prompt hash, image hash, timestamps |
| `manifest_*.json` | configuration and its hash, knowledge-base hash, model digest, Ollama version, package versions, git commit, test and stability image IDs |
| `gpu*.txt` | GPU used for each block of runs |
| `table_accuracy.csv` | accuracy with Wilson 95% CI, macro-F1, JSON validity and taxonomy compliance at first attempt and after the guardrail |
| `table_mcnemar.csv`, `confusion_*.csv` | paired McNemar tests, confusion matrices |
| `table_attributes.csv`, `table_attribute_per_attr.csv` | attribute precision/recall/F1 against ABZSL, false-absence rate |
| `table_explanations.csv` | faithfulness of explanations (unsupported and contradicted citations) |
| `table_stability.csv`, `interventions_summary.json`, `summary.md` | stability, interventions, overall summary |

## Code map

| path | role |
|------|------|
| `georsagent/vocab.py` | closed vocabularies (19 classes, 38 attributes) |
| `georsagent/data.py` | annotation loading with consistency checks |
| `georsagent/kb.py` | knowledge base (train only, hashed, leakage check) |
| `georsagent/scorer.py` | deterministic scoring and per-attribute contributions |
| `georsagent/prompts.py` | all prompts, verbatim |
| `georsagent/schemas.py` | JSON schemas, taxonomy and faithfulness validators |
| `georsagent/ollama_client.py` | logged Ollama client |
| `georsagent/pipeline.py` | conditions B0, B1, B2, ORACLE |
| `georsagent/metrics.py` | Wilson CI, macro-F1, McNemar, Fleiss' κ, attribute P/R/F1 |

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The WHU-RS19 images are subject to their original
terms and are not included.
