# Evaluation

The manuscript evaluates GeoRS-Agent in two separate settings (Sec. 4.6). They are kept apart because the controlled queries were built from known class–attribute associations and are not independent evidence of recognition performance.

| Setting | Inputs | Purpose | Location |
|---|---|---|---|
| Controlled attribute queries | 2 attribute sets | Illustrate constrained and ambiguity-aware reasoning | `examples/controlled/` |
| Image-based assessment | 10 WHU-RS19 images, one per category | Primary-class agreement and taxonomy compliance | this folder |

## Image-based protocol

- Categories: airport, beach, bridge, commercial, farmland, forest, industrial, parking, railway station, viaduct (10 of 19).
- Each image submitted in a separate conversation with the same instruction (`image_test_instruction.md`); ground truth not disclosed; configuration not modified between tests; structured output mode.
- All outputs retained, including non-canonical labels; nothing relabelled.

## Outcome measures (Sec. 4.7)

| Measure | Definition | Result |
|---|---|---|
| Exact-match accuracy | Primary label equals the reference class after case normalisation, without synonym replacement | 8/10 |
| Semantic class agreement | Exact match, or the manually defined equivalence *parking lot* ≡ *parking* (`label_equivalences.json`) | 9/10 |
| Full-response taxonomy compliance | Primary label **and** every alternative are canonical WHU-RS19 class names | 3/10 |

Compliance measures adherence to the output vocabulary, not whether the interpretation is correct. A canonical label can misdescribe a scene, and a non-canonical term (e.g. *overpass*) can describe a real scene element.

With n = 10, sampling uncertainty is large: Wilson 95% intervals are 49–94% for exact match, 60–98% for semantic agreement and 11–60% for full compliance. These intervals are printed by `georsagent metrics` as a descriptive addition and are not reported in the manuscript.

## Reproducing the numbers

```bash
georsagent metrics                                  # from the manuscript table
georsagent metrics --raw-dir evaluation/raw_outputs # recomputed from original outputs
georsagent metrics --json results.json              # machine-readable output
```

`image_tests.csv` encodes Table 3. Because the manuscript does not list every alternative label, full compliance in this file is taken as reported (`fully_compliant` column) and checked for internal consistency: a row cannot be compliant if its primary label or a listed alternative is non-canonical. For **T01 (airport)** the manuscript does not name the non-canonical label; its non-compliance follows from the reported total (3/10) and the three rows marked as having canonical alternatives. Adding the raw outputs to `raw_outputs/` removes this dependency.

## Qualitative observations (Sec. 5.4)

Error types observed in the image responses, reported descriptively (no expert rating study was conducted):

1. Non-canonical primary or alternative labels (*parking lot*, *overpass*, *island*).
2. Visual-evidence errors: the airport response stated that no aircraft were visible although aircraft appear in the image.
3. Over-narrow class assumptions: the viaduct response discounted the reference class because repeated piers or arches were not detected.

Other responses expressed uncertainty appropriately when distinguishing commercial from industrial or farmland from meadow.

## Limitations

Ten images across ten of nineteen classes; one image per class; no verified held-out sampling (the record does not establish whether the test images were excluded from profile preparation); no baseline (e.g. unconstrained LLM); no independent expert rating of attribute grounding or explanation quality; verbal confidence is uncalibrated. The results describe this sample and are not an estimate of generalisation.
