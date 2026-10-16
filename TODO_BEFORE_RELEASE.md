# Before making the repository public

Items that cannot be filled from the manuscript alone. Delete this file once they are done.

## Required for reproducibility (manuscript Sec. 4.1 and Sec. 7)

- [ ] **`agent/system_prompt.md`** — replace the reconstructed template with the verbatim instructions used in the experiments, and remove the STATUS comment.
- [ ] **`agent/model_config.yaml`** — fill platform, model, version, experiment dates, sampling parameters and enabled tools.
- [ ] **`evaluation/image_test_instruction.md`** — replace the paraphrase with the verbatim instruction sent with each image.
- [ ] **`knowledge_base/class_attribute_profiles.csv`** — replace with the profiles actually supplied to the agent (17 of 19 rows are empty), or regenerate them with `georsagent build-profiles` from the training split and state this in the manuscript.
- [ ] **`evaluation/raw_outputs/`** — add the ten original responses (format in the folder README), then run `georsagent metrics --raw-dir evaluation/raw_outputs` and confirm it reports no mismatch.
- [ ] **`evaluation/image_tests.csv`** — fill the `image_file` column with the WHU-RS19 file names.
- [ ] **T01 (airport)** — record the non-canonical alternative label. The manuscript implies one (3/10 compliant overall) but does not name it.
- [ ] **`examples/controlled/case_b_river_pond.json`** — replace the fields marked `[NOT REPORTED]` with the original output.

## Metadata

- [ ] Replace `<repository-url>` in `README.md` with the GitHub URL and add `repository-code:` to `CITATION.cff`.
- [ ] Add ORCID iDs to `CITATION.cff`.
- [ ] After acceptance: update the paper badge, BibTeX and `CITATION.cff` with volume, article number and DOI.
- [ ] Optionally archive a release on Zenodo to obtain a DOI for the repository and cite it in the Data Availability Statement.

## Check

```bash
pytest
georsagent metrics --raw-dir evaluation/raw_outputs
georsagent validate examples/controlled/*.json evaluation/raw_outputs/*.json
grep -rn "TODO\|NOT REPORTED\|RECONSTRUCTED\|PARAPHRASE" --include=*.md --include=*.yaml --include=*.csv --include=*.json .
```
