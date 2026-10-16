# Raw agent outputs

Place the original, unedited agent responses from the ten image-based tests here, one JSON file per test (e.g. `T01_airport.json`). With these files, every reported metric can be recomputed from the outputs themselves instead of from the manuscript table:

```bash
georsagent metrics --raw-dir evaluation/raw_outputs
```

The command also cross-checks the recomputed values against `evaluation/image_tests.csv` and reports any mismatch.

## File format

```json
{
  "test_id": "T01",
  "reference_class": "airport",
  "image_file": "<WHU-RS19 file name>",
  "conversation_date": "YYYY-MM-DD",
  "response": { "...": "the structured response exactly as returned by the agent" }
}
```

`response` must follow `agent/output_schema.json`. Keep non-canonical labels exactly as generated: the evaluation counts them as non-compliant and never relabels them. If the agent returned text around the JSON, store the JSON part in `response` and the full text in an additional `response_text` field.

Do not commit the WHU-RS19 images themselves; reference them by file name.
