# Agent configuration

GeoRS-Agent was implemented as a custom agent in a conversational LLM environment, using natural-language instructions, the reference files in `knowledge_base/` and a predefined response format (manuscript Sec. 4.1). No model was trained or fine-tuned.

| File | Purpose |
|---|---|
| `system_prompt.md` | Agent instructions (role, workflow, label constraints, output format) |
| `model_config.yaml` | Record of platform, model, version and evaluation protocol |
| `output_schema.json` | Structure of the structured-mode response |
| `output_schema.strict.json` | Strict variant restricting classes and attributes to the canonical vocabularies |

## How the logical workflow is realised

The stages of the framework (task recognition → semantic grounding → attribute reasoning → class interpretation → ambiguity handling → explanation) are logical functions of the instructed LLM, not separately deployed services. Schematically,

```
(ĉ, C_alt, e, u, q) = f_LLM(x, A(x), P, I)
```

where `x` is the image or query, `A(x)` the reported attributes, `P` the 19 class profiles, `I` the instructions, `ĉ` the primary label, `C_alt` the alternatives, `e` the explanation, `u` the uncertainty note and `q` the verbal confidence. `f_LLM` denotes model-conditioned generation; it is not a scoring function.

## What the prototype does **not** do

- It computes no attribute-overlap count, weighted similarity or numeric class ranking, and uses no attribute weights or thresholds.
- Its confidence words (*high*, *moderate*, …) are uncalibrated verbal judgements, not probabilities.
- It does not verify that an output label belongs to the taxonomy. Label constraints are prompt-level only, and the evaluation shows they can be violated.
- It has no out-of-distribution detector and no tested abstention ("none of the above") behaviour.

`output_schema.strict.json` and `georsagent validate` implement deterministic checks that the manuscript lists as future work. They are provided as audit tools and were **not** part of the agent evaluated in the manuscript.

## Interaction modes

| Mode | Output |
|---|---|
| Simple semantic analysis | Attributes, predicted class, alternatives, concise explanation |
| Structured interpretation | JSON object (`output_schema.json`) — used for all evaluation runs |
| Extended report | Narrative version of the structured interpretation |

The modes differ only in presentation; they share the same evidence and model.
