# Semantic knowledge base

GeoRS-Agent consults three reference resources derived from WHU-RS19 / WHU-RS19 ABZSL. They are supplied to the conversational model as textual and tabular context (manuscript Sec. 3.2 and 4.2). They are **reference knowledge, not classification rules**: the agent does not compute attribute overlaps, weights or ranking scores.

| File | Content | Status |
|---|---|---|
| `whu_rs19_classes.json` | The 19 canonical WHU-RS19 scene labels (closed-set output vocabulary) | Complete |
| `abzsl_attributes.json` | The 38 WHU-RS19 ABZSL attributes: 19 objects, 11 dominant colours, 8 geometric patterns, with original annotation column names and accepted aliases | Complete |
| `class_attribute_profiles.csv` | Per-class summary of commonly associated cues | **Template** — see below |

## Class vocabulary vs. attribute vocabulary

The two vocabularies play different roles. The class vocabulary defines which labels are permitted in the closed-set evaluation; the attribute vocabulary supplies the evidence used to interpret an image or query. Neither is a complete ontology of remote sensing concepts: a term outside the 19 classes (e.g. *harbour*, *wetland*, *overpass*) can be meaningful even though it is not a canonical label.

## Attribute naming

Canonical attribute names follow the dataset release. The manuscript uses two alternative terms, which are accepted as aliases:

| Manuscript term | ABZSL attribute |
|---|---|
| water vehicles | Water Transport |
| aircraft | Airplane |

`grey` is accepted as a spelling variant of `gray`.

## Class–attribute profiles

`class_attribute_profiles.csv` currently contains only the two qualitative profiles described in the manuscript (port, airport). The remaining rows must be filled with the profiles actually supplied to the agent before the repository can be used to reproduce the configuration.

If the profiles need to be regenerated from the annotations, `georsagent build-profiles` derives per-class attribute prevalence from the WHU-RS19 ABZSL annotation file (available on request from the dataset authors):

```bash
georsagent build-profiles path/to/annotations.csv \
    --class-column Class --split-column Split --train-value train \
    --out-dir knowledge_base/generated
```

Use the **training split only**. Profiles built from all images would include the images later used for testing, which is the leakage risk discussed in manuscript Sec. 4.6. The command writes:

- `class_attribute_prevalence.csv` — fraction of training images of each class in which each attribute is present;
- `class_profiles.md` — a readable summary listing, for each class, the attributes present in at least half of its training images and the attributes that are markedly more frequent in that class than elsewhere. This file can be supplied to the agent as reference context.
