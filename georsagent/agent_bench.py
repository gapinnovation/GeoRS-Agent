"""Verifiable question benchmark for the interactive agent.

PRE-SPECIFIED before any agent run (do not edit after running). Every gold
answer is computed deterministically with the same tools, from the B2
primary-run attribute evidence of the image.

Question types
  Q1 classify       "What type of scene ...?"            gold: reference label
                     secondary: agreement with the fixed pipeline (B2) label
  Q2 contrast       "Why <L> rather than <R>?"            L = B2 label, R = B2 runner-up
                     correct iff >=1 cited attribute AND every cited attribute
                     is in the support set (positive contribution for L over R,
                     same state as the evidence)
  Q3 counterfactual "If <a*> were not present ...?"      a* = present attribute with
                     the largest contribution to L; gold = re-scoring with a* absent
                     correct iff label == gold
  Q4 profile        "Which attributes characterise <C>?" C = reference class
                     gold = typical_present of the KB profile (p >= 0.7)
                     correct iff >=1 cited 'present' attribute AND precision >= 0.8;
                     F1 also reported

Required tools (a correct answer obtained without them counts as ungrounded):
  Q1 extract_attributes + score_classes; Q2 extract_attributes + compare_classes;
  Q3 extract_attributes + set_attribute + score_classes; Q4 get_class_profile
"""
from __future__ import annotations

from .agent import class_profile, compare
from .scorer import score_classes
from .vocab import ABSENT, PRESENT

REQUIRED_TOOLS = {
    "Q1": {"extract_attributes", "score_classes"},
    "Q2": {"extract_attributes", "compare_classes"},
    "Q3": {"extract_attributes", "set_attribute", "score_classes"},
    "Q4": {"get_class_profile"},
}


def make_questions(image_id, true_label, evidence, kb):
    s = score_classes(evidence, kb)
    L, R = s["label"], s["runner_up"]
    pos = [a for a, w in s["margin_contributions"].items() if w > 0 and evidence[a] == PRESENT]
    qs = [
        {"qtype": "Q1", "question": "What type of scene is shown in the current image? "
                                     "Give the scene class as label.",
         "gold": {"label": true_label, "pipeline_label": L}},
        {"qtype": "Q2", "question": f"Why is the current image interpreted as {L} rather than {R}? "
                                     "Cite the attributes that support this.",
         "gold": {"class_a": L, "class_b": R}},
        {"qtype": "Q4", "question": f"Which attributes typically characterise the class {true_label}? "
                                     "Cite them as present attributes; set label to none.",
         "gold": {"class": true_label, "typical_present": class_profile(kb, true_label)["typical_present"]}},
    ]
    if pos:
        a = pos[0]
        ev2 = {**evidence, a: ABSENT}
        qs.insert(2, {"qtype": "Q3",
                      "question": f"If the attribute {a} were not present in the current image, "
                                  "which scene class would be chosen? Give it as label.",
                      "gold": {"attribute": a, "label": score_classes(ev2, kb)["label"],
                               "original_label": L}})
    for q in qs:
        q["image_id"] = image_id
        q["qid"] = f"{image_id}:{q['qtype']}"
    return qs


def score_answer(q, final, evidence, kb, tools_used=None):
    """Return a dict of pre-specified metrics for one answer (agent or NOTOOLS)."""
    out = {"answered": final is not None}
    if final is None:
        # unanswered episodes count as wrong and NOT grounded (they stay in the denominator)
        out["correct"] = False
        if tools_used is not None:
            out["required_tools_used"] = REQUIRED_TOOLS[q["qtype"]] <= set(tools_used)
            out["grounded_correct"] = False
        return out
    cited = final.get("cited_attributes") or []
    label = final.get("label")
    g = q["gold"]
    if q["qtype"] == "Q1":
        out["correct"] = label == g["label"]
        out["agrees_with_pipeline"] = label == g["pipeline_label"]
    elif q["qtype"] == "Q2":
        support = {(x["attribute"], x["state"]) for x in compare(kb, evidence, g["class_a"], g["class_b"], k=99)}
        ok = [(c["attribute"], c["state"]) in support for c in cited]
        out["n_cited"] = len(cited)
        out["citation_precision"] = sum(ok) / len(ok) if ok else 0.0
        out["correct"] = bool(ok) and all(ok)
    elif q["qtype"] == "Q3":
        out["correct"] = label == g["label"]
        out["label_should_change"] = g["label"] != g["original_label"]
    elif q["qtype"] == "Q4":
        gold = set(g["typical_present"])
        pres = {c["attribute"] for c in cited if c["state"] == PRESENT}
        tp = len(pres & gold)
        prec = tp / len(pres) if pres else 0.0
        rec = tp / len(gold) if gold else 0.0
        out.update({"precision": prec, "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})
        out["correct"] = bool(pres) and prec >= 0.8
    # citations that contradict the image evidence (Q1-Q3) - hallucinated grounding
    if q["qtype"] in ("Q1", "Q2", "Q3") and evidence:
        skip = g.get("attribute") if q["qtype"] == "Q3" else None   # the hypothetically changed one
        out["contradicting_citations"] = sum(
            c["attribute"] != skip and evidence.get(c["attribute"]) != c["state"] for c in cited)
    if tools_used is not None:
        req = REQUIRED_TOOLS[q["qtype"]]
        out["required_tools_used"] = req <= set(tools_used)
        out["grounded_correct"] = out["correct"] and out["required_tools_used"]
    return out
