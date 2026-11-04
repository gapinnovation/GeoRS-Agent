"""Experimental conditions, all with the same underlying model.

B0      direct prompting (image -> label), no class list
B1      image + list of 19 admissible classes
B2      GeoRS-Agent: VLM attribute extraction -> deterministic KB scorer
        -> LLM explanation constrained to the scorer's evidence
ORACLE  reference ABZSL attributes -> deterministic KB scorer (no VLM);
        upper bound that separates extraction error from reasoning error

Every function returns a JSON-serialisable record containing the raw model
text of every attempt, so that nothing reported in the paper depends on
post-hoc reading of a chat window.
"""
from __future__ import annotations

from pathlib import Path

from . import prompts as P
from .schemas import (ATTRIBUTE_SCHEMA, EXPLANATION_SCHEMA, explanation_faithfulness,
                      parse_json, scene_schema, schema_errors, taxonomy_check)
from .scorer import score_classes
from .vocab import ATTR_IDS, PRESENT, UNCERTAIN


def _attempt(client, system, user, images, schema, enforce, seed, temperature):
    res = client.chat(system, user, images=images, fmt=schema if enforce else None,
                      seed=seed, temperature=temperature)
    obj, err = parse_json(res.text)
    errs = [err] if err else schema_errors(obj, schema)
    return {"raw": res.text, "meta": res.meta, "parsed": obj, "schema_errors": errs}


def run_scene_prompt(condition, client, image_bytes, cfg, seed, temperature, extra_text="", kb=None):
    """B0 / B1 / B1_KB. Retry (if enabled) on JSON/schema failure and, for the
    constrained conditions, on taxonomy failure. First-attempt and final
    outcomes are both recorded. B1_KB = B1 + the full knowledge base `kb` as text."""
    constrained = condition in ("B1", "B1_KB")
    if condition == "B1_KB":
        if kb is None:
            raise ValueError("B1_KB needs a knowledge base")
        user = P.user_b1_kb(kb) + extra_text
    else:
        user = (P.USER_B1 if constrained else P.USER_B0) + extra_text
    schema = scene_schema(constrained_labels=constrained and cfg["enforce_schema"])
    attempts = []
    for i in range(1 + (cfg["max_retries"] if cfg["guardrail"] else 0)):
        a = _attempt(client, P.SYSTEM_SCENE, user, [image_bytes], schema,
                     cfg["enforce_schema"], seed + i, temperature)
        a["taxonomy"] = taxonomy_check(a["parsed"]) if isinstance(a["parsed"], dict) else None
        attempts.append(a)
        ok = not a["schema_errors"] and (not constrained or a["taxonomy"]["full_compliance"])
        if ok:
            break
    final = attempts[-1]
    pred = final["parsed"].get("label") if isinstance(final["parsed"], dict) else None
    return {
        "condition": condition,
        "prompt_hash": P.prompt_hash(P.SYSTEM_SCENE, user),
        "kb_sha256": kb["sha256"] if kb else None,
        "attempts": attempts,
        "n_attempts": len(attempts),
        "first_valid_json": not attempts[0]["schema_errors"],
        "first_taxonomy": attempts[0]["taxonomy"],
        "final_valid_json": not final["schema_errors"],
        "final_taxonomy": final["taxonomy"],
        "pred_label": pred,
        "alternatives": (final["parsed"] or {}).get("alternatives") if isinstance(final["parsed"], dict) else None,
    }


def extract_attributes(client, image_bytes, cfg, seed, temperature):
    attempts = []
    for i in range(1 + (cfg["max_retries"] if cfg["guardrail"] else 0)):
        a = _attempt(client, P.SYSTEM_ATTR, P.USER_ATTR, [image_bytes], ATTRIBUTE_SCHEMA,
                     cfg["enforce_schema"], seed + i, temperature)
        attempts.append(a)
        if not a["schema_errors"]:
            break
    final = attempts[-1]
    evidence = final["parsed"] if not final["schema_errors"] else None
    return evidence, attempts


def explain(client, scored, evidence, cfg, seed, temperature, max_items=6):
    supporting = [(a, evidence[a], w) for a, w in scored["margin_contributions"].items()
                  if w > 0][:max_items]
    if not supporting:  # nothing discriminative was observed
        return {"skipped": "no supporting evidence", "attempts": []}
    user = P.user_explain(scored["label"], scored["runner_up"], supporting)
    attempts = []
    for i in range(1 + (cfg["max_retries"] if cfg["guardrail"] else 0)):
        a = _attempt(client, P.SYSTEM_EXPLAIN, user, None, EXPLANATION_SCHEMA,
                     cfg["enforce_schema"], seed + i, temperature)
        a["faithfulness"] = (explanation_faithfulness(a["parsed"]["cited_evidence"], evidence)
                             if not a["schema_errors"] else None)
        attempts.append(a)
        if a["faithfulness"] and a["faithfulness"]["faithful"]:
            break
    final = attempts[-1]
    return {
        "prompt_hash": P.prompt_hash(P.SYSTEM_EXPLAIN, user),
        "supporting_evidence": supporting,
        "attempts": attempts,
        "first_faithfulness": attempts[0]["faithfulness"],
        "final_faithfulness": final["faithfulness"],
        "explanation": final["parsed"]["explanation"] if not final["schema_errors"] else None,
    }


def run_agent(client, image_bytes, kb, cfg, seed, temperature, evidence_override=None,
              provenance="model_observed", with_explanation=True):
    """B2 (evidence from the VLM) or ORACLE/interventions (evidence supplied)."""
    rec = {"condition": "B2" if evidence_override is None else "ORACLE"}
    if evidence_override is None:
        evidence, attempts = extract_attributes(client, image_bytes, cfg, seed, temperature)
        rec.update({"attr_prompt_hash": P.prompt_hash(P.SYSTEM_ATTR, P.USER_ATTR),
                    "attr_attempts": attempts, "attr_n_attempts": len(attempts),
                    "attr_first_valid": not attempts[0]["schema_errors"]})
    else:
        evidence = dict(evidence_override)
    if evidence is None:
        rec.update({"pred_label": None, "failure": "attribute extraction invalid"})
        return rec
    rec["evidence"] = {a: {"state": evidence.get(a, UNCERTAIN), "provenance": provenance}
                       for a in ATTR_IDS}
    scored = score_classes(evidence, kb)
    rec.update({"pred_label": scored["label"],
                "alternatives": [r["label"] for r in scored["ranking"][1:]],
                "scoring": scored,
                "final_taxonomy": {"primary_in_taxonomy": True, "alternatives_in_taxonomy": True,
                                   "full_compliance": True}})  # by construction
    if with_explanation and client is not None:
        rec["explanation"] = explain(client, scored, evidence, cfg, seed + 100, temperature)
    return rec


def resolve_image(images_root: str | Path, filepath: str) -> Path:
    """Accept both layouts: <root>/Train/<Class>/<file> (as in the workbook) and
    <root>/<Class>/<file> (original WHU-RS19 folders). The split always comes
    from the workbook's Partition column, never from the folder layout."""
    root = Path(images_root)
    for cand in (root / filepath, root / filepath.split("/", 1)[1]):
        if cand.exists():
            return cand
        # case-insensitive fallback (e.g. Desert_01.jpg vs desert_01.jpg), only if unambiguous
        if cand.parent.is_dir():
            hits = [f for f in cand.parent.iterdir() if f.name.lower() == cand.name.lower()]
            if len(hits) == 1:
                return hits[0]
    raise FileNotFoundError(root / filepath)


def read_image(images_root: str | Path, filepath: str) -> bytes:
    return resolve_image(images_root, filepath).read_bytes()
