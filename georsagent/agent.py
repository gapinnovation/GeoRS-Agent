"""GeoRS-Agent interactive mode: an LLM that chooses which tool to call.

The agent never sees the image. Perception is a tool (extract_attributes);
reasoning over classes is done by deterministic tools on the knowledge base.
At each step the LLM returns ONE JSON object {thought, tool, args}; the
format is constrained by a JSON schema and every argument is re-validated in
Python. Every step (raw text, parsed action, tool result) is logged.
"""
from __future__ import annotations

import json

from .ollama_client import chat_messages
from .schemas import parse_json
from .scorer import score_classes
from .vocab import ABSENT, ATTR_IDS, CLASSES, PRESENT, UNCERTAIN

TOOLS = ["extract_attributes", "score_classes", "compare_classes", "get_class_profile",
         "set_attribute", "list_classes", "answer"]

SYSTEM_AGENT = f"""You are GeoRS-Agent, an assistant for remote sensing scene interpretation.
You cannot see the image. You must use tools and base every statement on tool results.
At each step reply with ONE JSON object: {{"thought": "<at most 25 words>", "tool": "<tool name>", "args": {{...}}}}

Tools:
- extract_attributes {{}}: returns the 38 visual attributes of the current image (present / absent / uncertain).
- score_classes {{}}: ranks the 19 scene classes from the current attribute evidence (top-3, relative scores, not probabilities).
- compare_classes {{"class_a": C, "class_b": C}}: attributes in the current evidence that favour class_a over class_b.
- get_class_profile {{"class": C}}: attributes that are typical or atypical of a class according to the knowledge base.
- set_attribute {{"attribute": A, "state": "present"|"absent"|"uncertain"}}: hypothetically change one attribute of the current evidence (for what-if questions); later tools use the changed evidence.
- list_classes {{}}: the 19 admissible classes.
- answer {{"text": "...", "label": C or "none", "cited_attributes": [{{"attribute": A, "state": "present"|"absent"}}]}}: final answer; ends the episode.

Rules: cite only attributes returned by tools, with the state given by the tools. Use class and attribute names exactly as written.
Classes: {", ".join(CLASSES)}
Attributes: {", ".join(ATTR_IDS)}"""

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "tool": {"type": "string", "enum": TOOLS},
        "args": {
            "type": "object",
            "properties": {
                "class_a": {"type": "string", "enum": CLASSES},
                "class_b": {"type": "string", "enum": CLASSES},
                "class": {"type": "string", "enum": CLASSES},
                "attribute": {"type": "string", "enum": ATTR_IDS},
                "state": {"type": "string", "enum": [PRESENT, ABSENT, UNCERTAIN]},
                "text": {"type": "string"},
                "label": {"type": "string", "enum": CLASSES + ["none"]},
                "cited_attributes": {
                    "type": "array",
                    "items": {"type": "object",
                              "properties": {"attribute": {"type": "string", "enum": ATTR_IDS},
                                             "state": {"type": "string", "enum": [PRESENT, ABSENT]}},
                              "required": ["attribute", "state"], "additionalProperties": False}},
            },
            "additionalProperties": False,
        },
    },
    "required": ["thought", "tool", "args"],
    "additionalProperties": False,
}

ANSWER_SCHEMA = {  # used by the NOTOOLS baseline: same answer format, single call
    "type": "object",
    "properties": {k: STEP_SCHEMA["properties"]["args"]["properties"][k]
                   for k in ("text", "label", "cited_attributes")},
    "required": ["text", "label", "cited_attributes"],
    "additionalProperties": False,
}

ALLOWED_ARGS = {"extract_attributes": set(), "score_classes": set(), "list_classes": set(),
                "compare_classes": {"class_a", "class_b"}, "get_class_profile": {"class"},
                "set_attribute": {"attribute", "state"},
                "answer": {"text", "label", "cited_attributes"}}

REQUIRED_ARGS = {"compare_classes": ["class_a", "class_b"], "get_class_profile": ["class"],
                 "set_attribute": ["attribute", "state"], "answer": ["text", "label", "cited_attributes"]}

PROFILE_PRESENT, PROFILE_ABSENT = 0.7, 0.1


def class_profile(kb, c):
    p = kb["profiles"][c]["p_present"]
    return {"class": c,
            "typical_present": sorted([a for a in ATTR_IDS if p[a] >= PROFILE_PRESENT], key=lambda a: -p[a]),
            "typical_absent": sorted([a for a in ATTR_IDS if p[a] <= PROFILE_ABSENT], key=lambda a: p[a])}


def compare(kb, evidence, a, b, k=6):
    import math
    pa, pb = kb["profiles"][a]["p_present"], kb["profiles"][b]["p_present"]
    out = []
    for att, s in evidence.items():
        if s == PRESENT:
            w = math.log(pa[att]) - math.log(pb[att])
        elif s == ABSENT:
            w = math.log(1 - pa[att]) - math.log(1 - pb[att])
        else:
            continue
        if w > 0:
            out.append((w, att, s))
    out.sort(reverse=True)
    return [{"attribute": att, "state": s, "weight": round(w, 2)} for w, att, s in out[:k]]


class Episode:
    """Tool state for one question about one image."""

    def __init__(self, kb, evidence_provider, strict_args=False):
        self.kb = kb
        self.strict_args = strict_args   # v2: reject arguments a tool does not accept
        self.provider = evidence_provider   # callable -> {attr: state}
        self.original = None
        self.evidence = None
        self.tools_used = []

    def run_tool(self, tool, args):
        self.tools_used.append(tool)
        extra = sorted(set(args) - ALLOWED_ARGS.get(tool, set()))
        if self.strict_args and extra:
            hint = (" To evaluate a hypothetical change, call set_attribute first, then score_classes."
                    if tool in ("score_classes", "compare_classes") and {"attribute", "state"} & set(extra) else "")
            return {"error": f"{tool} does not accept the arguments {extra}; "
                             f"allowed: {sorted(ALLOWED_ARGS.get(tool, set())) or 'none'}.{hint}"}
        missing = [a for a in REQUIRED_ARGS.get(tool, []) if a not in args]
        if missing:
            return {"error": f"missing arguments: {missing}"}
        if tool == "list_classes":
            return {"classes": CLASSES}
        if tool == "get_class_profile":
            return class_profile(self.kb, args["class"])
        if tool == "extract_attributes":
            if self.original is None:
                self.original = dict(self.provider())
                self.evidence = dict(self.original)
            return {"attributes": self.evidence}
        if self.evidence is None:
            return {"error": "no attribute evidence yet: call extract_attributes first"}
        if tool == "score_classes":
            s = score_classes(self.evidence, self.kb)
            return {"ranking": [{"label": r["label"], "relative_score": round(r["normalised_score"], 3)}
                                for r in s["ranking"]]}
        if tool == "compare_classes":
            return {"favouring": args["class_a"], "over": args["class_b"],
                    "attributes": compare(self.kb, self.evidence, args["class_a"], args["class_b"])}
        if tool == "set_attribute":
            self.evidence[args["attribute"]] = args["state"]
            return {"ok": True, "attribute": args["attribute"], "new_state": args["state"]}
        return {"error": f"unknown tool {tool}"}


def run_agent_episode(client, kb, question, evidence_provider, seed, temperature, max_steps=8,
                      strict_args=False):
    ep = Episode(kb, evidence_provider, strict_args=strict_args)
    messages = [{"role": "system", "content": SYSTEM_AGENT}, {"role": "user", "content": question}]
    steps, final, n_format_errors = [], None, 0
    for i in range(max_steps):
        res = chat_messages(client, messages, fmt=STEP_SCHEMA, seed=seed + i, temperature=temperature)
        obj, err = parse_json(res.text)
        step = {"i": i, "raw": res.text, "meta": res.meta}
        if err or not isinstance(obj, dict) or obj.get("tool") not in TOOLS:
            n_format_errors += 1
            result = {"error": "invalid step format; reply with one JSON object {thought, tool, args}"}
        elif obj["tool"] == "answer":
            missing = [a for a in REQUIRED_ARGS["answer"] if a not in obj.get("args", {})]
            if missing:
                result = ep.run_tool("answer", obj.get("args", {}))
            else:
                ep.tools_used.append("answer")
                final = obj["args"]
                step.update({"action": obj, "result": None})
                steps.append(step)
                break
        else:
            result = ep.run_tool(obj["tool"], obj.get("args", {}))
        step.update({"action": obj, "result": result})
        steps.append(step)
        messages.append({"role": "assistant", "content": res.text})
        messages.append({"role": "user", "content": "TOOL RESULT: " + json.dumps(result)})
    return {"final": final, "steps": steps, "n_steps": len(steps), "tools_used": ep.tools_used,
            "n_format_errors": n_format_errors, "hit_max_steps": final is None,
            "evidence_original": ep.original, "evidence_final": ep.evidence}


SYSTEM_NOTOOLS = (
    "You are an expert in remote sensing image interpretation. Answer the question about the image. "
    "Answer only with a JSON object: {\"text\": ..., \"label\": <class or \"none\">, "
    "\"cited_attributes\": [{\"attribute\": ..., \"state\": \"present\"|\"absent\"}]}.\n"
    f"Classes: {', '.join(CLASSES)}\nAttributes: {', '.join(ATTR_IDS)}"
)


def run_notools(client, question, image_bytes, seed, temperature):
    res = client.chat(SYSTEM_NOTOOLS, question, images=[image_bytes], fmt=ANSWER_SCHEMA,
                      seed=seed, temperature=temperature)
    obj, err = parse_json(res.text)
    return {"final": obj if isinstance(obj, dict) and not err else None, "raw": res.text, "meta": res.meta}
