"""Deterministic class scoring from attribute evidence.

Bernoulli naive-Bayes log-likelihood with uniform class prior:
    score(c) = sum_{a observed} [ x_a log p_ca + (1-x_a) log(1-p_ca) ]
Only attributes whose state is "present" or "absent" enter the sum; attributes
that are "uncertain" or not provided are skipped. This is what makes
"not mentioned" and "explicitly negated" behave differently.

The normalised score is a softmax of the log-likelihoods. It is NOT a
calibrated probability and must not be reported as one.
"""
from __future__ import annotations

import math

from .vocab import ABSENT, PRESENT


def _ll_terms(evidence: dict[str, str], profile: dict[str, float]) -> dict[str, float]:
    terms = {}
    for a, state in evidence.items():
        if a not in profile:
            raise KeyError(f"attribute not in vocabulary: {a}")
        p = profile[a]
        if state == PRESENT:
            terms[a] = math.log(p)
        elif state == ABSENT:
            terms[a] = math.log(1.0 - p)
    return terms


def score_classes(evidence: dict[str, str], kb: dict, top_k: int = 3) -> dict:
    """Rank all classes; return top-k with per-attribute contributions.

    Contribution of attribute a to the margin between the top class and the
    runner-up = term_top(a) - term_second(a). Positive values support the
    decision; these are the only facts the explanation is allowed to cite.
    """
    profiles = {c: kb["profiles"][c]["p_present"] for c in kb["classes"]}
    terms = {c: _ll_terms(evidence, p) for c, p in profiles.items()}
    ll = {c: sum(t.values()) for c, t in terms.items()}
    # tie-break alphabetically for full determinism
    ranked = sorted(ll, key=lambda c: (-ll[c], c))
    m = max(ll.values())
    z = sum(math.exp(v - m) for v in ll.values())
    norm = {c: math.exp(ll[c] - m) / z for c in ll}
    top, second = ranked[0], ranked[1]
    contrib = {a: terms[top][a] - terms[second][a] for a in terms[top]}
    return {
        "label": top,
        "ranking": [{"label": c, "log_likelihood": ll[c], "normalised_score": norm[c]}
                    for c in ranked[:top_k]],
        "n_evidence": len(terms[top]),
        "margin_contributions": dict(sorted(contrib.items(), key=lambda kv: -kv[1])),
        "runner_up": second,
    }


def evidence_from_binary(row: dict, attrs) -> dict[str, str]:
    """Reference (ground-truth) annotation row -> fully observed evidence."""
    return {a: PRESENT if int(row[a]) == 1 else ABSENT for a in attrs}
