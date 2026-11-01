"""Statistics reported in the paper. Pure functions, no model calls."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.stats import binomtest, norm


def wilson_ci(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def accuracy(y_true, y_pred) -> dict:
    k = sum(t == p for t, p in zip(y_true, y_pred))
    n = len(y_true)
    lo, hi = wilson_ci(k, n)
    return {"n": n, "correct": k, "accuracy": k / n if n else float("nan"),
            "ci95_low": lo, "ci95_high": hi}


def macro_f1(y_true, y_pred, labels) -> float:
    f1s = []
    for c in labels:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Paired comparison on the same images. b = A right & B wrong, c = A wrong & B right."""
    b = sum(a and not bb for a, bb in zip(correct_a, correct_b))
    c = sum(bb and not a for a, bb in zip(correct_a, correct_b))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only": b, "b_only": c, "p_exact": p,
            "acc_diff": (sum(correct_b) - sum(correct_a)) / len(correct_a)}


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """ratings[i] = labels assigned to item i by each run (same number per item)."""
    cats = sorted({l for r in ratings for l in r})
    n = len(ratings[0])
    if n < 2 or not all(len(r) == n for r in ratings):
        return float("nan")
    M = np.array([[Counter(r)[c] for c in cats] for r in ratings], float)
    P_i = ((M**2).sum(1) - n) / (n * (n - 1))
    p_j = M.sum(0) / M.sum()
    Pbar, Pe = P_i.mean(), (p_j**2).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else 1.0


def attribute_prf(gt: list[dict], pred: list[dict], attrs) -> dict:
    """Micro P/R/F1 of 'present' predictions against binary reference.

    Also returns:
      false_absence_rate = P(pred 'absent' | ref present)  (the airport error)
      uncertain_rate     = share of 'uncertain' answers
    """
    tp = fp = fn = fa = pos = unc = tot = 0
    for g, p in zip(gt, pred):
        for a in attrs:
            ref = int(g[a]) == 1
            s = p.get(a, "uncertain")
            tot += 1
            unc += s == "uncertain"
            if s == "present":
                tp += ref
                fp += not ref
            elif ref:
                fn += 1
            if ref:
                pos += 1
                fa += s == "absent"
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1,
            "false_absence_rate": fa / pos if pos else float("nan"),
            "uncertain_rate": unc / tot if tot else float("nan"), "n_judgements": tot}
