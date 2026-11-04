"""Figures for the GeoRS-Agent paper, generated from the released raw logs.

    python paper/make_figures.py --results results --data data/processed/annotations.csv --out paper/figures
"""
import argparse, json, math, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

# ---- style: minimal academic, Okabe-Ito, sized for MDPI text width ------------------
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#8C8C8C")
FULL, HALF = 6.9, 3.35            # inches (MDPI full text width ~17.5 cm)
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "svg.fonttype": "none"})

COND = {  # display name, colour, primary analysis?
    "B0": ("B0 direct prompting", OI["grey"], True),
    "B1": ("B1 + class list", OI["blue"], True),
    "B1_NOSCHEMA": ("B1 without schema", OI["sky"], False),
    "B1_KB": ("B1 + full KB in prompt (human KB)", OI["purple"], False),
    "B1_VLMKB": ("B1 + full KB in prompt (VLM KB)", "#9A8A00", False),
    "B2": ("B2 GeoRS-Agent (human-annotation KB)", OI["verm"], True),
    "B2_VLMKB": ("B2 GeoRS-Agent (VLM-attribute KB)", OI["orange"], False),
    "ORACLE": ("Oracle (reference attributes)", OI["green"], True),
}
CLASS_ORDER = ["Airport", "Beach", "Bridge", "Commercial", "Desert", "Farmland", "footballField",
               "Forest", "Industrial", "Meadow", "Mountain", "Park", "Parking", "Pond", "Port",
               "railwayStation", "Residential", "River", "Viaduct"]


def save(fig, out, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---- Figure 1: architecture (conceptual, matches the implementation) -----------------------
def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(FULL, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    learned, determ = "#FDF0D5", "#E3F1FA"

    def box(x, y, w, h, title, body, fc, ec, fs=6.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=7, weight="bold")
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=fs, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#333333"))

    y0, h = 14, 19
    xs = [(1, 11), (15, 21), (39.5, 22), (65, 17.5), (86, 13.5)]
    box(*xs[0][:1], y0, xs[0][1], h, "Input", "aerial\nimage\n(RGB)", "#F2F2F2", "#999999") if False else None
    box(xs[0][0], y0, xs[0][1], h, "Input", "aerial image\n(RGB,\n600×600)", "#F2F2F2", "#999999")
    box(xs[1][0], y0, xs[1][1], h, "Perception", "VLM (Qwen2.5-VL 7B)\n38 ABZSL attributes:\npresent / absent /\nuncertain (JSON)", learned, OI["orange"])
    box(xs[2][0], y0, xs[2][1], h, "Class inference", "Bernoulli naive Bayes\non class–attribute\nprofiles; returns label,\ntop-3, contributions", determ, OI["blue"])
    box(xs[3][0], y0, xs[3][1], h, "Explanation", "LLM; may cite only\nthe evidence used\nby the classifier", learned, OI["orange"])
    box(xs[4][0], y0, xs[4][1], h, "Output", "label,\nalternatives,\nexplanation,\nprovenance", "#F2F2F2", "#999999")
    for (xa, wa), (xb, _) in zip(xs[:-1], xs[1:]):
        arrow(xa + wa + 0.6, y0 + h / 2, xb - 0.6, y0 + h / 2)
    # knowledge base
    ax.add_patch(FancyBboxPatch((37, 37.5), 27, 6, boxstyle="round,pad=0.3,rounding_size=1", fc=determ, ec=OI["blue"], lw=0.8))
    ax.text(50.5, 41.8, "Knowledge base", ha="center", va="center", fontsize=6.6, weight="bold")
    ax.text(50.5, 39.0, "p(attribute | class), train split only", ha="center", va="center", fontsize=5.9)
    arrow(50.5, 37.2, 50.5, 33.6)
    # validators band
    ax.add_patch(FancyBboxPatch((12, 1.5), 73.5, 7.4, boxstyle="round,pad=0.3,rounding_size=1", fc="white", ec="#555555", lw=0.7, ls=(0, (3, 2))))
    ax.text(48.75, 6.6, "Deterministic validators", ha="center", va="center", fontsize=6.6, weight="bold")
    ax.text(48.75, 3.6, "JSON schema · 19-class taxonomy · citation consistency · bounded retry (≤ 2) · full logging", ha="center", va="center", fontsize=5.9)
    for x in (25.5, 50.5, 73.75):
        arrow(x, 9.2, x, 13.6)
    # legend
    ax.add_patch(FancyBboxPatch((1, 41.2), 2.2, 2.2, boxstyle="round,pad=0.1", fc=learned, ec=OI["orange"], lw=0.7))
    ax.text(4.2, 42.3, "model-based", va="center", fontsize=6.2)
    ax.add_patch(FancyBboxPatch((1, 37.4), 2.2, 2.2, boxstyle="round,pad=0.1", fc=determ, ec=OI["blue"], lw=0.7))
    ax.text(4.2, 38.5, "deterministic", va="center", fontsize=6.2)
    save(fig, out, "fig1_architecture")


# ---- Figure 2: accuracy with 95% CI -------------------------------------------------------
def fig_accuracy(res, out):
    t = pd.read_csv(res / "table_accuracy.csv")
    t = t[t.run == "primary"].set_index("condition")
    len_b0 = pd.read_csv(res / "table_b0_lenient_posthoc.csv").iloc[0]
    order = ["B0", "B1", "B1_NOSCHEMA", "B1_KB", "B1_VLMKB", "B2", "B2_VLMKB", "ORACLE"]
    fig, ax = plt.subplots(figsize=(HALF * 1.45, 2.9))
    for i, c in enumerate(order[::-1]):
        r = t.loc[c]; name, col, primary = COND[c]
        ax.plot([r.ci95_low * 100, r.ci95_high * 100], [i, i], color=col, lw=1.6, solid_capstyle="butt")
        ax.plot(r.accuracy * 100, i, "o", ms=6, color=col if primary else "white", mec=col, mew=1.4, zorder=3)
        ax.text(r.ci95_high * 100 + 1.2, i, f"{r.accuracy*100:.1f}%", va="center", fontsize=6.8)
        if c == "B0":
            ax.plot([len_b0.ci95_low * 100, len_b0.ci95_high * 100], [i - 0.28] * 2, color=col, lw=1, ls=(0, (2, 1.5)))
            ax.plot(len_b0.accuracy * 100, i - 0.28, "D", ms=3.6, color="white", mec=col, mew=1)
            ax.text(len_b0.ci95_high * 100 + 1.2, i - 0.28, f"{len_b0.accuracy*100:.1f}% (lenient, post-hoc)",
                    va="center", fontsize=6, color="#555555")
    ax.set_yticks(range(len(order)), [COND[c][0] for c in order[::-1]])
    ax.set_xlim(35, 100); ax.set_xlabel("Accuracy on 301 test images (%), 95% Wilson CI")
    ax.axvline(100 / 19, color="#cccccc", lw=0.5)
    ax.grid(axis="x", color="#eeeeee", lw=0.5); ax.set_axisbelow(True)
    ax.plot([], [], "o", color="#555555", ms=5, label="pre-specified"); ax.plot([], [], "o", color="white", mec="#555555", ms=5, label="added after primary analysis")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.3)
    save(fig, out, "fig2_accuracy")


# ---- Figure 3: per-class accuracy heatmap --------------------------------------------------
def per_class(recs, cond, run="primary"):
    d = defaultdict(lambda: [0, 0])
    for r in recs:
        if r["condition"] == cond and r["run"] == run:
            d[r["true_label"]][1] += 1; d[r["true_label"]][0] += r.get("pred_label") == r["true_label"]
    return {c: d[c][0] / d[c][1] for c in d}


def fig_per_class(recs, out):
    conds = ["B1", "B1_KB", "B2", "B2_VLMKB", "ORACLE"]
    M = np.array([[per_class(recs, c)[k] for c in conds] for k in CLASS_ORDER])
    fig, ax = plt.subplots(figsize=(HALF * 1.4, 4.1))
    im = ax.imshow(M * 100, cmap="cividis", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.3, color="white" if v < 55 else "black")
    ax.set_xticks(range(5), ["B1\nclass\nlist", "B1_KB\nKB in\nprompt", "B2\nhuman\nKB", "B2\nVLM\nKB", "Oracle"], fontsize=6.3)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("Per-class accuracy (%)", fontsize=7)
    cb.outline.set_linewidth(0.4)
    save(fig, out, "fig3_per_class")


# ---- Figure 4: attribute agreement + convention mismatch ---------------------------------
def fig_attributes(res, ann, out):
    pa = pd.read_csv(res / "table_attribute_per_attr.csv").sort_values("f1")
    gcol = {"object": OI["blue"], "color": OI["orange"], "shape": OI["purple"]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 4.0), gridspec_kw={"width_ratios": [1, 1.0], "wspace": 0.42})
    y = np.arange(len(pa))
    a1.barh(y, pa.f1, color=[gcol[g] for g in pa.group], height=0.72)
    a1.set_yticks(y, pa.attribute, fontsize=5.6); a1.set_xlim(0, 1)
    a1.set_xlabel("F1 vs. reference annotation (primary run)")
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=c, label=g) for g, c in gcol.items()], frameon=False,
              loc="lower right", title="group", title_fontsize=6.5)
    a1.set_title("(a) Agreement of VLM attributes with ABZSL", loc="left")
    # (b) class-level prevalence: human annotation vs VLM, train split
    tr = ann[ann.split == "train"]
    vlm = [json.loads(l) for l in open(res / "train_attributes.jsonl")]
    rows = []
    attrs = [c for c in ann.columns if c not in ("image_id", "filepath", "label", "split")]
    vdf = pd.DataFrame([{**{a: (r["evidence"] or {}).get(a) == "present" for a in attrs}, "label": r["true_label"]} for r in vlm])
    for c in CLASS_ORDER:
        h = tr[tr.label == c][attrs].mean(); v = vdf[vdf.label == c][attrs].mean()
        for a in attrs:
            rows.append((c, a, h[a], v[a]))
    df = pd.DataFrame(rows, columns=["cls", "attr", "human", "vlm"])
    df["gap"] = df.vlm - df.human
    a2.scatter(df.human, df.vlm, s=7, color="#9a9a9a", alpha=0.55, lw=0)
    a2.plot([0, 1], [0, 1], color="#444444", lw=0.6, ls=(0, (3, 2)))
    must = [("River", "sinusoid"), ("Residential", "building"), ("Mountain", "tree")]
    top = df.reindex(df.gap.abs().sort_values(ascending=False).index)
    sel = pd.concat([df[(df.cls == c) & (df.attr == a)] for c, a in must] +
                    [top[~top.set_index(["cls", "attr"]).index.isin(must)].head(5)])
    a2.scatter(sel.human, sel.vlm, s=20, color=OI["verm"], zorder=3, lw=0)
    lines = ["Largest disagreements", "(human annotation vs VLM, train split)", ""]
    lines += [f"{r.cls}: {r.attr}   {r.human:.0%} vs {r.vlm:.0%}" for _, r in sel.iterrows()]
    a2.text(1.06, 1.0, "\n".join(lines), transform=a2.transAxes, va="top", ha="left", fontsize=5.9,
            linespacing=1.45, color="#222222")
    a2.set_xlim(-0.03, 1.03); a2.set_ylim(-0.03, 1.03); a2.set_aspect("equal")
    a2.set_xlabel("Share of train images: human annotation")
    a2.set_ylabel("Share of train images: VLM reports 'present'")
    a2.set_title("(b) Attribute prevalence per class\n(19 classes × 38 attributes)", loc="left")
    save(fig, out, "fig4_attributes")
    df.sort_values("gap", key=abs, ascending=False).to_csv(out / "fig4b_prevalence_gap.csv", index=False)


# ---- Figure 5: interventions + stability --------------------------------------------------
def fig_interventions_stability(res, out):
    iv = pd.DataFrame([json.loads(l) for l in open(res / "interventions.jsonl")])
    iv["drop"] = iv.margin_before - iv.margin_after
    st = pd.read_csv(res / "table_stability.csv"); st = st[st.version == "v2_per_image_seed"].set_index("condition")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FULL, 2.35), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1, 1.1]})
    names = [("OMIT", "omit"), ("NEGATE", "negate"), ("CONTRADICT", "contra-\ndict")]
    rates = [iv[iv.variant == v].label_changed.mean() * 100 for v, _ in names]
    ns = [len(iv[iv.variant == v]) for v, _ in names]
    lo_hi = [wilson(int(round(r * n / 100)), n) for r, n in zip(rates, ns)]
    a1.bar(range(3), rates, color=[OI["sky"], OI["blue"], OI["verm"]], width=0.62)
    a1.errorbar(range(3), rates, yerr=[[r - l * 100 for r, (l, h) in zip(rates, lo_hi)], [h * 100 - r for r, (l, h) in zip(rates, lo_hi)]],
                fmt="none", ecolor="#333333", lw=0.7, capsize=2)
    a1.set_xticks(range(3), [n for _, n in names]); a1.set_ylabel("Label changed (%)"); a1.set_ylim(0, 50)
    a1.set_title("(a) Label changes after a\nsingle-attribute edit", loc="left")
    data = [iv[iv.variant == v]["drop"].values for v, _ in names]
    bp = a2.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=0.8))
    for p, c in zip(bp["boxes"], [OI["sky"], OI["blue"], OI["verm"]]):
        p.set_facecolor(c); p.set_alpha(0.75); p.set_linewidth(0.6)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_linewidth(0.6)
    a2.set_xticks([1, 2, 3], [n for _, n in names]); a2.set_ylabel("Log-likelihood margin drop")
    a2.set_title("(b) Drop in decision\nmargin", loc="left")
    conds = ["B0", "B1", "B2", "B2_VLMKB"]
    a3.barh(range(4)[::-1], [st.loc[c].fleiss_kappa for c in conds], color=[COND[c][1] for c in conds], height=0.6)
    for i, c in zip(range(4)[::-1], conds):
        a3.text(st.loc[c].fleiss_kappa + 0.02, i, f"{st.loc[c].fleiss_kappa:.2f}", va="center", fontsize=6.5)
    a3.set_yticks(range(4)[::-1], ["B0", "B1", "B2 human KB", "B2 VLM KB"]); a3.set_xlim(0, 1)
    a3.set_xlabel("Fleiss' κ, 3 runs (T = 0.7)"); a3.set_title("(c) Run-to-run\nstability (95 images)", loc="left")
    save(fig, out, "fig5_interventions_stability")


# ---- Figure 6: agent benchmark ------------------------------------------------------------
def fig_agent(res, out):
    t = pd.read_csv(res / "table_agent.csv")
    modes = [("NOTOOLS", "No tools", OI["grey"]), ("AGENT", "Agent v1", OI["blue"]), ("AGENT_V2", "Agent v2", OI["orange"])]
    qs = [("Q1", "Q1\nscene class"), ("Q2", "Q2\ncontrastive"), ("Q3", "Q3\ncounterfactual"), ("Q4", "Q4\nclass profile")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.7), gridspec_kw={"wspace": 0.38, "width_ratios": [1.7, 1]})
    w = 0.26
    for j, (m, lab, col) in enumerate(modes):
        sub = t[t["mode"] == m].set_index("qtype")
        acc = [sub.loc[q].accuracy * 100 for q, _ in qs]
        lo = [sub.loc[q].ci95_low * 100 for q, _ in qs]; hi = [sub.loc[q].ci95_high * 100 for q, _ in qs]
        x = np.arange(4) + (j - 1) * w
        a1.bar(x, acc, w, color=col, label=lab)
        a1.errorbar(x, acc, yerr=[np.subtract(acc, lo), np.subtract(hi, acc)], fmt="none", ecolor="#333333", lw=0.6, capsize=1.5)
        if m != "NOTOOLS":
            g = [sub.loc[q].grounded_accuracy * 100 for q, _ in qs]
            a1.scatter(x, g, marker="_", s=60, color="black", lw=1.2, zorder=4)
    a1.scatter([], [], marker="_", s=60, color="black", lw=1.2, label="correct and grounded")
    a1.set_xticks(range(4), [l for _, l in qs]); a1.set_ylim(0, 108); a1.set_ylabel("Correct answers (%)")
    a1.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.3, handlelength=1.4, columnspacing=1.0)
    a1.set_title("(a) Accuracy on 95 questions per type (95% CI)", loc="left")
    # (b) contradicting citations, Q1-Q3 pooled
    vals = []
    for m, lab, col in modes:
        sub = t[(t["mode"] == m) & t.qtype.isin(["Q1", "Q2", "Q3"])]
        vals.append((lab, sub.answers_with_contradicting_citation.mean() * 100, col))
    a2.barh(range(3)[::-1], [v for _, v, _ in vals], color=[c for *_, c in vals], height=0.55)
    for i, (_, v, _) in zip(range(3)[::-1], vals):
        a2.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=6.5)
    a2.set_yticks(range(3)[::-1], [l for l, *_ in vals]); a2.set_xlim(0, 60)
    a2.set_xlabel("Answers citing an attribute that\ncontradicts the evidence (%, Q1–Q3)")
    a2.set_title("(b) Unsupported citations", loc="left")
    save(fig, out, "fig6_agent")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(res / "eval_runs.jsonl")]
    ann = pd.read_csv(a.data)
    fig_architecture(out); fig_accuracy(res, out); fig_per_class(recs, out)
    fig_attributes(res, ann, out); fig_interventions_stability(res, out); fig_agent(res, out)
    print("figures written to", out)
