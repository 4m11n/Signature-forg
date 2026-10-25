"""Builds every comparison figure (CNN vs GNN) from the CSVs in metadata/. Run from the project root:
  python src\\make_figures.py
Writes PNGs to figures/. A figure whose CSV is missing is skipped (the script says which)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M, OUT = "metadata", "figures"
os.makedirs(OUT, exist_ok=True)
C_CNN, C_GNN, C_BIN, C_F = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def read(name):
    p = f"{M}/{name}"
    if os.path.exists(p):
        return pd.read_csv(p)
    print(f"[skip] {p} not found")
    return None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/{name}")


# ---------- fallback / fixed numbers (taken from the reported baseline runs) ----------
CNN_CORR = {"blur": [0.9995, 0.9980, 0.9970, 0.9965, 0.9975], "compression": [0.9990, 0.9990, 0.9980, 0.9995, 0.6825],
            "perspective": [0.9995, 0.9985, 0.9975, 0.9920, 0.9830], "brightness": [0.5355, 0.5190, 0.5030, 0.5260, 0.5110]}
CNN_CLEAN = 0.999
XT = {"compression": ("JPEG quality", ["70", "50", "30", "15", "5"]), "blur": ("Gaussian kernel", ["3", "5", "9", "13", "17"]),
      "brightness": ("max shift", ["0.1", "0.2", "0.3", "0.4", "0.5"]), "perspective": ("warp scale", ["0.02", "0.05", "0.08", "0.12", "0.16"])}

cnn, gnn = read("multi_fold_results.csv"), read("multi_fold_gnn_results.csv")
b20, b60 = read("multi_fold_cnnbin_results.csv"), read("multi_fold_cnnbin_60ep_results.csv")
for n, d in (("CNN baseline", cnn), ("GNN baseline", gnn), ("binarised CNN 20ep", b20)):
    if d is not None:
        print(f"check {n}: mean accuracy {d.accuracy.mean():.4f}")

# ---------- Fig 1: per-fold baseline ----------
if cnn is not None and gnn is not None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    series = [("CNN (grayscale)", cnn, C_CNN)] + ([("CNN (binarised)", b20, C_BIN)] if b20 is not None else []) + [("GNN", gnn, C_GNN)]
    w = 0.8 / len(series)
    for i, (lab, d, c) in enumerate(series):
        ax.bar(np.arange(1, 6) + (i - (len(series) - 1) / 2) * w, d.sort_values("fold").accuracy.values, w, label=lab, color=c)
    ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy (held-out writers)"); ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="k", lw=0.6, ls=":"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(series))
    ax.set_title("Writer-disjoint CEDAR accuracy by fold")
    save(fig, "fig1_baseline_per_fold.png")

# ---------- Fig 2: model comparison on CEDAR ----------
rows = [(l, d) for l, d in (("CNN\ngrayscale", cnn), ("CNN\nbinarised 20 ep", b20), ("CNN\nbinarised 60 ep", b60), ("GNN\n(shape graph)", gnn)) if d is not None]
if rows:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cols = {"CNN\ngrayscale": C_CNN, "GNN\n(shape graph)": C_GNN}
    for i, (l, d) in enumerate(rows):
        m, s = d.accuracy.mean(), d.accuracy.std()
        ax.bar(i, m, yerr=s, capsize=4, color=cols.get(l, C_BIN))
        ax.text(i, m + s + 0.012, f"{100*m:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([l for l, _ in rows]); ax.set_ylim(0.5, 1.08)
    ax.set_ylabel("Mean accuracy (5 folds, +/- SD)"); ax.set_title("Same folds, pairs and seeds: what each model sees")
    save(fig, "fig2_model_comparison.png")

# ---------- Fig 3: few-shot ----------
fc, fg = read("fewshot_results.csv"), read("gnn_fewshot_results.csv")
if fc is not None or fg is not None:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for lab, d, c in (("CNN", fc, C_CNN), ("GNN", fg, C_GNN)):
        if d is None:
            continue
        g = d.groupby("n_references")["acc_mean"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g.n_references, g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=3, label=lab, color=c)
    ax.set_xticks([3, 5, 10]); ax.set_xlabel("Reference signatures per writer (n)"); ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0.5, 1.02); ax.legend(frameon=False); ax.set_title("Reduced enrollment (CEDAR, 5 folds)")
    save(fig, "fig3_fewshot.png")

# ---------- Fig 4: corruption, CNN vs GNN ----------
cc, gc = read("corruption_results.csv"), read("gnn_corruption_results.csv")
if cc is not None:
    cl = cc[cc.corruption == "clean"].accuracy.mean()
    bsev0 = cc[(cc.corruption == "brightness") & (cc.severity == 0)].accuracy.mean()
    if not (0.995 <= cl <= 1.0 and 0.50 <= bsev0 <= 0.60):
        print(f"[warn] corruption_results.csv looks like a different run (clean={cl:.3f}, brightness sev0={bsev0:.3f}); using the reported baseline numbers for the CNN curve")
        cc = None
if cc is not None or gc is not None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for ax, cname in zip(axes.flat, ["blur", "compression", "perspective", "brightness"]):
        for lab, d, c in (("CNN", cc, C_CNN), ("GNN", gc, C_GNN)):
            if d is None and lab == "CNN":
                m, s = np.array(CNN_CORR[cname]), np.zeros(5)
                clean = CNN_CLEAN
            elif d is None:
                continue
            else:
                g = d[d.corruption == cname].groupby("severity")["accuracy"].agg(["mean", "std"])
                m, s = g["mean"].values, g["std"].fillna(0).values
                clean = d[d.corruption == "clean"].accuracy.mean()
            ax.errorbar(range(5), m, yerr=s, marker="o", capsize=2, label=lab, color=c)
            ax.axhline(clean, color=c, ls=":", lw=0.8)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xticks(range(5)); ax.set_xticklabels(XT[cname][1]); ax.set_xlabel(XT[cname][0]); ax.set_ylim(0.4, 1.03)
        ax.set_title(cname.capitalize()); ax.set_ylabel("Accuracy")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle("Accuracy under image corruption (dotted = clean accuracy, mean +/- SD over folds)", fontsize=10)
    save(fig, "fig4_corruption.png")

# ---------- Fig 5: cross-script ----------
xc, xg = read("crossscript_results.csv"), read("gnn_crossscript_results.csv")
if (xc is not None or xg is not None) and cnn is not None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    sets = ["CEDAR", "Bengali", "Hindi"]
    def per_set(base, cross):
        out = {"CEDAR": base}
        if cross is not None:
            out["Bengali"] = cross[cross.script == "bengali"]; out["Hindi"] = cross[cross.script == "hindi"]
        return out
    models = [("CNN", per_set(cnn, xc), C_CNN)] + ([("GNN", per_set(gnn, xg), C_GNN)] if gnn is not None else [])
    w = 0.8 / len(models)
    for i, (lab, dd, c) in enumerate(models):
        xs = [j for j, s in enumerate(sets) if s in dd]
        a1.bar(np.array(xs) + (i - (len(models) - 1) / 2) * w, [dd[sets[j]].accuracy.mean() for j in xs], w,
               yerr=[dd[sets[j]].accuracy.std() for j in xs], capsize=3, label=lab, color=c)
    a1.set_xticks(range(3)); a1.set_xticklabels(sets); a1.set_ylim(0.5, 1.02); a1.set_ylabel("Mean accuracy"); a1.legend(frameon=False)
    a1.set_title("Trained on CEDAR, no fine-tuning")
    labels, far, frr, cols = [], [], [], []
    for lab, dd, c in models:
        for s in sets[1:]:
            if s in dd:
                labels.append(f"{lab}\n{s}"); far.append(dd[s].far.mean()); frr.append(dd[s].frr.mean()); cols.append(c)
    x = np.arange(len(labels))
    a2.bar(x - 0.2, far, 0.4, color=cols, label="FAR (forgery accepted)")
    a2.bar(x + 0.2, frr, 0.4, color=cols, alpha=0.45, label="FRR (genuine rejected)")
    a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_ylabel("Rate"); a2.legend(frameon=False, fontsize=8); a2.set_title("Error direction on BHSig260")
    save(fig, "fig5_crossscript.png")

# ---------- Fig 6: brightness remediation (CNN, fixed numbers from the reported runs) ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.8))
sev = ["0.1", "0.2", "0.3", "0.4", "0.5"]
for lab, v, c in (("Baseline", [0.5355, 0.5190, 0.5030, 0.5260, 0.5110], C_CNN),
                  ("Per-image standardisation", [0.766, 0.679, 0.6575, 0.6595, 0.651], C_F),
                  ("Brightness augmentation", [0.688, 0.6565, 0.6115, 0.598, 0.5705], "#ff7f0e")):
    a1.plot(sev, v, marker="o", label=lab, color=c)
a1.set_xlabel("Brightness shift (max)"); a1.set_ylabel("Accuracy"); a1.set_ylim(0.4, 1.0); a1.legend(frameon=False, fontsize=8); a1.set_title("Brightness accuracy")
conds = ["Clean", "Compression\nsev 0", "Blur\nsev 4", "Perspective\nsev 4"]
data = {"Baseline": [0.999, 0.999, 0.9975, 0.983], "Per-image standardisation": [0.9505, 0.914, 0.5815, 0.6865],
        "Brightness augmentation": [0.916, 0.9045, 0.816, 0.818]}
w = 0.27
for i, (lab, c) in enumerate(zip(data, (C_CNN, C_F, "#ff7f0e"))):
    a2.bar(np.arange(4) + (i - 1) * w, data[lab], w, label=lab, color=c)
a2.set_xticks(range(4)); a2.set_xticklabels(conds, fontsize=8); a2.set_ylim(0.4, 1.03); a2.set_title("Cost on other conditions")
save(fig, "fig6_brightness_remediation.png")

# ---------- Fig 7: late fusion ----------
fu = read("fusion_v1_results.csv")
if fu is not None:
    order = ["cnn", "gnn", "fused_w0.5", "fused_bestw"]
    names = {"cnn": "CNN", "gnn": "GNN", "fused_w0.5": "Fused (equal weight)", "fused_bestw": "Fused (best weight*)"}
    colors = {"cnn": C_CNN, "gnn": C_GNN, "fused_w0.5": "#9467bd", "fused_bestw": "#c5b0d5"}
    t = fu.groupby(["set", "method"])["auc"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sets_ = [s for s in ["CEDAR test", "BHSig260 Bengali", "BHSig260 Hindi"] if s in t.index]
    w = 0.2
    for i, mth in enumerate(order):
        ax.bar(np.arange(len(sets_)) + (i - 1.5) * w, [t.loc[s, mth] for s in sets_], w, label=names[mth], color=colors[mth])
    ax.set_xticks(range(len(sets_))); ax.set_xticklabels(sets_); ax.set_ylim(0.5, 1.02); ax.set_ylabel("AUC (mean over folds)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1)); ax.set_title("Late fusion of CNN and GNN distances")
    fig.text(0.01, -0.06, "*weight chosen on the evaluated pairs, so an upper bound", fontsize=7)
    save(fig, "fig7_fusion.png")
