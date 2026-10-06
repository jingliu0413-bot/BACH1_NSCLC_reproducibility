"""Build the streamlined Figure 2 used in the final submission package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/manuscript_submission_20260914"
FIG_DIR = OUT / "figures"
PRIMARY = ROOT / "out/external_bach1_activity_primary/tables"
ROBUST = ROOT / "out/external_bach1_activity_robustness/tables"

DATASET_COLORS = {"GSE131907": "#4C78A8", "GSE274934": "#F58518"}
SCORE_ORDER = [
    "DOROTHEA_BACH1_ABC_TF_ACTIVITY__mean_z",
    "DOROTHEA_BACH1_ABC_TF_ACTIVITY_NO_HYPOXIA_OVERLAP__mean_z",
    "LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY__mean_z",
    "COLLECTRI_BACH1_TF_ACTIVITY__mean_z",
    "KLENJA2025_BACH1_INVERSE_ACTIVITY__mean_z",
]
SCORE_LABELS = {
    "DOROTHEA_BACH1_ABC_TF_ACTIVITY__mean_z": "DoRothEA mean-z",
    "DOROTHEA_BACH1_ABC_TF_ACTIVITY_NO_HYPOXIA_OVERLAP__mean_z": "DoRothEA mean-z\nshared genes excluded",
    "LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY__mean_z": "lung effector mean-z",
    "COLLECTRI_BACH1_TF_ACTIVITY__mean_z": "CollecTRI mean-z",
    "KLENJA2025_BACH1_INVERSE_ACTIVITY__mean_z": "Klenja inverse mean-z",
}


def superscript_int(n: int) -> str:
    table = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(n).translate(table)


def format_p(p: float, prefix: str = "P") -> str:
    if not np.isfinite(p):
        return f"{prefix}=NA"
    if p < 0.001:
        exp = int(np.floor(np.log10(p)))
        mant = p / (10**exp)
        return rf"${prefix}={mant:.1f}\times10^{{{exp}}}$"
    return rf"${prefix}={p:.3f}$"


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")


def save_all(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    patient = pd.read_csv(ROBUST / "external_bach1_activity_patient_scores_with_deoverlap.csv")
    deltas = pd.read_csv(ROBUST / "within_patient_bach1_detection_deltas_with_deoverlap.csv")
    tests = pd.read_csv(ROBUST / "figure2c_within_patient_score_comparison_holm_family.csv").set_index("score")
    cor = pd.read_csv(ROBUST / "dorothea_hypoxia_deoverlap_patient_correlations.csv")
    lopo = pd.read_csv(ROBUST / "dorothea_hypoxia_deoverlap_lopo_summary.csv")
    null_long = pd.read_csv(ROBUST / "matched_random_gene_set_hypoxia_correlations_long.csv")
    null_summary = pd.read_csv(ROBUST / "matched_random_gene_set_hypoxia_correlation_summary.csv")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(13.4, 7.4), constrained_layout=True)
    # Give the central covariance and matched-null panels more visual weight;
    # keep the score-selection comparison compact and the LOPO panel readable.
    gs = fig.add_gridspec(2, 12, height_ratios=[0.88, 1.12], wspace=0.75, hspace=0.50)

    ax = fig.add_subplot(gs[0, :5])
    p = patient.sort_values(["dataset", "BACH1_detected_fraction"]).reset_index(drop=True)
    x = np.arange(len(p))
    colors = [DATASET_COLORS.get(d, "#777777") for d in p["dataset"]]
    ax.bar(x, p["BACH1_detected_fraction"], color=colors, width=0.72)
    def compact_n(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(int(n))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{pid}\n(n={compact_n(n)})" for pid, n in zip(p["patient"], p["n_cells"])],
        rotation=45,
        ha="right",
        fontsize=6.0,
        linespacing=1.0,
    )
    ax.set_ylabel("BACH1-detected fraction")
    ax.set_xlabel("Patients ordered within dataset by BACH1-detected fraction", fontsize=7)
    ax.set_ylim(0, max(0.56, float(p["BACH1_detected_fraction"].max()) * 1.18))
    ax.set_title("Patient-level BACH1 transcript detection", loc="left", fontsize=8.5, pad=2)
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=DATASET_COLORS[k], markeredgecolor="none", label=k, markersize=7)
        for k in ["GSE131907", "GSE274934"]
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 5:])
    y_positions = np.arange(len(SCORE_ORDER))[::-1]
    jitter = {"GSE131907": -0.07, "GSE274934": 0.07}
    for yi, score in zip(y_positions, SCORE_ORDER):
        sub = deltas.loc[deltas["score"].eq(score)].dropna(subset=["delta_detected_minus_undetected"]).copy()
        for dataset, ds in sub.groupby("dataset"):
            ax.scatter(
                ds["delta_detected_minus_undetected"],
                np.full(len(ds), yi) + jitter.get(dataset, 0),
                s=18,
                color=DATASET_COLORS.get(dataset, "#777777"),
                alpha=0.82,
                edgecolor="none",
            )
        med = float(tests.loc[score, "median_delta_detected_minus_undetected"])
        ax.scatter([med], [yi], s=22, color="black", zorder=5)
        ax.text(
            1.015,
            yi,
            f"Δmed={med:.3f}; Holm {format_p(float(tests.loc[score, 'wilcoxon_holm_p']))}",
            ha="left",
            va="center",
            fontsize=6.5,
        )
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([SCORE_LABELS[s] for s in SCORE_ORDER])
    ax.set_xlabel("Patient-level delta: BACH1-detected - undetected")
    ax.set_xlim(-0.25, 1.30)
    ax.set_title("External BACH1 score comparison", loc="left", fontsize=8.5, pad=2)
    label_scores = SCORE_ORDER
    for tick, score in zip(ax.get_yticklabels(), label_scores):
        tick.set_color("#2F5C84" if score.startswith("DOROTHEA") else "#73777D")
        tick.set_fontweight("bold" if score.startswith("DOROTHEA") else "normal")
    ax.text(
        1.0,
        1.015,
        "Colours indicate dataset; black dots show medians across patients. n=15 except Klenja (n=14).",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#666666",
    )
    panel_label(ax, "b", x=-0.06)

    ax = fig.add_subplot(gs[1, :5])
    xcol = "DOROTHEA_BACH1_ABC_TF_ACTIVITY_NO_HYPOXIA_OVERLAP__mean_z"
    ycol = "HALLMARK_HYPOXIA_NO_DOROTHEA_OVERLAP__mean_z"
    for dataset, ds in patient.groupby("dataset"):
        ax.scatter(ds[xcol], ds[ycol], s=28, color=DATASET_COLORS.get(dataset, "#777777"), label=dataset, alpha=0.85, edgecolor="white", linewidth=0.4)
    fit = patient[[xcol, ycol]].dropna()
    if len(fit) >= 3:
        m, b = np.polyfit(fit[xcol], fit[ycol], 1)
        xs = np.linspace(float(fit[xcol].min()), float(fit[xcol].max()), 100)
        ax.plot(xs, m * xs + b, color="#333333", lw=0.8, alpha=0.8)
    row = cor.loc[cor["comparison"].eq("both_signatures_without_shared")].iloc[0]
    ax.text(
        0.04,
        0.96,
        f"ρ={row.spearman_rho:.3f}\nHolm-adjusted {format_p(float(row.spearman_holm_p))}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.set_xlabel("DoRothEA BACH1 mean-z")
    ax.set_ylabel("Hallmark hypoxia mean-z")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.98))
    ax.set_title("Patient-level BACH1–hypoxia covariance", loc="left", fontsize=8.5, pad=2)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 5:8])
    order = ["original", "both_signatures_without_shared"]
    labels = ["Original", "De-overlapped"]
    rows = lopo.set_index("comparison").loc[order].reset_index()
    full_rho = cor.set_index("comparison")["spearman_rho"]
    rows["full_rho"] = rows["comparison"].map(full_rho)
    y = np.arange(len(rows))
    ax.errorbar(
        rows["full_rho"],
        y,
        xerr=[rows["full_rho"] - rows["min_rho"], rows["max_rho"] - rows["full_rho"]],
        fmt="o",
        color="#333333",
        ecolor="#777777",
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Spearman ρ\npoint = full data; line = LOPO range", fontsize=6.6, linespacing=1.1)
    ax.set_xlim(0.82, 1.01)
    ax.invert_yaxis()
    for yi, label, row in zip(y, labels, rows.itertuples(index=False)):
        ax.text(
            0.997,
            yi,
            f"{label}: ρ={row.full_rho:.3f};\nLOPO range: {row.min_rho:.3f}–{row.max_rho:.3f}",
            va="center",
            ha="right",
            fontsize=6.2,
            color="#4A4A4A",
        )
    ax.set_title("Leave-one-patient-out robustness", loc="left", fontsize=8.5, pad=2)
    panel_label(ax, "d", x=-0.08)

    ax = fig.add_subplot(gs[1, 8:])
    ns = null_summary.loc[null_summary["comparison"].eq("deoverlapped_hypoxia")].iloc[0]
    nl = null_long.loc[null_long["comparison"].eq("deoverlapped_hypoxia")]
    ax.hist(nl["rho_random_vs_outcome"], bins=38, color="#BFCAD6", edgecolor="white", linewidth=0.4)
    observed_rho = float(ns.observed_rho)
    null_mean_rho = float(ns.null_mean_rho)
    ax.axvline(observed_rho, color="#D62728", lw=1.4)
    ax.axvline(null_mean_rho, color="#4C78A8", lw=1.0, ls="--")
    ax.text(observed_rho, 0.94, "Observed", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.2, color="#D62728")
    ax.text(null_mean_rho, 0.94, "Null mean", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.2, color="#4C78A8")
    ax.text(
        0.04,
        0.94,
        f"observed ρ={observed_rho:.2f}\nnull mean ρ={null_mean_rho:.2f}\nempirical {format_p(float(ns.empirical_p_greater_equal_observed))}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#333333",
    )
    ax.set_xlabel("Matched random score vs de-overlapped hypoxia ρ")
    ax.set_ylabel("Permutation count")
    ax.set_title("Matched-gene null benchmark", loc="left", fontsize=8.5, pad=2)
    panel_label(ax, "e")

    for axis in fig.axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    save_all(fig, "Figure2_BACH1_activity_hypoxia_robustness")


if __name__ == "__main__":
    main()
