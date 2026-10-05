"""Create the focused Figure 4 and the cohort-split supplementary panel."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "final_submission_figures"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / "source_data" / "manuscript_submission_20260911"
TF_TABLE = SOURCE / "dorothea_all_tf_hypoxia_contextual_benchmark.csv"
COHORT_TABLE = SOURCE / "scrna_cohort_split_dorothea_hypoxia_correlations.csv"
TCGA_SCORE_TABLE = SOURCE / "tcga_luad_lusc_bach1_hypoxia_patient_scores.csv"
TCGA_CORR_TABLE = SOURCE / "tcga_bach1_hypoxia_correlations.csv"
TCGA_MODEL_TABLE = SOURCE / "tcga_bach1_hypoxia_adjusted_models.csv"

DOR_NO = "DOROTHEA_BACH1_ABC_TF_ACTIVITY_NO_HYPOXIA_OVERLAP__mean_z"
HYP_NO = "HALLMARK_HYPOXIA_NO_DOROTHEA_OVERLAP__mean_z"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def format_p(value: float) -> str:
    if value < 0.001:
        exponent = math.floor(math.log10(value))
        mantissa = value / (10**exponent)
        return rf"$P$={mantissa:.1f}$\times$10$^{{{exponent}}}$"
    return rf"$P$={value:.3f}"


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(TF_TABLE),
        pd.read_csv(COHORT_TABLE),
        pd.read_csv(TCGA_SCORE_TABLE),
        pd.read_csv(TCGA_CORR_TABLE),
        pd.read_csv(TCGA_MODEL_TABLE),
    )


def plot_main() -> None:
    tf, cohort, scores, correlations, models = load_tables()
    top = tf.sort_values("rank_by_rho_desc").head(10)["tf"].tolist()
    comparators = ["JUN", "ATF4", "NFE2L2", "MYC"]
    show = tf[tf["tf"].isin(dict.fromkeys(top + comparators))].sort_values("rank_by_rho_desc").copy()

    style()
    fig = plt.figure(figsize=(10.5, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.03, 1.0], width_ratios=[1.08, 0.92])

    ax = fig.add_subplot(grid[0, :])
    y = np.arange(len(show))
    colors = ["#C44E52" if name == "BACH1" else "#4C72B0" for name in show["tf"]]
    ax.barh(y, show["spearman_rho_with_hypoxia"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(show["tf"], fontstyle="italic")
    ax.set_xlabel(r"TF-specific de-overlapped Spearman $\rho$ with hypoxia")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    for i, row in enumerate(show.itertuples(index=False)):
        ax.text(float(row.spearman_rho_with_hypoxia) + 0.01, i, f"rank {int(row.rank_by_rho_desc)}", va="center", fontsize=7, fontweight="bold" if row.tf == "BACH1" else "normal")
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "a")

    ax = fig.add_subplot(grid[1, 0])
    for cohort_name, color in [("LUAD", "#4C72B0"), ("LUSC", "#C44E52")]:
        subset = scores[scores["cohort"].eq(cohort_name)]
        ax.scatter(subset[DOR_NO], subset[HYP_NO], s=10, alpha=0.38, color=color, linewidths=0, label=cohort_name)
    x = scores[DOR_NO].to_numpy(float)
    yv = scores[HYP_NO].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(yv)
    slope, intercept = np.polyfit(x[ok], yv[ok], 1)
    xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
    ax.plot(xs, slope * xs + intercept, color="#222222", lw=1.1)
    row = correlations[(correlations["cohort"].eq("combined")) & (correlations["comparison"].eq("both_no_shared"))].iloc[0]
    ax.text(0.04, 0.96, rf"$\rho$={row.spearman_rho:.3f}" + "\n" + f"Holm {format_p(row.spearman_holm_p)}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlabel("TCGA de-overlapped DoRothEA BACH1 score")
    ax.set_ylabel("TCGA de-overlapped Hallmark hypoxia score")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "b")

    ax = fig.add_subplot(grid[1, 1])
    model_order = [
        ("combined_deoverlap_cohort_stage_smoking", "Combined"),
        ("LUAD_deoverlap_purity_stage_smoking", "LUAD"),
        ("LUSC_deoverlap_stage_smoking", "LUSC"),
    ]
    rows = []
    for model, label in model_order:
        row = models[models["model"].eq(model)].iloc[0]
        rows.append((label, row))
    y = np.arange(len(rows))
    for i, (label, row) in enumerate(rows):
        ax.errorbar(row.beta, i, xerr=[[row.beta - row.ci_low], [row.ci_high - row.beta]], fmt="o", color="#333333", capsize=2, lw=1.1)
        ax.text(row.ci_high + 0.015, i, format_p(row.p_value), va="center", fontsize=7)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_xlabel("Adjusted standardized beta")
    ax.set_xlim(0, 0.48)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "c")

    fig.savefig(OUT / "Figure4_pan_tf_tcga_focused.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure4_pan_tf_tcga_focused.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_supplement() -> None:
    _, cohort, _, _, _ = load_tables()
    style()
    order = {"All patients": 0, "GSE131907": 1, "GSE274934": 2}
    cohort = cohort.sort_values("dataset", key=lambda s: s.map(order))
    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    y = np.arange(len(cohort))
    ax.errorbar(
        cohort["rho"],
        y,
        xerr=[cohort["rho"] - cohort["ci_low"], cohort["ci_high"] - cohort["rho"]],
        fmt="o",
        color="#333333",
        ecolor="#333333",
        capsize=2,
        lw=1.1,
    )
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.dataset} (n={int(row.n_patients)})" for row in cohort.itertuples()])
    ax.set_xlabel(r"Patient-level Spearman $\rho$")
    ax.set_title("Supplementary Figure 4. scRNA-seq cohort-stratified association", fontsize=10, fontweight="bold")
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    for i, row in enumerate(cohort.itertuples()):
        ax.text(0.03, i, format_p(row.p_value), transform=ax.get_yaxis_transform(), va="center", fontsize=7)
    fig.savefig(OUT / "Supplementary_Figure4_scrna_cohort_split.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Supplementary_Figure4_scrna_cohort_split.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_main()
    plot_cohort_supplement()
