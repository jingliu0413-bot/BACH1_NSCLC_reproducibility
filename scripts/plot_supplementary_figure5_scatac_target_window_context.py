#!/usr/bin/env python3
"""Plot descriptive scATAC target-window BACH1-family motif context."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "out" / "bach1_atac_motif_support_consensus_full_tf_5seed_stable_ge60_ucsc_bedmode_10kb"
TABLE_DIR = IN_DIR / "tables"
FIG_DIR = IN_DIR / "figures"
SRC_DIR = IN_DIR / "source_data"

COL = {
    "blue": "#4E79A7",
    "teal": "#59A14F",
    "red": "#E15759",
    "gold": "#F2C14E",
    "grey": "#68707D",
    "dark": "#222222",
    "light": "#E7EAF0",
    "blue_soft": "#DBE8F6",
    "teal_soft": "#DDEFE8",
    "gold_soft": "#F8EBC5",
    "rose_soft": "#F4D9D9",
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#4F5663",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_workflow(ax, summary: dict) -> None:
    ax.axis("off")
    steps = [
        ("Tumour\nscATAC peaks", COL["blue_soft"]),
        ("BACH1-family\nmotifs", COL["teal_soft"]),
        ("TSS +/-10 kb\nlinkage", COL["gold_soft"]),
        ("Target-window\nsummary", COL["rose_soft"]),
    ]
    x0s = np.linspace(0.06, 0.76, len(steps))
    for i, (label, color) in enumerate(steps):
        x0 = float(x0s[i])
        box = FancyBboxPatch(
            (x0, 0.46),
            0.18,
            0.30,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.8,
            edgecolor="#55606E",
            facecolor=color,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x0 + 0.09, 0.61, label, ha="center", va="center", fontsize=8, transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + 0.19, 0.61),
                    (float(x0s[i + 1]) - 0.01, 0.61),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    lw=0.8,
                    color="#55606E",
                    transform=ax.transAxes,
                )
            )
    note = (
        f"{summary['n_atac_samples']} samples; "
        f"{summary['n_motif_positive_sample_specific_peaks_in_target_windows']:,} motif-positive peaks; "
        f"{summary['n_bach1_targets_with_atac_motif_support_in_window']}/{summary['n_targets_with_gencode_coordinates']} mapped targets"
    )
    ax.text(0.06, 0.23, note, ha="left", va="center", fontsize=7, color=COL["grey"], transform=ax.transAxes)


def plot_qc(ax, qc: pd.DataFrame) -> None:
    df = qc.copy()
    x = np.arange(len(df))
    cells = pd.to_numeric(df["n_cells_in_filtered_peak_matrix"], errors="coerce") / 1000
    peak_frag = pd.to_numeric(df["median_peak_region_fragments"], errors="coerce") / 1000
    ax.bar(x, cells, color=COL["blue_soft"], edgecolor="white", linewidth=0.5, width=0.62, label="cells")
    ax.plot(x, peak_frag, color=COL["blue"], marker="o", ms=4, lw=1.2, label="peak fragments")
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], fontsize=7)
    ax.set_ylabel("cells / median peak\nfragments (x10^3)")
    ax.grid(axis="y", color=COL["light"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    despine(ax)


def plot_thresholds(ax, thresholds: pd.DataFrame) -> None:
    df = thresholds.copy()
    x = pd.to_numeric(df["ucsc_motif_score_threshold"], errors="coerce")
    y_window = pd.to_numeric(df["n_target_genes_supported_in_window"], errors="coerce")
    y_prom = pd.to_numeric(df["n_target_genes_supported_promoter_2kb"], errors="coerce")
    ax.plot(x, y_window, color=COL["blue"], marker="o", ms=4, lw=1.3, label="TSS +/-10 kb")
    ax.plot(x, y_prom, color=COL["teal"], marker="o", ms=4, lw=1.3, label="promoter +/-2 kb")
    for xi, yi in zip(x, y_window):
        ax.text(xi, yi + 1.0, f"{int(yi)}/45", ha="center", fontsize=6)
    ax.set_ylim(0, 48)
    ax.set_xticks(list(x))
    ax.set_xlabel("UCSC score threshold")
    ax.set_ylabel("supported mapped targets")
    ax.grid(axis="y", color=COL["light"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    despine(ax)


def plot_motif_models(ax, links: pd.DataFrame) -> None:
    df = (
        links.groupby(["motif_model", "motif_name"])
        .agg(n_peak_gene_links=("peak_id", "size"), n_unique_peaks=("peak_id", "nunique"))
        .reset_index()
        .sort_values("n_peak_gene_links")
    )
    y = np.arange(len(df))
    colors = [COL["teal"], COL["blue"]][: len(df)]
    ax.barh(y, df["n_peak_gene_links"], color=colors, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(df["motif_name"] + "\n" + df["motif_model"], fontsize=7)
    ax.set_xlabel("target-window peak-gene links")
    ax.grid(axis="x", color=COL["light"], linewidth=0.6)
    ax.set_axisbelow(True)
    for yi, value in zip(y, df["n_peak_gene_links"]):
        ax.text(value + max(df["n_peak_gene_links"]) * 0.025, yi, f"{int(value):,}", va="center", fontsize=6)
    despine(ax)
    df.to_csv(SRC_DIR / "supplementary_figure5_panel_d_motif_model_links.csv", index=False)


def plot_candidate_bubble(ax, targets: pd.DataFrame) -> None:
    df = targets[targets["has_atac_bach1_motif_support_in_window"].astype(str).str.lower() == "true"].copy()
    numeric = [
        "n_atac_motif_peak_links_in_window",
        "n_samples_with_motif_peak_in_window",
        "closest_motif_peak_distance_to_tss",
        "max_ucsc_motif_score",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(
        ["n_atac_motif_peak_links_in_window", "n_samples_with_motif_peak_in_window", "max_ucsc_motif_score"],
        ascending=False,
    ).head(16)
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    dist_kb = df["closest_motif_peak_distance_to_tss"].abs().fillna(10000) / 1000
    sizes = 26 + df["n_samples_with_motif_peak_in_window"].fillna(0) * 22
    sc = ax.scatter(
        df["n_atac_motif_peak_links_in_window"],
        y,
        s=sizes,
        c=dist_kb,
        cmap="viridis_r",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=6.5)
    ax.set_xlabel("target-window motif-peak links")
    ax.grid(axis="x", color=COL["light"], linewidth=0.6)
    ax.set_axisbelow(True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("closest peak\n(kb to TSS)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, width=0.4)
    cb.outline.set_linewidth(0.4)
    despine(ax)
    df.to_csv(SRC_DIR / "supplementary_figure5_panel_e_candidate_target_support.csv", index=False)


def plot_example_loci(ax, links: pd.DataFrame, targets: pd.DataFrame) -> None:
    candidates = targets[targets["has_atac_bach1_motif_support_in_window"].astype(str).str.lower() == "true"].copy()
    candidates["n_atac_motif_peak_links_in_window"] = pd.to_numeric(
        candidates["n_atac_motif_peak_links_in_window"], errors="coerce"
    )
    chosen = list(candidates.sort_values("n_atac_motif_peak_links_in_window", ascending=False)["gene"].head(6))
    sub = links[links["gene"].isin(chosen)].copy()
    sub["abs_distance"] = pd.to_numeric(sub["distance_to_tss"], errors="coerce").abs()
    sub = sub.sort_values(["gene", "abs_distance", "sample"]).groupby(["gene", "sample", "motif_name"], as_index=False).head(2)
    order = chosen[::-1]
    y_map = {gene: i for i, gene in enumerate(order)}
    colors = {"BACH1": COL["blue"], "Bach1::Mafk": COL["teal"]}
    for gene, yi in y_map.items():
        ax.hlines(yi, -10, 10, color=COL["light"], lw=1.0)
        ax.plot(0, yi, marker="|", color=COL["dark"], ms=9, mew=1.0)
    for row in sub.itertuples(index=False):
        x = float(row.distance_to_tss) / 1000
        yi = y_map[str(row.gene)]
        color = colors.get(str(row.motif_name), COL["grey"])
        ax.vlines(x, yi - 0.26, yi + 0.26, color=color, lw=0.65, alpha=0.7)
        ax.scatter(
            x,
            yi,
            s=10 + float(row.frac_cells_accessible) * 130,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
    ax.axvline(0, color=COL["grey"], lw=0.8, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.5)
    ax.set_xlim(-10.5, 10.5)
    ax.set_xlabel("distance from TSS (kb)")
    ax.grid(axis="x", color=COL["light"], linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", label=label, markersize=5)
        for label, color in colors.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="upper right")
    despine(ax)
    sub.to_csv(SRC_DIR / "supplementary_figure5_panel_f_example_loci.csv", index=False)


def main() -> None:
    setup_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    summary = json.loads((IN_DIR / "bach1_atac_motif_support_ucsc_targeted_summary.json").read_text(encoding="utf-8"))
    qc = pd.read_csv(TABLE_DIR / "gse274934_atac_sample_qc_summary.csv")
    thresholds = pd.read_csv(TABLE_DIR / "ucsc_jaspar2026_score_threshold_sensitivity.csv")
    targets = pd.read_csv(TABLE_DIR / "pyscenic_bach1_targets_with_ucsc_jaspar2026_atac_support.csv")
    links = pd.read_csv(TABLE_DIR / "bach1_regulon_targets_atac_motif_peak_links_target_windows.csv.gz")

    qc.to_csv(SRC_DIR / "supplementary_figure5_panel_b_scatac_sample_qc.csv", index=False)
    thresholds.to_csv(SRC_DIR / "supplementary_figure5_panel_c_ucsc_score_thresholds.csv", index=False)

    fig = plt.figure(figsize=(10.8, 7.6))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.70, 1.0, 1.28], width_ratios=[1.0, 1.0, 1.2], hspace=0.52, wspace=0.48)

    ax_a = fig.add_subplot(gs[0, :])
    draw_workflow(ax_a, summary)
    panel_label(ax_a, "a", x=0.0, y=1.02)

    ax_b = fig.add_subplot(gs[1, 0])
    plot_qc(ax_b, qc)
    panel_label(ax_b, "b", x=0.02, y=0.98)

    ax_c = fig.add_subplot(gs[1, 1])
    plot_thresholds(ax_c, thresholds)
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 2])
    plot_motif_models(ax_d, links)
    panel_label(ax_d, "d", x=-0.06)

    ax_e = fig.add_subplot(gs[2, :2])
    plot_candidate_bubble(ax_e, targets)
    panel_label(ax_e, "e", x=-0.035)

    ax_f = fig.add_subplot(gs[2, 2])
    plot_example_loci(ax_f, links, targets)
    panel_label(ax_f, "f", x=-0.06)

    for ext in ("pdf", "png", "svg"):
        path = FIG_DIR / f"supplementary_figure5_scatac_target_window_context.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(FIG_DIR / "supplementary_figure5_scatac_target_window_context.pdf")


if __name__ == "__main__":
    main()
