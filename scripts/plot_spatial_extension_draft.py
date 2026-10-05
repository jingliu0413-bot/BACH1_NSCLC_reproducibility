"""Create a restrained draft figure for the GSE292299 spatial extension."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "spatial_state_context_analysis"
SPOT_TABLE = OUT / "GSE292299_spatial_spot_scores.csv.gz"
SPATIAL_DIR = ROOT / "data" / "spatial_candidates" / "GSE292299_spatial_extracted"
RAW = OUT / "GSE292299_spatial_moran_permutation_summary.csv"
QC = OUT / "GSE292299_spatial_residualized_sensitivity.csv"
STATE = OUT / "GSE292299_spatial_state_pc_block_null.csv"
BLOCK_SENSITIVITY = OUT / "GSE292299_spatial_block_size_sensitivity.csv"

SAMPLE_COLORS = {
    "NSCLC_P5": "#0072B2",
    "NSCLC_P8": "#D55E00",
    "NSCLC_P12": "#009E73",
    "NSCLC_P15": "#CC79A7",
}


def find_spatial_files(sample: str) -> dict[str, Path]:
    root = SPATIAL_DIR / sample
    positions = next(root.rglob("tissue_positions.csv"))
    scalefactors = next(root.rglob("scalefactors_json.json"))
    hires = next(root.rglob("tissue_hires_image.png"))
    lowres = next(root.rglob("tissue_lowres_image.png"))
    return {"positions": positions, "scalefactors": scalefactors, "hires": hires, "lowres": lowres}


def add_letter(ax, letter: str) -> None:
    ax.text(-0.08, 1.10, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def representative_maps(fig, axes, spots: pd.DataFrame) -> None:
    p12 = spots[spots["sample"].eq("NSCLC_P12")].copy()
    files = find_spatial_files("NSCLC_P12")
    image = np.asarray(Image.open(files["lowres"]).convert("RGB"))
    with open(files["scalefactors"], encoding="utf-8") as handle:
        scale = json.load(handle)["tissue_lowres_scalef"]
    x = p12["pxl_col_in_fullres"] * scale
    y = p12["pxl_row_in_fullres"] * scale
    for ax, column, cmap, title in zip(
        axes,
        ["DOROTHEA_BACH1", "HALLMARK_HYPOXIA", "Stress_AP1"],
        ["viridis", "magma", "plasma"],
        ["DoRothEA BACH1", "Hypoxia", "Stress-associated programme"],
    ):
        ax.imshow(image, origin="upper")
        values = p12[column].to_numpy(float)
        lo, hi = np.quantile(values, [0.02, 0.98])
        norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)
        ax.scatter(x, y, c=values, s=2.8, alpha=0.76, cmap=cmap, norm=norm, linewidths=0)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_title(title, fontsize=7.8, pad=4)
        ax.axis("off")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(values)
        colorbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, shrink=0.72)
        colorbar.ax.tick_params(labelsize=5.8, length=2)
        colorbar.set_label("Score", fontsize=5.8, labelpad=1)


def paired_plot(ax, table: pd.DataFrame, raw_col: str, state_col: str, ylabel: str, title: str) -> None:
    samples = ["NSCLC_P5", "NSCLC_P8", "NSCLC_P12", "NSCLC_P15"]
    x_raw, x_state = 0.0, 1.0
    for sample in samples:
        row = table[table["sample"].eq(sample)].iloc[0]
        color = SAMPLE_COLORS[sample]
        ax.plot([x_raw, x_state], [row[raw_col], row[state_col]], color=color, lw=1.1, alpha=0.75)
        ax.scatter([x_raw, x_state], [row[raw_col], row[state_col]], color=color, s=26, zorder=3)
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set_xticks([x_raw, x_state], ["Raw", "State-PC\nadjusted"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.25, 1.25)
    ax.spines[["top", "right"]].set_visible(False)


def raw_program_plot(ax, raw: pd.DataFrame) -> None:
    samples = ["NSCLC_P5", "NSCLC_P8", "NSCLC_P12", "NSCLC_P15"]
    comparisons = ["DoRothEA_vs_hypoxia", "Stress_AP1_vs_hypoxia"]
    labels = ["BACH1 score–hypoxia", "Stress programme–hypoxia"]
    for sample in samples:
        values = []
        for comparison in comparisons:
            row = raw[(raw["sample"].eq(sample)) & (raw["comparison"].eq(comparison))].iloc[0]
            values.append(row["spearman_rho_descriptive"])
        color = SAMPLE_COLORS[sample]
        ax.scatter([0, 1], values, color=color, s=28, zorder=3, label=sample.replace("NSCLC_", ""))
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Raw Spearman ρ")
    ax.set_title("Positive raw associations across all spatial sections", fontsize=8.6, fontweight="semibold")
    ax.set_xlim(-0.25, 1.25)
    ax.spines[["top", "right"]].set_visible(False)


def observed_vs_block_null_plot(ax, raw: pd.DataFrame, state: pd.DataFrame) -> None:
    samples = ["NSCLC_P5", "NSCLC_P8", "NSCLC_P12", "NSCLC_P15"]
    for sample in samples:
        raw_row = raw[raw["sample"].eq(sample)].iloc[0]
        null_row = state[state["sample"].eq(sample)].iloc[0]
        color = SAMPLE_COLORS[sample]
        observed = raw_row["bivariate_moran_I"]
        null_mean = null_row["block_null_mean"]
        null_low = null_row.get("block_null_q025", null_mean - 1.96 * null_row["block_null_sd"])
        null_high = null_row.get("block_null_q975", null_mean + 1.96 * null_row["block_null_sd"])
        values = [observed, null_mean]
        ax.plot([0, 1], values, color=color, lw=1.1, alpha=0.75)
        ax.scatter([0], [observed], color=color, s=28, zorder=3)
        ax.errorbar(
            [1],
            [null_mean],
            yerr=[[null_mean - null_low], [null_high - null_mean]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            markersize=5,
            zorder=3,
        )
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set_xticks([0, 1], ["Observed", "Block-null\nmean (20×20)"])
    ax.set_ylabel("Bivariate Moran's I")
    ax.set_title("Spatial concordance exceeds\nlocal block-null expectation", fontsize=8.6, fontweight="semibold")
    ax.set_xlim(-0.25, 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.96, "Error bars: 95% permutation interval", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)


def block_size_plot(ax, sensitivity: pd.DataFrame) -> None:
    samples = ["NSCLC_P5", "NSCLC_P8", "NSCLC_P12", "NSCLC_P15"]
    dor = sensitivity[sensitivity["comparison"].eq("DoRothEA_vs_hypoxia")].copy()
    dor["observed_minus_null"] = dor["bivariate_moran_I_state_pc_residualized"] - dor["block_null_mean"]
    for sample in samples:
        row = dor[dor["sample"].eq(sample)].sort_values("block_size_array_coordinates")
        ax.plot(
            row["block_size_array_coordinates"],
            row["observed_minus_null"],
            marker="o",
            lw=1.1,
            color=SAMPLE_COLORS[sample],
            label=sample.replace("NSCLC_", ""),
        )
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set_xlabel("Block size (array coordinates)")
    ax.set_ylabel("Observed − block-null mean Moran's I")
    ax.set_title("Block-size sensitivity", fontsize=9)
    ax.set_xticks([10, 15, 20, 30])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper right")


def main() -> None:
    spots = pd.read_csv(SPOT_TABLE)
    raw = pd.read_csv(RAW)
    qc = pd.read_csv(QC)
    state = pd.read_csv(STATE)
    sensitivity = pd.read_csv(BLOCK_SENSITIVITY)
    raw_pair = raw[raw["comparison"].eq("DoRothEA_vs_hypoxia")].copy()
    state_pair = state[state["comparison"].eq("DoRothEA_vs_hypoxia")].copy()
    stress_state = state[state["comparison"].eq("Stress_AP1_vs_hypoxia")].copy()
    qc_pair = qc[(qc["model"].eq("QC_only")) & (qc["comparison"].eq("DoRothEA_vs_hypoxia"))].copy()
    attenuation = raw_pair[["sample", "spearman_rho_descriptive"]].rename(columns={"spearman_rho_descriptive": "raw"})
    attenuation = attenuation.merge(
        qc_pair[["sample", "spearman_rho_residualized"]].rename(columns={"spearman_rho_residualized": "qc"}),
        on="sample",
    ).merge(
        state_pair[["sample", "spearman_rho_state_pc_residualized"]].rename(columns={"spearman_rho_state_pc_residualized": "state"}),
        on="sample",
    )
    attenuation_moran = raw_pair[["sample", "bivariate_moran_I"]].rename(columns={"bivariate_moran_I": "raw"})
    attenuation_moran = attenuation_moran.merge(
        qc_pair[["sample", "bivariate_moran_I_residualized"]].rename(columns={"bivariate_moran_I_residualized": "qc"}),
        on="sample",
    ).merge(
        state_pair[["sample", "bivariate_moran_I_state_pc_residualized"]].rename(columns={"bivariate_moran_I_state_pc_residualized": "state"}),
        on="sample",
    )

    # Keep the figure close to a journal-page aspect ratio.  The earlier 12-column
    # layout made the three representative maps span a very wide top strip.
    fig = plt.figure(figsize=(10.8, 7.8))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.10, 0.98, 1.05],
        hspace=0.44,
        wspace=0.42,
    )

    top_grid = grid[0, :].subgridspec(1, 2, width_ratios=[0.56, 1.44], wspace=0.24)
    ax_a = fig.add_subplot(top_grid[0, 0])
    add_letter(ax_a, "a")
    ax_a.axis("off")
    # The panel mixes scatter points with axes-fraction-like text coordinates;
    # fix the data limits so tight bounding-box export cannot expand the canvas.
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.text(0.0, 0.98, "Independent spatial cohort", fontsize=9.2, fontweight="semibold", va="top")
    ax_a.text(0.0, 0.89, "GSE292299", fontsize=7.2, color="#555555", va="top")
    ax_a.text(0.08, 0.82, "Section", fontsize=7, fontweight="semibold", va="center")
    ax_a.text(0.32, 0.82, "Histology", fontsize=7, fontweight="semibold", va="center")
    ax_a.text(0.66, 0.82, "Spots", fontsize=7, fontweight="semibold", va="center")
    rows = [
        ("P5", "LUAD", "5,388 spots"),
        ("P8", "LUAD", "5,218 spots"),
        ("P12", "LUAD", "2,131 spots"),
        ("P15", "LUSC", "13,002 spots"),
    ]
    for i, (sample, hist, n) in enumerate(rows):
        y = 0.72 - i * 0.16
        ax_a.scatter(0.03, y, color=SAMPLE_COLORS[f"NSCLC_{sample}"], s=32)
        ax_a.text(0.08, y, sample, fontsize=8, va="center")
        ax_a.text(0.32, y, hist, fontsize=8, va="center")
        ax_a.text(0.66, y, n, fontsize=8, va="center")

    map_grid = top_grid[0, 1].subgridspec(1, 3, wspace=0.04)
    map_axes = [fig.add_subplot(map_grid[0, i]) for i in range(3)]
    add_letter(map_axes[0], "b")
    representative_maps(fig, map_axes, spots)
    map_axes[1].text(
        0.5,
        1.24,
        "Representative spatial projections in P12 (LUAD)",
        transform=map_axes[1].transAxes,
        ha="center",
        fontsize=9.2,
        fontweight="semibold",
    )

    ax_c = fig.add_subplot(grid[1, 0])
    add_letter(ax_c, "c")
    raw_program_plot(ax_c, raw)

    ax_d = fig.add_subplot(grid[1, 1])
    add_letter(ax_d, "d")
    observed_vs_block_null_plot(ax_d, raw_pair, state_pair)

    ax_e = fig.add_subplot(grid[2, :])
    add_letter(ax_e, "e")
    for sample in ["NSCLC_P5", "NSCLC_P8", "NSCLC_P12", "NSCLC_P15"]:
        row = attenuation[attenuation["sample"].eq(sample)].iloc[0]
        x = np.arange(3)
        label = "P15 (LUSC)" if sample == "NSCLC_P15" else sample.replace("NSCLC_", "")
        ax_e.plot(x, [row["raw"], row["qc"], row["state"]], marker="o", lw=1.2, color=SAMPLE_COLORS[sample], label=label)
    ax_e.axhline(0, color="#777777", lw=0.7)
    ax_e.set_xticks([0, 1, 2], ["Raw", "QC", "QC + state PCs"])
    ax_e.set_ylabel("DoRothEA–hypoxia ρ")
    ax_e.set_title(
        "Epithelial-state adjustment attenuates\nBACH1–hypoxia covariance",
        fontsize=9.2,
        fontweight="semibold",
    )
    ax_e.spines[["top", "right"]].set_visible(False)
    ax_e.legend(frameon=False, ncol=4, fontsize=7, loc="upper right")

    fig.savefig(OUT / "GSE292299_spatial_extension_draft.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUT / "GSE292299_spatial_extension_draft.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure5_spatial_state_context.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUT / "Figure5_spatial_state_context.pdf", bbox_inches="tight")
    plt.close(fig)

    # Block-size sensitivity is retained as a reviewer-facing methodological
    # check rather than a main biological panel.
    fig_sup, ax_sup = plt.subplots(figsize=(5.4, 3.8))
    add_letter(ax_sup, "a")
    block_size_plot(ax_sup, sensitivity)
    fig_sup.suptitle("Supplementary Figure 5. Spatial block-size sensitivity", fontsize=10, fontweight="bold")
    fig_sup.tight_layout(rect=[0, 0, 1, 0.94])
    fig_sup.savefig(OUT / "Supplementary_Figure5_spatial_block_size_sensitivity.png", dpi=250, bbox_inches="tight")
    fig_sup.savefig(OUT / "Supplementary_Figure5_spatial_block_size_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig_sup)


if __name__ == "__main__":
    main()
