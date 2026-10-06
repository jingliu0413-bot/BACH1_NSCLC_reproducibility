#!/usr/bin/env python3
"""Rebuild the final five-figure manuscript architecture from locked tables.

This script reorganizes the already computed manuscript-submission results into
the final story-line figures requested for submission. It does not recompute the
underlying statistics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib import colors as mpl_colors
from matplotlib import ticker as mpl_ticker
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "manuscript_submission_20260914"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
TCGA_SCORE_DIR = ROOT / "outputs" / "manuscript_submission_20260911" / "tables"
STABILITY_DIR = ROOT / "out" / "revision_diagnostics" / "pyscenic_bach1_consensus_full_tf_seed_stability_v1"
FIG1_SOURCE = ROOT / "out" / "nature_story_4figures_10kb" / "source_data"

DOR_NO = "DOROTHEA_BACH1_ABC_TF_ACTIVITY_NO_HYPOXIA_OVERLAP__mean_z"
HYP_NO = "HALLMARK_HYPOXIA_NO_DOROTHEA_OVERLAP__mean_z"

STATE_ORDER = ["Stress_AP1", "Ciliated", "Epithelial_core", "Proliferative", "AT2_like", "Other"]
STATE_DISPLAY = {
    "Stress_AP1": "Stress-associated",
    "Ciliated": "Ciliated-like",
    "Epithelial_core": "Epithelial core",
    "Proliferative": "Proliferative",
    "AT2_like": "AT2-like",
    "Other": "Other",
}
STATE_COLORS = {
    "Stress_AP1": "#D55E00",
    "Ciliated": "#0072B2",
    "Epithelial_core": "#009E73",
    "Proliferative": "#CC79A7",
    "AT2_like": "#56B4E9",
    "Other": "#999999",
}
COHORT_COLORS = {"LUAD": "#4C72B0", "LUSC": "#C44E52", "combined": "#333333", "Combined": "#333333"}
PRIORITY_CANDIDATES = ["SOCS5", "SPECC1", "PDK4", "SETD1B", "SEMA4A", "FSTL1"]
TIER_ORDER = {
    "Tier 1: 5/5 seeds + 5-sample ATAC + promoter support": 1,
    "Tier 2: 4/5 seeds + 5-sample ATAC + promoter support": 2,
    "Tier 1: 5/5 seeds + 5-sample ATAC": 1,
    "Tier 2: 4/5 seeds + promoter ATAC": 2,
    "Context-limited: recurrent with limited ATAC": 3,
    "Context-limited: recurrent without mapped ATAC": 4,
    "Tier 3: recurrent with limited ATAC": 3,
    "Tier 3: recurrent without mapped ATAC": 4,
}
CELLTYPE_ORDER = ["Epithelial", "T/NK", "Myeloid", "B", "Plasma", "Fibroblast", "Mast", "Endothelial"]
CELLTYPE_COLORS = {
    "Epithelial": "#D9544D",
    "T/NK": "#4E79A7",
    "Myeloid": "#59A14F",
    "B": "#76B7B2",
    "Plasma": "#B07AA1",
    "Fibroblast": "#E28E2C",
    "Mast": "#9C755F",
    "Endothelial": "#8F8F8F",
}
MARKER_GENES = [
    "EPCAM",
    "KRT8",
    "KRT18",
    "CD3D",
    "NKG7",
    "MS4A1",
    "CD79A",
    "MZB1",
    "JCHAIN",
    "LYZ",
    "S100A8",
    "COL1A1",
    "DCN",
    "PECAM1",
    "VWF",
    "TPSAB1",
    "CPA3",
]


def set_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=13, ha="left", va="bottom")


def format_p(p: float | str | None) -> str:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return ""
    p = float(p)
    if p >= 0.9995:
        return r"$P$=1.00"
    if p < 0.001 and p > 0:
        exp = math.floor(math.log10(p))
        mant = p / (10**exp)
        return rf"$P$={mant:.1f}$\times$10$^{{{exp}}}$"
    return rf"$P$={p:.3f}"


def format_q(q: float | str | None) -> str:
    if q is None or (isinstance(q, float) and not np.isfinite(q)):
        return ""
    q = float(q)
    if q < 0.001 and q > 0:
        exp = math.floor(math.log10(q))
        mant = q / (10**exp)
        return rf"FDR={mant:.1f}$\times$10$^{{{exp}}}$"
    return f"FDR={q:.3f}"


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    for col in out.columns:
        sd = out[col].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out[col] = 0.0
        else:
            out[col] = (out[col] - out[col].mean()) / sd
    return out


def format_umap_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl_ticker.MaxNLocator(4))
    ax.set_aspect("equal", adjustable="datalim")
    sns.despine(ax=ax)


def draw_workflow(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.03, "Study design", ha="left", va="bottom", fontsize=9, fontweight="semibold")
    boxes = [
        ("NSCLC\nscRNA atlas", "GSE131907 + GSE274934\n176,294 cells"),
        ("Malignant\nepithelium", "13,694 cells\n16 patients"),
        ("BACH1–hypoxia\ncovariance", "DoRothEA score +\nHallmark hypoxia"),
        ("Epithelial-state\narchitecture", "State programmes +\nresidualization"),
        ("Pan-transcription\nfactor / TCGA", "TCGA-LUAD/LUSC\nn = 1,017"),
        ("Independent spatial\nreplication", "GSE292299\n4 spatial sections"),
    ]
    groups = {
        "Discovery (scRNA-seq)": {"indices": range(0, 4), "face": "#E9EFF6", "edge": "#7A8CA5", "dark": "#3E5C84"},
        "Validation (bulk / spatial)": {"indices": range(4, 6), "face": "#E8F2EC", "edge": "#6E9983", "dark": "#2F6B50"},
    }
    group_for = {i: group for group, spec in groups.items() for i in spec["indices"]}
    margin, in_group_gap, between_group_gap = 0.005, 0.018, 0.04
    box_width = (1 - 2 * margin - 4 * in_group_gap - between_group_gap) / 6
    y0, box_height = 0.04, 0.50
    top = y0 + box_height
    xs, x = [], margin
    for i in range(6):
        xs.append(x)
        x += box_width + (between_group_gap if i == 3 else in_group_gap)

    for i, (title, detail) in enumerate(boxes, start=1):
        spec = groups[group_for[i - 1]]
        x = xs[i - 1]
        ax.add_patch(
            FancyBboxPatch(
                (x, y0),
                box_width,
                box_height,
                boxstyle="round,pad=0.010,rounding_size=0.010",
                fc=spec["face"],
                ec=spec["edge"],
                lw=0.55,
                zorder=2,
            )
        )
        marker_x, marker_y = x + 0.018, top
        ax.scatter([marker_x], [marker_y], s=62, color=spec["dark"], edgecolor="white", linewidth=0.7, transform=ax.transAxes, zorder=4)
        ax.text(marker_x, marker_y, str(i), ha="center", va="center", fontsize=5.8, fontweight="bold", color="white", transform=ax.transAxes, zorder=5)
        ax.text(x + box_width / 2, y0 + 0.31, title, ha="center", va="center", fontsize=6.9, fontweight="bold", linespacing=1.15, zorder=3)
        ax.text(x + box_width / 2, y0 + 0.105, detail, ha="center", va="center", fontsize=5.6, color="#4A4A4A", linespacing=1.25, zorder=3)
    for start, end in [
        (xs[0] + box_width + 0.005, xs[1] - 0.005),
        (xs[1] + box_width + 0.005, xs[2] - 0.005),
        (xs[2] + box_width + 0.005, xs[3] - 0.005),
        (xs[3] + box_width + 0.005, xs[4] - 0.005),
        (xs[4] + box_width + 0.005, xs[5] - 0.005),
    ]:
        ax.add_patch(FancyArrowPatch((start, y0 + box_height / 2), (end, y0 + box_height / 2), arrowstyle="-|>", mutation_scale=10, lw=0.75, color="#5A5A5A", shrinkA=0, shrinkB=0, zorder=3))
    for label, spec in groups.items():
        first, last = min(spec["indices"]), max(spec["indices"])
        xa, xb = xs[first], xs[last] + box_width
        bracket_y = top + 0.085
        ax.plot([xa, xa, xb, xb], [bracket_y - 0.025, bracket_y, bracket_y, bracket_y - 0.025], color=spec["edge"], lw=0.6, solid_capstyle="butt", zorder=1)
        ax.text((xa + xb) / 2, bracket_y + 0.035, label, ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=spec["dark"])


def plot_umap_categorical(ax: plt.Axes, df: pd.DataFrame, column: str, palette: dict[str, str], point_size: float, alpha: float) -> None:
    for value, color in palette.items():
        sub = df.loc[df[column].astype(str).eq(value)]
        if not sub.empty:
            ax.scatter(sub["UMAP1"], sub["UMAP2"], s=point_size, alpha=alpha, color=color, linewidths=0, rasterized=True)
    format_umap_axes(ax)


def plot_figure1() -> None:
    """Rebuild Figure 1 from current cached source data."""
    atlas = pd.read_csv(FIG1_SOURCE / "figure1_panel_b_atlas_umap_current.csv.gz")
    markers = pd.read_csv(FIG1_SOURCE / "figure1_panel_d_celltype_marker_dotplot_current.csv")
    epithelial = pd.read_csv(FIG1_SOURCE / "figure1_panel_e_epithelial_malignancy_umap_source.csv.gz")
    bach1 = pd.read_csv(FIG1_SOURCE / "figure1_panel_f_bach1_expression_umap_source.csv.gz")

    set_style()
    fig = plt.figure(figsize=(13.4, 7.9))
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[0.48, 1.10, 1.10],
        width_ratios=[1.0, 1.0, 1.06, 1.06],
        # Leave enough room for the upper-row x labels before the lower-row
        # titles; the same physical gap is used between all three rows.
        hspace=0.48,
        wspace=0.38,
    )

    ax = fig.add_subplot(gs[0, :])
    draw_workflow(ax)
    # The workflow spans the full figure width, so compensate for the wider
    # axes when aligning its label with the half-width panels below.
    panel_label(ax, "a", x=-0.035)

    ax = fig.add_subplot(gs[1, :2])
    plot_umap_categorical(ax, atlas, "major_celltype_auto", CELLTYPE_COLORS, point_size=0.45, alpha=0.68)
    centroids = atlas.groupby("major_celltype_auto", observed=True)[["UMAP1", "UMAP2"]].median()
    label_offsets = {
        "B": (-0.55, 0.55),
        "T/NK": (-0.55, 0.05),
        "Plasma": (0.35, -0.75),
        "Mast": (-0.35, -0.75),
        "Fibroblast": (0.35, -0.75),
        "Epithelial": (0.35, -0.75),
        "Endothelial": (1.00, 0.35),
        "Myeloid": (0.55, 0.70),
    }
    for ct in CELLTYPE_ORDER:
        if ct in centroids.index:
            x0 = centroids.loc[ct, "UMAP1"]
            y0 = centroids.loc[ct, "UMAP2"]
            dx, dy = label_offsets.get(ct, (0.0, 0.0))
            ax.annotate(
                ct,
                xy=(x0, y0),
                xytext=(x0 + dx, y0 + dy),
                fontsize=6.1,
                ha="right" if dx < 0 else "left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.35, "shrinkA": 2, "shrinkB": 2},
            )
    ax.set_title("Integrated NSCLC single-cell atlas", loc="left", fontsize=9, pad=2)
    ax.text(0.01, 0.02, f"{len(atlas):,} cells", transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 2:])
    groups = [ct for ct in CELLTYPE_ORDER if ct in set(markers["cell_type"])]
    genes = [g for g in MARKER_GENES if g in set(markers["gene"])]
    x_map = {g: i for i, g in enumerate(genes)}
    y_map = {g: i for i, g in enumerate(groups[::-1])}
    plot_df = markers.loc[markers["gene"].isin(genes) & markers["cell_type"].isin(groups)].copy()
    scale = pd.to_numeric(plot_df["fraction_expressing"], errors="coerce")
    sc = ax.scatter(
        plot_df["gene"].map(x_map),
        plot_df["cell_type"].map(y_map),
        s=8 + scale * 0.72,
        c=plot_df["mean_expression_scaled"],
        cmap=mpl_colors.LinearSegmentedColormap.from_list("marker_scale", ["#4E79A7", "#F3F3F3", "#D9544D"]),
        vmin=-1.8,
        vmax=1.8,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xticks(np.arange(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=7.6)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups[::-1], fontsize=7.6)
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_title("Canonical marker validation", loc="left", fontsize=9, pad=2)
    ax.grid(color="#E5E7EB", lw=0.45)
    ax.set_axisbelow(True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.040, pad=0.015)
    cb.set_label("scaled mean\nexpression", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.2, width=0.4)
    cb.outline.set_linewidth(0.4)
    for pct, xp in zip([25, 50, 75], [0.08, 0.18, 0.30]):
        ax.scatter(xp, -0.18, s=8 + pct * 0.72, transform=ax.transAxes, color="#AEB7C2", edgecolor="white", linewidth=0.3, clip_on=False)
        ax.text(xp + 0.035, -0.18, f"{pct}%", transform=ax.transAxes, va="center", ha="left", fontsize=6, color="#767676")
    ax.text(0.08, -0.29, "fraction expressing", transform=ax.transAxes, va="center", ha="left", fontsize=6, color="#767676")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[2, :2])
    plot_umap_categorical(
        ax,
        epithelial,
        "malignant_call",
        {"non-malignant": "#B8C0CC", "malignant": "#D9544D"},
        point_size=1.15,
        alpha=0.75,
    )
    ax.set_title("Malignant epithelial selection", loc="left", fontsize=9, pad=2)
    ax.add_patch(
        FancyBboxPatch(
            (0.70, 0.835),
            0.25,
            0.115,
            transform=ax.transAxes,
            boxstyle="round,pad=0.014,rounding_size=0.012",
            facecolor="white",
            edgecolor="none",
            alpha=0.84,
            zorder=4,
        )
    )
    ax.scatter([0.725], [0.915], transform=ax.transAxes, s=26, color="#D9544D", edgecolor="none", zorder=5)
    ax.text(0.755, 0.915, "retained malignant", transform=ax.transAxes, fontsize=6.8, va="center", ha="left", zorder=5)
    ax.scatter([0.725], [0.865], transform=ax.transAxes, s=26, color="#B8C0CC", edgecolor="none", zorder=5)
    ax.text(0.755, 0.865, "other epithelial cells", transform=ax.transAxes, fontsize=6.8, va="center", ha="left", zorder=5)
    ax.text(
        0.01,
        0.02,
        f"{(epithelial['malignant_call'] == 'malignant').sum():,} malignant epithelial cells · 16 patients",
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="bottom",
        bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.0},
    )
    panel_label(ax, "d")

    ax = fig.add_subplot(gs[2, 2:])
    vals = pd.to_numeric(bach1["BACH1_expr"], errors="coerce")
    zero = vals.fillna(0).eq(0)
    ax.scatter(
        bach1.loc[zero, "UMAP1"],
        bach1.loc[zero, "UMAP2"],
        s=2.8,
        c="#D9D9D9",
        alpha=0.58,
        linewidths=0,
        rasterized=True,
    )
    positive = vals.gt(0)
    positive_vals = vals.loc[positive]
    sc = ax.scatter(
        bach1.loc[positive, "UMAP1"],
        bach1.loc[positive, "UMAP2"],
        s=2.8,
        c=positive_vals,
        cmap="magma",
        vmin=0,
        vmax=np.nanquantile(positive_vals, 0.98),
        alpha=0.86,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("BACH1 transcript heterogeneity", loc="left", fontsize=9, pad=2)
    format_umap_axes(ax)
    ax.set_xlim(epithelial["UMAP1"].min(), epithelial["UMAP1"].max())
    ax.set_ylim(epithelial["UMAP2"].min(), epithelial["UMAP2"].max())
    ax.text(
        0.98,
        0.96,
        f"BACH1 detected: {positive.mean() * 100:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"fc": "white", "ec": "none", "alpha": 0.82, "pad": 1.0},
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label("log-normalized\nBACH1", fontsize=7)
    panel_label(ax, "e")

    fig.subplots_adjust(top=0.95, bottom=0.08)
    save_all(fig, "Figure1_cellular_context")


def plot_figure3() -> None:
    """Malignant epithelial-state architecture figure."""
    umap = pd.read_csv(TABLE_DIR / "figure4a_primary_malignant_state_umap.csv.gz")
    heat_z = pd.read_csv(TABLE_DIR / "figure4b_state_programme_summary_column_z.csv").set_index("state")
    state_summary = pd.read_csv(TABLE_DIR / "figure4b_state_programme_summary.csv").set_index("state")
    attenuation = pd.read_csv(TABLE_DIR / "figure4d_patient_level_association_attenuation.csv")
    within_state = pd.read_csv(TABLE_DIR / "figure4c_within_state_patient_aware_associations.csv")

    set_style()
    fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.05, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    for state in STATE_ORDER:
        sub = umap.loc[umap["state"].eq(state)]
        if not sub.empty:
            ax.scatter(
                sub["UMAP1"],
                sub["UMAP2"],
                s=2.2,
                alpha=0.62,
                color=STATE_COLORS[state],
                label=STATE_DISPLAY[state],
                rasterized=True,
                linewidths=0,
            )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(frameon=False, markerscale=5, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    idx = [s for s in STATE_ORDER if s in heat_z.index]
    heat_df = heat_z.loc[idx]
    sns.heatmap(
        heat_df,
        cmap="vlag",
        center=0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Column z"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    if "BACH1 detected" in heat_df.columns:
        ax.axvline(1, color="#222222", lw=1.0)
    labels = []
    for state in heat_df.index:
        row = state_summary.loc[state]
        labels.append(f"{STATE_DISPLAY[state]} | {int(row.n_cells):,} cells | {int(row.n_patients)} patients")
    ax.set_yticklabels(labels, rotation=0, fontsize=7)
    panel_label(ax, "b", x=-0.08)

    sub = gs[1, 0].subgridspec(1, 2, width_ratios=[0.66, 0.34], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    y = np.arange(len(attenuation))
    ax.axvline(0, color="0.72", lw=0.8)
    ax.errorbar(
        attenuation["rho"],
        y,
        xerr=[attenuation["rho"] - attenuation["ci_low"], attenuation["ci_high"] - attenuation["rho"]],
        fmt="o",
        color="#333333",
        ecolor="#333333",
        capsize=2,
        lw=1.1,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(attenuation["comparison"], fontsize=7)
    ax.set_xlabel(r"Spearman $\rho$")
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    for i, r in attenuation.iterrows():
        ax_ann.text(
            0,
            i,
            rf"$\rho$={r.rho:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]"
            + "\n"
            + f"n={int(r.n_patients)}; {format_p(r.p_value)}",
            ha="left",
            va="center",
            fontsize=7,
        )
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    panel_label(ax, "c")

    sub = gs[1, 1].subgridspec(1, 2, width_ratios=[0.62, 0.38], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    ws = within_state.dropna(subset=["spearman_rho"]).copy()
    ws["state"] = pd.Categorical(ws["state"], STATE_ORDER, ordered=True)
    ws = ws.sort_values("state")
    y = np.arange(len(ws))
    ax.axvline(0, color="0.72", lw=0.8)
    ax.errorbar(
        ws["spearman_rho"],
        y,
        xerr=[ws["spearman_rho"] - ws["ci_low"], ws["ci_high"] - ws["spearman_rho"]],
        fmt="o",
        color="#4C72B0",
        ecolor="#4C72B0",
        capsize=2,
        lw=1.1,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([STATE_DISPLAY[str(s)] for s in ws["state"]], fontsize=7)
    ax.set_xlabel(r"Within-state patient-level $\rho$")
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    for i, r in enumerate(ws.itertuples(index=False)):
        ax_ann.text(
            0,
            i,
            rf"$\rho$={r.spearman_rho:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]"
            + "\n"
            + f"n={int(r.n_patient_state_units)}; Holm {format_p(r.holm_adjusted_p_across_states)}",
            ha="left",
            va="center",
            fontsize=7,
        )
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    panel_label(ax, "d", x=-0.08)

    save_all(fig, "Figure3_epithelial_state_architecture")


def plot_figure4() -> None:
    """Specificity, cohort split and TCGA score-association replication figure."""
    tfbench = pd.read_csv(TABLE_DIR / "dorothea_all_tf_hypoxia_contextual_benchmark.csv")
    cohort_split = pd.read_csv(TABLE_DIR / "scrna_cohort_split_dorothea_hypoxia_correlations.csv")
    tcga_scores = pd.read_csv(TCGA_SCORE_DIR / "tcga_luad_lusc_bach1_hypoxia_patient_scores.csv")
    tcga_corr = pd.read_csv(TCGA_SCORE_DIR / "tcga_bach1_hypoxia_correlations.csv")
    tcga_models = pd.read_csv(TCGA_SCORE_DIR / "tcga_bach1_hypoxia_adjusted_models.csv")

    top_tfs = tfbench.sort_values("rank_by_rho_desc").head(10)["tf"].tolist()
    comparators = ["JUN", "ATF4", "NFE2L2", "MYC"]
    display_tfs = list(dict.fromkeys(top_tfs + comparators))
    tf_show = tfbench.loc[tfbench["tf"].isin(display_tfs)].copy().sort_values("rank_by_rho_desc")

    set_style()
    fig = plt.figure(figsize=(12.4, 8.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.05, 1.0])

    sub = gs[0, 0].subgridspec(1, 2, width_ratios=[0.78, 0.22], wspace=0.01)
    ax = fig.add_subplot(sub[0, 0])
    ax_rank = fig.add_subplot(sub[0, 1], sharey=ax)
    y = np.arange(len(tf_show))
    bar_colors = ["#C44E52" if tf == "BACH1" else "#4C72B0" for tf in tf_show["tf"]]
    ax.barh(y, tf_show["spearman_rho_with_hypoxia"], color=bar_colors)
    ax.axvline(0, color="0.75", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(tf_show["tf"], fontsize=7)
    for tick in ax.get_yticklabels():
        tick.set_fontstyle("italic")
    ax.set_xlabel(r"De-overlapped $\rho$ with hypoxia")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    for i, r in enumerate(tf_show.itertuples(index=False)):
        rank = int(r.rank_by_rho_desc)
        ax_rank.text(0.0, i, f"rank {rank}", ha="left", va="center", fontsize=6.5, fontweight="bold" if r.tf == "BACH1" else "normal")
    ax_rank.set_xlim(0, 1)
    ax_rank.set_ylim(ax.get_ylim())
    ax_rank.axis("off")
    panel_label(ax, "a", x=-0.12)

    sub = gs[0, 1].subgridspec(1, 2, width_ratios=[0.64, 0.36], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    cs = cohort_split.copy()
    cs["_order"] = cs["dataset"].map({"All patients": 0, "GSE131907": 1, "GSE274934": 2})
    cs = cs.sort_values("_order")
    y = np.arange(len(cs))
    ax.axvline(0, color="0.75", lw=0.8)
    ax.errorbar(
        cs["rho"],
        y,
        xerr=[cs["rho"] - cs["ci_low"], cs["ci_high"] - cs["rho"]],
        fmt="o",
        color="#333333",
        capsize=2,
        lw=1.1,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(cs["dataset"], fontsize=7)
    ax.set_xlabel(r"scRNA patient-level $\rho$")
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    for i, r in enumerate(cs.itertuples(index=False)):
        ax_ann.text(
            0,
            i,
            rf"$\rho$={r.rho:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]"
            + "\n"
            + f"n={int(r.n_patients)}; {format_p(r.p_value)}",
            ha="left",
            va="center",
            fontsize=7,
        )
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    panel_label(ax, "b", x=-0.12)

    ax = fig.add_subplot(gs[1, 0])
    for cohort, color in [("LUAD", COHORT_COLORS["LUAD"]), ("LUSC", COHORT_COLORS["LUSC"])]:
        subdf = tcga_scores.loc[tcga_scores["cohort"].eq(cohort)]
        ax.scatter(subdf[DOR_NO], subdf[HYP_NO], s=12, alpha=0.45, color=color, linewidths=0, label=cohort)
    x = tcga_scores[DOR_NO].to_numpy()
    yv = tcga_scores[HYP_NO].to_numpy()
    ok = np.isfinite(x) & np.isfinite(yv)
    if ok.sum() > 2:
        b1, b0 = np.polyfit(x[ok], yv[ok], 1)
        xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
        ax.plot(xs, b1 * xs + b0, color="#222222", lw=1.2)
    corr = tcga_corr.loc[(tcga_corr["cohort"].eq("combined")) & (tcga_corr["comparison"].eq("both_no_shared"))].iloc[0]
    ax.text(
        0.04,
        0.96,
        rf"$\rho$={corr.spearman_rho:.3f}" + "\n" + f"Holm {format_p(corr.spearman_holm_p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("TCGA DoRothEA BACH1 score")
    ax.set_ylabel("TCGA Hallmark hypoxia score")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "c")

    sub = gs[1, 1].subgridspec(1, 2, width_ratios=[0.62, 0.38], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    model_order = [
        ("combined_deoverlap_cohort_stage_smoking", "Combined"),
        ("LUAD_deoverlap_purity_stage_smoking", "LUAD"),
        ("LUSC_deoverlap_stage_smoking", "LUSC"),
    ]
    rows = []
    for model, label in model_order:
        row = tcga_models.loc[tcga_models["model"].eq(model)].iloc[0].copy()
        row["label"] = label
        rows.append(row)
    forest = pd.DataFrame(rows)
    y = np.arange(len(forest))
    ax.axvline(0, color="0.75", lw=0.8)
    ax.errorbar(
        forest["beta"],
        y,
        xerr=[forest["beta"] - forest["ci_low"], forest["ci_high"] - forest["beta"]],
        fmt="o",
        color="#333333",
        capsize=2,
        lw=1.1,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(forest["label"], fontsize=7)
    ax.set_xlabel("Adjusted standardized beta")
    ax.set_xlim(0, max(0.45, forest["ci_high"].max() + 0.05))
    ax.invert_yaxis()
    for i, r in enumerate(forest.itertuples(index=False)):
        ax_ann.text(
            0,
            i,
            f"{r.beta:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"
            + "\n"
            + format_p(r.p_value),
            ha="left",
            va="center",
            fontsize=7,
        )
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    panel_label(ax, "d", x=-0.12)

    save_all(fig, "Figure4_pan_tf_cross_cohort_validation")


def plot_figure5() -> None:
    """Multi-omic candidate-prioritisation figure."""
    candidates = pd.read_csv(TABLE_DIR / "figure4e_candidate_evidence_matrix.csv")
    stability = json.loads((STABILITY_DIR / "pyscenic_bach1_seed_stability_summary.json").read_text())

    cand = candidates.copy()
    if "candidate_tier" in cand.columns:
        cand["_tier_order"] = cand["candidate_tier"].map(TIER_ORDER).fillna(9)
        cand = cand.sort_values(
            [
                "_tier_order",
                "n_runs_detected",
                "weighted_importance",
                "n_ucsc_score_ge_300_motif_peak_links_in_window",
            ],
            ascending=[True, False, False, False],
        )
    else:
        cand = cand.sort_values(["n_runs_detected", "weighted_importance"], ascending=False)
    cand = cand.loc[pd.to_numeric(cand["n_runs_detected"], errors="coerce") >= 4].copy()
    tier_counts = cand.get("candidate_tier", pd.Series(index=cand.index, dtype=str)).value_counts()
    n_tier1 = int(
        tier_counts.get("Tier 1: 5/5 seeds + 5-sample ATAC + promoter support", 0)
        + tier_counts.get("Tier 1: 5/5 seeds + 5-sample ATAC", 0)
    )
    n_tier2 = int(
        tier_counts.get("Tier 2: 4/5 seeds + 5-sample ATAC + promoter support", 0)
        + tier_counts.get("Tier 2: 4/5 seeds + promoter ATAC", 0)
    )
    n_lower = int(len(cand) - n_tier1 - n_tier2)

    set_style()
    fig = plt.figure(figsize=(11.8, 8.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.05], width_ratios=[0.92, 1.38])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    funnel = [
        ("Seed-union targets", stability["n_union_targets"], 0.88, "#DCE6F2"),
        ("≥3/5 seed recurrence", stability["n_targets_detected_in_at_least_60pct_runs"], 0.66, "#CFE6D8"),
        ("≥4/5 seed recurrence", stability["n_targets_detected_in_at_least_70pct_runs"], 0.48, "#F3E4C7"),
    ]
    y0 = 0.83
    h = 0.13
    for i, (label, n, width, color) in enumerate(funnel):
        y = y0 - i * 0.2
        x_left = 0.5 - width / 2
        x_right = 0.5 + width / 2
        poly = Polygon(
            [[x_left, y], [x_right, y], [x_right - 0.04, y - h], [x_left + 0.04, y - h]],
            closed=True,
            facecolor=color,
            edgecolor="#666666",
            linewidth=0.8,
        )
        ax.add_patch(poly)
        ax.text(0.5, y - h / 2 + 0.012, f"{n:,}", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(0.5, y - h / 2 - 0.037, label, ha="center", va="center", fontsize=7)
        if i < len(funnel) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (0.5, y - h - 0.018),
                    (0.5, y - 0.18),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    lw=0.8,
                    color="#666666",
                )
            )
    tier_y = 0.08
    tier_boxes = [
        (0.10, tier_y, 0.22, 0.13, f"Tier 1\nn={n_tier1}", "#F0CBC8"),
        (0.37, tier_y, 0.22, 0.13, f"Tier 2\nn={n_tier2}", "#F5D9C8"),
        (0.64, tier_y, 0.28, 0.13, f"Context-\nlimited\nn={n_lower}", "#E8E1EF"),
    ]
    ax.add_patch(FancyArrowPatch((0.5, y0 - 2 * 0.2 - h - 0.018), (0.5, tier_y + 0.13), arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#666666"))
    ax.text(
        0.72,
        tier_y + 0.18,
        "Recurrence/chromatin\ncontext tiering",
        ha="left",
        va="bottom",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )
    for x0, y0_box, w, h_box, text, color in tier_boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0_box),
                w,
                h_box,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=color,
                edgecolor="#666666",
                linewidth=0.8,
            )
        )
        ax.text(x0 + w / 2, y0_box + h_box / 2, text, ha="center", va="center", fontsize=8, linespacing=0.9)
    panel_label(ax, "a", x=-0.02)

    ax = fig.add_subplot(gs[0, 1])
    motif_links = pd.to_numeric(cand["n_ucsc_score_ge_300_motif_peak_links_in_window"], errors="coerce").fillna(0)
    sample_links = pd.to_numeric(cand["n_samples_with_motif_peak_in_window"], errors="coerce").fillna(0)
    promoter_links = pd.to_numeric(cand.get("n_promoter_2kb_motif_peak_links", 0), errors="coerce").fillna(0)
    y = np.arange(len(cand))
    max_links = max(1.0, float(motif_links.max()))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.85, len(cand) - 0.15)
    for i, gene in enumerate(cand["gene"]):
        ax.hlines(i, 0.04, 0.42, color="#E6E6E6", lw=4.0, zorder=0)
        bar_end = 0.04 + 0.38 * float(motif_links.iloc[i]) / max_links
        ax.hlines(i, 0.04, bar_end, color="#55A868", lw=4.0, zorder=1)
        ax.text(min(0.50, bar_end + 0.018), i, f"{int(motif_links.iloc[i])}", ha="left", va="center", fontsize=6.5, color="#444444")
        for j in range(5):
            face = "#4C72B0" if j < int(sample_links.iloc[i]) else "white"
            ax.scatter(0.56 + j * 0.035, i, s=18, facecolor=face, edgecolor="#4C72B0", lw=0.7, zorder=2)
        if promoter_links.iloc[i] > 0:
            ax.scatter(0.82, i, marker="v", s=22, color="#333333", zorder=3)
            ax.text(0.87, i, f"{int(promoter_links.iloc[i])}", ha="left", va="center", fontsize=6.5)
        else:
            ax.text(0.87, i, "0", ha="left", va="center", fontsize=6.5, color="#777777")
    for boundary in (1.5, 10.5):
        ax.axhline(boundary, xmin=0.0, xmax=0.94, color="#D8D8D8", lw=0.7, zorder=4)
    ax.text(0.04, -0.64, "Motif-peak\nlinks (n)", ha="left", va="center", fontsize=6.7, fontweight="bold", linespacing=0.95)
    ax.text(0.56, -0.64, "ATAC samples", ha="left", va="center", fontsize=6.7, fontweight="bold")
    ax.text(0.80, -0.64, "Promoter-window\nlinks (n)", ha="left", va="center", fontsize=6.7, fontweight="bold", linespacing=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(cand["gene"], fontsize=7, fontstyle="italic")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, :])
    mat = cand.set_index("gene")[
        [
            "n_runs_detected",
            "weighted_importance",
            "partial_pearson_r",
            "stress_ap1_expression_delta",
            "n_ucsc_score_ge_300_motif_peak_links_in_window",
            "n_promoter_2kb_motif_peak_links",
            "n_samples_with_motif_peak_in_window",
        ]
    ].copy()
    display = zscore_columns(mat)
    display.columns = ["Seeds", "Importance", "Partial r\n(relative z)", "Stress delta", "Motif-peak\nlinks (n)", "Promoter-window\nlinks (n)", "ATAC samples"]
    sns.heatmap(
        display,
        cmap="vlag",
        center=0,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Column z (display)"},
        ax=ax,
    )
    for boundary in (2, 11):
        ax.axhline(boundary, color="#D8D8D8", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    for tick in ax.get_yticklabels():
        tick.set_fontstyle("italic")
    panel_label(ax, "c", x=-0.05)

    save_all(fig, "Figure5_multiomic_candidate_prioritisation")


def plot_supplementary_clinical_projection() -> None:
    """Move TCGA clinical and OS projection out of the main figure set."""
    assoc = pd.read_csv(TABLE_DIR / "tcga_stress_ap1_state_signature_clinical_associations.csv")
    cox = pd.read_csv(TABLE_DIR / "tcga_stress_ap1_state_signature_os_cox.csv")

    set_style()
    fig = plt.figure(figsize=(10.8, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.9])

    sub = gs[0, 0].subgridspec(1, 2, width_ratios=[0.75, 0.25], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    clinical = assoc.loc[assoc["endpoint"].isin(["Stage III-IV vs I-II", "N+ vs N0"])].copy()
    clinical["label"] = clinical["cohort"] + ": " + clinical["endpoint"].str.replace(" vs ", " vs\n", regex=False)
    clinical["_endpoint_order"] = clinical["endpoint"].map({"Stage III-IV vs I-II": 0, "N+ vs N0": 1})
    clinical["_cohort_order"] = clinical["cohort"].map({"Combined": 0, "LUAD": 1, "LUSC": 2})
    clinical = clinical.sort_values(["_endpoint_order", "_cohort_order"])
    y = np.arange(len(clinical))
    ax.axvline(0, color="0.75", lw=0.8)
    ax.errorbar(
        clinical["median_z_high_minus_low"],
        y,
        xerr=[
            clinical["median_z_high_minus_low"] - clinical["bootstrap_ci_low"],
            clinical["bootstrap_ci_high"] - clinical["median_z_high_minus_low"],
        ],
        fmt="o",
        color="#55A868",
        ecolor="#55A868",
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(clinical["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("State-signature median z difference")
    finite_ci = clinical[["bootstrap_ci_low", "bootstrap_ci_high"]].to_numpy().ravel()
    finite_ci = finite_ci[np.isfinite(finite_ci)]
    if finite_ci.size:
        ax.set_xlim(min(-0.55, float(finite_ci.min()) - 0.08), max(0.55, float(finite_ci.max()) + 0.08))
    for i, r in enumerate(clinical.itertuples(index=False)):
        ax_ann.text(0, i, f"Holm {format_p(r.holm_adjusted_p_12_tests)}", ha="left", va="center", fontsize=7)
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    panel_label(ax, "a", x=-0.08)

    sub = gs[0, 1].subgridspec(1, 2, width_ratios=[0.62, 0.38], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    c = cox.dropna(subset=["hr_per_sd"]).copy()
    c["_order"] = c["cohort"].map({"Combined": 0, "LUAD": 1, "LUSC": 2})
    c = c.sort_values("_order")
    y = np.arange(len(c))
    ax.axvline(1, color="0.75", lw=0.8)
    ax.errorbar(c["hr_per_sd"], y, xerr=[c["hr_per_sd"] - c["ci_low"], c["ci_high"] - c["hr_per_sd"]], fmt="o", color="#333333", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(c["cohort"], fontsize=7)
    ax.set_xlabel("OS hazard ratio per SD")
    ax.set_xlim(0.75, max(1.45, float(c["ci_high"].max()) + 0.08))
    ax.invert_yaxis()
    for i, r in enumerate(c.itertuples(index=False)):
        ax_ann.text(0, i, f"Holm {format_p(r.holm_adjusted_p_across_os_models)}", ha="left", va="center", fontsize=7)
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    panel_label(ax, "b", x=-0.08)

    save_all(fig, "Supplementary_Figure4_TCGA_clinical_context_projection")


def main() -> None:
    plot_figure1()
    plot_figure3()
    plot_figure4()
    plot_figure5()


if __name__ == "__main__":
    main()
