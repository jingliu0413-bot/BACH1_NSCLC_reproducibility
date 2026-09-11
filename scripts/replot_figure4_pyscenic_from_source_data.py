"""Replot Figure 4 from the current restricted-consensus pySCENIC results.

The legacy Figure 4 source_data tables were generated from a 6,073-cell
analysis set. The current manuscript defines the pySCENIC sensitivity material
as the restricted consensus malignant epithelial set (4,906 cells), so this
script rebuilds the figure directly from the current h5ad, AUCell, motif
pruning and five-seed recurrence outputs.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
CURRENT_H5AD = ROOT / "out" / "bach1_malignant_epithelial_consensus" / "nsclc_malignant_epithelial_consensus_bach1_analysis_object.h5ad"
CURRENT_AUCELL = (
    ROOT
    / "out"
    / "bach1_malignant_epithelial_consensus_pyscenic_all_tfs"
    / "tables"
    / "pyscenic_selected_tf_regulon_aucell_by_cell.csv.gz"
)
CURRENT_MOTIFS = (
    ROOT
    / "out"
    / "bach1_malignant_epithelial_consensus_pyscenic_all_tfs"
    / "tables"
    / "pyscenic_all_tfs_motif_enrichment_summary.csv"
)
CURRENT_SUMMARY = (
    ROOT
    / "out"
    / "bach1_malignant_epithelial_consensus_pyscenic_all_tfs"
    / "pyscenic_all_tfs_selected_regulon_summary.json"
)
SEED_STABILITY_DIR = ROOT / "out" / "revision_diagnostics" / "pyscenic_bach1_consensus_full_tf_seed_stability_v1"
CURRENT_TARGET_FREQUENCY = SEED_STABILITY_DIR / "pyscenic_bach1_seed_stability_target_frequency.csv"
CURRENT_RUN_SUMMARY = SEED_STABILITY_DIR / "pyscenic_bach1_seed_stability_run_summary.csv"
CURRENT_JACCARD = SEED_STABILITY_DIR / "pyscenic_bach1_seed_stability_pairwise_jaccard.csv"

OUT_FIG_DIR = ROOT / "out" / "nature_story_4figures_10kb" / "figures"
OUT_FINAL_DIR = OUT_FIG_DIR / "publication_figures"
PACKAGE_DIR = ROOT / "outputs" / "submission_ready_20260911"
PACKAGE_FIG_DIR = PACKAGE_DIR / "figures"
PACKAGE_TABLE_DIR = PACKAGE_DIR / "tables"
PACKAGE_REPORT_DIR = PACKAGE_DIR / "reports"

COL = {
    "dark": "#272727",
    "grey": "#767676",
    "light": "#E5E7EB",
    "blue": "#4E79A7",
    "blue_soft": "#B4C0E4",
    "red": "#D9544D",
    "violet": "#9A4D8E",
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)


def wrap(value: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def add_panel_label(ax, label: str, x: float = -0.09, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


def format_umap_axes(ax) -> None:
    ax.set_xlabel("UMAP1", labelpad=1)
    ax.set_ylabel("UMAP2", labelpad=1)
    ax.tick_params(axis="both", labelsize=5, width=0.45, length=2.2)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")


def motif_context_label(context: str) -> str:
    tags: list[str] = []
    text = str(context)
    if "10kbp_up_10kbp_down" in text:
        tags.append("10 kb")
    if "500bp_up_100bp_down" in text:
        tags.append("promoter")
    if "top10perTarget" in text:
        tags.append("top10")
    if "top50" in text:
        tags.append("top50")
    if "weight>90.0%" in text:
        tags.append("w>90%")
    elif "weight>75.0%" in text:
        tags.append("w>75%")
    return ", ".join(tags) if tags else "cisTarget"


def numeric_gene_matrix(expr_source, genes: list[str]) -> np.ndarray:
    mat = expr_source[:, genes].X
    if sparse.issparse(mat):
        mat = mat.toarray()
    return np.asarray(mat, dtype=float)


def build_umap_aucell(adata, aucell: pd.DataFrame) -> pd.DataFrame:
    if "cell_id" not in aucell.columns or "BACH1" not in aucell.columns:
        raise ValueError("Current AUCell table must contain cell_id and BACH1 columns.")
    aucell = aucell.set_index("cell_id", drop=False)
    missing = adata.obs_names.difference(aucell.index)
    if len(missing):
        raise ValueError(f"AUCell table is missing {len(missing)} restricted-consensus cells.")

    coords = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names, columns=["UMAP1", "UMAP2"])
    keep_obs = ["dataset", "sample_id", "sample", "patient", "tissue_status", "BACH1_expr"]
    obs_cols = [c for c in keep_obs if c in adata.obs.columns]
    out = coords.join(adata.obs[obs_cols])
    out.insert(0, "cell_id", out.index.astype(str))
    out["BACH1_expr"] = pd.to_numeric(out["BACH1_expr"], errors="coerce")
    out["BACH1_detected"] = out["BACH1_expr"].gt(0)
    out["BACH1_regulon_AUC"] = pd.to_numeric(aucell.loc[out.index, "BACH1"], errors="coerce").to_numpy()
    return out.reset_index(drop=True)


def select_recurrent_genes(target_frequency: pd.DataFrame, var_names: pd.Index, n: int = 18) -> list[str]:
    stable = target_frequency[target_frequency["frequency"].ge(0.6)].copy()
    stable = stable.sort_values(["frequency", "weighted_importance", "max_importance"], ascending=False)
    genes = [gene for gene in stable["gene"].astype(str) if gene in var_names]
    if len(genes) < n:
        fallback = target_frequency.sort_values(["frequency", "weighted_importance", "max_importance"], ascending=False)
        for gene in fallback["gene"].astype(str):
            if gene in var_names and gene not in genes:
                genes.append(gene)
            if len(genes) >= n:
                break
    if len(genes) < 8:
        raise ValueError("Too few recurrent BACH1 candidate genes were found in the expression matrix.")
    return genes[:n]


def build_coexpression_heatmap(adata, target_frequency: pd.DataFrame) -> pd.DataFrame:
    expr_source = adata.raw if adata.raw is not None else adata
    var_names = pd.Index(expr_source.var_names.astype(str))
    genes = select_recurrent_genes(target_frequency, var_names, n=18)

    bach1_expr = pd.to_numeric(adata.obs["BACH1_expr"], errors="coerce").fillna(0)
    ordered_cells = bach1_expr.sort_values(kind="mergesort").index
    n_cells = len(ordered_cells)
    labels = np.array(["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
    bin_index = np.minimum((np.arange(n_cells) * len(labels) / n_cells).astype(int), len(labels) - 1)
    bin_by_cell = pd.Series(labels[bin_index], index=ordered_cells, name="BACH1_expression_bin")

    mat = numeric_gene_matrix(expr_source, genes)
    expr = pd.DataFrame(mat, index=adata.obs_names, columns=genes).loc[ordered_cells]
    means = expr.groupby(bin_by_cell, sort=False).mean().T
    means = means.reindex(columns=list(labels))
    row_sd = means.std(axis=1, ddof=0).replace(0, np.nan)
    z = means.sub(means.mean(axis=1), axis=0).div(row_sd, axis=0).fillna(0)
    z.insert(0, "gene", z.index)
    return z.reset_index(drop=True)


def build_motif_pruning(motif_summary: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    motifs = motif_summary[motif_summary["TF"].astype(str).eq("BACH1")].copy()
    if motifs.empty:
        raise ValueError("No BACH1 motif-pruning rows were found in the current motif summary.")
    motifs["NES"] = pd.to_numeric(motifs["NES"], errors="coerce")
    motifs["n_targets"] = pd.to_numeric(motifs["n_targets"], errors="coerce")
    motifs = motifs.sort_values("NES", ascending=False).head(n).reset_index(drop=True)
    motifs["short_context"] = motifs["Context"].map(motif_context_label)
    return motifs


def build_recurrent_targets(target_frequency: pd.DataFrame, n: int = 18) -> pd.DataFrame:
    targets = target_frequency[target_frequency["frequency"].ge(0.6)].copy()
    if targets.empty:
        raise ValueError("No BACH1 targets recurrent in >=3/5 seeds were found.")
    for col in ["frequency", "weighted_importance", "max_importance", "n_runs_detected", "max_NES"]:
        targets[col] = pd.to_numeric(targets[col], errors="coerce")
    targets = targets.sort_values(["frequency", "weighted_importance", "max_importance"], ascending=False)
    return targets.head(n).reset_index(drop=True)


def plot_coexpression(ax, heatmap: pd.DataFrame) -> None:
    df = heatmap.set_index("gene")
    cols = [c for c in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"] if c in df.columns]
    z = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(["low", "Q2", "Q3", "Q4", "Q5", "high"], fontsize=6)
    ax.set_yticks(np.arange(z.shape[0]))
    ax.set_yticklabels(z.index, fontsize=5.6)
    ax.set_xlabel("BACH1 expression bin")
    ax.set_title("Co-expression", loc="left", fontsize=8, pad=2)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.020)
    cb.set_label("row z-score", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)


def plot_umap(ax, umap: pd.DataFrame) -> None:
    vals = pd.to_numeric(umap["BACH1_regulon_AUC"], errors="coerce")
    ax.scatter(umap["UMAP1"], umap["UMAP2"], s=2.1, c="#D9D9D9", alpha=0.25, linewidths=0, rasterized=True)
    sca = ax.scatter(
        umap["UMAP1"],
        umap["UMAP2"],
        s=2.4,
        c=vals,
        cmap="magma",
        vmin=np.nanquantile(vals, 0.02),
        vmax=np.nanquantile(vals, 0.98),
        alpha=0.86,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("AUCell UMAP", loc="left", fontsize=8, pad=2)
    format_umap_axes(ax)
    cb = plt.colorbar(sca, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)


def plot_aucell_distribution(ax, umap: pd.DataFrame) -> None:
    auc = pd.to_numeric(umap["BACH1_regulon_AUC"], errors="coerce").dropna()
    ax.hist(auc, bins=48, color=COL["blue_soft"], edgecolor="white", linewidth=0.35)
    ax.set_xlabel("BACH1 regulon AUCell")
    ax.set_ylabel("cells")
    ax.set_title("AUCell distribution", loc="left", fontsize=8, pad=2)
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)


def plot_motif_pruning(ax, motif_df: pd.DataFrame) -> None:
    df = motif_df.copy()
    df["label"] = [f"{row.MotifID} ({row.short_context})" for row in df.itertuples(index=False)]
    df = df.sort_values("NES", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    ax.barh(y, df["NES"], color=COL["violet"], height=0.58)
    ax.axvline(3.0, color=COL["grey"], lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(x, 30) for x in df["label"]], fontsize=5.4)
    ax.set_xlabel("cisTarget NES")
    ax.set_title("Motif pruning", loc="left", fontsize=8, pad=2)
    ax.set_xlim(2.95, max(4.05, df["NES"].max() + 0.08))
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    for yi, nes, n_targets in zip(y, df["NES"], df["n_targets"]):
        ax.text(nes + 0.014, yi, f"{int(n_targets)} targets", va="center", fontsize=5.3)


def plot_targets(ax, targets: pd.DataFrame) -> None:
    df = targets.copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    x = pd.to_numeric(df["weighted_importance"], errors="coerce")
    sizes = 18 + pd.to_numeric(df["n_runs_detected"], errors="coerce").fillna(0) * 12
    sca = ax.scatter(
        x,
        y,
        s=sizes,
        c=pd.to_numeric(df["frequency"], errors="coerce"),
        cmap=mpl.colors.LinearSegmentedColormap.from_list("freq", [COL["blue_soft"], COL["red"]]),
        vmin=0.6,
        vmax=1.0,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.hlines(y, 0, x, color=COL["light"], lw=1.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=5.8)
    ax.set_xlabel("recurrence-weighted importance")
    ax.set_title("Recurrent targets", loc="left", fontsize=8, pad=2)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    cb = plt.colorbar(sca, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("seed frequency", fontsize=5.4)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)


def save_outputs(fig) -> None:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure2_pyscenic_regulon_reordered"
    for ext in ["pdf", "png", "svg"]:
        path = OUT_FIG_DIR / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    shutil.copy2(OUT_FIG_DIR / f"{stem}.pdf", OUT_FINAL_DIR / "figure2_pyscenic_regulon.pdf")
    shutil.copy2(OUT_FIG_DIR / f"{stem}.pdf", PACKAGE_FIG_DIR / "Figure4_BACH1_coexpression_pyscenic.pdf")
    shutil.copy2(OUT_FIG_DIR / f"{stem}.png", PACKAGE_FIG_DIR / "Figure4_BACH1_coexpression_pyscenic.png")
    shutil.copy2(OUT_FIG_DIR / f"{stem}.svg", PACKAGE_FIG_DIR / "Figure4_BACH1_coexpression_pyscenic.svg")


def write_source_tables(
    heatmap: pd.DataFrame,
    umap: pd.DataFrame,
    motifs: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    PACKAGE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    heatmap.to_csv(PACKAGE_TABLE_DIR / "figure4a_current_restricted_consensus_coexpression_heatmap.csv", index=False)
    umap.to_csv(PACKAGE_TABLE_DIR / "figure4b_current_restricted_consensus_umap_aucell.csv.gz", index=False)
    umap[["cell_id", "BACH1_regulon_AUC"]].to_csv(
        PACKAGE_TABLE_DIR / "figure4c_current_restricted_consensus_aucell_distribution.csv", index=False
    )
    motifs.to_csv(PACKAGE_TABLE_DIR / "figure4d_current_bach1_motif_pruning.csv", index=False)
    targets.to_csv(PACKAGE_TABLE_DIR / "figure4e_current_bach1_recurrent_targets.csv", index=False)


def write_audit(umap: pd.DataFrame, target_frequency: pd.DataFrame) -> None:
    PACKAGE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(CURRENT_SUMMARY.read_text())
    run_summary = pd.read_csv(CURRENT_RUN_SUMMARY)
    jaccard = pd.read_csv(CURRENT_JACCARD)
    n_cells = len(umap)
    n_detected = int(umap["BACH1_detected"].sum())
    n_undetected = n_cells - n_detected
    n_union = len(target_frequency)
    n_ge3 = int(target_frequency["frequency"].ge(0.6).sum())
    n_core = int(target_frequency["frequency"].eq(1.0).sum())
    mean_jaccard = float(pd.to_numeric(jaccard["jaccard"], errors="coerce").mean())
    audit = f"""# Figure 4 source-data version audit

This audit records the inputs used to rebuild Figure 4 after removing the legacy 6,073-cell BACH1 transcript-detection panel.

Restricted consensus cell set: {n_cells:,} cells ({n_undetected:,} BACH1-undetected and {n_detected:,} BACH1-detected by transcript count >0).

| Panel | Input file | Analysis set | Cells | Random seed / summary |
| --- | --- | --- | ---: | --- |
| 4a | `{CURRENT_H5AD.relative_to(ROOT)}` plus `{CURRENT_TARGET_FREQUENCY.relative_to(ROOT)}` | restricted consensus malignant epithelium | {n_cells:,} | deterministic expression-bin summary; genes selected from current five-seed recurrent candidates |
| 4b | `{CURRENT_H5AD.relative_to(ROOT)}` plus `{CURRENT_AUCELL.relative_to(ROOT)}` | restricted consensus malignant epithelium | {n_cells:,} | seed777 full-TF pySCENIC run, shown for visualization |
| 4c | `{CURRENT_AUCELL.relative_to(ROOT)}` | restricted consensus malignant epithelium | {n_cells:,} | seed777 continuous AUCell distribution; no legacy 187-cell threshold annotation displayed |
| 4d | `{CURRENT_MOTIFS.relative_to(ROOT)}` | current BACH1 motif-pruned regulon table | - | seed777 full-TF motif-pruning output; {summary['n_bach1_unique_targets']} BACH1 unique targets |
| 4e | `{CURRENT_TARGET_FREQUENCY.relative_to(ROOT)}` | five-seed full-TF recurrence audit | - | seeds {', '.join(run_summary['run'].astype(str))}; {n_union} union targets, {n_core} core targets, {n_ge3} targets in >=3/5 seeds; mean pairwise Jaccard={mean_jaccard:.3f} |

Legacy files in `source_data/figure2_panel_*` are not used by the revised Figure 4. Seed 777 is shown for panels 4b-d as an illustrative visualization run, whereas cross-seed robustness is summarized independently in panel 4e. The former cell-level AUCell-by-BACH1-transcript panel was deleted instead of redrawn because it was descriptive, added limited support to the main inference, and carried the highest version-mismatch risk.
"""
    (PACKAGE_REPORT_DIR / "figure4_source_data_version_audit.md").write_text(audit)
    (OUT_FIG_DIR / "figure4_source_data_version_audit.md").write_text(audit)


def main() -> None:
    adata = sc.read_h5ad(CURRENT_H5AD)
    aucell = pd.read_csv(CURRENT_AUCELL)
    motif_summary = pd.read_csv(CURRENT_MOTIFS)
    target_frequency = pd.read_csv(CURRENT_TARGET_FREQUENCY)

    if adata.n_obs != 4906:
        raise ValueError(f"Expected 4,906 restricted-consensus cells, found {adata.n_obs}.")

    umap = build_umap_aucell(adata, aucell)
    heatmap = build_coexpression_heatmap(adata, target_frequency)
    motifs = build_motif_pruning(motif_summary)
    targets = build_recurrent_targets(target_frequency)

    write_source_tables(heatmap, umap, motifs, targets)
    write_audit(umap, target_frequency)

    fig = plt.figure(figsize=(12.2, 7.1))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.04], hspace=0.48, wspace=0.76)

    ax = fig.add_subplot(gs[0, 0:2])
    plot_coexpression(ax, heatmap)
    add_panel_label(ax, "a", x=-0.10)

    ax = fig.add_subplot(gs[0, 2:4])
    plot_umap(ax, umap)
    add_panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 4:6])
    plot_aucell_distribution(ax, umap)
    add_panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 0:3])
    plot_motif_pruning(ax, motifs)
    add_panel_label(ax, "d", x=-0.07)

    ax = fig.add_subplot(gs[1, 3:6])
    plot_targets(ax, targets)
    add_panel_label(ax, "e", x=-0.09)

    fig.subplots_adjust(top=0.96)
    save_outputs(fig)
    plt.close(fig)
    print(PACKAGE_FIG_DIR / "Figure4_BACH1_coexpression_pyscenic.pdf")


if __name__ == "__main__":
    main()
