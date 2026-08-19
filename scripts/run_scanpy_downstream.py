from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns

from project_paths import OUTPUT_DIR

BASE_DIR = OUTPUT_DIR
QC_H5AD = BASE_DIR / "scanpy_qc" / "nsclc_gse131907_gse274934_scanpy_qc_raw_counts_qc_filtered.h5ad"
OUT_DIR = BASE_DIR / "scanpy_downstream_scrublet"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PREFIX = "nsclc_gse131907_gse274934"


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}


MAJOR_MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1", "KRT17"],
    "T/NK": ["CD3D", "CD3E", "TRAC", "NKG7", "GNLY", "KLRD1", "IL7R"],
    "B": ["MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "HLA-DRA"],
    "Plasma": ["MZB1", "JCHAIN", "XBP1", "SDC1", "IGKC", "IGHG1"],
    "Myeloid": ["LYZ", "LST1", "TYROBP", "FCER1G", "C1QA", "C1QB", "S100A8", "S100A9"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2"],
    "Endothelial": ["PECAM1", "VWF", "KDR", "ENG", "EMCN", "ESAM"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "TAGLN", "ACTA2"],
    "Cycling": ["MKI67", "TOP2A", "CENPF", "STMN1", "UBE2C"],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["legend.frameon"] = False


def save_pub(fig: plt.Figure, stem: Path, dpi: int = 450) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def category_palette(categories: list[str]) -> dict[str, str]:
    base = [
        "#0F4D92",
        "#8BCF8B",
        "#B64342",
        "#42949E",
        "#9A4D8E",
        "#E28E2C",
        "#5B8FD6",
        "#7BAA5B",
        "#C45AD6",
        "#767676",
        "#D9544D",
        "#5B7FCA",
        "#B89BD9",
        "#33B5A5",
        "#D8A03D",
        "#6B5B95",
        "#88B04B",
        "#F0C0CC",
        "#7884B4",
        "#606060",
    ]
    if len(categories) <= len(base):
        colors = base[: len(categories)]
    else:
        cmap = plt.get_cmap("hsv")
        extra = [matplotlib.colors.to_hex(cmap(i / (len(categories) - len(base)))) for i in range(len(categories) - len(base))]
        colors = base + extra
    return dict(zip(categories, colors))


def run_scrublet_and_filter(adata: ad.AnnData) -> ad.AnnData:
    """Run Scanpy's built-in Scrublet by sample and remove predicted doublets."""
    log("Running Scanpy Scrublet with batch_key='sample_id'")
    sc.pp.scrublet(
        adata,
        batch_key="sample_id",
        expected_doublet_rate=0.05,
        n_prin_comps=30,
        random_state=7,
        verbose=True,
    )

    summary = (
        adata.obs.groupby(["dataset", "sample", "sample_id", "tissue_status"], observed=True)
        .agg(
            n_cells=("predicted_doublet", "size"),
            n_predicted_doublet=("predicted_doublet", "sum"),
            median_doublet_score=("doublet_score", "median"),
            q95_doublet_score=("doublet_score", lambda x: x.quantile(0.95)),
            q99_doublet_score=("doublet_score", lambda x: x.quantile(0.99)),
        )
        .reset_index()
    )
    summary["predicted_doublet_fraction"] = summary["n_predicted_doublet"] / summary["n_cells"]
    summary.to_csv(TABLE_DIR / f"{PREFIX}_scrublet_doublet_summary.csv", index=False)

    n_doublets = int(adata.obs["predicted_doublet"].sum())
    log(f"Removing {n_doublets:,} Scrublet-predicted doublets")
    return adata[~adata.obs["predicted_doublet"].to_numpy()].copy()


def preprocess_and_integrate(adata: ad.AnnData) -> ad.AnnData:
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    adata.layers["counts"] = adata.X.copy()

    log("Filtering genes detected in fewer than 20 cells")
    sc.pp.filter_genes(adata, min_cells=20)

    log("Normalizing total counts and log-transforming")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    log("Scoring broad marker programs for automatic major-group annotation")
    for group, genes in MAJOR_MARKERS.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) >= 2:
            sc.tl.score_genes(adata, present, score_name=f"score_{group}", use_raw=False)
        else:
            adata.obs[f"score_{group}"] = 0.0

    log("Selecting highly variable genes")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=3000,
        flavor="seurat",
        batch_key="sample_id",
        subset=False,
    )

    log("Running PCA")
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True, svd_solver="arpack")

    log("Running Harmony integration on PCA using sample_id")
    try:
        import scanpy.external as sce

        sce.pp.harmony_integrate(
            adata,
            key="sample_id",
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
            max_iter_harmony=30,
        )
        use_rep = "X_pca_harmony"
        adata.uns["integration_note"] = "Harmony integration succeeded; neighbors/UMAP use X_pca_harmony."
    except Exception as exc:
        log(f"WARNING: Harmony failed: {type(exc).__name__}: {exc}")
        use_rep = "X_pca"
        adata.uns["integration_note"] = f"Harmony failed; neighbors/UMAP use X_pca. Error: {type(exc).__name__}: {exc}"

    log(f"Computing neighbors using {use_rep}")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40, use_rep=use_rep)

    log("Running UMAP")
    sc.tl.umap(adata, min_dist=0.35, spread=1.0, random_state=7)

    log("Running Leiden clustering: major resolution 0.3, subcluster resolution 0.8")
    sc.tl.leiden(adata, resolution=0.3, key_added="leiden_major", random_state=7)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden_sub", random_state=7)

    assign_major_celltypes(adata)
    return adata


def assign_major_celltypes(adata: ad.AnnData) -> None:
    score_cols = [f"score_{k}" for k in MAJOR_MARKERS]
    cluster_score = adata.obs.groupby("leiden_major", observed=True)[score_cols].mean()
    rows = []
    cluster_to_type = {}
    for cluster, row in cluster_score.iterrows():
        best_col = row.astype(float).idxmax()
        best_type = best_col.replace("score_", "")
        best_score = float(row[best_col])
        cluster_to_type[str(cluster)] = best_type
        rows.append(
            {
                "leiden_major": cluster,
                "major_celltype_auto": best_type,
                "best_score": best_score,
                **{c: float(row[c]) for c in score_cols},
            }
        )
    adata.obs["major_celltype_auto"] = adata.obs["leiden_major"].astype(str).map(cluster_to_type).astype("category")
    pd.DataFrame(rows).to_csv(TABLE_DIR / f"{PREFIX}_major_celltype_auto_annotation.csv", index=False)


def run_marker_tests(adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("Running marker differential expression for major clusters")
    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden_major",
        method="wilcoxon",
        use_raw=True,
        pts=True,
        n_genes=200,
        key_added="rank_genes_major",
    )
    major_df = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_major")
    major_df.to_csv(TABLE_DIR / f"{PREFIX}_major_leiden_markers_top200.csv", index=False)

    log("Running marker differential expression for subclusters")
    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden_sub",
        method="wilcoxon",
        use_raw=True,
        pts=True,
        n_genes=200,
        key_added="rank_genes_sub",
    )
    sub_df = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_sub")
    sub_df.to_csv(TABLE_DIR / f"{PREFIX}_subcluster_leiden_markers_top200.csv", index=False)

    log("Running marker differential expression for automatic major cell types")
    sc.tl.rank_genes_groups(
        adata,
        groupby="major_celltype_auto",
        method="wilcoxon",
        use_raw=True,
        pts=True,
        n_genes=200,
        key_added="rank_genes_major_celltype_auto",
    )
    celltype_df = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_major_celltype_auto")
    celltype_df.to_csv(TABLE_DIR / f"{PREFIX}_major_celltype_auto_markers_top200.csv", index=False)

    excel_path = TABLE_DIR / f"{PREFIX}_marker_tables_top200.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        major_df.to_excel(writer, sheet_name="leiden_major", index=False)
        sub_df.to_excel(writer, sheet_name="leiden_sub", index=False)
        celltype_df.to_excel(writer, sheet_name="major_celltype_auto", index=False)
    return major_df, sub_df, celltype_df


def write_cluster_tables(adata: ad.AnnData, major_df: pd.DataFrame, sub_df: pd.DataFrame) -> None:
    obs = adata.obs.copy()
    for key in ["leiden_major", "leiden_sub", "major_celltype_auto"]:
        counts = (
            obs.groupby([key, "dataset", "tissue_status"], observed=True)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        counts.to_csv(TABLE_DIR / f"{PREFIX}_{key}_cell_counts_by_dataset_status.csv", index=False)

    top_major = (
        major_df.sort_values(["group", "scores"], ascending=[True, False])
        .groupby("group", observed=True)
        .head(10)
        .groupby("group", observed=True)["names"]
        .apply(lambda x: ", ".join(x.astype(str)))
        .reset_index(name="top10_marker_genes")
    )
    top_sub = (
        sub_df.sort_values(["group", "scores"], ascending=[True, False])
        .groupby("group", observed=True)
        .head(10)
        .groupby("group", observed=True)["names"]
        .apply(lambda x: ", ".join(x.astype(str)))
        .reset_index(name="top10_marker_genes")
    )
    top_major.to_csv(TABLE_DIR / f"{PREFIX}_leiden_major_top10_markers_by_cluster.csv", index=False)
    top_sub.to_csv(TABLE_DIR / f"{PREFIX}_leiden_sub_top10_markers_by_cluster.csv", index=False)


def plot_umap(
    adata: ad.AnnData,
    color_key: str,
    title: str,
    filename: str,
    point_size: float = 1.0,
    alpha: float = 0.72,
) -> None:
    coords = adata.obsm["X_umap"]
    series = adata.obs[color_key].astype(str)
    categories = sorted(series.unique(), key=lambda x: (len(x), x))
    pal = category_palette(categories)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for cat in categories:
        mask = series.to_numpy() == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            c=pal[cat],
            label=cat,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title, fontsize=8, pad=6)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ncol = 1 if len(categories) <= 12 else 2
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        markerscale=4,
        handletextpad=0.3,
        columnspacing=0.8,
        ncol=ncol,
        fontsize=6,
    )
    save_pub(fig, FIG_DIR / filename)


def plot_marker_dotplot(adata: ad.AnnData) -> None:
    genes = []
    for marker_list in MAJOR_MARKERS.values():
        genes.extend(marker_list[:4])
    genes = [g for g in dict.fromkeys(genes) if g in adata.raw.var_names]
    if not genes:
        return
    log("Saving marker dotplot")
    old_figdir = sc.settings.figdir
    sc.settings.figdir = str(FIG_DIR)
    fig = sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby="major_celltype_auto",
        use_raw=True,
        dendrogram=False,
        standard_scale="var",
        return_fig=True,
    )
    fig.savefig(FIG_DIR / f"{PREFIX}_major_celltype_marker_dotplot.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{PREFIX}_major_celltype_marker_dotplot.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{PREFIX}_major_celltype_marker_dotplot.svg", bbox_inches="tight")
    plt.close("all")
    sc.settings.figdir = old_figdir


def save_figures(adata: ad.AnnData) -> None:
    log("Saving Nature-style UMAP figures")
    plot_umap(
        adata,
        "major_celltype_auto",
        "Major cell groups after Scanpy QC and Harmony integration",
        f"{PREFIX}_umap_major_celltype_auto_nature",
        point_size=1.0,
    )
    plot_umap(
        adata,
        "leiden_major",
        "Coarse Leiden clusters",
        f"{PREFIX}_umap_leiden_major_nature",
        point_size=0.9,
    )
    plot_umap(
        adata,
        "leiden_sub",
        "Subcluster Leiden structure",
        f"{PREFIX}_umap_leiden_subclusters_nature",
        point_size=0.75,
        alpha=0.68,
    )
    plot_umap(
        adata,
        "dataset",
        "Dataset mixing on integrated UMAP",
        f"{PREFIX}_umap_dataset_nature",
        point_size=0.8,
        alpha=0.65,
    )
    plot_umap(
        adata,
        "tissue_status",
        "Tumor and adjacent-normal distribution",
        f"{PREFIX}_umap_tissue_status_nature",
        point_size=0.8,
        alpha=0.65,
    )
    plot_marker_dotplot(adata)


def write_report(adata: ad.AnnData) -> None:
    n_cells, n_genes = adata.shape
    n_major = adata.obs["leiden_major"].nunique()
    n_sub = adata.obs["leiden_sub"].nunique()
    n_auto = adata.obs["major_celltype_auto"].nunique()

    report = [
        "# Scanpy Downstream Analysis Report",
        "",
        "## Executed Steps",
        "",
        "1. Loaded Scanpy-QC-filtered raw-count object.",
        "2. Preserved raw counts in `layers['counts']`.",
        "3. Ran Scanpy's built-in Scrublet by `sample_id` and removed predicted doublets.",
        "4. Filtered genes detected in fewer than 20 cells.",
        "5. Ran `normalize_total(target_sum=1e4)` and `log1p`.",
        "6. Selected 3,000 highly variable genes with `batch_key='sample_id'`.",
        "7. Ran PCA and Harmony using `sample_id`.",
        "8. Computed neighbors, UMAP, Leiden major clusters (`resolution=0.3`) and subclusters (`resolution=0.8`).",
        "9. Assigned automatic broad cell groups from canonical marker scores.",
        "10. Exported marker/DE tables for major clusters, subclusters, and automatic major cell groups.",
        "",
        "## Notes",
        "",
        "- Scrublet was run through Scanpy's built-in `sc.pp.scrublet`. The standalone `scrublet` package could not be installed because its `annoy` dependency requires Microsoft C++ Build Tools.",
        "- Marker differential expression was computed with Scanpy Wilcoxon tests on log-normalized expression (`use_raw=True`), top 200 genes per group.",
        "",
        "## Result",
        "",
        f"- Analysis object: {n_cells:,} cells x {n_genes:,} genes after gene min-cell filtering.",
        f"- Coarse Leiden clusters: {n_major}",
        f"- Subcluster Leiden clusters: {n_sub}",
        f"- Automatic major cell groups: {n_auto}",
        "- Scrublet-predicted doublets were removed before normalization and clustering.",
        "",
        "## Key Files",
        "",
        f"- Analysis object: `{PREFIX}_scanpy_downstream_analysis.h5ad`",
        f"- Major cluster marker table: `tables/{PREFIX}_major_leiden_markers_top200.csv`",
        f"- Subcluster marker table: `tables/{PREFIX}_subcluster_leiden_markers_top200.csv`",
        f"- Excel marker workbook: `tables/{PREFIX}_marker_tables_top200.xlsx`",
        f"- Major-group UMAP: `figures/{PREFIX}_umap_major_celltype_auto_nature.png/.pdf/.svg`",
        f"- Subcluster UMAP: `figures/{PREFIX}_umap_leiden_subclusters_nature.png/.pdf/.svg`",
        "",
    ]
    (OUT_DIR / f"{PREFIX}_scanpy_downstream_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    apply_publication_style()
    sns.set_theme(style="white", context="paper")
    sc.settings.verbosity = 2

    log(f"Reading {QC_H5AD}")
    adata = sc.read_h5ad(QC_H5AD)
    adata = run_scrublet_and_filter(adata)

    singlet_path = OUT_DIR / f"{PREFIX}_qc_counts_scrublet_singlets.h5ad"
    log(f"Writing raw-count singlet object: {singlet_path}")
    adata.write_h5ad(singlet_path, compression="lzf")

    adata = preprocess_and_integrate(adata)
    major_df, sub_df, _ = run_marker_tests(adata)
    write_cluster_tables(adata, major_df, sub_df)
    save_figures(adata)
    write_report(adata)

    out_path = OUT_DIR / f"{PREFIX}_scanpy_downstream_analysis.h5ad"
    log(f"Writing analysis object: {out_path}")
    adata.write_h5ad(out_path, compression="lzf")
    log("Done")


if __name__ == "__main__":
    main()
