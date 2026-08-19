from __future__ import annotations

import json
import time
from pathlib import Path

import infercnvpy as cnv
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from project_paths import OUTPUT_DIR

BASE_DIR = OUTPUT_DIR
INPUT_H5AD = (
    BASE_DIR
    / "scanpy_downstream_scrublet"
    / "nsclc_gse131907_gse274934_scanpy_downstream_analysis.h5ad"
)
OUT_DIR = BASE_DIR / "epithelial_cnv_infercnvpy"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PREFIX = "nsclc_gse131907_gse274934_epithelial_cnv"

RANDOM_STATE = 7
REFERENCE_FRACTION = 0.05
AUTOSOMES = [str(i) for i in range(1, 23)]
AUTOSOMES_CHR = [f"chr{i}" for i in range(1, 23)]


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


def sample_tnk_reference(obs: pd.DataFrame) -> list[str]:
    rng = np.random.default_rng(RANDOM_STATE)
    tnk_obs = obs.loc[obs["major_celltype_auto"] == "T/NK"].copy()
    selected: list[str] = []
    for _, sub in tnk_obs.groupby("sample_id", sort=False):
        n = int(round(len(sub) * REFERENCE_FRACTION))
        if len(sub) > 0:
            n = max(1, min(n, len(sub)))
        if n:
            selected.extend(rng.choice(sub.index.to_numpy(), size=n, replace=False).tolist())
    return selected


def load_epithelial_and_reference() -> sc.AnnData:
    log(f"Reading metadata from {INPUT_H5AD}")
    backed = sc.read_h5ad(INPUT_H5AD, backed="r")
    obs = backed.obs.copy()

    epithelial_cells = obs.index[obs["major_celltype_auto"] == "Epithelial"].tolist()
    reference_cells = sample_tnk_reference(obs)
    selected_cells = epithelial_cells + reference_cells

    ref_table = obs.loc[reference_cells, ["dataset", "sample", "sample_id", "tissue_status"]].copy()
    ref_table.insert(0, "cell_id", ref_table.index)
    ref_table.to_csv(TABLE_DIR / f"{PREFIX}_selected_tnk_reference_cells.csv", index=False)

    log(
        "Subsetting to "
        f"{len(epithelial_cells):,} epithelial cells and "
        f"{len(reference_cells):,} T/NK reference cells"
    )
    adata = backed[selected_cells, :].to_memory()
    backed.file.close()

    adata.obs["cnv_input_group"] = "Epithelial"
    adata.obs.loc[reference_cells, "cnv_input_group"] = "T/NK_reference"
    adata.obs["cnv_input_group"] = pd.Categorical(
        adata.obs["cnv_input_group"], categories=["T/NK_reference", "Epithelial"]
    )

    # Keep the integrated UMAP for later projection plots, but discard large
    # neighborhood/PCA payloads from the full-object workflow.
    for key in list(adata.obsm.keys()):
        if key != "X_umap":
            del adata.obsm[key]
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]
    for key in list(adata.uns.keys()):
        if key.startswith("rank_genes_groups") or key.endswith("_colors") or key == "neighbors":
            del adata.uns[key]

    return adata


def biomart_gene_positions() -> pd.DataFrame:
    cache = TABLE_DIR / f"{PREFIX}_biomart_gene_positions_raw.csv.gz"
    if cache.exists():
        log(f"Reading cached BioMart gene positions: {cache}")
        annot = pd.read_csv(cache)
    else:
        log("Querying Ensembl BioMart gene positions")
        annot = sc.queries.biomart_annotations(
            "hsapiens",
            [
                "ensembl_gene_id",
                "hgnc_symbol",
                "start_position",
                "end_position",
                "chromosome_name",
            ],
            use_cache=True,
        )
        annot.to_csv(cache, index=False)

    annot = annot.rename(
        columns={
            "start_position": "start",
            "end_position": "end",
            "chromosome_name": "chromosome",
        }
    )
    annot["chromosome"] = annot["chromosome"].astype(str)
    annot = annot.loc[annot["chromosome"].isin(AUTOSOMES)].copy()
    annot["chromosome"] = "chr" + annot["chromosome"]
    annot["start"] = pd.to_numeric(annot["start"], errors="coerce")
    annot["end"] = pd.to_numeric(annot["end"], errors="coerce")
    annot = annot.dropna(subset=["start", "end", "chromosome"])
    annot["start"] = annot["start"].astype(int)
    annot["end"] = annot["end"].astype(int)
    annot["chrom_order"] = annot["chromosome"].str.replace("chr", "", regex=False).astype(int)
    annot = annot.sort_values(["chrom_order", "start", "end"])

    ensembl_map = (
        annot.dropna(subset=["ensembl_gene_id"])
        .drop_duplicates("ensembl_gene_id", keep="first")
        .set_index("ensembl_gene_id")[["chromosome", "start", "end", "hgnc_symbol"]]
    )
    symbol_map = (
        annot.dropna(subset=["hgnc_symbol"])
        .loc[lambda x: x["hgnc_symbol"].astype(str).str.len() > 0]
        .drop_duplicates("hgnc_symbol", keep="first")
        .set_index("hgnc_symbol")[["chromosome", "start", "end", "ensembl_gene_id"]]
    )

    return ensembl_map, symbol_map


def add_gene_positions(adata: sc.AnnData) -> sc.AnnData:
    ensembl_map, symbol_map = biomart_gene_positions()
    var = adata.var.copy()

    var["cnv_gene_id_source"] = pd.NA
    for col in ["chromosome", "start", "end"]:
        var[col] = pd.NA

    if "gene_ids" in var.columns:
        ens_join = var[["gene_ids"]].join(ensembl_map, on="gene_ids")
        hit = ens_join["chromosome"].notna()
        for col in ["chromosome", "start", "end"]:
            var.loc[hit, col] = ens_join.loc[hit, col]
        var.loc[hit, "cnv_gene_id_source"] = "ensembl_gene_id"

    miss = var["chromosome"].isna()
    symbol_join = var.loc[miss, ["gene_symbol"]].join(symbol_map, on="gene_symbol")
    hit_symbol = symbol_join["chromosome"].notna()
    hit_index = symbol_join.index[hit_symbol]
    for col in ["chromosome", "start", "end"]:
        var.loc[hit_index, col] = symbol_join.loc[hit_index, col]
    var.loc[hit_index, "cnv_gene_id_source"] = "hgnc_symbol_fallback"

    var["start"] = pd.to_numeric(var["start"], errors="coerce")
    var["end"] = pd.to_numeric(var["end"], errors="coerce")
    var["chrom_order"] = (
        var["chromosome"].astype(str).str.replace("chr", "", regex=False).replace("nan", np.nan)
    )
    var["chrom_order"] = pd.to_numeric(var["chrom_order"], errors="coerce")
    adata.var = var

    mapped = adata.var["chromosome"].notna()
    summary = pd.DataFrame(
        [
            {"metric": "input_genes", "value": int(adata.n_vars)},
            {"metric": "mapped_autosomal_genes", "value": int(mapped.sum())},
            {"metric": "unmapped_or_non_autosomal_genes", "value": int((~mapped).sum())},
            {
                "metric": "mapped_by_ensembl_gene_id",
                "value": int((adata.var["cnv_gene_id_source"] == "ensembl_gene_id").sum()),
            },
            {
                "metric": "mapped_by_hgnc_symbol_fallback",
                "value": int((adata.var["cnv_gene_id_source"] == "hgnc_symbol_fallback").sum()),
            },
        ]
    )
    summary.to_csv(TABLE_DIR / f"{PREFIX}_gene_position_mapping_summary.csv", index=False)

    keep = mapped.to_numpy()
    log(f"Keeping {int(keep.sum()):,} autosomal genes with genomic positions for CNV")
    adata = adata[:, keep].copy()
    order = adata.var.sort_values(["chrom_order", "start", "end"]).index
    adata = adata[:, order].copy()
    adata.var["start"] = adata.var["start"].astype(int)
    adata.var["end"] = adata.var["end"].astype(int)
    adata.var["chromosome"] = pd.Categorical(
        adata.var["chromosome"], categories=AUTOSOMES_CHR, ordered=True
    )
    return adata


def dense_row_mean_abs(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(abs(matrix).mean(axis=1)).ravel()
    return np.mean(np.abs(matrix), axis=1)


def classify_cnv(adata: sc.AnnData) -> dict[str, float]:
    adata.obs["cnv_burden"] = dense_row_mean_abs(adata.obsm["X_cnv"])

    ref_mask = adata.obs["cnv_input_group"].astype(str).to_numpy() == "T/NK_reference"
    ref_scores = adata.obs.loc[ref_mask, "cnv_burden"].to_numpy()
    ref_median = float(np.median(ref_scores))
    ref_mad = float(np.median(np.abs(ref_scores - ref_median)))
    ref_mad_scaled = 1.4826 * ref_mad
    threshold_q99 = float(np.quantile(ref_scores, 0.99))
    threshold_mad = float(ref_median + 3 * ref_mad_scaled)
    threshold = float(max(threshold_q99, threshold_mad))

    adata.obs["cnv_cellwise_high"] = adata.obs["cnv_burden"] >= threshold

    obs = adata.obs.copy()
    obs["is_epithelial"] = obs["cnv_input_group"].astype(str) == "Epithelial"
    cluster_summary = (
        obs.groupby("cnv_leiden", observed=True)
        .agg(
            n_cells=("cnv_burden", "size"),
            n_epithelial=("is_epithelial", "sum"),
            n_reference=("is_epithelial", lambda x: int((~x).sum())),
            mean_cnv_burden=("cnv_burden", "mean"),
            median_cnv_burden=("cnv_burden", "median"),
            q90_cnv_burden=("cnv_burden", lambda x: float(np.quantile(x, 0.90))),
            fraction_cellwise_high=("cnv_cellwise_high", "mean"),
        )
        .reset_index()
    )
    cluster_summary["fraction_epithelial"] = (
        cluster_summary["n_epithelial"] / cluster_summary["n_cells"]
    )
    cluster_summary["cnv_cluster_call"] = np.where(
        (cluster_summary["n_epithelial"] >= 20)
        & (cluster_summary["median_cnv_burden"] >= threshold),
        "malignant_candidate_cluster",
        "reference_like_or_low_cnv_cluster",
    )
    cluster_summary.to_csv(TABLE_DIR / f"{PREFIX}_cnv_cluster_summary.csv", index=False)

    malignant_clusters = set(
        cluster_summary.loc[
            cluster_summary["cnv_cluster_call"] == "malignant_candidate_cluster", "cnv_leiden"
        ].astype(str)
    )
    is_epi = adata.obs["cnv_input_group"].astype(str) == "Epithelial"
    in_malignant_cluster = adata.obs["cnv_leiden"].astype(str).isin(malignant_clusters)
    adata.obs["cnv_malignancy_call"] = "T/NK_reference"
    adata.obs.loc[is_epi & in_malignant_cluster, "cnv_malignancy_call"] = "malignant_epithelial"
    adata.obs.loc[is_epi & ~in_malignant_cluster, "cnv_malignancy_call"] = "nonmalignant_epithelial"
    adata.obs["cnv_malignancy_call"] = pd.Categorical(
        adata.obs["cnv_malignancy_call"],
        categories=["T/NK_reference", "nonmalignant_epithelial", "malignant_epithelial"],
    )

    thresholds = {
        "reference_median": ref_median,
        "reference_mad_scaled": ref_mad_scaled,
        "reference_q99": threshold_q99,
        "reference_median_plus_3mad": threshold_mad,
        "primary_threshold": threshold,
    }
    with open(TABLE_DIR / f"{PREFIX}_cnv_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)
    return thresholds


def write_result_tables(adata: sc.AnnData) -> None:
    obs_cols = [
        "dataset",
        "sample",
        "sample_id",
        "tissue_status",
        "major_celltype_auto",
        "leiden_major",
        "leiden_sub",
        "cnv_input_group",
        "cnv_leiden",
        "cnv_score",
        "cnv_burden",
        "cnv_cellwise_high",
        "cnv_malignancy_call",
    ]
    available = [c for c in obs_cols if c in adata.obs.columns]
    cell_calls = adata.obs[available].copy()
    cell_calls.insert(0, "cell_id", cell_calls.index)
    cell_calls.to_csv(TABLE_DIR / f"{PREFIX}_cell_cnv_calls.csv.gz", index=False)

    epithelial_calls = cell_calls.loc[cell_calls["cnv_input_group"].astype(str) == "Epithelial"]
    epithelial_calls.to_csv(
        TABLE_DIR / f"{PREFIX}_epithelial_cell_cnv_calls.csv.gz", index=False
    )

    malignant = epithelial_calls.loc[
        epithelial_calls["cnv_malignancy_call"].astype(str) == "malignant_epithelial",
        "cell_id",
    ]
    malignant.to_csv(
        TABLE_DIR / f"{PREFIX}_malignant_epithelial_cell_ids.txt",
        index=False,
        header=False,
    )

    sample_summary = (
        epithelial_calls.groupby(["dataset", "sample_id", "tissue_status", "cnv_malignancy_call"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    sample_summary.to_csv(
        TABLE_DIR / f"{PREFIX}_epithelial_cnv_calls_by_sample.csv", index=False
    )

    call_summary = (
        cell_calls.groupby(["cnv_input_group", "cnv_malignancy_call"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    call_summary.to_csv(TABLE_DIR / f"{PREFIX}_cnv_call_summary.csv", index=False)


def plot_embedding(
    adata: sc.AnnData,
    basis_key: str,
    color: str,
    stem: Path,
    title: str,
    categorical: bool = True,
) -> None:
    coords = adata.obsm[basis_key]
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    if categorical:
        palette = {
            "T/NK_reference": "#777777",
            "nonmalignant_epithelial": "#4DBBD5",
            "malignant_epithelial": "#E64B35",
        }
        values = adata.obs[color].astype(str)
        for label in [k for k in palette if k in set(values)]:
            idx = values == label
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                s=1.2,
                c=palette[label],
                label=label,
                linewidths=0,
                alpha=0.85,
                rasterized=True,
            )
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=4, handletextpad=0.2)
    else:
        sca = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=1.2,
            c=adata.obs[color].to_numpy(),
            cmap="magma",
            linewidths=0,
            alpha=0.85,
            rasterized=True,
        )
        cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color)
    ax.set_title(title, pad=8)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    save_pub(fig, stem)


def plot_burden_violin(adata: sc.AnnData) -> None:
    order = ["T/NK_reference", "nonmalignant_epithelial", "malignant_epithelial"]
    data = [
        adata.obs.loc[adata.obs["cnv_malignancy_call"].astype(str) == group, "cnv_burden"].to_numpy()
        for group in order
    ]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    parts = ax.violinplot(data, showmeans=False, showextrema=False, showmedians=True)
    colors = ["#777777", "#4DBBD5", "#E64B35"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.75)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Mean absolute CNV signal")
    ax.set_title("CNV burden by inferred status", pad=8)
    save_pub(fig, FIG_DIR / f"{PREFIX}_cnv_burden_violin")


def plot_chromosome_summary(adata: sc.AnnData) -> None:
    try:
        cnv.pl.chromosome_heatmap_summary(
            adata,
            groupby="cnv_malignancy_call",
            use_rep="cnv",
            figsize=(8.5, 2.8),
            show=False,
        )
        fig = plt.gcf()
        save_pub(fig, FIG_DIR / f"{PREFIX}_chromosome_heatmap_summary_by_call")
    except Exception as exc:
        log(f"Could not save chromosome heatmap summary by call: {type(exc).__name__}: {exc}")

    try:
        cnv.pl.chromosome_heatmap_summary(
            adata,
            groupby="cnv_leiden",
            use_rep="cnv",
            figsize=(10.0, 4.8),
            show=False,
        )
        fig = plt.gcf()
        save_pub(fig, FIG_DIR / f"{PREFIX}_chromosome_heatmap_summary_by_cnv_leiden")
    except Exception as exc:
        log(f"Could not save chromosome heatmap summary by CNV Leiden: {type(exc).__name__}: {exc}")


def write_report(adata: sc.AnnData, thresholds: dict[str, float]) -> None:
    counts = adata.obs["cnv_malignancy_call"].value_counts().rename_axis("call").reset_index(name="n_cells")
    counts_md = ["| call | n_cells |", "|---|---:|"]
    for row in counts.itertuples(index=False):
        counts_md.append(f"| {row.call} | {int(row.n_cells)} |")
    report = [
        "# Epithelial CNV Inference Report",
        "",
        f"- Input object: `{INPUT_H5AD}`",
        f"- Output object: `{OUT_DIR / (PREFIX + '_analysis.h5ad')}`",
        f"- Cells in CNV input: {adata.n_obs:,}",
        f"- Genes used for CNV: {adata.n_vars:,} autosomal genes with genomic positions",
        f"- T/NK reference fraction: {REFERENCE_FRACTION:.1%}, stratified by sample_id",
        f"- Reference q99 threshold: {thresholds['reference_q99']:.6f}",
        f"- Reference median + 3*MAD threshold: {thresholds['reference_median_plus_3mad']:.6f}",
        f"- Primary CNV burden threshold: {thresholds['primary_threshold']:.6f}",
        "",
        "## CNV Calls",
        "",
        "\n".join(counts_md),
        "",
        "Primary malignant calls are epithelial cells assigned to CNV Leiden clusters whose median CNV burden exceeds the T/NK-reference-derived threshold.",
        "Cellwise CNV-high flags and continuous CNV burden values are retained for sensitivity checks and later figure generation.",
        "",
    ]
    (OUT_DIR / f"{PREFIX}_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    apply_publication_style()
    params = {
        "input_h5ad": str(INPUT_H5AD),
        "reference_fraction": REFERENCE_FRACTION,
        "random_state": RANDOM_STATE,
        "infercnv": {
            "window_size": 100,
            "step": 10,
            "dynamic_threshold": 1.5,
            "lfc_clip": 3,
            "reference_key": "cnv_input_group",
            "reference_cat": "T/NK_reference",
            "n_jobs": 1,
            "chunksize": 4000,
        },
    }
    with open(OUT_DIR / f"{PREFIX}_params.json", "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)

    adata = load_epithelial_and_reference()
    adata = add_gene_positions(adata)

    input_h5ad = OUT_DIR / f"{PREFIX}_input_epithelial_tnk_reference_autosomes.h5ad"
    log(f"Writing CNV input subset: {input_h5ad}")
    adata.write_h5ad(input_h5ad)

    log("Running infercnvpy.tl.infercnv with T/NK reference")
    cnv.tl.infercnv(
        adata,
        reference_key="cnv_input_group",
        reference_cat="T/NK_reference",
        lfc_clip=3,
        window_size=100,
        step=10,
        dynamic_threshold=1.5,
        exclude_chromosomes=None,
        chunksize=4000,
        n_jobs=1,
        inplace=True,
        key_added="cnv",
        calculate_gene_values=False,
    )

    log("Running CNV PCA, neighbors, UMAP and Leiden")
    cnv.tl.pca(adata, n_comps=30, random_state=RANDOM_STATE)
    cnv.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    cnv.tl.leiden(adata, resolution=0.5, random_state=RANDOM_STATE)
    cnv.tl.umap(adata, random_state=RANDOM_STATE)
    cnv.tl.cnv_score(adata, groupby="cnv_leiden", use_rep="cnv")

    log("Classifying candidate malignant epithelial cells from CNV burden")
    thresholds = classify_cnv(adata)

    log("Writing CNV result tables")
    write_result_tables(adata)

    log("Saving CNV figures")
    plot_embedding(
        adata,
        "X_umap",
        "cnv_malignancy_call",
        FIG_DIR / f"{PREFIX}_integrated_umap_cnv_call",
        "Integrated UMAP: epithelial CNV calls",
        categorical=True,
    )
    plot_embedding(
        adata,
        "X_umap",
        "cnv_burden",
        FIG_DIR / f"{PREFIX}_integrated_umap_cnv_burden",
        "Integrated UMAP: CNV burden",
        categorical=False,
    )
    plot_embedding(
        adata,
        "X_cnv_umap",
        "cnv_malignancy_call",
        FIG_DIR / f"{PREFIX}_cnv_umap_cnv_call",
        "CNV UMAP: inferred epithelial status",
        categorical=True,
    )
    plot_embedding(
        adata,
        "X_cnv_umap",
        "cnv_burden",
        FIG_DIR / f"{PREFIX}_cnv_umap_cnv_burden",
        "CNV UMAP: CNV burden",
        categorical=False,
    )
    plot_burden_violin(adata)
    plot_chromosome_summary(adata)

    output_h5ad = OUT_DIR / f"{PREFIX}_analysis.h5ad"
    log(f"Writing final CNV object: {output_h5ad}")
    adata.write_h5ad(output_h5ad)
    write_report(adata, thresholds)
    log("Done")


if __name__ == "__main__":
    main()
