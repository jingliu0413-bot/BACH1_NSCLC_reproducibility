from __future__ import annotations

import importlib.util
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
SOURCE_SCRIPT = Path(__file__).with_name("run_epithelial_cnv_infercnvpy.py")
INPUT_H5AD = (
    BASE_DIR
    / "epithelial_cnv_infercnvpy"
    / "nsclc_gse131907_gse274934_epithelial_cnv_input_epithelial_tnk_reference_autosomes.h5ad"
)
OUT_DIR = BASE_DIR / "epithelial_cnv_adjacent_epithelial_reference"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PREFIX = "nsclc_gse131907_gse274934_epithelial_cnv_adjacent_epithelial_ref"
RANDOM_STATE = 7


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_helpers():
    spec = importlib.util.spec_from_file_location("epithelial_cnv_helpers", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT_DIR = OUT_DIR
    module.FIG_DIR = FIG_DIR
    module.TABLE_DIR = TABLE_DIR
    module.PREFIX = PREFIX
    module.INPUT_H5AD = INPUT_H5AD
    return module


def save_pub(fig: plt.Figure, stem: Path, dpi: int = 450) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def mean_abs(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(abs(matrix).mean(axis=1)).ravel()
    return np.mean(np.abs(matrix), axis=1)


def classify_with_adjacent_reference(adata: sc.AnnData) -> dict[str, float]:
    adata.obs["cnv_burden_adjacent_ref"] = mean_abs(adata.obsm["X_cnv"])
    ref_mask = adata.obs["cnv_reference_group"].astype(str).to_numpy() == "adjacent_normal_epithelial_reference"
    ref_scores = adata.obs.loc[ref_mask, "cnv_burden_adjacent_ref"].to_numpy()
    ref_median = float(np.median(ref_scores))
    ref_mad = float(np.median(np.abs(ref_scores - ref_median)))
    ref_mad_scaled = 1.4826 * ref_mad
    threshold_q99 = float(np.quantile(ref_scores, 0.99))
    threshold_mad = float(ref_median + 3 * ref_mad_scaled)
    threshold = float(max(threshold_q99, threshold_mad))

    adata.obs["cnv_adjacent_ref_cellwise_high"] = adata.obs["cnv_burden_adjacent_ref"] >= threshold
    obs = adata.obs.copy()
    obs["is_tumor_epithelial"] = obs["cnv_reference_group"].astype(str) == "tumor_epithelial_query"
    obs["is_adjacent_ref"] = obs["cnv_reference_group"].astype(str) == "adjacent_normal_epithelial_reference"

    cluster_summary = (
        obs.groupby("cnv_leiden", observed=True)
        .agg(
            n_cells=("cnv_burden_adjacent_ref", "size"),
            n_tumor_epithelial=("is_tumor_epithelial", "sum"),
            n_adjacent_reference=("is_adjacent_ref", "sum"),
            mean_cnv_burden=("cnv_burden_adjacent_ref", "mean"),
            median_cnv_burden=("cnv_burden_adjacent_ref", "median"),
            q90_cnv_burden=("cnv_burden_adjacent_ref", lambda x: float(np.quantile(x, 0.90))),
            fraction_cellwise_high=("cnv_adjacent_ref_cellwise_high", "mean"),
        )
        .reset_index()
    )
    cluster_summary["fraction_adjacent_reference"] = (
        cluster_summary["n_adjacent_reference"] / cluster_summary["n_cells"]
    )
    cluster_summary["cnv_cluster_call_adjacent_ref"] = np.where(
        (cluster_summary["n_tumor_epithelial"] >= 20)
        & (cluster_summary["median_cnv_burden"] >= threshold)
        & (cluster_summary["fraction_adjacent_reference"] <= 0.10),
        "malignant_candidate_cluster",
        "reference_like_or_low_cnv_cluster",
    )
    cluster_summary.to_csv(TABLE_DIR / f"{PREFIX}_cnv_cluster_summary.csv", index=False)

    malignant_clusters = set(
        cluster_summary.loc[
            cluster_summary["cnv_cluster_call_adjacent_ref"] == "malignant_candidate_cluster",
            "cnv_leiden",
        ].astype(str)
    )
    is_tumor = adata.obs["cnv_reference_group"].astype(str) == "tumor_epithelial_query"
    in_malignant_cluster = adata.obs["cnv_leiden"].astype(str).isin(malignant_clusters)
    adata.obs["cnv_adjacent_ref_malignancy_call"] = "adjacent_normal_epithelial_reference"
    adata.obs.loc[is_tumor & in_malignant_cluster, "cnv_adjacent_ref_malignancy_call"] = (
        "malignant_epithelial"
    )
    adata.obs.loc[is_tumor & ~in_malignant_cluster, "cnv_adjacent_ref_malignancy_call"] = (
        "nonmalignant_or_low_cnv_tumor_epithelial"
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


def write_tables(adata: sc.AnnData) -> None:
    cols = [
        "dataset",
        "sample",
        "sample_id",
        "tissue_status",
        "major_celltype_auto",
        "leiden_major",
        "leiden_sub",
        "cnv_reference_group",
        "cnv_leiden",
        "cnv_score",
        "cnv_burden_adjacent_ref",
        "cnv_adjacent_ref_cellwise_high",
        "cnv_adjacent_ref_malignancy_call",
    ]
    cols = [c for c in cols if c in adata.obs]
    df = adata.obs[cols].copy()
    df.insert(0, "cell_id", df.index)
    df.to_csv(TABLE_DIR / f"{PREFIX}_cell_cnv_calls.csv.gz", index=False)
    df.loc[
        df["cnv_adjacent_ref_malignancy_call"] == "malignant_epithelial", "cell_id"
    ].to_csv(TABLE_DIR / f"{PREFIX}_malignant_epithelial_cell_ids.txt", index=False, header=False)
    df.groupby(["dataset", "sample_id", "tissue_status", "cnv_adjacent_ref_malignancy_call"], observed=True).size().reset_index(
        name="n_cells"
    ).to_csv(TABLE_DIR / f"{PREFIX}_calls_by_sample.csv", index=False)
    df["cnv_adjacent_ref_malignancy_call"].value_counts().rename_axis("call").reset_index(
        name="n_cells"
    ).to_csv(TABLE_DIR / f"{PREFIX}_call_summary.csv", index=False)


def plot_embedding(adata: sc.AnnData, basis: str, color: str, stem: Path, title: str, continuous: bool) -> None:
    coords = adata.obsm[basis]
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    if continuous:
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
    else:
        palette = {
            "adjacent_normal_epithelial_reference": "#777777",
            "nonmalignant_or_low_cnv_tumor_epithelial": "#4DBBD5",
            "malignant_epithelial": "#E64B35",
        }
        values = adata.obs[color].astype(str)
        for label, col in palette.items():
            idx = values == label
            if idx.any():
                ax.scatter(
                    coords[idx, 0],
                    coords[idx, 1],
                    s=1.2,
                    c=col,
                    label=label,
                    linewidths=0,
                    alpha=0.85,
                    rasterized=True,
                )
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=4, handletextpad=0.2)
    ax.set_title(title, pad=8)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    save_pub(fig, stem)


def plot_chromosome_summary(adata: sc.AnnData) -> None:
    for groupby, suffix, figsize in [
        ("cnv_adjacent_ref_malignancy_call", "by_call", (8.5, 2.8)),
        ("cnv_leiden", "by_cnv_leiden", (10.0, 5.2)),
    ]:
        try:
            cnv.pl.chromosome_heatmap_summary(
                adata,
                groupby=groupby,
                use_rep="cnv",
                figsize=figsize,
                show=False,
            )
            save_pub(plt.gcf(), FIG_DIR / f"{PREFIX}_chromosome_heatmap_summary_{suffix}")
        except Exception as exc:
            log(f"Could not plot chromosome heatmap summary {suffix}: {type(exc).__name__}: {exc}")


def write_report(adata: sc.AnnData, thresholds: dict[str, float]) -> None:
    counts = adata.obs["cnv_adjacent_ref_malignancy_call"].value_counts()
    lines = [
        "# Adjacent-Normal Epithelial Reference CNV Report",
        "",
        f"- Input object: `{INPUT_H5AD}`",
        f"- Cells: {adata.n_obs:,}",
        f"- Genes: {adata.n_vars:,}",
        f"- Adjacent-normal epithelial reference cells: {(adata.obs['cnv_reference_group'].astype(str) == 'adjacent_normal_epithelial_reference').sum():,}",
        f"- Tumor epithelial query cells: {(adata.obs['cnv_reference_group'].astype(str) == 'tumor_epithelial_query').sum():,}",
        f"- Primary threshold: {thresholds['primary_threshold']:.6f}",
        "",
        "| call | n_cells |",
        "|---|---:|",
    ]
    for k, v in counts.items():
        lines.append(f"| {k} | {int(v)} |")
    lines.extend(
        [
            "",
            "This analysis uses adjacent-normal epithelial cells as the normal reference.",
            "Because adjacent-normal epithelial cells are available only from GSE131907, use this result as a complementary check rather than a replacement for the T/NK-reference analysis.",
        ]
    )
    (OUT_DIR / f"{PREFIX}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    helper = load_helpers()
    helper.ensure_dirs()
    helper.apply_publication_style()
    params = {
        "input_h5ad": str(INPUT_H5AD),
        "reference": "adjacent_normal_epithelial_reference",
        "dynamic_threshold": None,
        "window_size": 100,
        "step": 10,
        "lfc_clip": 3,
        "random_state": RANDOM_STATE,
    }
    with open(OUT_DIR / f"{PREFIX}_params.json", "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)

    log(f"Reading saved epithelial + T/NK CNV input: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)
    adata = adata[adata.obs["cnv_input_group"].astype(str) == "Epithelial"].copy()
    adata.obs["cnv_reference_group"] = np.where(
        adata.obs["tissue_status"].astype(str) == "adjacent_normal",
        "adjacent_normal_epithelial_reference",
        "tumor_epithelial_query",
    )

    log(
        "Running CNV with "
        f"{(adata.obs['cnv_reference_group'] == 'adjacent_normal_epithelial_reference').sum():,} "
        "adjacent-normal epithelial reference cells"
    )
    cnv.tl.infercnv(
        adata,
        reference_key="cnv_reference_group",
        reference_cat="adjacent_normal_epithelial_reference",
        lfc_clip=3,
        window_size=100,
        step=10,
        dynamic_threshold=None,
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

    log("Classifying candidate malignant epithelial cells")
    thresholds = classify_with_adjacent_reference(adata)

    log("Writing tables and figures")
    write_tables(adata)
    plot_embedding(
        adata,
        "X_umap",
        "cnv_adjacent_ref_malignancy_call",
        FIG_DIR / f"{PREFIX}_integrated_umap_cnv_call",
        "Integrated UMAP: CNV calls with adjacent epithelial reference",
        continuous=False,
    )
    plot_embedding(
        adata,
        "X_umap",
        "cnv_burden_adjacent_ref",
        FIG_DIR / f"{PREFIX}_integrated_umap_cnv_burden",
        "Integrated UMAP: CNV burden with adjacent epithelial reference",
        continuous=True,
    )
    plot_embedding(
        adata,
        "X_cnv_umap",
        "cnv_adjacent_ref_malignancy_call",
        FIG_DIR / f"{PREFIX}_cnv_umap_cnv_call",
        "CNV UMAP: adjacent epithelial reference calls",
        continuous=False,
    )
    plot_embedding(
        adata,
        "X_cnv_umap",
        "cnv_burden_adjacent_ref",
        FIG_DIR / f"{PREFIX}_cnv_umap_cnv_burden",
        "CNV UMAP: adjacent epithelial reference burden",
        continuous=True,
    )
    plot_chromosome_summary(adata)

    output_h5ad = OUT_DIR / f"{PREFIX}_analysis.h5ad"
    log(f"Writing final object: {output_h5ad}")
    adata.write_h5ad(output_h5ad)
    write_report(adata, thresholds)
    log("Done")


if __name__ == "__main__":
    main()
