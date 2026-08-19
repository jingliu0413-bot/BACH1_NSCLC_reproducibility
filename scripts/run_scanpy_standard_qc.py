from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from project_paths import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR
QC_DIR = OUT_DIR / "scanpy_qc"
FIG_DIR = QC_DIR / "figures"
RAW_H5AD = OUT_DIR / "nsclc_gse131907_gse274934_raw_qc.h5ad"

PREFIX = "nsclc_gse131907_gse274934_scanpy_qc"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def metric_frame(obs: pd.DataFrame, pass_only: bool | None = None) -> pd.DataFrame:
    metrics = [
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
        "pct_counts_hb",
        "pct_counts_ribo",
    ]
    cols = ["dataset", "sample", "sample_id", "tissue_status", "qc_pass_scanpy"] + metrics
    df = obs[cols].copy()
    if pass_only is True:
        df = df.loc[df["qc_pass_scanpy"]]
    elif pass_only is False:
        df = df.loc[~df["qc_pass_scanpy"]]
    return df


def add_qc_flags(adata: ad.AnnData) -> OrderedDict[str, pd.Series]:
    obs = adata.obs
    criteria: OrderedDict[str, pd.Series] = OrderedDict(
        [
            ("n_genes_by_counts >= 200", obs["n_genes_by_counts"] >= 200),
            ("n_genes_by_counts < 5000", obs["n_genes_by_counts"] < 5000),
            ("total_counts >= 500", obs["total_counts"] >= 500),
            ("pct_counts_mt < 15", obs["pct_counts_mt"] < 15),
            ("pct_counts_hb < 1", obs["pct_counts_hb"] < 1),
        ]
    )

    for name, mask in criteria.items():
        col = "qc_" + (
            name.replace(" ", "_")
            .replace(">=", "ge")
            .replace("<=", "le")
            .replace("<", "lt")
            .replace(">", "gt")
            .replace("%", "pct")
        )
        col = col.replace("__", "_").replace("=", "")
        adata.obs[col] = mask.to_numpy()

    pass_mask = np.logical_and.reduce([mask.to_numpy() for mask in criteria.values()])
    adata.obs["qc_pass_scanpy"] = pass_mask
    adata.obs["qc_failed_scanpy"] = ~pass_mask
    adata.obs["qc_ribo_lt_1"] = (obs["pct_counts_ribo"] < 1).to_numpy()
    adata.obs["qc_note"] = (
        "Scanpy QC: n_genes_by_counts >=200 and <5000, total_counts >=500, "
        "pct_counts_mt <15, pct_counts_hb <1; pct_counts_ribo was summarized but not filtered."
    )
    return criteria


def write_tables(adata: ad.AnnData, criteria: OrderedDict[str, pd.Series]) -> None:
    log("Writing QC summary tables")
    obs = adata.obs.copy()

    sample_summary = (
        obs.groupby(["dataset", "sample", "sample_id", "tissue_status", "sample_origin"], observed=True)
        .agg(
            n_cells=("sample_id", "size"),
            n_pass=("qc_pass_scanpy", "sum"),
            median_genes=("n_genes_by_counts", "median"),
            median_counts=("total_counts", "median"),
            median_pct_mt=("pct_counts_mt", "median"),
            median_pct_hb=("pct_counts_hb", "median"),
            median_pct_ribo=("pct_counts_ribo", "median"),
            q95_pct_ribo=("pct_counts_ribo", lambda x: x.quantile(0.95)),
            q99_pct_ribo=("pct_counts_ribo", lambda x: x.quantile(0.99)),
        )
        .reset_index()
    )
    sample_summary["pass_fraction"] = sample_summary["n_pass"] / sample_summary["n_cells"]
    sample_summary.to_csv(QC_DIR / f"{PREFIX}_sample_summary.csv", index=False)

    filter_records = []
    filter_masks = {f"fail: {name}": ~mask for name, mask in criteria.items()}
    filter_masks["would_fail_if_pct_counts_ribo >= 1"] = obs["pct_counts_ribo"] >= 1
    filter_masks["failed_any_scanpy_qc"] = obs["qc_failed_scanpy"]

    for sample_id, frame_index in obs.groupby("sample_id", observed=True).groups.items():
        frame = obs.loc[frame_index]
        for filter_name, mask in filter_masks.items():
            sample_mask = mask.loc[frame_index] if isinstance(mask, pd.Series) else pd.Series(mask, index=obs.index).loc[frame_index]
            filter_records.append(
                {
                    "dataset": frame["dataset"].iloc[0],
                    "sample": frame["sample"].iloc[0],
                    "sample_id": sample_id,
                    "tissue_status": frame["tissue_status"].iloc[0],
                    "filter": filter_name,
                    "n_cells": int(sample_mask.sum()),
                    "fraction": float(sample_mask.mean()),
                }
            )
    pd.DataFrame(filter_records).to_csv(QC_DIR / f"{PREFIX}_threshold_counts.csv", index=False)

    quantiles = [0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0]
    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_hb", "pct_counts_ribo"]
    quantile_records = []
    for sample_id, frame in obs.groupby("sample_id", observed=True):
        base = {
            "dataset": frame["dataset"].iloc[0],
            "sample": frame["sample"].iloc[0],
            "sample_id": sample_id,
            "tissue_status": frame["tissue_status"].iloc[0],
        }
        for metric in metrics:
            values = frame[metric].astype(float)
            for q in quantiles:
                quantile_records.append({**base, "metric": metric, "quantile": q, "value": values.quantile(q)})
    pd.DataFrame(quantile_records).to_csv(QC_DIR / f"{PREFIX}_qc_quantiles.csv", index=False)

    metadata_cols = [
        "dataset",
        "sample",
        "sample_id",
        "gsm",
        "patient",
        "tissue_status",
        "sample_origin",
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
        "pct_counts_hb",
        "pct_counts_ribo",
        "qc_pass_scanpy",
        "qc_failed_scanpy",
        "qc_ribo_lt_1",
    ]
    obs[metadata_cols].to_csv(QC_DIR / f"{PREFIX}_cell_qc_metadata.csv.gz", index=True, compression="gzip")


def save_metric_violin(df: pd.DataFrame, filename: str, title: str) -> None:
    metrics = [
        ("n_genes_by_counts", "Genes detected"),
        ("total_counts", "UMI counts"),
        ("pct_counts_mt", "Mitochondrial %"),
        ("pct_counts_hb", "Hemoglobin %"),
        ("pct_counts_ribo", "Ribosomal %"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(18, 18), sharex=True)
    order = sorted(df["sample_id"].unique())
    for ax, (metric, label) in zip(axes, metrics):
        sns.violinplot(
            data=df,
            x="sample_id",
            y=metric,
            order=order,
            hue="tissue_status",
            dodge=False,
            cut=0,
            inner="quartile",
            linewidth=0.4,
            ax=ax,
        )
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.legend_.remove() if ax.legend_ else None
        if metric in {"n_genes_by_counts", "total_counts"}:
            ax.set_yscale("log")
    axes[0].set_title(title)
    axes[-1].tick_params(axis="x", rotation=90, labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, 0.98, 0.98))
    fig.savefig(FIG_DIR / filename, dpi=220)
    plt.close(fig)


def save_metric_density(df: pd.DataFrame, filename: str, title: str) -> None:
    metrics = [
        ("n_genes_by_counts", "Genes detected", True),
        ("total_counts", "UMI counts", True),
        ("pct_counts_mt", "Mitochondrial %", False),
        ("pct_counts_hb", "Hemoglobin %", False),
        ("pct_counts_ribo", "Ribosomal %", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, (metric, label, log_scale) in zip(axes, metrics):
        sns.histplot(
            data=df,
            x=metric,
            hue="dataset",
            bins=80,
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
    axes[-1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=220)
    plt.close(fig)


def save_scatter_panels(df: pd.DataFrame, filename: str, title: str) -> None:
    if len(df) > 100_000:
        df = df.sample(n=100_000, random_state=7)
    color_metrics = [
        ("pct_counts_mt", "Mitochondrial %"),
        ("pct_counts_hb", "Hemoglobin %"),
        ("pct_counts_ribo", "Ribosomal %"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes, color_metrics):
        sca = ax.scatter(
            df["total_counts"],
            df["n_genes_by_counts"],
            c=df[metric],
            s=1,
            alpha=0.45,
            cmap="viridis",
            rasterized=True,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("total_counts")
        ax.set_title(label)
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("n_genes_by_counts")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=220)
    plt.close(fig)


def save_filter_impact(obs: pd.DataFrame) -> None:
    summary = (
        obs.groupby(["dataset", "sample_id", "tissue_status"], observed=True)
        .agg(n_cells=("qc_pass_scanpy", "size"), n_pass=("qc_pass_scanpy", "sum"))
        .reset_index()
    )
    summary["pass_fraction"] = summary["n_pass"] / summary["n_cells"]
    summary = summary.sort_values(["dataset", "sample_id"])
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.barplot(data=summary, x="sample_id", y="pass_fraction", hue="tissue_status", dodge=False, ax=ax)
    ax.axhline(0.9, color="0.3", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("QC pass fraction")
    ax.set_xlabel("")
    ax.set_title("Scanpy QC Pass Fraction by Sample")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{PREFIX}_filter_impact_by_sample.png", dpi=220)
    plt.close(fig)


def save_ribo_context(obs: pd.DataFrame) -> None:
    df = metric_frame(obs)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(data=df, x="dataset", y="pct_counts_ribo", hue="tissue_status", showfliers=False, ax=axes[0])
    axes[0].axhline(1, color="red", linestyle="--", linewidth=0.8, label="1%")
    axes[0].axhline(20, color="0.4", linestyle="--", linewidth=0.8, label="20%")
    axes[0].axhline(40, color="0.2", linestyle="--", linewidth=0.8, label="40%")
    axes[0].set_title("Ribosomal Percentage by Dataset")
    axes[0].set_ylabel("pct_counts_ribo")
    axes[0].legend(frameon=False)

    sns.histplot(
        data=df,
        x="pct_counts_ribo",
        hue="dataset",
        bins=100,
        stat="density",
        common_norm=False,
        element="step",
        ax=axes[1],
    )
    axes[1].axvline(1, color="red", linestyle="--", linewidth=0.8)
    axes[1].axvline(20, color="0.4", linestyle="--", linewidth=0.8)
    axes[1].axvline(40, color="0.2", linestyle="--", linewidth=0.8)
    axes[1].set_title("Ribosomal Percentage Distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{PREFIX}_ribo_distribution_context.png", dpi=220)
    plt.close(fig)


def save_figures(adata: ad.AnnData) -> None:
    log("Saving QC figures")
    pre_df = metric_frame(adata.obs)
    post_df = metric_frame(adata.obs, pass_only=True)

    save_metric_violin(pre_df, f"{PREFIX}_pre_qc_violin_by_sample.png", "Pre-QC Metrics by Sample")
    save_metric_violin(post_df, f"{PREFIX}_post_qc_violin_by_sample.png", "Post-QC Metrics by Sample")
    save_metric_density(pre_df, f"{PREFIX}_pre_qc_density_by_dataset.png", "Pre-QC Metric Distributions")
    save_metric_density(post_df, f"{PREFIX}_post_qc_density_by_dataset.png", "Post-QC Metric Distributions")
    save_scatter_panels(pre_df, f"{PREFIX}_pre_qc_scatter_total_counts_vs_genes.png", "Pre-QC Counts vs Genes")
    save_scatter_panels(post_df, f"{PREFIX}_post_qc_scatter_total_counts_vs_genes.png", "Post-QC Counts vs Genes")
    save_filter_impact(adata.obs)
    save_ribo_context(adata.obs)


def write_report(adata: ad.AnnData) -> None:
    obs = adata.obs
    n_total = adata.n_obs
    n_pass = int(obs["qc_pass_scanpy"].sum())
    n_ribo_lt_1 = int((obs["pct_counts_ribo"] < 1).sum())
    group_summary = (
        obs.groupby(["dataset", "tissue_status"], observed=True)
        .agg(n_cells=("qc_pass_scanpy", "size"), n_pass=("qc_pass_scanpy", "sum"))
        .reset_index()
    )
    group_summary["pass_fraction"] = group_summary["n_pass"] / group_summary["n_cells"]
    group_lines = ["| dataset | tissue_status | n_cells | n_pass | pass_fraction |", "|---|---:|---:|---:|---:|"]
    for row in group_summary.itertuples(index=False):
        group_lines.append(
            f"| {row.dataset} | {row.tissue_status} | {int(row.n_cells):,} | "
            f"{int(row.n_pass):,} | {row.pass_fraction:.4%} |"
        )

    report = [
        "# Scanpy Standard QC Report",
        "",
        "## Filters Applied",
        "",
        "- `n_genes_by_counts >= 200`",
        "- `n_genes_by_counts < 5000`",
        "- `total_counts >= 500`",
        "- `pct_counts_mt < 15`",
        "- `pct_counts_hb < 1`",
        "- `pct_counts_ribo` was summarized and plotted, but not used as a hard filter.",
        "",
        "## Result",
        "",
        f"- Input cells: {n_total:,}",
        f"- QC-pass cells: {n_pass:,}",
        f"- QC-pass fraction: {n_pass / n_total:.4%}",
        f"- Cells with `pct_counts_ribo < 1`: {n_ribo_lt_1:,} ({n_ribo_lt_1 / n_total:.4%}); this threshold was not applied.",
        "",
        "## Group Summary",
        "",
        "\n".join(group_lines),
        "",
        "## Files",
        "",
        f"- Filtered raw-count object: `{PREFIX}_raw_counts_qc_filtered.h5ad`",
        f"- Cell QC metadata: `{PREFIX}_cell_qc_metadata.csv.gz`",
        f"- Sample summary: `{PREFIX}_sample_summary.csv`",
        f"- Threshold counts: `{PREFIX}_threshold_counts.csv`",
        f"- QC quantiles: `{PREFIX}_qc_quantiles.csv`",
        f"- Figures: `figures/`",
        "",
        "## Boundary",
        "",
        "This run stopped after QC filtering. No normalization, HVG selection, PCA, integration, UMAP, or clustering was performed.",
        "",
    ]
    (QC_DIR / f"{PREFIX}_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sc.settings.verbosity = 2
    sns.set_theme(style="whitegrid", context="notebook")

    log(f"Reading {RAW_H5AD}")
    adata = sc.read_h5ad(RAW_H5AD)

    log("Recomputing Scanpy QC metrics from raw-count matrix")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=[20, 50, 100, 200],
        log1p=False,
        inplace=True,
    )

    log("Applying Scanpy QC filters")
    criteria = add_qc_flags(adata)
    write_tables(adata, criteria)
    save_figures(adata)
    write_report(adata)

    filtered = adata[adata.obs["qc_pass_scanpy"].to_numpy()].copy()
    filtered.uns["qc_note"] = (
        "Filtered only by Scanpy QC thresholds: n_genes_by_counts >=200 and <5000, "
        "total_counts >=500, pct_counts_mt <15, pct_counts_hb <1. "
        "pct_counts_ribo was plotted/summarized but not used as a hard filter."
    )
    out_path = QC_DIR / f"{PREFIX}_raw_counts_qc_filtered.h5ad"
    log(f"Writing filtered raw-count object: {out_path}")
    filtered.write_h5ad(out_path, compression="lzf")

    log("Done")


if __name__ == "__main__":
    main()
