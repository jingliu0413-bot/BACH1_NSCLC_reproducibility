from __future__ import annotations

import argparse
import gzip
import re
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from project_paths import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
OUT_DIR = OUTPUT_DIR

GSE131907_DIR = DATA_DIR / "GSE131907"
GSE274934_DIR = DATA_DIR / "GSE274934"

GSE131907_ANN = GSE131907_DIR / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"
GSE131907_COUNTS = GSE131907_DIR / "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
GSE131907_SERIES = GSE131907_DIR / "GSE131907_series_matrix.txt.gz"

GSE274934_H5AD = GSE274934_DIR / "GSE274934_NSCLC_GEX_filtered_merged.h5ad"
GSE274934_SERIES = GSE274934_DIR / "GSE274934_series_matrix.txt.gz"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def clean_geo_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value


def clean_column_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "characteristics"


def parse_series_matrix(path: Path) -> pd.DataFrame:
    """Parse GEO series matrix sample-level metadata into one row per GSM sample."""
    sample_rows: dict[str, list[str]] = {}
    characteristic_rows: list[tuple[str, list[str]]] = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            key = parts[0].lstrip("!")
            values = [clean_geo_value(v) for v in parts[1:]]

            if key == "Sample_characteristics_ch1":
                parsed_keys = []
                parsed_values = []
                for value in values:
                    if ":" in value:
                        char_key, char_value = value.split(":", 1)
                        parsed_keys.append(clean_column_name(char_key))
                        parsed_values.append(char_value.strip())
                    else:
                        parsed_keys.append("characteristics")
                        parsed_values.append(value)

                if len(set(parsed_keys)) == 1:
                    column = parsed_keys[0]
                else:
                    column = f"characteristics_{len(characteristic_rows) + 1}"
                characteristic_rows.append((column, parsed_values))
            else:
                sample_rows[clean_column_name(key.replace("Sample_", ""))] = values

    if "title" not in sample_rows:
        raise ValueError(f"No !Sample_title row found in {path}")

    n_samples = len(sample_rows["title"])
    data = {}
    for key, values in sample_rows.items():
        if len(values) == n_samples:
            data[key] = values

    for key, values in characteristic_rows:
        column = key
        suffix = 2
        while column in data:
            column = f"{key}_{suffix}"
            suffix += 1
        if len(values) == n_samples:
            data[column] = values

    meta = pd.DataFrame(data)
    if "geo_accession" in meta.columns:
        meta = meta.rename(columns={"geo_accession": "gsm"})
    return meta


def annotate_gene_flags(var_names: pd.Index) -> pd.DataFrame:
    gene_upper = var_names.astype(str).str.upper()
    hb_genes = {
        "HBA1",
        "HBA2",
        "HBB",
        "HBD",
        "HBE1",
        "HBG1",
        "HBG2",
        "HBM",
        "HBQ1",
        "HBZ",
    }
    var = pd.DataFrame(index=var_names)
    var["mt"] = np.asarray(gene_upper.str.startswith("MT-"), dtype=bool)
    var["ribo"] = np.asarray(gene_upper.str.match(r"^RP[SL][0-9A-Z]+"), dtype=bool)
    var["hb"] = np.asarray(gene_upper.isin(hb_genes), dtype=bool)
    return var


def build_gse131907(series_meta: pd.DataFrame) -> ad.AnnData:
    log("Loading GSE131907 cell annotation")
    annotation = pd.read_csv(GSE131907_ANN, sep="\t", compression="gzip")
    annotation = annotation.loc[annotation["Sample_Origin"].isin(["nLung", "tLung"])].copy()
    annotation = annotation.set_index("Index", drop=False)

    sample_meta = series_meta.set_index("title", drop=False)

    log("Scanning GSE131907 raw UMI matrix header")
    with gzip.open(GSE131907_COUNTS, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if not header or header[0] != "Index":
            raise ValueError("Unexpected GSE131907 raw UMI header")

        all_cells = np.array(header[1:], dtype=object)
        keep_set = set(annotation.index)
        keep_mask = np.fromiter((cell in keep_set for cell in all_cells), dtype=bool, count=len(all_cells))
        keep_positions = np.flatnonzero(keep_mask).astype(np.int64)
        kept_cells = all_cells[keep_positions].tolist()

        if len(kept_cells) != len(annotation):
            missing = sorted(keep_set.difference(kept_cells))[:10]
            raise ValueError(
                f"GSE131907 selected cells mismatch: annotation={len(annotation)}, matrix={len(kept_cells)}, "
                f"example_missing={missing}"
            )

        obs = annotation.loc[kept_cells].copy()
        obs["dataset"] = "GSE131907"
        obs["sample"] = obs["Sample"].astype(str)
        obs["sample_id"] = "GSE131907_" + obs["sample"].astype(str)
        obs["sample_origin"] = obs["Sample_Origin"].astype(str)
        obs["tissue"] = "lung"
        obs["tissue_status"] = obs["sample_origin"].map({"nLung": "adjacent_normal", "tLung": "tumor"})
        obs["modality"] = "scRNA-seq"
        obs["barcode_original"] = obs["Barcode"].astype(str)
        obs["cell_id_original"] = obs["Index"].astype(str)
        obs["cell_type_author"] = obs["Cell_type.refined"].astype(str)
        obs["cell_subtype_author"] = obs["Cell_subtype"].astype(str)
        obs["gsm"] = obs["sample"].map(sample_meta["gsm"]) if "gsm" in sample_meta.columns else pd.NA
        obs["patient"] = obs["sample"].map(sample_meta["patient_id"]) if "patient_id" in sample_meta.columns else pd.NA
        obs["tumor_stage"] = (
            obs["sample"].map(sample_meta["tumor_stage"]) if "tumor_stage" in sample_meta.columns else pd.NA
        )
        obs["tumor_type"] = (
            obs["sample"].map(sample_meta["lung_cancer_subtype"])
            if "lung_cancer_subtype" in sample_meta.columns
            else "lung adenocarcinoma"
        )
        obs.index = "GSE131907_" + obs.index.astype(str)
        obs.index.name = None

        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        data_chunks: list[np.ndarray] = []
        genes: list[str] = []
        n_cells_total = len(all_cells)

        log(f"Extracting {len(kept_cells):,} selected cells from GSE131907 raw UMI matrix")
        for gene_idx, line in enumerate(handle):
            line = line.rstrip("\n")
            gene, values_text = line.split("\t", 1)
            values = np.fromstring(values_text, dtype=np.int32, sep="\t")
            if values.size != n_cells_total:
                raise ValueError(f"Gene {gene} has {values.size} values; expected {n_cells_total}")

            selected_values = values[keep_positions]
            nonzero = np.flatnonzero(selected_values)
            if nonzero.size:
                row_chunks.append(nonzero.astype(np.int32, copy=False))
                col_chunks.append(np.full(nonzero.size, gene_idx, dtype=np.int32))
                data_chunks.append(selected_values[nonzero].astype(np.int32, copy=False))
            genes.append(gene)

            if (gene_idx + 1) % 1000 == 0:
                nnz_so_far = sum(chunk.size for chunk in data_chunks)
                log(f"GSE131907 parsed {gene_idx + 1:,} genes; selected nnz={nnz_so_far:,}")

    if data_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        data_values = np.concatenate(data_chunks)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        data_values = np.array([], dtype=np.int32)

    matrix = sp.coo_matrix(
        (data_values, (rows, cols)),
        shape=(len(kept_cells), len(genes)),
        dtype=np.int32,
    ).tocsr()
    del row_chunks, col_chunks, data_chunks, rows, cols, data_values

    var = annotate_gene_flags(pd.Index(genes, name=None))
    var["gene_symbol"] = var.index.astype(str)

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    log(f"GSE131907 AnnData shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


def build_gse274934(series_meta: pd.DataFrame) -> ad.AnnData:
    log("Loading GSE274934 merged GEX h5ad")
    adata = sc.read_h5ad(GSE274934_H5AD)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()

    sample_meta = series_meta.copy()
    sample_meta["sample"] = sample_meta["title"].str.extract(r"^([^,]+)", expand=False)
    sample_meta = sample_meta.loc[sample_meta["title"].str.contains("scRNA-seq", regex=False)].set_index("sample")

    adata.var_names_make_unique()
    existing_var = adata.var.drop(columns=[c for c in ["mt", "ribo", "hb"] if c in adata.var.columns], errors="ignore")
    adata.var = annotate_gene_flags(adata.var_names).join(existing_var, how="left")
    adata.var["gene_symbol"] = adata.var_names.astype(str)

    obs = adata.obs.copy()
    obs["dataset"] = "GSE274934"
    obs["sample"] = obs["sample"].astype(str)
    obs["sample_id"] = "GSE274934_" + obs["sample"].astype(str)
    obs["sample_origin"] = "tLung"
    obs["tissue"] = "lung"
    obs["tissue_status"] = "tumor"
    obs["modality"] = "scRNA-seq"
    obs["barcode_original"] = obs["barcode"].astype(str) if "barcode" in obs.columns else obs.index.astype(str)
    obs["cell_id_original"] = obs.index.astype(str)
    obs["cell_type_author"] = pd.NA
    obs["cell_subtype_author"] = pd.NA
    obs["tumor_stage"] = pd.NA
    obs["tumor_type"] = obs["sample"].map(sample_meta["tumor_type"]) if "tumor_type" in sample_meta.columns else pd.NA
    if "patient" not in obs.columns:
        obs["patient"] = obs["sample"]
    obs.index = "GSE274934_" + obs.index.astype(str)
    obs.index.name = None
    adata.obs = obs

    log(f"GSE274934 AnnData shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


def standardize_obs_columns(adata: ad.AnnData) -> ad.AnnData:
    desired = [
        "dataset",
        "sample",
        "sample_id",
        "gsm",
        "patient",
        "tissue",
        "tissue_status",
        "sample_origin",
        "tumor_stage",
        "tumor_type",
        "modality",
        "barcode_original",
        "cell_id_original",
        "cell_type_author",
        "cell_subtype_author",
    ]
    for column in desired:
        if column not in adata.obs.columns:
            adata.obs[column] = pd.NA
    adata.obs = adata.obs[desired + [c for c in adata.obs.columns if c not in desired]]
    return adata


def calculate_qc_and_flags(adata: ad.AnnData) -> ad.AnnData:
    log("Calculating QC metrics on merged object")
    adata.var = annotate_gene_flags(adata.var_names).join(
        adata.var.drop(columns=[c for c in ["mt", "ribo", "hb"] if c in adata.var.columns], errors="ignore"),
        how="left",
    )
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    by_sample = adata.obs.groupby("sample_id", observed=True)
    upper_genes = by_sample["n_genes_by_counts"].quantile(0.995)
    upper_counts = by_sample["total_counts"].quantile(0.995)

    obs = adata.obs
    adata.obs["qc_upper_genes_99_5"] = obs["sample_id"].map(upper_genes).astype(float)
    adata.obs["qc_upper_counts_99_5"] = obs["sample_id"].map(upper_counts).astype(float)
    adata.obs["qc_pass_basic"] = (
        (obs["n_genes_by_counts"] >= 200)
        & (obs["total_counts"] >= 500)
        & (obs["pct_counts_mt"] <= 20)
        & (obs["pct_counts_hb"] <= 5)
        & (obs["n_genes_by_counts"] <= adata.obs["qc_upper_genes_99_5"])
        & (obs["total_counts"] <= adata.obs["qc_upper_counts_99_5"])
    )
    return adata


def write_qc_tables(adata: ad.AnnData, prefix: str) -> None:
    log("Writing sample and QC summary tables")
    obs = adata.obs.copy()

    sample_summary = (
        obs.groupby(["dataset", "sample", "sample_id", "tissue_status", "sample_origin"], observed=True)
        .agg(
            n_cells=("sample_id", "size"),
            n_qc_pass=("qc_pass_basic", "sum"),
            median_genes=("n_genes_by_counts", "median"),
            median_counts=("total_counts", "median"),
            median_pct_mt=("pct_counts_mt", "median"),
            median_pct_ribo=("pct_counts_ribo", "median"),
            median_pct_hb=("pct_counts_hb", "median"),
            q99_5_genes=("n_genes_by_counts", lambda x: x.quantile(0.995)),
            q99_5_counts=("total_counts", lambda x: x.quantile(0.995)),
        )
        .reset_index()
    )
    sample_summary["qc_pass_fraction"] = sample_summary["n_qc_pass"] / sample_summary["n_cells"]
    sample_summary.to_csv(OUT_DIR / f"{prefix}_sample_summary.csv", index=False)

    quantile_records = []
    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"]
    quantiles = [0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995, 1.0]
    for sample_id, frame in obs.groupby("sample_id", observed=True):
        base = {
            "dataset": frame["dataset"].iloc[0],
            "sample": frame["sample"].iloc[0],
            "sample_id": sample_id,
            "tissue_status": frame["tissue_status"].iloc[0],
            "sample_origin": frame["sample_origin"].iloc[0],
        }
        for metric in metrics:
            values = frame[metric].astype(float)
            for q in quantiles:
                record = dict(base)
                record.update({"metric": metric, "quantile": q, "value": values.quantile(q)})
                quantile_records.append(record)
    pd.DataFrame(quantile_records).to_csv(OUT_DIR / f"{prefix}_qc_quantiles.csv", index=False)

    threshold_counts = []
    filters = {
        "n_genes_by_counts < 200": obs["n_genes_by_counts"] < 200,
        "total_counts < 500": obs["total_counts"] < 500,
        "pct_counts_mt > 20": obs["pct_counts_mt"] > 20,
        "pct_counts_hb > 5": obs["pct_counts_hb"] > 5,
        "n_genes_by_counts > sample q99.5": obs["n_genes_by_counts"] > obs["qc_upper_genes_99_5"],
        "total_counts > sample q99.5": obs["total_counts"] > obs["qc_upper_counts_99_5"],
        "failed_any_basic_filter": ~obs["qc_pass_basic"],
    }
    for sample_id, frame_index in obs.groupby("sample_id", observed=True).groups.items():
        frame = obs.loc[frame_index]
        for filter_name, mask in filters.items():
            threshold_counts.append(
                {
                    "dataset": frame["dataset"].iloc[0],
                    "sample": frame["sample"].iloc[0],
                    "sample_id": sample_id,
                    "tissue_status": frame["tissue_status"].iloc[0],
                    "filter": filter_name,
                    "n_cells": int(mask.loc[frame_index].sum()),
                    "fraction": float(mask.loc[frame_index].mean()),
                }
            )
    pd.DataFrame(threshold_counts).to_csv(OUT_DIR / f"{prefix}_qc_threshold_counts.csv", index=False)

    included_samples = sample_summary[
        [
            "dataset",
            "sample",
            "sample_id",
            "tissue_status",
            "sample_origin",
            "n_cells",
            "n_qc_pass",
            "qc_pass_fraction",
        ]
    ].copy()
    included_samples.to_csv(OUT_DIR / f"{prefix}_included_samples.csv", index=False)


def make_analysis_object(adata: ad.AnnData, prefix: str, run_harmony: bool) -> ad.AnnData:
    log("Creating QC-filtered normalized PCA object")
    analysis = adata[adata.obs["qc_pass_basic"].to_numpy()].copy()
    analysis.layers["counts"] = analysis.X.copy()

    sc.pp.filter_genes(analysis, min_cells=20)
    sc.pp.normalize_total(analysis, target_sum=1e4)
    sc.pp.log1p(analysis)

    log("Selecting highly variable genes")
    sc.pp.highly_variable_genes(
        analysis,
        n_top_genes=3000,
        flavor="seurat",
        batch_key="sample_id",
        subset=False,
    )

    log("Running PCA on highly variable genes")
    sc.tl.pca(
        analysis,
        n_comps=50,
        use_highly_variable=True,
        svd_solver="arpack",
    )

    if run_harmony:
        try:
            import scanpy.external as sce

            log("Running Harmony on PCA coordinates with sample_id as batch key")
            sce.pp.harmony_integrate(
                analysis,
                key="sample_id",
                basis="X_pca",
                adjusted_basis="X_pca_harmony",
                max_iter_harmony=20,
            )
            analysis.uns["integration_note"] = (
                "Harmony was run on X_pca using sample_id as the batch key. "
                "Use X_pca_harmony for integrated neighbors/UMAP after confirming biology is preserved."
            )
        except Exception as exc:  # Keep the main output even if optional integration fails.
            log(f"WARNING: Harmony failed: {type(exc).__name__}: {exc}")
            analysis.uns["integration_note"] = f"Harmony failed: {type(exc).__name__}: {exc}"
    else:
        analysis.uns["integration_note"] = "Harmony was skipped by command-line option."

    analysis_path = OUT_DIR / f"{prefix}_qc_filtered_pca.h5ad"
    log(f"Writing {analysis_path}")
    analysis.write_h5ad(analysis_path, compression="lzf")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate selected NSCLC scRNA-seq samples.")
    parser.add_argument("--prefix", default="nsclc_gse131907_gse274934", help="Output filename prefix.")
    parser.add_argument("--skip-harmony", action="store_true", help="Skip optional Harmony integration.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sc.settings.verbosity = 2

    log("Parsing GEO sample metadata")
    gse131907_series = parse_series_matrix(GSE131907_SERIES)
    gse274934_series = parse_series_matrix(GSE274934_SERIES)

    gse131907 = standardize_obs_columns(build_gse131907(gse131907_series))
    gse274934 = standardize_obs_columns(build_gse274934(gse274934_series))

    log("Concatenating datasets with outer gene union")
    merged = ad.concat(
        [gse131907, gse274934],
        axis=0,
        join="outer",
        merge="first",
        fill_value=0,
        index_unique=None,
    )
    if not sp.issparse(merged.X):
        merged.X = sp.csr_matrix(merged.X)
    else:
        merged.X = merged.X.tocsr()
    merged.X = merged.X.astype(np.float32)
    merged.var_names_make_unique()

    merged = calculate_qc_and_flags(merged)
    write_qc_tables(merged, args.prefix)

    raw_path = OUT_DIR / f"{args.prefix}_raw_qc.h5ad"
    log(f"Writing {raw_path}")
    merged.write_h5ad(raw_path, compression="lzf")

    make_analysis_object(merged, args.prefix, run_harmony=not args.skip_harmony)

    log("Done")


if __name__ == "__main__":
    main()
