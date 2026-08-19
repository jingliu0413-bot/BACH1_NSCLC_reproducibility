"""Merge the nine author-filtered GSE274934 scRNA-seq matrices."""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from project_paths import DATA_DIR


INPUT_DIR = DATA_DIR / "GSE274934" / "GEX_filtered_h5"
OUTPUT_H5AD = DATA_DIR / "GSE274934" / "GSE274934_NSCLC_GEX_filtered_merged.h5ad"
SUMMARY_CSV = DATA_DIR / "GSE274934" / "GSE274934_NSCLC_GEX_filtered_merged_sample_summary.csv"


def parse_sample(path: Path) -> tuple[str, str]:
    match = re.match(r"^(GSM\d+)_(AL\d+)_GEX_filtered_feature_bc_matrix\.h5$", path.name)
    if not match:
        raise ValueError(f"Unexpected GEX filename: {path.name}")
    return match.group(1), match.group(2)


def gene_flags(var_names: pd.Index) -> pd.DataFrame:
    symbols = var_names.astype(str).str.upper()
    hb_genes = {"HBA1", "HBA2", "HBB", "HBD", "HBE1", "HBG1", "HBG2", "HBM", "HBQ1", "HBZ"}
    return pd.DataFrame(
        {
            "mt": symbols.str.startswith("MT-").to_numpy(),
            "ribo": symbols.str.match(r"^RP[SL][0-9A-Z]+").to_numpy(),
            "hb": symbols.isin(hb_genes).to_numpy(),
        },
        index=var_names,
    )


def main() -> None:
    files = sorted(INPUT_DIR.glob("GSM*_AL*_GEX_filtered_feature_bc_matrix.h5"))
    if len(files) != 9:
        raise FileNotFoundError(f"Expected nine filtered GEX matrices in {INPUT_DIR}; found {len(files)}")

    objects = []
    for path in files:
        gsm, sample = parse_sample(path)
        print(f"Reading {path.name}", flush=True)
        obj = sc.read_10x_h5(path, genome=None, gex_only=True)
        obj.var_names_make_unique()
        obj.X = obj.X.tocsr().astype(np.float32) if sparse.issparse(obj.X) else sparse.csr_matrix(obj.X, dtype=np.float32)
        obj.obs["barcode"] = obj.obs_names.astype(str)
        obj.obs["sample"] = sample
        obj.obs["gsm"] = gsm
        obj.obs["patient"] = sample
        obj.obs["source_file"] = path.name
        obj.obs_names = pd.Index([f"{sample}_{barcode}" for barcode in obj.obs_names])
        objects.append(obj)

    merged = ad.concat(objects, join="outer", merge="first", fill_value=0, index_unique=None)
    merged.X = merged.X.tocsr().astype(np.float32)
    merged.var_names_make_unique()
    merged.var = gene_flags(merged.var_names).join(merged.var, how="left")
    sc.pp.calculate_qc_metrics(
        merged,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    OUTPUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(OUTPUT_H5AD, compression="lzf")
    summary = (
        merged.obs.groupby(["sample", "gsm", "patient", "source_file"], observed=True)
        .agg(
            n_cells=("sample", "size"),
            median_genes=("n_genes_by_counts", "median"),
            median_counts=("total_counts", "median"),
            median_pct_mt=("pct_counts_mt", "median"),
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {OUTPUT_H5AD}: {merged.n_obs:,} cells x {merged.n_vars:,} genes", flush=True)


if __name__ == "__main__":
    main()
