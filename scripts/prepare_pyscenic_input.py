from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scanpy as sc
from scipy import sparse

from project_paths import PROJECT_ROOT

BASE = PROJECT_ROOT
RES = BASE / "resources" / "pyscenic"
DEFAULT_IN_H5AD = BASE / "out" / "bach1_malignant_epithelial" / "nsclc_malignant_epithelial_bach1_analysis_object.h5ad"
DEFAULT_OUT = BASE / "out" / "bach1_malignant_epithelial_pyscenic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_IN_H5AD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    expr_csv = out / "malignant_epithelial_counts_filtered_for_pyscenic.csv"
    tfs_txt = out / "hs_hgnc_tfs_expressed_in_malignant_epithelial.txt"
    meta_csv = out / "malignant_epithelial_cell_metadata_for_pyscenic.csv"
    summary_json = out / "pyscenic_input_summary.json"

    ad = sc.read_h5ad(args.input_h5ad)
    counts = ad.layers["counts"]
    if sparse.issparse(counts):
        detected = np.asarray((counts > 0).sum(axis=0)).ravel()
        total = np.asarray(counts.sum(axis=0)).ravel()
    else:
        detected = (counts > 0).sum(axis=0)
        total = counts.sum(axis=0)

    db_files = sorted(RES.glob("*.rankings.feather"))
    if not db_files:
        raise FileNotFoundError("No cisTarget ranking feather files found.")
    db_gene_sets = []
    for p in db_files:
        names = feather.read_table(p, columns=None).schema.names
        db_gene_sets.append(set(names) - {"motifs", "tracks"})
    db_genes = set.intersection(*db_gene_sets)

    tf_all = [line.strip() for line in (RES / "hs_hgnc_tfs.txt").read_text().splitlines() if line.strip()]
    tf_set = set(tf_all)

    min_cells = max(30, int(np.ceil(ad.n_obs * 0.01)))
    min_total = 3 * min_cells
    genes = np.asarray(ad.var_names)
    expressed = (detected >= min_cells) & (total >= min_total)
    in_db = np.array([g in db_genes for g in genes])
    expressed_tf = np.array([(g in tf_set) and (detected[i] > 0) for i, g in enumerate(genes)])
    keep = (expressed & in_db) | (expressed_tf & in_db)
    if "BACH1" in genes:
        keep[np.where(genes == "BACH1")[0][0]] = True

    kept_genes = genes[keep].tolist()
    tf_keep = [tf for tf in tf_all if tf in kept_genes]
    if "BACH1" not in tf_keep:
        raise RuntimeError("BACH1 was not retained in the pySCENIC TF list.")

    X = counts[:, keep]
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    expr = pd.DataFrame(X, index=ad.obs_names, columns=kept_genes)
    expr.to_csv(expr_csv)
    pd.Series(tf_keep).to_csv(tfs_txt, index=False, header=False)

    meta_cols = [
        c for c in [
            "sample_id",
            "patient",
            "dataset",
            "tissue_status",
            "tumor_type",
            "cell_type_author",
            "cell_subtype_author",
            "gse131907_author_tS_state",
            "epi_leiden",
            "epi_marker_only_label",
            "epi_marker_only_broad_class",
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_mt",
            "score_Epithelial",
            "score_Immune_contamination",
            "cnv_consensus_call",
            "n_cnv_methods_malignant",
            "recommended_working_malignant",
            "strict_cnv_malignancy_call",
            "sensitive_cnv_malignancy_call",
            "cnv_adjacent_ref_malignancy_call",
            "analysis_malignant_set",
            "BACH1_expr",
            "BACH1_group",
        ]
        if c in ad.obs.columns
    ]
    ad.obs[meta_cols].to_csv(meta_csv)

    summary = {
        "input_h5ad": str(args.input_h5ad),
        "n_cells": int(ad.n_obs),
        "n_genes_original": int(ad.n_vars),
        "min_cells": int(min_cells),
        "min_total_counts": int(min_total),
        "n_db_intersection_genes": int(len(db_genes)),
        "n_genes_retained": int(len(kept_genes)),
        "n_tfs_original": int(len(tf_all)),
        "n_tfs_retained": int(len(tf_keep)),
        "BACH1_detected_cells": int(detected[np.where(genes == "BACH1")[0][0]]) if "BACH1" in genes else None,
        "BACH1_retained": "BACH1" in kept_genes,
        "expr_csv": str(expr_csv),
        "tfs_txt": str(tfs_txt),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
