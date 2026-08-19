from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scanpy as sc
from scipy import sparse

from project_paths import PROJECT_ROOT

BASE = PROJECT_ROOT
IN_H5AD = BASE / "out" / "bach1_malignant_epithelial" / "nsclc_malignant_epithelial_bach1_analysis_object.h5ad"
RES = BASE / "resources" / "pyscenic"
OUT = BASE / "out" / "bach1_malignant_epithelial_pyscenic"
OUT.mkdir(parents=True, exist_ok=True)

EXPR_CSV = OUT / "malignant_epithelial_counts_filtered_for_pyscenic.csv"
TFS_TXT = OUT / "hs_hgnc_tfs_expressed_in_malignant_epithelial.txt"
META_CSV = OUT / "malignant_epithelial_cell_metadata_for_pyscenic.csv"
SUMMARY_JSON = OUT / "pyscenic_input_summary.json"


def main() -> None:
    ad = sc.read_h5ad(IN_H5AD)
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
    expr.to_csv(EXPR_CSV)
    pd.Series(tf_keep).to_csv(TFS_TXT, index=False, header=False)

    meta_cols = [
        c for c in [
            "sample_id",
            "dataset",
            "tissue_status",
            "epi_leiden",
            "epi_marker_only_label",
            "epi_marker_only_broad_class",
            "BACH1_expr",
            "BACH1_group",
        ]
        if c in ad.obs.columns
    ]
    ad.obs[meta_cols].to_csv(META_CSV)

    summary = {
        "input_h5ad": str(IN_H5AD),
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
        "expr_csv": str(EXPR_CSV),
        "tfs_txt": str(TFS_TXT),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
