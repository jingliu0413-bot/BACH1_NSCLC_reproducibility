"""Prepare lightweight scRNA and Visium inputs for spacexr/RCTD.

The exported matrices use genes as rows and cells/spots as columns, matching
spacexr's expected input orientation. The reference is balanced by major cell
type and marks cells from the revised primary malignant epithelial set as
Malignant_epithelial.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite

from project_paths import OUTPUT_DIR


SCRNA = OUTPUT_DIR / "scanpy_downstream_scrublet" / "nsclc_gse131907_gse274934_scanpy_downstream_analysis.h5ad"
PRIMARY_META = (
    OUTPUT_DIR
    / "bach1_malignant_epithelial_primary"
    / "tables"
    / "nsclc_malignant_epithelial_primary_bach1_cell_metadata.csv.gz"
)
SPATIAL_H5AD = OUTPUT_DIR / "emtab13530_spatial_bach1_nod" / "h5ad_by_sample"
DEFAULT_OUT = OUTPUT_DIR / "emtab13530_spatial_bach1_nod" / "rctd_spacexr_inputs"


MARKER_GENES = {
    "Malignant_epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "TACSTD2", "KRT17"],
    "Other_epithelial": ["AGER", "SFTPC", "SCGB1A1", "FOXJ1", "KRT5", "MUC1"],
    "T_NK": ["CD3D", "CD3E", "TRAC", "NKG7", "GNLY", "KLRD1", "IL7R"],
    "B": ["MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "HLA-DRA"],
    "Plasma": ["MZB1", "JCHAIN", "XBP1", "SDC1", "IGKC", "IGHG1"],
    "Myeloid": ["LYZ", "LST1", "TYROBP", "FCER1G", "C1QA", "C1QB", "S100A8", "S100A9"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2"],
    "Endothelial": ["PECAM1", "VWF", "KDR", "ENG", "EMCN", "ESAM"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "TAGLN", "ACTA2"],
}


NOD_GENES = [
    "NOD1",
    "NOD2",
    "RIPK2",
    "RELA",
    "NFKB1",
    "NFKBIA",
    "TNFAIP3",
    "TBK1",
    "ATG16L1",
    "IL1B",
]


def read_h5ad_compat(path: Path) -> ad.AnnData:
    try:
        return ad.read_h5ad(path)
    except Exception as err:
        if "encoding_type='null'" not in str(err) and "uns/log1p" not in str(err):
            raise
    with tempfile.TemporaryDirectory(prefix="h5ad_compat_") as tmpdir:
        tmp_path = Path(tmpdir) / path.name
        shutil.copy2(path, tmp_path)
        with h5py.File(tmp_path, "a") as handle:
            if "uns/log1p/base" in handle:
                del handle["uns/log1p/base"]
        return ad.read_h5ad(tmp_path)


def sanitize_label(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace("-", "_")


def read_primary_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    meta = pd.read_csv(path, usecols=[0])
    return set(meta.iloc[:, 0].astype(str))


def choose_reference_genes(adata: ad.AnnData, spatial_genes: set[str], max_genes: int) -> list[str]:
    common = pd.Index([g for g in adata.var_names.astype(str) if g in spatial_genes])
    marker = []
    for genes in MARKER_GENES.values():
        marker.extend(genes)
    marker.extend(["BACH1", "MAFK", "NFE2L2", "FOS", "JUN"])
    marker.extend(NOD_GENES)
    selected = [g for g in dict.fromkeys(marker) if g in set(common)]

    var = adata.var.loc[common].copy()
    if "highly_variable" in var:
        hvg = var.index[var["highly_variable"].fillna(False).astype(bool)].astype(str).tolist()
    else:
        hvg = []
    for gene in hvg:
        if gene not in selected:
            selected.append(gene)
        if len(selected) >= max_genes:
            return selected[:max_genes]

    rank_col = "n_cells_by_counts" if "n_cells_by_counts" in var else "n_cells"
    if rank_col in var:
        ranked = var.sort_values(rank_col, ascending=False).index.astype(str).tolist()
    else:
        ranked = common.astype(str).tolist()
    for gene in ranked:
        if gene not in selected:
            selected.append(gene)
        if len(selected) >= max_genes:
            break
    return selected[:max_genes]


def write_matrix_market(matrix, matrix_path: Path) -> None:
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    mtx_plain = matrix_path.with_suffix("")
    mmwrite(mtx_plain, sparse.coo_matrix(matrix))
    with mtx_plain.open("rb") as src, gzip.open(matrix_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    mtx_plain.unlink()


def export_reference(args: argparse.Namespace, spatial_genes: set[str], out_dir: Path) -> dict:
    adata_b = sc.read_h5ad(args.scrna, backed="r")
    obs = adata_b.obs.copy()
    primary_ids = read_primary_ids(args.primary_meta)
    labels = obs["major_celltype_auto"].astype(str).map(sanitize_label)
    labels.loc[obs.index.isin(primary_ids)] = "Malignant_epithelial"
    labels.loc[(obs["major_celltype_auto"].astype(str) == "Epithelial") & ~obs.index.isin(primary_ids)] = "Other_epithelial"
    keep_types = [
        "Malignant_epithelial",
        "Other_epithelial",
        "T_NK",
        "B",
        "Plasma",
        "Myeloid",
        "Mast",
        "Endothelial",
        "Fibroblast",
    ]
    labels = labels.where(labels.isin(keep_types))
    obs = obs.assign(rctd_cell_type=labels)
    obs = obs.dropna(subset=["rctd_cell_type"])
    obs = obs[pd.to_numeric(obs["total_counts"], errors="coerce").fillna(0) >= args.min_reference_umi]

    rng = np.random.default_rng(args.seed)
    selected_cells = []
    ref_summary_rows = []
    for cell_type in keep_types:
        candidates = obs.index[obs["rctd_cell_type"].eq(cell_type)].to_numpy(str)
        if candidates.size == 0:
            continue
        if candidates.size > args.max_cells_per_type:
            candidates = rng.choice(candidates, size=args.max_cells_per_type, replace=False)
        candidates = np.sort(candidates)
        selected_cells.extend(candidates.tolist())
        ref_summary_rows.append(
            {
                "cell_type": cell_type,
                "n_available": int(obs["rctd_cell_type"].eq(cell_type).sum()),
                "n_selected": int(len(candidates)),
            }
        )

    selected_genes = choose_reference_genes(adata_b, spatial_genes, args.max_genes)
    cell_idx = adata_b.obs_names.get_indexer(selected_cells)
    gene_idx = adata_b.var_names.get_indexer(selected_genes)
    valid_cells = cell_idx >= 0
    valid_genes = gene_idx >= 0
    selected_cells = [cell for cell, ok in zip(selected_cells, valid_cells) if ok]
    selected_genes = [gene for gene, ok in zip(selected_genes, valid_genes) if ok]
    sub = adata_b[cell_idx[valid_cells], gene_idx[valid_genes]].to_memory()
    adata_b.file.close()
    counts = sub.layers["counts"] if "counts" in sub.layers else sub.X
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    counts = counts.astype(np.int64).T.tocoo()

    ref_dir = out_dir / "reference"
    write_matrix_market(counts, ref_dir / "reference_counts.mtx.gz")
    pd.Series(selected_genes).to_csv(ref_dir / "genes.tsv", sep="\t", index=False, header=False)
    pd.Series(selected_cells).to_csv(ref_dir / "cells.tsv", sep="\t", index=False, header=False)
    cell_table = pd.DataFrame(
        {
            "cell": selected_cells,
            "cell_type": obs.loc[selected_cells, "rctd_cell_type"].astype(str).to_numpy(),
            "total_counts": pd.to_numeric(obs.loc[selected_cells, "total_counts"], errors="coerce").to_numpy(),
            "sample_id": obs.loc[selected_cells, "sample_id"].astype(str).to_numpy(),
            "patient": obs.loc[selected_cells, "patient"].astype(str).to_numpy(),
        }
    )
    cell_table.to_csv(ref_dir / "cell_types.csv", index=False)
    pd.DataFrame(ref_summary_rows).to_csv(ref_dir / "reference_cell_type_summary.csv", index=False)
    return {
        "n_reference_cells": int(len(selected_cells)),
        "n_reference_genes": int(len(selected_genes)),
        "reference_cell_type_summary": ref_summary_rows,
    }


def export_spatial(args: argparse.Namespace, genes: list[str], out_dir: Path) -> list[dict]:
    sample_rows = []
    for h5ad_path in sorted(args.spatial_h5ad_dir.glob("*_bach1_nod_scored.h5ad")):
        sample = h5ad_path.name.replace("_bach1_nod_scored.h5ad", "")
        adata = read_h5ad_compat(h5ad_path)
        present = [gene for gene in genes if gene in adata.var_names]
        if len(present) < args.min_spatial_genes:
            sample_rows.append(
                {
                    "sample": sample,
                    "status": "skipped",
                    "reason": f"only {len(present)} selected genes present",
                    "n_spots": int(adata.n_obs),
                    "n_genes": int(len(present)),
                }
            )
            continue
        sub = adata[:, present].copy()
        counts = sub.layers["counts"] if "counts" in sub.layers else sub.X
        if not sparse.issparse(counts):
            counts = sparse.csr_matrix(counts)
        counts = counts.astype(np.int64).T.tocoo()

        sample_dir = out_dir / "spatial" / sample
        write_matrix_market(counts, sample_dir / "spatial_counts.mtx.gz")
        pd.Series(present).to_csv(sample_dir / "genes.tsv", sep="\t", index=False, header=False)
        pd.Series(sub.obs_names.astype(str)).to_csv(sample_dir / "spots.tsv", sep="\t", index=False, header=False)
        coords = pd.DataFrame(
            {
                "spot": sub.obs_names.astype(str),
                "x": pd.to_numeric(sub.obs["array_col"], errors="coerce").to_numpy(),
                "y": pd.to_numeric(sub.obs["array_row"], errors="coerce").to_numpy(),
                "sample": sub.obs["sample"].astype(str).to_numpy(),
                "patient_id": sub.obs["patient_id"].astype(str).to_numpy(),
                "tissue_group": sub.obs["tissue_group"].astype(str).to_numpy(),
                "total_counts": pd.to_numeric(sub.obs["total_counts"], errors="coerce").to_numpy(),
                "n_genes_by_counts": pd.to_numeric(sub.obs["n_genes_by_counts"], errors="coerce").to_numpy(),
                "BACH1_log_norm": pd.to_numeric(sub.obs["BACH1_log_norm"], errors="coerce").to_numpy(),
                "BACH1_detected": sub.obs["BACH1_detected"].astype(bool).to_numpy(),
                "BACH1_high": sub.obs["BACH1_high"].astype(bool).to_numpy(),
                "NOD_like_score_scanpy": pd.to_numeric(sub.obs["NOD_like_score_scanpy"], errors="coerce").to_numpy(),
                "NOD_like_high": sub.obs["NOD_like_high"].astype(bool).to_numpy(),
                "BACH1_NOD_cohigh": sub.obs["BACH1_NOD_cohigh"].astype(bool).to_numpy(),
            }
        )
        coords.to_csv(sample_dir / "spot_metadata.csv", index=False)
        sample_rows.append(
            {
                "sample": sample,
                "status": "ready",
                "path": str(sample_dir),
                "n_spots": int(sub.n_obs),
                "n_genes": int(len(present)),
                "patient_id": str(sub.obs["patient_id"].iloc[0]),
                "tissue_group": str(sub.obs["tissue_group"].iloc[0]),
            }
        )
    return sample_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrna", type=Path, default=SCRNA)
    parser.add_argument("--primary-meta", type=Path, default=PRIMARY_META)
    parser.add_argument("--spatial-h5ad-dir", type=Path, default=SPATIAL_H5AD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-cells-per-type", type=int, default=800)
    parser.add_argument("--max-genes", type=int, default=4000)
    parser.add_argument("--min-reference-umi", type=int, default=100)
    parser.add_argument("--min-spatial-genes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_spatial = next(iter(sorted(args.spatial_h5ad_dir.glob("*_bach1_nod_scored.h5ad"))))
    adata_sp = read_h5ad_compat(first_spatial)
    spatial_genes = set(adata_sp.var_names.astype(str))
    ref_summary = export_reference(args, spatial_genes, args.output_dir)
    genes = pd.read_csv(args.output_dir / "reference" / "genes.tsv", sep="\t", header=None)[0].astype(str).tolist()
    samples = export_spatial(args, genes, args.output_dir)
    manifest = {
        "reference": ref_summary,
        "n_spatial_sections": int(sum(row.get("status") == "ready" for row in samples)),
        "spatial_sections": samples,
        "outputs": {
            "reference_dir": str(args.output_dir / "reference"),
            "spatial_dir": str(args.output_dir / "spatial"),
            "manifest": str(args.output_dir / "rctd_input_manifest.json"),
        },
    }
    (args.output_dir / "rctd_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(samples).to_csv(args.output_dir / "spatial_sample_manifest.csv", index=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
