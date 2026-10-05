"""Project the frozen BACH1/hypoxia scores onto external Visium matrices.

This is a score-coverage feasibility audit for the lung samples in GSE292299.
It intentionally does not make spatial-neighbourhood or patient-level claims,
because the first pass uses expression H5 files without the matching image/
coordinate archives.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "manuscript_submission_20260911"
MATRIX_DIR = ROOT / "data" / "spatial_candidates" / "GSE292299_lung_h5"
METADATA = ROOT / "data" / "spatial_candidates" / "GSE292299_sample_metadata.csv.gz"
OUT = ROOT / "out" / "spatial_external_score_feasibility"
OUT.mkdir(parents=True, exist_ok=True)

SIGNATURES = [
    "DOROTHEA_BACH1_ABC_TF_ACTIVITY",
    "HALLMARK_HYPOXIA",
    "KLENJA2025_BACH1_INVERSE_ACTIVITY",
]
DEOVERLAP = {"ALDOA", "HMOX1", "IL6"}


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    return np.zeros_like(x) if not np.isfinite(sd) or sd == 0 else (x - np.nanmean(x)) / sd


def score(adata: sc.AnnData, genes: list[str], weights: list[float]) -> tuple[np.ndarray, int, list[str]]:
    gene_names = pd.Index(adata.var_names.astype(str)).str.upper()
    parts = []
    present = []
    for gene, weight in zip(genes, weights):
        hits = np.flatnonzero(gene_names.to_numpy() == gene)
        if len(hits) == 0:
            continue
        values = adata.X[:, hits]
        if sparse.issparse(values):
            values = values.toarray()
        values = np.nanmedian(np.asarray(values, dtype=float), axis=1)
        parts.append(float(weight) * zscore(values))
        present.append(gene)
    if not parts:
        return np.full(adata.n_obs, np.nan), 0, []
    return np.sum(parts, axis=0) / max(np.sum(np.abs([w for g, w in zip(genes, weights) if g in present])), 1e-12), len(present), present


def main() -> None:
    targets = pd.read_csv(SOURCE / "external_bach1_signature_targets.csv")
    targets["target"] = targets["target"].astype(str).str.upper()
    pathways = pd.read_csv(SOURCE / "phenotype_pathway_signature_targets.csv")
    pathways["target"] = pathways["target"].astype(str).str.upper()
    metadata = pd.read_csv(METADATA).set_index("ID_Sample")
    target_rows = {}
    for signature in SIGNATURES:
        source = pathways if signature == "HALLMARK_HYPOXIA" else targets
        sub = source[source["signature"].eq(signature)].copy()
        sub = sub[sub["present_in_primary"].astype(str).str.lower().isin(["true", "1", "yes"])]
        sub = sub[sub["target_in_adata"].notna()]
        sub["target"] = sub["target_in_adata"].astype(str).str.upper()
        sub = sub.drop_duplicates("target")
        if signature in {"DOROTHEA_BACH1_ABC_TF_ACTIVITY", "HALLMARK_HYPOXIA"}:
            sub = sub[~sub["target"].isin(DEOVERLAP)]
        target_rows[signature] = (sub["target"].tolist(), sub["weight"].astype(float).tolist())

    rows = []
    correlation_rows = []
    skipped = []
    for path in sorted(MATRIX_DIR.glob("*.h5")):
        try:
            adata = sc.read_10x_h5(path, gex_only=True)
        except (OSError, ValueError, EOFError) as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
            continue
        adata.var_names = adata.var_names.astype(str).str.upper()
        adata.var_names_make_unique()
        sc.pp.normalize_total(adata, target_sum=10_000)
        sc.pp.log1p(adata)
        match = re.search(r"_(NSCLC_P\d+)_filtered", path.name)
        sample = match.group(1) if match else path.stem
        accession = path.name.split("_")[0]
        row = {
            "sample": sample,
            "accession": accession,
            "n_spots": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
        }
        if sample in metadata.index:
            sample_meta = metadata.loc[sample]
            row.update(
                {
                    "biopsy_tissue": sample_meta.get("Bx_Tissue"),
                    "histology": sample_meta.get("Diagnosis_Subtype"),
                    "treatment_response": sample_meta.get("Tx_Response"),
                }
            )
        score_values = {}
        for signature, (genes, weights) in target_rows.items():
            values, n_present, present = score(adata, genes, weights)
            score_values[signature] = values
            row[f"{signature}__n_targets"] = len(genes)
            row[f"{signature}__n_present"] = n_present
            row[f"{signature}__coverage"] = n_present / len(genes) if genes else np.nan
            row[f"{signature}__mean"] = float(np.nanmean(values))
            row[f"{signature}__sd"] = float(np.nanstd(values))
            row[f"{signature}__finite_fraction"] = float(np.mean(np.isfinite(values)))
            row[f"{signature}__present_genes"] = ";".join(present)
        rows.append(row)
        comparisons = [
            ("dorothea_vs_hypoxia", SIGNATURES[0], SIGNATURES[1]),
            ("klenja_vs_hypoxia", SIGNATURES[2], SIGNATURES[1]),
            ("dorothea_vs_klenja", SIGNATURES[0], SIGNATURES[2]),
        ]
        for comparison, left, right in comparisons:
            x = score_values[left]
            y = score_values[right]
            ok = np.isfinite(x) & np.isfinite(y)
            rho = stats.spearmanr(x[ok], y[ok]).statistic if ok.sum() >= 4 else np.nan
            correlation_rows.append(
                {
                    "sample": sample,
                    "accession": accession,
                    "comparison": comparison,
                    "spearman_rho_descriptive": float(rho),
                    "n_spots": int(ok.sum()),
                    "inference_note": "Descriptive only; spatial spots are not independent.",
                }
            )
    table = pd.DataFrame(rows)
    correlations = pd.DataFrame(correlation_rows)
    table.to_csv(OUT / "GSE292299_lung_score_coverage.csv", index=False)
    correlations.to_csv(OUT / "GSE292299_lung_score_correlations_descriptive.csv", index=False)
    summary = {
        "dataset": "GSE292299",
        "sample_scope": "lung biopsy samples with complete H5 matrices; spatial coordinates/images not included in this first pass",
        "n_samples": int(len(table)),
        "skipped_files": skipped,
        "coverage": {
            signature: {
                "min": float(table[f"{signature}__coverage"].min()),
                "max": float(table[f"{signature}__coverage"].max()),
                "mean": float(table[f"{signature}__coverage"].mean()),
            }
            for signature in SIGNATURES
        },
    }
    (OUT / "GSE292299_lung_score_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(correlations.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
