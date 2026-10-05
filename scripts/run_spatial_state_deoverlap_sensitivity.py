"""Recompute spatial state scores after de-overlapping BACH1 and hypoxia genes.

The primary spatial scores already exclude the three DoRothEA/Hallmark overlap
genes. This sensitivity removes the full 81-gene de-overlapped DoRothEA target
set plus the 191-gene de-overlapped Hallmark hypoxia set from the four projected
epithelial-state programmes before rebuilding state PCs and residuals.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "manuscript_submission_20260911"
MATRIX_DIR = ROOT / "data" / "spatial_candidates" / "GSE292299_lung_h5"
SPOT_TABLE = ROOT / "out" / "spatial_state_context_analysis" / "GSE292299_spatial_spot_scores.csv.gz"
STRESS_SIGNATURE = ROOT / "outputs" / "manuscript_submission_20260914" / "tables" / "stress_ap1_state_signature_genes.csv"
OUT = ROOT / "out" / "spatial_state_context_analysis"

TARGET_SUM = 10_000.0
STATE_COLUMNS = ["Stress_AP1", "Epithelial_core", "AT2_like", "Ciliated"]
STATE_MARKERS = {
    "Epithelial_core": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "CLDN4", "KRT7"],
    "AT2_like": ["SFTPA1", "SFTPA2", "SFTPB", "SFTPC", "SLC34A2", "ABCA3", "NAPSA", "LAMP3"],
    "Ciliated": ["FOXJ1", "TPPP3", "PIFO", "DNAH5", "DNAI1", "CAPS"],
}


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / sd


def load_signature_targets() -> tuple[dict[str, tuple[list[str], list[float]]], set[str]]:
    external = pd.read_csv(SOURCE / "external_bach1_signature_targets.csv")
    phenotype = pd.read_csv(SOURCE / "phenotype_pathway_signature_targets.csv")
    external["target"] = external["target"].astype(str).str.upper()
    phenotype["target"] = phenotype["target"].astype(str).str.upper()
    out: dict[str, tuple[list[str], list[float]]] = {}
    for name, table, signature in [
        ("DOROTHEA_BACH1", external, "DOROTHEA_BACH1_ABC_TF_ACTIVITY"),
        ("HALLMARK_HYPOXIA", phenotype, "HALLMARK_HYPOXIA"),
    ]:
        source = table[table["signature"].eq(signature)].copy()
        source = source[source["present_in_primary"].astype(str).str.lower().isin(["true", "1", "yes"])]
        source = source[source["target_in_adata"].notna()]
        source["target"] = source["target_in_adata"].astype(str).str.upper()
        source = source.drop_duplicates("target")
        source = source[~source["target"].isin({"ALDOA", "HMOX1", "IL6"})]
        out[name] = (source["target"].tolist(), source["weight"].astype(float).tolist())
    overlap = set(out["DOROTHEA_BACH1"][0]) | set(out["HALLMARK_HYPOXIA"][0])
    stress = pd.read_csv(STRESS_SIGNATURE)["gene"].astype(str).str.upper().tolist()
    out["Stress_AP1"] = (stress, [1.0] * len(stress))
    for state, genes in STATE_MARKERS.items():
        out[state] = ([gene.upper() for gene in genes], [1.0] * len(genes))
    return out, overlap


def load_10x_h5(path: Path) -> tuple[sparse.csc_matrix, list[str], list[str]]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(x) for x in matrix["shape"][...])
        data = matrix["data"][...]
        indices = matrix["indices"][...]
        indptr = matrix["indptr"][...]
        genes = [x.decode() if isinstance(x, bytes) else str(x) for x in matrix["features"]["name"][...]]
        barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in matrix["barcodes"][...]]
    return sparse.csc_matrix((data, indices, indptr), shape=shape), genes, barcodes


def score_matrix(
    counts: sparse.csc_matrix,
    gene_index: dict[str, int],
    genes: list[str],
    weights: list[float],
    library_size: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    parts = []
    used_weights = []
    present = []
    for gene, weight in zip(genes, weights):
        idx = gene_index.get(gene)
        if idx is None:
            continue
        values = counts.getrow(idx).toarray().ravel().astype(float)
        values = np.log1p(values / np.maximum(library_size, 1.0) * TARGET_SUM)
        parts.append(float(weight) * zscore(values))
        used_weights.append(float(weight))
        present.append(gene)
    if not parts:
        return np.full(counts.shape[1], np.nan), []
    return np.sum(parts, axis=0) / max(np.sum(np.abs(used_weights)), 1e-12), present


def pooled_state_pcs(table: pd.DataFrame, columns: list[str], n_pcs: int) -> pd.DataFrame:
    state = table[columns].copy()
    for column in columns:
        state[column] = state.groupby(table["sample"], observed=True)[column].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
        )
    centered = state.to_numpy(dtype=float)
    centered -= centered.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    pcs = u[:, :n_pcs] * singular_values[:n_pcs]
    return pd.DataFrame(pcs, columns=[f"pc{i}" for i in range(1, n_pcs + 1)], index=table.index)


def residualize(values: np.ndarray, covariates: pd.DataFrame) -> np.ndarray:
    x = np.column_stack([np.ones(len(covariates)), covariates.to_numpy(dtype=float)])
    beta = np.linalg.lstsq(x, np.asarray(values, dtype=float), rcond=None)[0]
    return np.asarray(values, dtype=float) - x @ beta


def rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def main() -> None:
    signatures, deoverlap_genes = load_signature_targets()
    original = pd.read_csv(SPOT_TABLE)
    original["log_total_counts"] = np.log1p(original["raw_total_counts"].astype(float))

    state_audit = []
    for state in STATE_COLUMNS:
        genes = signatures[state][0]
        retained = [gene for gene in genes if gene not in deoverlap_genes]
        state_audit.append(
            {
                "state": state,
                "n_original_genes": len(genes),
                "n_removed_due_to_bach1_or_hypoxia_overlap": len(genes) - len(retained),
                "removed_genes": ";".join(sorted(set(genes) & deoverlap_genes)),
                "n_retained_genes": len(retained),
                "deoverlap_gene_pool": len(deoverlap_genes),
            }
        )

    recomputed = []
    score_audit = []
    for path in sorted(MATRIX_DIR.glob("*.h5")):
        match = re.search(r"_(NSCLC_P\d+)_filtered", path.name)
        if not match:
            continue
        sample = match.group(1)
        counts, genes, barcodes = load_10x_h5(path)
        gene_index = {}
        for idx, gene in enumerate(pd.Index(genes).astype(str).str.upper()):
            gene_index.setdefault(gene, idx)
        library_size = np.asarray(counts.sum(axis=0)).ravel().astype(float)

        values = {}
        for signature in ["DOROTHEA_BACH1", "HALLMARK_HYPOXIA"]:
            vals, present = score_matrix(counts, gene_index, *signatures[signature], library_size)
            values[signature] = vals
            score_audit.append(
                {
                    "sample": sample,
                    "signature": signature,
                    "n_targets": len(signatures[signature][0]),
                    "n_present": len(present),
                    "score_version": "canonical de-overlapped score",
                }
            )
        for state in STATE_COLUMNS:
            genes_state = [gene for gene in signatures[state][0] if gene not in deoverlap_genes]
            weights_state = [1.0] * len(genes_state)
            values[state], present = score_matrix(counts, gene_index, genes_state, weights_state, library_size)
            score_audit.append(
                {
                    "sample": sample,
                    "signature": state,
                    "n_targets": len(genes_state),
                    "n_present": len(present),
                    "score_version": "state signature after full BACH1+hypoxia de-overlap",
                }
            )
        current = original[original["sample"].eq(sample)].copy().set_index("barcode")
        order = [barcode.decode() if isinstance(barcode, bytes) else str(barcode) for barcode in barcodes]
        order = [barcode for barcode in order if barcode in current.index]
        current = current.loc[order].copy()
        rows = pd.DataFrame({"sample": sample, "barcode": order}, index=order)
        for name, vals in values.items():
            value_map = dict(zip([barcode.decode() if isinstance(barcode, bytes) else str(barcode) for barcode in barcodes], vals))
            rows[name] = [value_map[barcode] for barcode in order]
        for column in ["raw_total_counts", "raw_n_genes", *STATE_COLUMNS]:
            rows[f"original_{column}"] = current[column].to_numpy()
        for state in STATE_COLUMNS:
            rows[f"deoverlap_{state}"] = rows[state]
        rows["log_total_counts"] = np.log1p(rows["original_raw_total_counts"].astype(float))
        recomputed.append(rows.reset_index(drop=True))

    table = pd.concat(recomputed, ignore_index=True)
    results = []
    comparisons = [
        ("DoRothEA_vs_hypoxia", "DOROTHEA_BACH1", "HALLMARK_HYPOXIA", STATE_COLUMNS, 3),
        ("Stress_AP1_vs_hypoxia", "Stress_AP1", "HALLMARK_HYPOXIA", ["Epithelial_core", "AT2_like", "Ciliated"], 2),
    ]
    for comparison, left, right, pc_states, n_pcs in comparisons:
        original_pc = pooled_state_pcs(table, [f"original_{state}" for state in pc_states], n_pcs)
        deoverlap_pc = pooled_state_pcs(table, [f"deoverlap_{state}" for state in pc_states], n_pcs)
        for sample, group in table.groupby("sample", sort=True):
            idx = group.index
            base_cov = group[["log_total_counts", "original_raw_n_genes"]].rename(
                columns={"original_raw_n_genes": "raw_n_genes"}
            )
            original_left = group[f"original_{left}"].to_numpy(float) if left in STATE_COLUMNS else group[left].to_numpy(float)
            deoverlap_left = group[f"deoverlap_{left}"].to_numpy(float) if left in STATE_COLUMNS else group[left].to_numpy(float)
            right_values = group[right].to_numpy(float)
            original_cov = pd.concat([base_cov.reset_index(drop=True), original_pc.loc[idx].reset_index(drop=True)], axis=1)
            deoverlap_cov = pd.concat([base_cov.reset_index(drop=True), deoverlap_pc.loc[idx].reset_index(drop=True)], axis=1)
            results.append(
                {
                    "sample": sample,
                    "comparison": comparison,
                    "n_spots": len(group),
                    "raw_rho": rho(original_left, right_values),
                    "original_state_pc_residual_rho": rho(residualize(original_left, original_cov), residualize(right_values, original_cov)),
                    "deoverlap_state_pc_residual_rho": rho(residualize(deoverlap_left, deoverlap_cov), residualize(right_values, deoverlap_cov)),
                    "n_state_pcs": n_pcs,
                    "state_pc_source": ";".join(pc_states),
                    "deoverlap_pool_size": len(deoverlap_genes),
                    "normalization_target_sum": TARGET_SUM,
                }
            )

    # Confirm that the recomputed canonical scores agree with the locked spot table.
    score_checks = []
    for column in ["DOROTHEA_BACH1", "HALLMARK_HYPOXIA"]:
        left = table[column].to_numpy(float)
        right = original.set_index(["sample", "barcode"]).loc[
            pd.MultiIndex.from_frame(table[["sample", "barcode"]]), column
        ].to_numpy(float)
        score_checks.append({"score": column, "max_abs_difference_vs_locked_table": float(np.nanmax(np.abs(left - right)))})

    pd.DataFrame(state_audit).to_csv(OUT / "GSE292299_spatial_state_deoverlap_gene_audit.csv", index=False)
    pd.DataFrame(score_audit).to_csv(OUT / "GSE292299_spatial_state_deoverlap_score_coverage.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "GSE292299_spatial_state_deoverlap_residualization.csv", index=False)
    pd.DataFrame(score_checks).to_csv(OUT / "GSE292299_spatial_deoverlap_score_reproducibility_check.csv", index=False)
    print(pd.DataFrame(state_audit).to_string(index=False))
    print(pd.DataFrame(results).to_string(index=False))
    print(pd.DataFrame(score_checks).to_string(index=False))


if __name__ == "__main__":
    main()
