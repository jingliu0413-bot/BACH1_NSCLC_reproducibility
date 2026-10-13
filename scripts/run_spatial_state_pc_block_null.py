"""State-PC residualization and locally restricted spatial-null sensitivity.

DoRothEA–hypoxia uses PCs built from all four projected epithelial programmes.
Stress_AP1–hypoxia uses leave-one-program-out PCs built from the other three
programmes, avoiding self-adjustment. The null permutes score labels within
array-coordinate blocks, preserving coarse spatial gradients and the
within-block score distribution. This is a conservative, transparent
block-restricted null rather than a full variogram-preserving surrogate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
SPOT_TABLE = ROOT / "out" / "spatial_state_context_analysis" / "GSE292299_spatial_spot_scores.csv.gz"
OUT = ROOT / "out" / "spatial_state_context_analysis"
N_PERMUTATIONS = 1000
RANDOM_SEED = 20261004 + 177
PRIMARY_BLOCK_SIZE = 20
SENSITIVITY_BLOCK_SIZES = [10, 15, 20, 30]
STATE_COLUMNS = ["Stress_AP1", "Epithelial_core", "AT2_like", "Ciliated"]
STATE_COLUMNS_BY_COMPARISON = {
    "DoRothEA_vs_hypoxia": STATE_COLUMNS,
    "Stress_AP1_vs_hypoxia": ["Epithelial_core", "AT2_like", "Ciliated"],
}
N_PCS_BY_COMPARISON = {
    "DoRothEA_vs_hypoxia": 3,
    "Stress_AP1_vs_hypoxia": 2,
}

def six_neighbour_graph(coordinates: np.ndarray) -> sparse.csr_matrix:
    n_spots = len(coordinates)
    k = min(7, n_spots)
    _, neighbours = cKDTree(coordinates).query(coordinates, k=k)
    rows = np.repeat(np.arange(n_spots), k - 1)
    columns = neighbours[:, 1:].reshape(-1)
    values = np.full(len(rows), 1.0 / max(k - 1, 1))
    return sparse.csr_matrix((values, (rows, columns)), shape=(n_spots, n_spots))


def bivariate_moran(x: np.ndarray, y: np.ndarray, weights: sparse.csr_matrix) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return float("nan")
    if not valid.all():
        raise ValueError("Scores must be finite before spatial graph construction")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denominator == 0:
        return float("nan")
    return float((len(x) / weights.sum()) * np.dot(x_centered, weights @ y_centered) / denominator)


def residualize(y: np.ndarray, covariates: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(covariates)), covariates.to_numpy(dtype=float)])
    beta = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)[0]
    return np.asarray(y, dtype=float) - X @ beta


def state_pcs(
    df: pd.DataFrame, columns: list[str], requested_n_pcs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = df[columns].copy()
    for column in columns:
        state[column] = state.groupby(df["sample"], observed=True)[column].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
        )
    centered = state.to_numpy(dtype=float)
    centered = centered - centered.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    variance_explained = variance / variance.sum() if variance.sum() else variance
    n_pcs = min(requested_n_pcs, len(columns))
    pcs = u[:, :n_pcs] * singular_values[:n_pcs]
    pcs_df = pd.DataFrame(
        pcs,
        columns=[f"state_pc{i}" for i in range(1, n_pcs + 1)],
        index=df.index,
    )
    variance_df = pd.DataFrame(
        {
            "pc": [f"PC{i}" for i in range(1, len(variance_explained) + 1)],
            "variance_explained": variance_explained,
            "cumulative_variance_explained": np.cumsum(variance_explained),
            "n_state_features": len(columns),
            "state_pc_source": ";".join(columns),
            "n_pcs_used_for_residualization": n_pcs,
        }
    )
    return pcs_df, variance_df


def block_permutation(
    x: np.ndarray,
    y: np.ndarray,
    weights,
    block_ids: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[float, float, float, float, float, float]:
    observed = bivariate_moran(x, y, weights)
    null = np.empty(n_permutations, dtype=float)
    groups = [np.flatnonzero(block_ids == block) for block in pd.unique(block_ids)]
    for i in range(n_permutations):
        y_perm = np.asarray(y, dtype=float).copy()
        for idx in groups:
            if len(idx) > 1:
                y_perm[idx] = y_perm[rng.permutation(idx)]
        null[i] = bivariate_moran(x, y_perm, weights)
    p_positive = (1 + np.sum(null >= observed)) / (n_permutations + 1)
    return (
        float(observed),
        float(p_positive),
        float(np.nanmean(null)),
        float(np.nanstd(null)),
        float(np.nanquantile(null, 0.025)),
        float(np.nanquantile(null, 0.975)),
    )


def main() -> None:
    df = pd.read_csv(SPOT_TABLE)
    df["log_total_counts"] = np.log1p(df["raw_total_counts"].astype(float))
    rows = []
    raw_rows = []
    sensitivity_rows = []
    variance_rows = []

    # Fit each state-PC basis once on the pooled spatial table after
    # sample-wise score standardisation, matching the current analysis basis.
    pc_tables = {}
    for comparison, columns in STATE_COLUMNS_BY_COMPARISON.items():
        pcs, variance = state_pcs(df, columns, N_PCS_BY_COMPARISON[comparison])
        pc_tables[comparison] = pcs
        variance.insert(0, "sample", "pooled_sections")
        variance.insert(1, "comparison", comparison)
        variance_rows.append(variance)

    for sample, group in df.groupby("sample", sort=True):
        group = group.copy()
        original_index = group.index.to_numpy()
        group = group.reset_index(drop=True)
        coordinates = group[["array_row", "array_col"]].to_numpy(dtype=float)
        weights = six_neighbour_graph(coordinates)

        # Figure 5d uses the raw score pair. The state-PC residualized pair is
        # reserved for the supplementary block-size sensitivity analysis.
        raw_block_ids = (
            (group["array_row"] // PRIMARY_BLOCK_SIZE).astype(str)
            + "_"
            + (group["array_col"] // PRIMARY_BLOCK_SIZE).astype(str)
        ).to_numpy()
        raw_rng = np.random.default_rng(RANDOM_SEED)
        raw_moran, raw_p, raw_null_mean, raw_null_sd, raw_q025, raw_q975 = block_permutation(
            group["DOROTHEA_BACH1"].to_numpy(dtype=float),
            group["HALLMARK_HYPOXIA"].to_numpy(dtype=float),
            weights,
            raw_block_ids,
            raw_rng,
        )
        raw_rows.append(
            {
                "sample": sample,
                "comparison": "DoRothEA_vs_hypoxia",
                "n_spots": len(group),
                "bivariate_moran_I_raw": raw_moran,
                "block_permutation_p_positive": raw_p,
                "block_null_mean": raw_null_mean,
                "block_null_sd": raw_null_sd,
                "block_null_q025": raw_q025,
                "block_null_q975": raw_q975,
                "block_size_array_coordinates": PRIMARY_BLOCK_SIZE,
                "n_permutations": N_PERMUTATIONS,
                "random_seed": RANDOM_SEED,
                "inference_note": "Raw-score section-level block-restricted null for Figure 5d; not a patient-level inferential test.",
            }
        )

        for comparison, left, right in [
            ("DoRothEA_vs_hypoxia", "DOROTHEA_BACH1", "HALLMARK_HYPOXIA"),
            ("Stress_AP1_vs_hypoxia", "Stress_AP1", "HALLMARK_HYPOXIA"),
        ]:
            columns = STATE_COLUMNS_BY_COMPARISON[comparison]
            pcs = pc_tables[comparison].loc[original_index].reset_index(drop=True)
            group_with_pcs = pd.concat([group.reset_index(drop=True), pcs.reset_index(drop=True)], axis=1)
            n_pcs = N_PCS_BY_COMPARISON[comparison]
            pc_columns = [f"state_pc{i}" for i in range(1, min(n_pcs, len(columns)) + 1)]
            covariates = group_with_pcs[["log_total_counts", "raw_n_genes", *pc_columns]]
            residuals = {
                score: residualize(group_with_pcs[score].to_numpy(dtype=float), covariates)
                for score in ["DOROTHEA_BACH1", "HALLMARK_HYPOXIA", "Stress_AP1"]
            }
            x = residuals[left]
            y = residuals[right]
            rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))

            for block_size in [PRIMARY_BLOCK_SIZE]:
                block_ids = (
                    (group["array_row"] // block_size).astype(str)
                    + "_"
                    + (group["array_col"] // block_size).astype(str)
                ).to_numpy()
                rng = np.random.default_rng(RANDOM_SEED)
                moran, p_value, null_mean, null_sd, null_q025, null_q975 = block_permutation(
                    x, y, weights, block_ids, rng
                )
                rows.append(
                    {
                        "sample": sample,
                        "comparison": comparison,
                        "n_spots": len(group),
                        "spearman_rho_state_pc_residualized": rho,
                        "bivariate_moran_I_state_pc_residualized": moran,
                        "block_permutation_p_positive": p_value,
                        "block_null_mean": null_mean,
                        "block_null_sd": null_sd,
                        "block_null_q025": null_q025,
                        "block_null_q975": null_q975,
                        "block_size_array_coordinates": block_size,
                        "n_permutations": N_PERMUTATIONS,
                        "random_seed": RANDOM_SEED,
                        "state_pc_source": ";".join(columns),
                        "state_pc_adjustment_note": (
                            "Full four-programme PCs"
                            if comparison == "DoRothEA_vs_hypoxia"
                            else "Leave-one-programme-out PCs excluding Stress_AP1"
                        ),
                        "inference_note": "Exploratory section-level block-restricted null; not a patient-level inferential test.",
                    }
                )

            for block_size in SENSITIVITY_BLOCK_SIZES:
                block_ids = (
                    (group["array_row"] // block_size).astype(str)
                    + "_"
                    + (group["array_col"] // block_size).astype(str)
                ).to_numpy()
                sensitivity_seed = RANDOM_SEED + block_size
                rng = np.random.default_rng(sensitivity_seed)
                moran, p_value, null_mean, null_sd, null_q025, null_q975 = block_permutation(
                    x, y, weights, block_ids, rng
                )
                sensitivity_rows.append(
                    {
                        "sample": sample,
                        "comparison": comparison,
                        "n_spots": len(group),
                        "bivariate_moran_I_state_pc_residualized": moran,
                        "block_permutation_p_positive": p_value,
                        "block_null_mean": null_mean,
                        "block_null_sd": null_sd,
                        "block_null_q025": null_q025,
                        "block_null_q975": null_q975,
                        "block_size_array_coordinates": block_size,
                        "n_permutations": N_PERMUTATIONS,
                        "random_seed": sensitivity_seed,
                        "state_pc_source": ";".join(columns),
                        "inference_note": "Exploratory section-level block-restricted null; not a patient-level inferential test.",
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "GSE292299_spatial_state_pc_block_null.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(OUT / "GSE292299_spatial_raw_block_null.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(
        OUT / "GSE292299_spatial_block_size_sensitivity.csv", index=False
    )
    pd.concat(variance_rows, ignore_index=True).to_csv(
        OUT / "GSE292299_spatial_state_pc_variance_explained.csv", index=False
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
