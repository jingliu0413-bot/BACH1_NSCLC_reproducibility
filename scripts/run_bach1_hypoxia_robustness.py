"""Robustness checks for the DoRothEA-BACH1/hypoxia association.

This script addresses reviewer concerns about gene-set overlap, sequencing
depth, dataset/epithelial-state structure, and method dependence in the
external BACH1 activity analysis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[1]
H5AD = ROOT / "out/bach1_malignant_epithelial_primary/nsclc_malignant_epithelial_primary_bach1_analysis_object.h5ad"
EXT = ROOT / "out/external_bach1_activity_primary"
EXT_TABLE = EXT / "tables"
OUT = ROOT / "out/external_bach1_activity_robustness"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"

DOR_SIG = "DOROTHEA_BACH1_ABC_TF_ACTIVITY"
COL_SIG = "COLLECTRI_BACH1_TF_ACTIVITY"
KLENJA_SIG = "KLENJA2025_BACH1_INVERSE_ACTIVITY"
EFFECTOR_SIG = "LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY"
HYPOXIA_SIG = "HALLMARK_HYPOXIA"

DOR_COL = f"{DOR_SIG}__mean_z"
COL_COL = f"{COL_SIG}__mean_z"
KLENJA_COL = f"{KLENJA_SIG}__mean_z"
EFFECTOR_COL = f"{EFFECTOR_SIG}__mean_z"
HYPOXIA_COL = f"{HYPOXIA_SIG}__mean_z"
DOR_NO_OVERLAP_COL = f"{DOR_SIG}_NO_HYPOXIA_OVERLAP__mean_z"
HYPOXIA_NO_OVERLAP_COL = f"{HYPOXIA_SIG}_NO_DOROTHEA_OVERLAP__mean_z"

RANDOM_SEED = 1701
N_PERMUTATIONS = 1000


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def bh_adjust(pvalues: Iterable[float]) -> list[float]:
    p = np.asarray([np.nan if x is None else float(x) for x in pvalues], dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out.tolist()
    pv = p[ok]
    order = np.argsort(pv)
    ranks = np.arange(1, len(pv) + 1)
    q = np.empty_like(pv)
    q[order] = pv[order] * len(pv) / ranks
    q[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    out[ok] = np.clip(q, 0, 1)
    return out.tolist()


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    p = np.asarray([np.nan if x is None else float(x) for x in pvalues], dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out.tolist()
    pv = p[ok]
    order = np.argsort(pv)
    adjusted_sorted = np.maximum.accumulate((len(pv) - np.arange(len(pv))) * pv[order])
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    adjusted = np.empty_like(pv)
    adjusted[order] = adjusted_sorted
    out[ok] = adjusted
    return out.tolist()


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, int]:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or len(np.unique(a[ok])) < 2 or len(np.unique(b[ok])) < 2:
        return np.nan, np.nan, int(ok.sum())
    rho, pvalue = stats.spearmanr(a[ok], b[ok])
    return float(rho), float(pvalue), int(ok.sum())


def safe_wilcoxon(delta: Iterable[float]) -> tuple[float, float, int]:
    d = pd.to_numeric(pd.Series(delta), errors="coerce").dropna().to_numpy(dtype=float)
    d = d[np.abs(d) > 1e-12]
    if d.size < 3:
        return np.nan, np.nan, int(d.size)
    try:
        stat, pvalue = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return np.nan, np.nan, int(d.size)
    return float(stat), float(pvalue), int(d.size)


def dense_vector(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x).ravel()
    return np.asarray(x).ravel()


def get_gene_index(ad: sc.AnnData) -> dict[str, int]:
    return {str(g).upper(): i for i, g in enumerate(ad.var_names.astype(str))}


def present_signature_genes(targets: pd.DataFrame, signature: str) -> pd.DataFrame:
    sub = targets[targets["signature"].astype(str).eq(signature)].copy()
    sub = sub[sub["present_in_primary"].astype(str).str.lower().isin(["true", "1", "yes"])]
    sub = sub[sub["target_in_adata"].astype(str).ne("")]
    sub = sub.drop_duplicates(["signature", "target_in_adata"])
    sub["weight"] = pd.to_numeric(sub.get("weight", 1.0), errors="coerce").fillna(1.0)
    return sub


def gene_stats(ad: sc.AnnData) -> pd.DataFrame:
    x = ad.X
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    x = x.tocsr()
    n = x.shape[0]
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    sd = np.sqrt(variance)
    detected = np.asarray((x > 0).mean(axis=0)).ravel()
    out = pd.DataFrame(
        {
            "gene": ad.var_names.astype(str),
            "gene_index": np.arange(x.shape[1]),
            "mean_log_expression": mean,
            "sd_log_expression": sd,
            "detected_fraction": detected,
            "n_detected_cells": np.round(detected * n).astype(int),
        }
    )
    out.to_csv(TABLE_DIR / "primary_gene_expression_detection_statistics.csv", index=False)
    return out


def score_sparse(ad: sc.AnnData, genes: list[str], weights: list[float], stats_df: pd.DataFrame) -> np.ndarray:
    idx_map = dict(zip(stats_df["gene"].astype(str), stats_df["gene_index"].astype(int)))
    mean_map = dict(zip(stats_df["gene"].astype(str), stats_df["mean_log_expression"].astype(float)))
    sd_map = dict(zip(stats_df["gene"].astype(str), stats_df["sd_log_expression"].astype(float)))
    idx: list[int] = []
    use_weights: list[float] = []
    means: list[float] = []
    sds: list[float] = []
    for gene, weight in zip(genes, weights):
        if gene not in idx_map:
            continue
        sd = float(sd_map[gene])
        if not np.isfinite(sd) or sd <= 0:
            continue
        idx.append(int(idx_map[gene]))
        use_weights.append(float(weight))
        means.append(float(mean_map[gene]))
        sds.append(sd)
    if not idx:
        return np.full(ad.n_obs, np.nan)
    w = np.asarray(use_weights, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    sds_arr = np.asarray(sds, dtype=float)
    denom = np.sum(np.abs(w))
    if denom <= 0:
        return np.full(ad.n_obs, np.nan)
    coeff = (w / sds_arr) / denom
    offset = float(np.sum(w * means_arr / sds_arr) / denom)
    values = np.asarray(ad.X[:, idx].dot(coeff)).ravel() - offset
    return values.astype(float)


def patient_mean(values: np.ndarray, patients: pd.Series) -> pd.Series:
    return pd.Series(values, index=patients.index).groupby(patients.astype(str), observed=True).mean()


def overlap_tables(bach1_targets: pd.DataFrame, phenotype_targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bach1_sigs = [DOR_SIG, COL_SIG, KLENJA_SIG, EFFECTOR_SIG]
    phen_sigs = sorted(phenotype_targets["signature"].astype(str).unique())
    rows = []
    gene_rows = []
    for bsig in bach1_sigs:
        bgenes = set(present_signature_genes(bach1_targets, bsig)["target_in_adata"].astype(str))
        for psig in phen_sigs:
            pgenes = set(present_signature_genes(phenotype_targets, psig)["target_in_adata"].astype(str))
            inter = sorted(bgenes & pgenes)
            union = bgenes | pgenes
            rows.append(
                {
                    "bach1_signature": bsig,
                    "phenotype_signature": psig,
                    "n_bach1_genes": len(bgenes),
                    "n_phenotype_genes": len(pgenes),
                    "n_overlap": len(inter),
                    "jaccard_index": len(inter) / len(union) if union else np.nan,
                    "overlap_coefficient": len(inter) / min(len(bgenes), len(pgenes)) if bgenes and pgenes else np.nan,
                    "overlap_genes": ";".join(inter),
                }
            )
            for gene in inter:
                gene_rows.append({"bach1_signature": bsig, "phenotype_signature": psig, "gene": gene})
    summary = pd.DataFrame(rows).sort_values(["bach1_signature", "phenotype_signature"])
    genes = pd.DataFrame(gene_rows)
    summary.to_csv(TABLE_DIR / "bach1_phenotype_gene_set_overlap_summary.csv", index=False)
    genes.to_csv(TABLE_DIR / "bach1_phenotype_gene_set_overlap_genes.csv", index=False)
    return summary, genes


def deoverlap_scores(ad: sc.AnnData, targets: pd.DataFrame, pathways: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    dor = present_signature_genes(targets, DOR_SIG)
    hyp = present_signature_genes(pathways, HYPOXIA_SIG)
    overlap = sorted(set(dor["target_in_adata"].astype(str)) & set(hyp["target_in_adata"].astype(str)))
    dor_no = dor[~dor["target_in_adata"].astype(str).isin(overlap)].copy()
    hyp_no = hyp[~hyp["target_in_adata"].astype(str).isin(overlap)].copy()
    cell = pd.DataFrame(index=ad.obs_names.astype(str))
    cell[DOR_NO_OVERLAP_COL] = score_sparse(
        ad,
        dor_no["target_in_adata"].astype(str).tolist(),
        dor_no["weight"].astype(float).tolist(),
        stats_df,
    )
    cell[HYPOXIA_NO_OVERLAP_COL] = score_sparse(
        ad,
        hyp_no["target_in_adata"].astype(str).tolist(),
        hyp_no["weight"].astype(float).tolist(),
        stats_df,
    )
    cell.to_csv(TABLE_DIR / "deoverlapped_dorothea_hypoxia_cell_scores.csv.gz")
    return cell


def deoverlap_correlations(patient_scores: pd.DataFrame) -> pd.DataFrame:
    tests = [
        ("original", DOR_COL, HYPOXIA_COL),
        ("dorothea_without_shared_vs_original_hypoxia", DOR_NO_OVERLAP_COL, HYPOXIA_COL),
        ("original_dorothea_vs_hypoxia_without_shared", DOR_COL, HYPOXIA_NO_OVERLAP_COL),
        ("both_signatures_without_shared", DOR_NO_OVERLAP_COL, HYPOXIA_NO_OVERLAP_COL),
    ]
    rows = []
    lopo_rows = []
    for label, xcol, ycol in tests:
        rho, pvalue, n = safe_spearman(patient_scores[xcol], patient_scores[ycol])
        rows.append({"comparison": label, "x": xcol, "y": ycol, "n_patients": n, "spearman_rho": rho, "spearman_p": pvalue})
        for held in patient_scores.index.astype(str):
            keep = patient_scores[patient_scores.index.astype(str) != held]
            lrho, lp, ln = safe_spearman(keep[xcol], keep[ycol])
            lopo_rows.append(
                {
                    "comparison": label,
                    "held_out_patient": held,
                    "n_patients": ln,
                    "spearman_rho": lrho,
                    "spearman_p": lp,
                    "sign": "positive" if np.isfinite(lrho) and lrho > 0 else ("negative" if np.isfinite(lrho) and lrho < 0 else "zero_or_nan"),
                }
            )
    out = pd.DataFrame(rows)
    out["spearman_holm_p"] = holm_adjust(out["spearman_p"])
    out["spearman_bh_q"] = bh_adjust(out["spearman_p"])
    lopo = pd.DataFrame(lopo_rows)
    lopo_summary = (
        lopo.groupby("comparison")
        .agg(
            n_lopo=("held_out_patient", "nunique"),
            min_rho=("spearman_rho", "min"),
            max_rho=("spearman_rho", "max"),
            median_rho=("spearman_rho", "median"),
            max_p=("spearman_p", "max"),
            n_positive=("sign", lambda s: int((s == "positive").sum())),
            n_negative=("sign", lambda s: int((s == "negative").sum())),
        )
        .reset_index()
    )
    lopo_summary["sign_stability"] = np.where(
        lopo_summary["n_positive"].eq(lopo_summary["n_lopo"]),
        "all_positive",
        np.where(lopo_summary["n_negative"].eq(lopo_summary["n_lopo"]), "all_negative", "mixed_or_zero"),
    )
    out.to_csv(TABLE_DIR / "dorothea_hypoxia_deoverlap_patient_correlations.csv", index=False)
    lopo.to_csv(TABLE_DIR / "dorothea_hypoxia_deoverlap_lopo.csv", index=False)
    lopo_summary.to_csv(TABLE_DIR / "dorothea_hypoxia_deoverlap_lopo_summary.csv", index=False)
    return out


def make_bins(values: pd.Series, n_bins: int) -> pd.Series:
    ranks = values.rank(method="first")
    try:
        return pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop").astype(int)
    except ValueError:
        return pd.Series(np.zeros(values.shape[0], dtype=int), index=values.index)


def matched_random_gene_sets(
    ad: sc.AnnData,
    stats_df: pd.DataFrame,
    targets: pd.DataFrame,
    pathways: pd.DataFrame,
    patient_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dor = present_signature_genes(targets, DOR_SIG)
    hyp = present_signature_genes(pathways, HYPOXIA_SIG)
    shared = sorted(set(dor["target_in_adata"].astype(str)) & set(hyp["target_in_adata"].astype(str)))
    dor_no = dor[~dor["target_in_adata"].astype(str).isin(shared)].copy()
    hyp_no = hyp[~hyp["target_in_adata"].astype(str).isin(shared)].copy()

    # Use a conservative common exclusion pool for both null families so that
    # shared genes cannot re-enter the de-overlapped random sets.
    exclude = set(dor["target_in_adata"].astype(str)) | set(hyp["target_in_adata"].astype(str)) | {"BACH1"}

    stats_df = stats_df.copy()
    stats_df["mean_bin"] = make_bins(stats_df["mean_log_expression"], 20)
    stats_df["detect_bin"] = make_bins(stats_df["detected_fraction"], 20)
    stats_df["is_candidate"] = ~stats_df["gene"].astype(str).isin(exclude)
    bin_to_genes: dict[tuple[int, int], list[str]] = {}
    for _, row in stats_df[stats_df["is_candidate"]].iterrows():
        key = (int(row["mean_bin"]), int(row["detect_bin"]))
        bin_to_genes.setdefault(key, []).append(str(row["gene"]))

    stat_map = stats_df.set_index("gene")[["mean_bin", "detect_bin"]]
    patient_labels = ad.obs["patient"].astype(str)

    all_mean_bins = sorted(stats_df["mean_bin"].dropna().astype(int).unique())
    all_detect_bins = sorted(stats_df["detect_bin"].dropna().astype(int).unique())
    specs = [
        {
            "comparison": "original_hypoxia",
            "source_score_col": DOR_COL,
            "outcome_score_col": HYPOXIA_COL,
            "targets": dor,
            "random_seed": RANDOM_SEED,
            "wide_prefix": "original_hypoxia",
        },
        {
            "comparison": "deoverlapped_hypoxia",
            "source_score_col": DOR_NO_OVERLAP_COL,
            "outcome_score_col": HYPOXIA_NO_OVERLAP_COL,
            "targets": dor_no,
            "random_seed": RANDOM_SEED + 1,
            "wide_prefix": "deoverlapped_hypoxia",
        },
    ]

    long_rows = []
    match_rows = []
    summary = []
    wide_by_perm: dict[int, dict[str, float | int]] = {
        perm + 1: {"permutation": perm + 1} for perm in range(N_PERMUTATIONS)
    }
    input_rows = []
    for spec in specs:
        rng = np.random.default_rng(int(spec["random_seed"]))
        source = spec["targets"]
        source_genes = source["target_in_adata"].astype(str).tolist()
        source_weights = source["weight"].astype(float).tolist()
        observed = safe_spearman(patient_scores[str(spec["source_score_col"])], patient_scores[str(spec["outcome_score_col"])])[0]
        input_rows.append(
            {
                "comparison": spec["comparison"],
                "source_score_col": spec["source_score_col"],
                "outcome_score_col": spec["outcome_score_col"],
                "random_seed": spec["random_seed"],
                "n_source_genes": len(source_genes),
                "excluded_from_candidate_pool": ";".join(sorted(exclude)),
                "shared_genes_removed_from_source_and_outcome": ";".join(shared) if spec["comparison"] == "deoverlapped_hypoxia" else "",
            }
        )
        comparison_rhos = []
        comparison_abs = []
        for perm in range(N_PERMUTATIONS):
            selected: list[str] = []
            selected_set: set[str] = set()
            radii_used: list[int] = []
            for gene in source_genes:
                mb = int(stat_map.loc[gene, "mean_bin"])
                db = int(stat_map.loc[gene, "detect_bin"])
                candidates: list[str] = []
                radius_used = 0
                for radius in range(0, 21):
                    keys = []
                    for mbi in all_mean_bins:
                        if abs(mbi - mb) > radius:
                            continue
                        for dbi in all_detect_bins:
                            if abs(dbi - db) <= radius:
                                keys.append((mbi, dbi))
                    candidates = []
                    for key in keys:
                        candidates.extend(bin_to_genes.get(key, []))
                    candidates = [g for g in candidates if g not in selected_set]
                    if candidates:
                        radius_used = radius
                        break
                if not candidates:
                    raise RuntimeError(f"No matched random-gene candidate found for {gene} in {spec['comparison']}")
                choice = str(rng.choice(candidates))
                selected.append(choice)
                selected_set.add(choice)
                radii_used.append(radius_used)
            score = score_sparse(ad, selected, source_weights, stats_df)
            pscore = patient_mean(score, patient_labels)
            rho, pvalue, n_patients = safe_spearman(
                pscore.reindex(patient_scores.index),
                patient_scores[str(spec["outcome_score_col"])],
            )
            long_rows.append(
                {
                    "comparison": spec["comparison"],
                    "permutation": perm + 1,
                    "random_seed": spec["random_seed"],
                    "source_score_col": spec["source_score_col"],
                    "outcome_score_col": spec["outcome_score_col"],
                    "n_source_genes": len(source_genes),
                    "n_random_genes": len(selected),
                    "max_bin_expansion_radius": int(max(radii_used)),
                    "mean_bin_expansion_radius": float(np.mean(radii_used)),
                    "rho_random_vs_outcome": rho,
                    "p_random_vs_outcome": pvalue,
                    "n_patients": n_patients,
                }
            )
            comparison_rhos.append(rho)
            comparison_abs.append(abs(rho))
            wide = wide_by_perm[perm + 1]
            prefix = str(spec["wide_prefix"])
            wide[f"n_genes_{prefix}"] = len(selected)
            wide[f"max_bin_expansion_radius_{prefix}"] = int(max(radii_used))
            wide[f"mean_bin_expansion_radius_{prefix}"] = float(np.mean(radii_used))
            wide[f"rho_random_vs_{prefix}"] = rho
            wide[f"p_random_vs_{prefix}"] = pvalue
            wide[f"n_patients_{prefix}"] = n_patients
            if perm < 50:
                for source_gene, random_gene, radius in zip(source_genes, selected, radii_used):
                    match_rows.append(
                        {
                            "comparison": spec["comparison"],
                            "permutation": perm + 1,
                            "random_seed": spec["random_seed"],
                            "source_score_col": spec["source_score_col"],
                            "outcome_score_col": spec["outcome_score_col"],
                            "dorothea_target_gene": source_gene,
                            "matched_random_gene": random_gene,
                            "bin_expansion_radius": radius,
                        }
                    )
        vals = np.asarray(comparison_rhos, dtype=float)
        vals = vals[np.isfinite(vals)]
        empirical_p_greater = (1 + np.sum(vals >= observed)) / (len(vals) + 1)
        empirical_p_abs = (1 + np.sum(np.abs(vals) >= abs(observed))) / (len(vals) + 1)
        summary.append(
            {
                "comparison": spec["comparison"],
                "source_score_col": spec["source_score_col"],
                "outcome_score_col": spec["outcome_score_col"],
                "random_seed": spec["random_seed"],
                "n_source_genes": len(source_genes),
                "observed_rho": observed,
                "n_permutations": len(vals),
                "null_mean_rho": float(np.mean(vals)),
                "null_sd_rho": float(np.std(vals, ddof=1)),
                "null_95pct_low": float(np.quantile(vals, 0.025)),
                "null_95pct_high": float(np.quantile(vals, 0.975)),
                "empirical_p_greater_equal_observed": float(empirical_p_greater),
                "empirical_p_abs_greater_equal_observed": float(empirical_p_abs),
            }
        )
    long_null = pd.DataFrame(long_rows)
    wide_null = pd.DataFrame([wide_by_perm[i + 1] for i in range(N_PERMUTATIONS)])
    # Backward-compatible column names used by the existing Figure 2 plotting code.
    wide_null = wide_null.rename(
        columns={
            "rho_random_vs_original_hypoxia": "rho_random_vs_original_hypoxia",
            "p_random_vs_original_hypoxia": "p_random_vs_original_hypoxia",
            "n_patients_original_hypoxia": "n_patients_original_hypoxia",
            "rho_random_vs_deoverlapped_hypoxia": "rho_random_vs_deoverlapped_hypoxia",
            "p_random_vs_deoverlapped_hypoxia": "p_random_vs_deoverlapped_hypoxia",
            "n_patients_deoverlapped_hypoxia": "n_patients_deoverlapped_hypoxia",
        }
    )
    wide_null.to_csv(TABLE_DIR / "matched_random_gene_set_hypoxia_correlations.csv", index=False)
    long_null.to_csv(TABLE_DIR / "matched_random_gene_set_hypoxia_correlations_long.csv", index=False)
    pd.DataFrame(match_rows).to_csv(TABLE_DIR / "matched_random_gene_set_first50_matches.csv", index=False)
    pd.DataFrame(input_rows).to_csv(TABLE_DIR / "matched_random_gene_set_input_gene_counts.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(TABLE_DIR / "matched_random_gene_set_hypoxia_correlation_summary.csv", index=False)
    return wide_null, summary_df


def qc_adjusted_models(cell_scores: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    model_df = obs.copy()
    model_df["BACH1_detected"] = model_df["BACH1_group"].astype(str).eq("BACH1_high").astype(int)
    model_df["log_total_counts"] = np.log1p(pd.to_numeric(model_df["total_counts"], errors="coerce"))
    for col in ["log_total_counts", "n_genes_by_counts", "pct_counts_mt"]:
        values = pd.to_numeric(model_df[col], errors="coerce")
        sd = values.std(ddof=0)
        model_df[f"{col}_z"] = (values - values.mean()) / sd if sd and np.isfinite(sd) else 0.0
    state_counts = model_df["epi_subtype_auto"].astype(str).value_counts()
    keep_states = set(state_counts[state_counts >= 30].index)
    model_df["epi_state_model"] = model_df["epi_subtype_auto"].astype(str).where(
        model_df["epi_subtype_auto"].astype(str).isin(keep_states), "Other_rare"
    )
    for col in cell_scores.columns:
        model_df[col] = pd.to_numeric(cell_scores[col], errors="coerce").to_numpy()

    scores = [
        c
        for c in [DOR_COL, DOR_NO_OVERLAP_COL, EFFECTOR_COL, COL_COL, KLENJA_COL]
        if c in model_df.columns
    ]
    rows = []
    for score in scores:
        for model_name, formula_rhs in [
            ("patient_fixed_effect_qc_state", "BACH1_detected + log_total_counts_z + n_genes_by_counts_z + pct_counts_mt_z + C(epi_state_model) + C(patient)"),
            ("dataset_fixed_effect_qc_state", "BACH1_detected + log_total_counts_z + n_genes_by_counts_z + pct_counts_mt_z + C(epi_state_model) + C(dataset)"),
        ]:
            df = model_df[
                [
                    score,
                    "BACH1_detected",
                    "log_total_counts_z",
                    "n_genes_by_counts_z",
                    "pct_counts_mt_z",
                    "epi_state_model",
                    "patient",
                    "dataset",
                ]
            ].dropna()
            df = df.rename(columns={score: "activity_score"})
            try:
                fit = smf.ols(f"activity_score ~ {formula_rhs}", data=df).fit(
                    cov_type="cluster", cov_kwds={"groups": df["patient"].astype(str)}
                )
                term = "BACH1_detected"
                rows.append(
                    {
                        "score": score,
                        "model": model_name,
                        "n_cells": int(df.shape[0]),
                        "n_patients": int(df["patient"].nunique()),
                        "n_bach1_detected": int(df["BACH1_detected"].sum()),
                        "beta_bach1_detected": float(fit.params.get(term, np.nan)),
                        "se_cluster_patient": float(fit.bse.get(term, np.nan)),
                        "ci_low": float(fit.conf_int().loc[term, 0]) if term in fit.params.index else np.nan,
                        "ci_high": float(fit.conf_int().loc[term, 1]) if term in fit.params.index else np.nan,
                        "p_value": float(fit.pvalues.get(term, np.nan)),
                        "r_squared": float(getattr(fit, "rsquared", np.nan)),
                        "note": "Cluster-robust standard errors by patient. Dataset is absorbed by patient fixed effects in the patient model.",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "score": score,
                        "model": model_name,
                        "n_cells": int(df.shape[0]),
                        "n_patients": int(df["patient"].nunique()),
                        "n_bach1_detected": int(df["BACH1_detected"].sum()),
                        "beta_bach1_detected": np.nan,
                        "se_cluster_patient": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": np.nan,
                        "r_squared": np.nan,
                        "note": f"Model failed: {type(exc).__name__}: {exc}",
                    }
                )
    out = pd.DataFrame(rows)
    out["p_holm_within_model_family"] = out.groupby("model")["p_value"].transform(holm_adjust)
    out["p_bh_within_model_family"] = out.groupby("model")["p_value"].transform(bh_adjust)
    out.to_csv(TABLE_DIR / "qc_adjusted_bach1_detection_activity_models.csv", index=False)
    model_df[["patient", "dataset", "BACH1_detected", "log_total_counts", "n_genes_by_counts", "pct_counts_mt", "epi_state_model"]].to_csv(
        TABLE_DIR / "qc_adjusted_model_cell_covariates.csv.gz", index=True
    )
    return out


def method_dependence(patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bach1_cols = [
        c
        for c in patient.columns
        if (
            str(c).startswith("DOROTHEA_BACH1")
            or str(c).startswith("COLLECTRI_BACH1")
            or str(c).startswith("KLENJA2025_BACH1")
        )
        and "_p" not in str(c)
    ]
    phenotype_cols = [c for c in patient.columns if str(c).startswith("HALLMARK_HYPOXIA") and "_p" not in str(c)]
    phenotype_cols += [HYPOXIA_NO_OVERLAP_COL] if HYPOXIA_NO_OVERLAP_COL in patient.columns else []
    bach1_cols = list(dict.fromkeys(bach1_cols))
    phenotype_cols = list(dict.fromkeys(phenotype_cols))
    rows = []
    lopo_rows = []
    for xcol in bach1_cols:
        for ycol in phenotype_cols:
            if xcol == ycol:
                continue
            rho, pvalue, n = safe_spearman(patient[xcol], patient[ycol])
            rows.append({"bach1_score": xcol, "phenotype_score": ycol, "n_patients": n, "spearman_rho": rho, "spearman_p": pvalue})
            for held in patient.index.astype(str):
                keep = patient[patient.index.astype(str) != held]
                lrho, lp, ln = safe_spearman(keep[xcol], keep[ycol])
                lopo_rows.append(
                    {
                        "bach1_score": xcol,
                        "phenotype_score": ycol,
                        "held_out_patient": held,
                        "n_patients": ln,
                        "spearman_rho": lrho,
                        "spearman_p": lp,
                        "sign": "positive" if np.isfinite(lrho) and lrho > 0 else ("negative" if np.isfinite(lrho) and lrho < 0 else "zero_or_nan"),
                    }
                )
    out = pd.DataFrame(rows)
    out["spearman_holm_p"] = holm_adjust(out["spearman_p"])
    out["spearman_bh_q"] = bh_adjust(out["spearman_p"])
    out = out.sort_values(["spearman_holm_p", "spearman_p", "bach1_score", "phenotype_score"])
    out.to_csv(TABLE_DIR / "bach1_resource_method_hypoxia_correlations.csv", index=False)
    lopo = pd.DataFrame(lopo_rows)
    lopo_summary = (
        lopo.groupby(["bach1_score", "phenotype_score"])
        .agg(
            n_lopo=("held_out_patient", "nunique"),
            min_rho=("spearman_rho", "min"),
            max_rho=("spearman_rho", "max"),
            median_rho=("spearman_rho", "median"),
            max_p=("spearman_p", "max"),
            n_positive=("sign", lambda s: int((s == "positive").sum())),
            n_negative=("sign", lambda s: int((s == "negative").sum())),
        )
        .reset_index()
    )
    lopo_summary["sign_stability"] = np.where(
        lopo_summary["n_positive"].eq(lopo_summary["n_lopo"]),
        "all_positive",
        np.where(lopo_summary["n_negative"].eq(lopo_summary["n_lopo"]), "all_negative", "mixed_or_zero"),
    )
    lopo_summary.to_csv(TABLE_DIR / "bach1_resource_method_hypoxia_lopo_summary.csv", index=False)
    return out, lopo_summary


def within_patient_deltas(cell_scores: pd.DataFrame, obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = cell_scores.copy()
    tmp["patient"] = obs["patient"].astype(str).to_numpy()
    tmp["dataset"] = obs["dataset"].astype(str).to_numpy()
    tmp["BACH1_detected"] = obs["BACH1_group"].astype(str).eq("BACH1_high").to_numpy()
    scores = [DOR_COL, DOR_NO_OVERLAP_COL, EFFECTOR_COL, COL_COL, KLENJA_COL]
    scores = [s for s in scores if s in tmp.columns]
    rows = []
    for (patient, dataset), sub in tmp.groupby(["patient", "dataset"], observed=True):
        high = sub[sub["BACH1_detected"]]
        low = sub[~sub["BACH1_detected"]]
        for score in scores:
            rows.append(
                {
                    "patient": patient,
                    "dataset": dataset,
                    "score": score,
                    "n_detected": int(high.shape[0]),
                    "n_undetected": int(low.shape[0]),
                    "mean_detected": float(high[score].mean()) if not high.empty else np.nan,
                    "mean_undetected": float(low[score].mean()) if not low.empty else np.nan,
                    "delta_detected_minus_undetected": float(high[score].mean() - low[score].mean()) if not high.empty and not low.empty else np.nan,
                }
            )
    deltas = pd.DataFrame(rows)
    tests = []
    for score, sub in deltas.groupby("score", observed=True):
        stat, pvalue, n = safe_wilcoxon(sub["delta_detected_minus_undetected"])
        tests.append(
            {
                "score": score,
                "n_patients_with_nonzero_delta": n,
                "median_delta_detected_minus_undetected": float(pd.to_numeric(sub["delta_detected_minus_undetected"], errors="coerce").median()),
                "wilcoxon_stat": stat,
                "wilcoxon_p": pvalue,
            }
        )
    tests_df = pd.DataFrame(tests)
    tests_df["wilcoxon_holm_p"] = holm_adjust(tests_df["wilcoxon_p"])
    tests_df["wilcoxon_bh_q"] = bh_adjust(tests_df["wilcoxon_p"])
    deltas.to_csv(TABLE_DIR / "within_patient_bach1_detection_deltas_with_deoverlap.csv", index=False)
    tests_df.to_csv(TABLE_DIR / "within_patient_bach1_detection_delta_tests_with_deoverlap.csv", index=False)
    tests_df.to_csv(TABLE_DIR / "figure2c_within_patient_score_comparison_holm_family.csv", index=False)
    return deltas, tests_df


SHORT_LABELS = {
    DOR_COL: "DoRothEA mean-z",
    DOR_NO_OVERLAP_COL: "DoRothEA mean-z\nshared genes excluded",
    COL_COL: "CollecTRI mean-z",
    KLENJA_COL: "Klenja inverse mean-z",
    EFFECTOR_COL: "lung effector mean-z",
    HYPOXIA_COL: "Hypoxia mean-z",
    HYPOXIA_NO_OVERLAP_COL: "Hypoxia\nshared genes excluded",
    f"{DOR_SIG}__decoupler_ulm": "DoRothEA ULM",
    f"{DOR_SIG}__decoupler_mlm": "DoRothEA MLM",
    f"{DOR_SIG}__ssgsea_NES": "DoRothEA ssGSEA",
    f"{COL_SIG}__decoupler_ulm": "CollecTRI ULM",
    f"{COL_SIG}__decoupler_mlm": "CollecTRI MLM",
    f"{COL_SIG}__ssgsea_NES": "CollecTRI ssGSEA",
    f"{KLENJA_SIG}__ssgsea_NES": "Klenja ssGSEA",
}


def label(value: str) -> str:
    return SHORT_LABELS.get(str(value), str(value).replace("_", " ").replace("__mean z", ""))


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_p_mpl(value: float) -> str:
    p = float(value)
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / (10 ** exponent)
        mantissa = round(mantissa, 1)
        if mantissa >= 10:
            mantissa /= 10
            exponent += 1
        return f"{mantissa:.1f}\\times10^{{{exponent}}}"
    if p < 0.01:
        return f"{p:.4f}".rstrip("0").rstrip(".")
    return f"{p:.3f}".rstrip("0").rstrip(".")


def format_p_text(value: float) -> str:
    p = float(value)
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / (10 ** exponent)
        mantissa = round(mantissa, 1)
        if mantissa >= 10:
            mantissa /= 10
            exponent += 1
        superscript = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{mantissa:.1f}×10{superscript}"
    if p < 0.01:
        return f"{p:.4f}".rstrip("0").rstrip(".")
    return f"{p:.3f}".rstrip("0").rstrip(".")


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if vals.size < 2:
        return np.nan, np.nan
    meds = [np.median(rng.choice(vals, size=vals.size, replace=True)) for _ in range(n)]
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def make_figure(
    patient: pd.DataFrame,
    deltas: pd.DataFrame,
    delta_tests: pd.DataFrame,
    method_cor: pd.DataFrame,
    deoverlap_cor: pd.DataFrame,
    random_null: pd.DataFrame,
    random_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    targets: pd.DataFrame,
    pathways: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.82)
    fig = plt.figure(figsize=(14.2, 9.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.44)

    patient = patient.copy()
    patient["dataset"] = patient["dataset"].astype(str)
    order = patient.sort_values(["dataset", "BACH1_detected_fraction", "patient"]).index.astype(str).tolist()
    dataset_palette = {"GSE131907": "#4C78A8", "GSE274934": "#F58518"}
    colors = [dataset_palette.get(str(patient.loc[p, "dataset"]).split(";")[0], "#888888") for p in order]

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(np.arange(len(order)), patient.loc[order, "BACH1_detected_fraction"], color=colors, edgecolor="white", linewidth=0.4)
    for i, p in enumerate(order):
        n = int(patient.loc[p, "n_cells"])
        ax.text(i, patient.loc[p, "BACH1_detected_fraction"] + 0.012, str(n), ha="center", va="bottom", rotation=90, fontsize=5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_ylabel("BACH1-detected fraction")
    ax.set_xlabel("Patient; numbers show malignant epithelial cells")
    ax.set_ylim(0, max(0.52, float(patient["BACH1_detected_fraction"].max()) * 1.18))
    add_panel_label(ax, "a")
    for ds, color in dataset_palette.items():
        ax.bar([], [], color=color, label=ds)
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    heat_cols = [
        KLENJA_COL,
        DOR_COL,
        DOR_NO_OVERLAP_COL,
        COL_COL,
        f"{DOR_SIG}__decoupler_ulm",
        f"{DOR_SIG}__decoupler_mlm",
    ]
    heat_cols = [c for c in heat_cols if c in patient.columns]
    heat = patient.loc[order, heat_cols].apply(pd.to_numeric, errors="coerce")
    heat = (heat - heat.mean(axis=0)) / heat.std(axis=0, ddof=0).replace(0, np.nan)
    target_counts = {}
    for sig in [DOR_SIG, COL_SIG, KLENJA_SIG]:
        target_counts[sig] = present_signature_genes(targets, sig).shape[0]
    target_counts[f"{DOR_SIG}_NO_HYPOXIA_OVERLAP"] = present_signature_genes(targets, DOR_SIG).shape[0] - int(
        overlap_summary[
            overlap_summary["bach1_signature"].eq(DOR_SIG) & overlap_summary["phenotype_signature"].eq(HYPOXIA_SIG)
        ]["n_overlap"].iloc[0]
    )
    ylabels = []
    for c in heat_cols:
        sig = c.split("__")[0]
        n = target_counts.get(sig, target_counts.get(f"{sig}_NO_HYPOXIA_OVERLAP"))
        ylabels.append(f"{label(c)}\n(n={n})" if n is not None else label(c))
    sns.heatmap(heat.T, cmap="vlag", center=0, xticklabels=order, yticklabels=ylabels, cbar_kws={"label": "patient z-score"}, ax=ax)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    add_panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    plot_scores = [DOR_COL, DOR_NO_OVERLAP_COL, EFFECTOR_COL, COL_COL, KLENJA_COL]
    plot_deltas = deltas[deltas["score"].isin(plot_scores)].copy()
    plot_deltas["score_label"] = pd.Categorical(plot_deltas["score"].map(label), categories=[label(s) for s in plot_scores], ordered=True)
    sns.stripplot(data=plot_deltas, y="score_label", x="delta_detected_minus_undetected", hue="dataset", palette=dataset_palette, dodge=False, size=4, alpha=0.85, ax=ax)
    ax.axvline(0, color="black", lw=0.7)
    xmin, xmax = ax.get_xlim()
    annotation_pad = max((xmax - xmin) * 0.28, 0.08)
    ax.set_xlim(xmin, xmax + annotation_pad)
    annotation_x = xmax + annotation_pad * 0.05
    for yi, score in enumerate(plot_scores):
        vals = plot_deltas.loc[plot_deltas["score"].eq(score), "delta_detected_minus_undetected"].dropna().to_numpy(float)
        if vals.size:
            med = float(np.median(vals))
            lo, hi = bootstrap_ci(vals, rng)
            ax.plot([lo, hi], [yi, yi], color="black", lw=1.6)
            ax.plot(med, yi, marker="D", color="black", markersize=4)
            test = delta_tests[delta_tests["score"].eq(score)]
            if not test.empty:
                p = float(test["wilcoxon_holm_p"].iloc[0])
                n = int(test["n_patients_with_nonzero_delta"].iloc[0])
                ax.text(
                    annotation_x,
                    yi,
                    f"n={n}, med={med:.3f}\nHolm-adjusted $P={format_p_mpl(p)}$",
                    va="center",
                    fontsize=5.6,
                    linespacing=1.05,
                )
    ax.set_xlabel("Patient-level delta: BACH1-detected - undetected")
    ax.set_ylabel("")
    add_panel_label(ax, "c")
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax = fig.add_subplot(gs[1, 0])
    core = [
        (DOR_COL, HYPOXIA_COL),
        (DOR_NO_OVERLAP_COL, HYPOXIA_NO_OVERLAP_COL),
        (COL_COL, HYPOXIA_COL),
        (KLENJA_COL, HYPOXIA_COL),
        (f"{DOR_SIG}__decoupler_ulm", HYPOXIA_COL),
        (f"{DOR_SIG}__decoupler_mlm", HYPOXIA_COL),
    ]
    core_rows = []
    for x, y in core:
        row = method_cor[(method_cor["bach1_score"].eq(x)) & (method_cor["phenotype_score"].eq(y))]
        if not row.empty:
            r = row.iloc[0].to_dict()
            if x == DOR_NO_OVERLAP_COL and y == HYPOXIA_NO_OVERLAP_COL:
                r["comparison"] = "DoRothEA mean-z vs hypoxia\nshared genes excluded"
            else:
                r["comparison"] = f"{label(x)}\nvs {label(y)}"
            core_rows.append(r)
    core_df = pd.DataFrame(core_rows)
    if not core_df.empty:
        mat = core_df.set_index("comparison")[["spearman_rho"]]
        annot = core_df.apply(lambda r: f"{r['spearman_rho']:.2f}{stars(float(r['spearman_holm_p']))}", axis=1).to_numpy().reshape(-1, 1)
        sns.heatmap(mat, cmap="vlag", vmin=-1, vmax=1, center=0, annot=annot, fmt="", cbar_kws={"label": "Spearman rho"}, ax=ax)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=7)
    add_panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    de = deoverlap_cor.set_index("comparison").loc[
        ["original", "both_signatures_without_shared"],
        :,
    ].reset_index()
    y = np.arange(de.shape[0])
    ax.errorbar(de["spearman_rho"], y, fmt="o", color="#333333", markersize=5)
    lopo_path = TABLE_DIR / "dorothea_hypoxia_deoverlap_lopo_summary.csv"
    if lopo_path.exists():
        lopo = pd.read_csv(lopo_path).set_index("comparison")
        for i, comp in enumerate(de["comparison"]):
            if comp in lopo.index:
                ax.plot([lopo.loc[comp, "min_rho"], lopo.loc[comp, "max_rho"]], [i, i], color="#777777", lw=2)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(["Original\nDoRothEA vs hypoxia", "Shared genes removed\nfrom both scores"], fontsize=7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho; line shows LOPO range")
    add_panel_label(ax, "e")
    ax.set_title("LOPO sensitivity", loc="left", fontsize=8, pad=2)

    ax = fig.add_subplot(gs[1, 2])
    vals = pd.to_numeric(random_null["rho_random_vs_deoverlapped_hypoxia"], errors="coerce").dropna()
    ax.hist(vals, bins=35, color="#B9C6D3", edgecolor="white")
    obs = float(random_summary[random_summary["comparison"].eq("deoverlapped_hypoxia")]["observed_rho"].iloc[0])
    emp = float(random_summary[random_summary["comparison"].eq("deoverlapped_hypoxia")]["empirical_p_greater_equal_observed"].iloc[0])
    ax.axvline(obs, color="#D62728", lw=2, label=f"observed rho={obs:.2f}\nempirical $P={format_p_mpl(emp)}$")
    ax.set_xlabel("Random matched-gene score vs de-overlapped hypoxia rho")
    ax.set_ylabel("Permutation count")
    ax.legend(frameon=False, fontsize=7)
    add_panel_label(ax, "f")
    ax.set_title("Matched-gene null", loc="left", fontsize=8, pad=2)

    fig.subplots_adjust(top=0.95)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"figure2_bach1_activity_hypoxia_robustness.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict, tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# BACH1-Hypoxia Robustness Report",
        "",
        "## Key results",
        "",
        f"- DoRothEA-Hallmark hypoxia overlap: {summary['dorothea_hypoxia_overlap']['n_overlap']} shared genes; Jaccard={summary['dorothea_hypoxia_overlap']['jaccard_index']:.4f}; overlap coefficient={summary['dorothea_hypoxia_overlap']['overlap_coefficient']:.4f}.",
        "- Figure 2c within-patient score shifts use Holm correction across the five score comparisons displayed in that panel.",
        f"- Original patient-level DoRothEA mean-z versus hypoxia mean-z: Spearman ρ={summary['deoverlap']['original']['spearman_rho']:.4f}, Holm-adjusted *P*={format_p_text(summary['deoverlap']['original']['spearman_holm_p'])}.",
        f"- Shared genes removed from both scores: Spearman ρ={summary['deoverlap']['both_signatures_without_shared']['spearman_rho']:.4f}, Holm-adjusted *P*={format_p_text(summary['deoverlap']['both_signatures_without_shared']['spearman_holm_p'])}.",
        f"- Expression/detection-matched random sets versus de-overlapped hypoxia: empirical one-sided *P*={format_p_text(summary['random_matched']['deoverlapped_hypoxia']['empirical_p_greater_equal_observed'])}.",
        f"- QC-adjusted patient fixed-effect beta for DoRothEA mean-z: beta={summary['qc_adjusted']['dorothea_patient_fixed_beta']:.4f}, 95% CI {summary['qc_adjusted']['dorothea_patient_fixed_ci_low']:.4f} to {summary['qc_adjusted']['dorothea_patient_fixed_ci_high']:.4f}, *P*={format_p_text(summary['qc_adjusted']['dorothea_patient_fixed_p'])}.",
        "",
        "## Interpretation",
        "",
        "The DoRothEA-derived association with hypoxia is not explained solely by direct gene-set overlap. However, not all BACH1 resources and scoring methods reproduce the same strength of association, so the manuscript should refer to database-inferred or DoRothEA-derived BACH1-associated transcriptional scores rather than direct BACH1 activity.",
        "",
        "## Main tables",
        "",
    ]
    for name, df in tables.items():
        lines.append(f"### {name}")
        lines.append("")
        view = df.head(12).copy()
        lines.append("```csv")
        lines.append(view.to_csv(index=False).strip())
        lines.append("```")
        lines.append("")
    (OUT / "bach1_hypoxia_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    log("Loading primary malignant epithelial h5ad")
    ad = sc.read_h5ad(H5AD)
    ad.obs_names = ad.obs_names.astype(str)
    targets = pd.read_csv(EXT_TABLE / "external_bach1_signature_targets.csv")
    pathways = pd.read_csv(EXT_TABLE / "phenotype_pathway_signature_targets.csv")
    base_cell = pd.read_csv(EXT_TABLE / "external_bach1_activity_scores_by_cell.csv.gz", index_col=0)
    base_cell = base_cell.reindex(ad.obs_names.astype(str))
    base_patient = pd.read_csv(EXT_TABLE / "external_bach1_activity_patient_scores.csv").set_index("patient")

    log("Computing gene-set overlap and sparse gene statistics")
    overlap_summary, overlap_genes = overlap_tables(targets, pathways)
    stats_df = gene_stats(ad)

    log("Scoring DoRothEA and hypoxia signatures after removing shared genes")
    de_cell = deoverlap_scores(ad, targets, pathways, stats_df)
    cell_scores = pd.concat([base_cell, de_cell], axis=1)
    obs = ad.obs.copy()
    patient_de = pd.DataFrame(index=base_patient.index.astype(str))
    patient_de[DOR_NO_OVERLAP_COL] = patient_mean(de_cell[DOR_NO_OVERLAP_COL].to_numpy(float), obs["patient"].astype(str)).reindex(patient_de.index)
    patient_de[HYPOXIA_NO_OVERLAP_COL] = patient_mean(de_cell[HYPOXIA_NO_OVERLAP_COL].to_numpy(float), obs["patient"].astype(str)).reindex(patient_de.index)
    patient = base_patient.join(patient_de, how="left")
    patient.to_csv(TABLE_DIR / "external_bach1_activity_patient_scores_with_deoverlap.csv")

    log("Running de-overlap correlations")
    de_cor = deoverlap_correlations(patient)

    log("Running expression/detection-matched random gene-set test")
    random_null, random_summary = matched_random_gene_sets(ad, stats_df, targets, pathways, patient)

    log("Running QC-adjusted BACH1-detection models")
    qc_models = qc_adjusted_models(cell_scores, obs)

    log("Summarising method dependence")
    method_cor, method_lopo = method_dependence(patient)
    deltas, delta_tests = within_patient_deltas(cell_scores, obs)

    dor_hyp = overlap_summary[
        overlap_summary["bach1_signature"].eq(DOR_SIG) & overlap_summary["phenotype_signature"].eq(HYPOXIA_SIG)
    ].iloc[0]
    de_idx = de_cor.set_index("comparison")
    rand_idx = random_summary.set_index("comparison")
    qc_pf = qc_models[
        qc_models["score"].eq(DOR_COL) & qc_models["model"].eq("patient_fixed_effect_qc_state")
    ].iloc[0]
    summary = {
        "n_cells": int(ad.n_obs),
        "n_patients": int(obs["patient"].astype(str).nunique()),
        "dorothea_hypoxia_overlap": {
            "n_dorothea_genes": int(dor_hyp["n_bach1_genes"]),
            "n_hypoxia_genes": int(dor_hyp["n_phenotype_genes"]),
            "n_overlap": int(dor_hyp["n_overlap"]),
            "jaccard_index": float(dor_hyp["jaccard_index"]),
            "overlap_coefficient": float(dor_hyp["overlap_coefficient"]),
            "overlap_genes": str(dor_hyp["overlap_genes"]),
        },
        "deoverlap": {
            comp: {
                "spearman_rho": float(de_idx.loc[comp, "spearman_rho"]),
                "spearman_p": float(de_idx.loc[comp, "spearman_p"]),
                "spearman_holm_p": float(de_idx.loc[comp, "spearman_holm_p"]),
            }
            for comp in de_idx.index
        },
        "random_matched": {
            comp: {
                "observed_rho": float(rand_idx.loc[comp, "observed_rho"]),
                "n_permutations": int(rand_idx.loc[comp, "n_permutations"]),
                "null_mean_rho": float(rand_idx.loc[comp, "null_mean_rho"]),
                "null_sd_rho": float(rand_idx.loc[comp, "null_sd_rho"]),
                "empirical_p_greater_equal_observed": float(rand_idx.loc[comp, "empirical_p_greater_equal_observed"]),
                "empirical_p_abs_greater_equal_observed": float(rand_idx.loc[comp, "empirical_p_abs_greater_equal_observed"]),
            }
            for comp in rand_idx.index
        },
        "qc_adjusted": {
            "dorothea_patient_fixed_beta": float(qc_pf["beta_bach1_detected"]),
            "dorothea_patient_fixed_ci_low": float(qc_pf["ci_low"]),
            "dorothea_patient_fixed_ci_high": float(qc_pf["ci_high"]),
            "dorothea_patient_fixed_p": float(qc_pf["p_value"]),
            "dorothea_patient_fixed_holm_p": float(qc_pf["p_holm_within_model_family"]),
        },
    }
    with (OUT / "bach1_hypoxia_robustness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    log("Creating robustness Figure 2")
    make_figure(patient, deltas, delta_tests, method_cor, de_cor, random_null, random_summary, overlap_summary, targets, pathways)

    write_report(
        summary,
        {
            "DoRothEA-hypoxia overlap": overlap_summary[
                overlap_summary["bach1_signature"].eq(DOR_SIG)
                & overlap_summary["phenotype_signature"].str.contains("HYPOXIA", regex=False)
            ],
            "De-overlap correlations": de_cor,
            "Matched random summary": random_summary,
            "QC-adjusted models": qc_models,
            "Method dependence": method_cor,
        },
    )
    log(f"Wrote outputs under {OUT}")


if __name__ == "__main__":
    main()
