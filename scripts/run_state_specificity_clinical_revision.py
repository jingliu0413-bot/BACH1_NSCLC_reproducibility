#!/usr/bin/env python3
"""Revision analyses for epithelial-state context and TCGA clinical projection.

This script keeps the additional reviewer-facing analyses deliberately narrow:
it reuses existing malignant epithelial objects, published external score tables
and TCGA Xena matrices to avoid introducing a second analysis universe.
"""

from __future__ import annotations

import json
import math
import re
from itertools import permutations
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import mannwhitneyu, norm, rankdata, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "manuscript_submission_20260914"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"

PRIMARY_H5AD = ROOT / "out/bach1_malignant_epithelial_primary/nsclc_malignant_epithelial_primary_bach1_analysis_object.h5ad"
CELL_SCORES = ROOT / "out/external_bach1_activity_primary/tables/external_bach1_activity_scores_by_cell.csv.gz"
DEOVERLAP_SCORES = ROOT / "out/external_bach1_activity_robustness/tables/deoverlapped_dorothea_hypoxia_cell_scores.csv.gz"
PATIENT_SCORES = ROOT / "out/external_bach1_activity_robustness/tables/external_bach1_activity_patient_scores_with_deoverlap.csv"
STATE_PCS = ROOT / "out/final_inference_sensitivity/tables/patient_level_epithelial_state_composition_pcs.csv"
DOROTHEA = ROOT / "resources/bach1_external_signatures/dorothea_human_ABC_decoupler.csv"
PHENO_TARGETS = ROOT / "out/external_bach1_activity_primary/tables/phenotype_pathway_signature_targets.csv"
RECURRENT_TARGETS = ROOT / "source_data/manuscript_submission_20260911/figure4e_current_bach1_recurrent_targets.csv"
CONSENSUS_CANDIDATES = ROOT / "out/bach1_malignant_epithelial_consensus/tables/nsclc_malignant_epithelial_consensus_bach1_candidate_targets_integrated_evidence.csv"
ATAC_SUPPORT = ROOT / "out/bach1_atac_motif_support_consensus_full_tf_5seed_stable_ge60_matched_background_10kb_neighbors15/tables/stable_bach1_targets_with_matched_background_atac_support.csv"
TCGA_SCORES = ROOT / "outputs/manuscript_submission_20260911/tables/tcga_luad_lusc_bach1_hypoxia_patient_scores.csv"


DOR_NO = "DOROTHEA_BACH1_ABC_TF_ACTIVITY_NO_HYPOXIA_OVERLAP__mean_z"
HYP_NO = "HALLMARK_HYPOXIA_NO_DOROTHEA_OVERLAP__mean_z"
DOR_ORIG = "DOROTHEA_BACH1_ABC_TF_ACTIVITY__mean_z"
HYP_ORIG = "HALLMARK_HYPOXIA__mean_z"

PROGRAMME_COLS = [
    ("BACH1 detected", "bach1_detected_fraction"),
    ("DoRothEA BACH1", DOR_NO),
    ("Hypoxia", HYP_NO),
    ("Glycolysis", "HALLMARK_GLYCOLYSIS__mean_z"),
    ("NRF2/oxidative", "CURATED_NRF2_ANTIOXIDANT_RESPONSE__mean_z"),
    ("Heme metabolism", "HALLMARK_HEME_METABOLISM__mean_z"),
    ("ROS pathway", "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY__mean_z"),
    ("OXPHOS", "HALLMARK_OXIDATIVE_PHOSPHORYLATION__mean_z"),
    ("EMT", "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION__mean_z"),
    ("Proliferation", "score_Proliferative"),
]

STATE_ORDER = ["Stress_AP1", "Ciliated", "Epithelial_core", "Proliferative", "AT2_like", "Other"]
WITHIN_STATE_DISPLAY_ORDER = ["Stress_AP1", "Epithelial_core", "AT2_like"]
IMMEDIATE_EARLY_GENES = {
    "FOS",
    "FOSB",
    "JUN",
    "JUNB",
    "JUND",
    "EGR1",
    "EGR2",
    "EGR3",
    "ATF3",
    "DUSP1",
    "IER2",
    "IER3",
    "ZFP36",
}
STATE_DISPLAY = {
    "Stress_AP1": "Stress-associated",
    "Ciliated": "Ciliated-like",
    "Epithelial_core": "Epithelial_core",
    "Proliferative": "Proliferative",
    "AT2_like": "AT2_like",
    "Other": "Other",
}
COLORS = {
    "Stress_AP1": "#C44E52",
    "Ciliated": "#55A868",
    "Epithelial_core": "#4C72B0",
    "Proliferative": "#8172B2",
    "AT2_like": "#CCB974",
    "Other": "#8C8C8C",
}


def ensure_dirs() -> None:
    for d in [TABLE_DIR, FIG_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def simplify_state(x: object) -> str:
    s = str(x)
    for key in ["Stress_AP1", "Proliferative", "Epithelial_core", "Ciliated", "AT2_like"]:
        if key in s:
            return key
    return "Other"


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        mantissa, exponent = f"{p:.1e}".split("e")
        exponent = int(exponent)
        return rf"${mantissa}\times10^{{{exponent}}}$"
    if p < 0.01:
        return f"{p:.4f}".rstrip("0").rstrip(".")
    return f"{p:.3f}".rstrip("0").rstrip(".")


def add_adjusted_p(
    df: pd.DataFrame,
    p_col: str = "p_value",
    out_col: str = "holm_adjusted_p",
    method: str = "holm",
) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    mask = np.isfinite(pd.to_numeric(out[p_col], errors="coerce"))
    if mask.any():
        out.loc[mask, out_col] = multipletests(out.loc[mask, p_col].astype(float), method=method)[1]
    return out


def spearman_exact_permutation(x: pd.Series, y: pd.Series, max_exact_n: int = 9, n_perm: int = 200000) -> tuple[float, float, str]:
    g = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(g)
    if n < 4:
        return (np.nan, np.nan, "not_evaluable")
    rx = rankdata(g["x"].to_numpy(dtype=float))
    ry = rankdata(g["y"].to_numpy(dtype=float))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if not np.isfinite(rho):
        return (np.nan, np.nan, "not_evaluable")

    def corr_with_ref(perm_y: np.ndarray) -> float:
        return float(np.corrcoef(rx, perm_y)[0, 1])

    if n <= max_exact_n:
        total = math.factorial(n)
        extreme = 0
        for perm in permutations(ry):
            r = corr_with_ref(np.asarray(perm, dtype=float))
            if abs(r) >= abs(rho) - 1e-12:
                extreme += 1
        p = extreme / total
        method = "exact_permutation"
    else:
        rng = np.random.default_rng(20260914)
        extreme = 0
        for _ in range(n_perm):
            r = corr_with_ref(rng.permutation(ry))
            if abs(r) >= abs(rho) - 1e-12:
                extreme += 1
        p = (extreme + 1) / (n_perm + 1)
        method = f"monte_carlo_permutation_{n_perm}"
    return (rho, float(p), method)


def bootstrap_median_difference(
    x1: pd.Series,
    x0: pd.Series,
    rng: np.random.Generator,
    n_boot: int = 10000,
) -> tuple[float, float]:
    a = pd.to_numeric(x1, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(x0, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.median(aa) - np.median(bb)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return (float(lo), float(hi))


def load_cell_data() -> tuple[ad.AnnData, pd.DataFrame]:
    adata = ad.read_h5ad(PRIMARY_H5AD)
    obs = adata.obs.copy()
    obs["state"] = obs["epi_marker_only_broad_class"].map(simplify_state)
    obs["bach1_detected"] = pd.to_numeric(obs["BACH1_expr"], errors="coerce").fillna(0) > 0
    scores = pd.read_csv(CELL_SCORES, index_col=0)
    de = pd.read_csv(DEOVERLAP_SCORES, index_col=0)
    keep_obs = [
        "patient",
        "dataset",
        "sample_id",
        "tumor_type",
        "state",
        "bach1_detected",
        "BACH1_expr",
        "score_Proliferative",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "n_cnv_methods_malignant",
        "cnv_consensus_call",
        "sensitive_cnv_burden",
        "strict_cnv_burden",
        "cnv_burden_adjacent_ref",
    ]
    df = obs[keep_obs].join(scores, how="left").join(de, how="left")
    return adata, df


def state_context_tables(adata: ad.AnnData, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    umap = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
    umap = umap.join(df[["patient", "dataset", "state", "bach1_detected", DOR_NO, HYP_NO]])
    umap.to_csv(TABLE_DIR / "figure4a_primary_malignant_state_umap.csv.gz")

    rows = []
    for state, g in df.groupby("state", observed=True):
        row = {
            "state": state,
            "n_cells": int(len(g)),
            "n_patients": int(g["patient"].nunique()),
            "bach1_detected_fraction": float(g["bach1_detected"].mean()),
        }
        for _, col in PROGRAMME_COLS[1:]:
            row[col] = float(pd.to_numeric(g[col], errors="coerce").mean())
        rows.append(row)
    state_summary = pd.DataFrame(rows)
    state_summary["state"] = pd.Categorical(state_summary["state"], STATE_ORDER, ordered=True)
    state_summary = state_summary.sort_values("state")
    state_summary.to_csv(TABLE_DIR / "figure4b_state_programme_summary.csv", index=False)

    heat = state_summary.set_index("state")[[col for _, col in PROGRAMME_COLS]]
    heat.columns = [label for label, _ in PROGRAMME_COLS]
    heat_z = heat.apply(zscore, axis=0)
    heat_z.to_csv(TABLE_DIR / "figure4b_state_programme_summary_column_z.csv")

    patient_state = (
        df.groupby(["patient", "dataset", "state"], observed=True)
        .agg(
            n_cells=("patient", "size"),
            dorothea_bach1_no_overlap=(DOR_NO, "mean"),
            hypoxia_no_overlap=(HYP_NO, "mean"),
            bach1_detected_fraction=("bach1_detected", "mean"),
        )
        .reset_index()
    )
    patient_state.to_csv(TABLE_DIR / "patient_state_score_summary.csv", index=False)

    assoc = []
    for state, g0 in patient_state.groupby("state", observed=True):
        g = g0[g0["n_cells"] >= 10].copy()
        if len(g) >= 4:
            rho, p, method = spearman_exact_permutation(g["dorothea_bach1_no_overlap"], g["hypoxia_no_overlap"])
            lo, hi = fisher_ci(rho, len(g))
        else:
            rho = p = lo = hi = np.nan
            method = "not_evaluable"
        assoc.append(
            {
                "state": state,
                "n_patient_state_units": int(len(g)),
                "spearman_rho": rho,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "permutation_method": method,
            }
        )
    within_state = pd.DataFrame(assoc)
    within_state = add_adjusted_p(within_state, p_col="p_value", out_col="holm_adjusted_p_across_states", method="holm")
    within_state["state"] = pd.Categorical(within_state["state"], STATE_ORDER, ordered=True)
    within_state = within_state.sort_values("state")
    within_state.to_csv(TABLE_DIR / "figure4c_within_state_patient_aware_associations.csv", index=False)

    return {
        "umap": umap,
        "state_summary": state_summary,
        "heat_z": heat_z,
        "patient_state": patient_state,
        "within_state": within_state,
    }


def state_malignancy_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, g in df.groupby("state", observed=True):
        n_cnv = pd.to_numeric(g["n_cnv_methods_malignant"], errors="coerce").fillna(0)
        consensus = g["cnv_consensus_call"].astype(str).str.lower().isin(["true", "1"])
        patient_frac = g["patient"].value_counts(normalize=True)
        rows.append(
            {
                "state": state,
                "display_state": STATE_DISPLAY.get(state, state),
                "n_cells": int(len(g)),
                "n_patients": int(g["patient"].nunique()),
                "pct_from_gse131907_author_malignant": float(g["dataset"].eq("GSE131907").mean() * 100),
                "pct_from_gse274934_cnv_supported": float(g["dataset"].eq("GSE274934").mean() * 100),
                "pct_with_any_cnv_method_malignant": float((n_cnv > 0).mean() * 100),
                "pct_with_cnv_consensus_call": float(consensus.mean() * 100),
                "median_sensitive_cnv_burden": float(pd.to_numeric(g.get("sensitive_cnv_burden"), errors="coerce").median()),
                "median_strict_cnv_burden": float(pd.to_numeric(g.get("strict_cnv_burden"), errors="coerce").median()),
                "median_adjacent_ref_cnv_burden": float(pd.to_numeric(g.get("cnv_burden_adjacent_ref"), errors="coerce").median()),
                "largest_patient": str(patient_frac.index[0]) if len(patient_frac) else "",
                "largest_patient_fraction_pct": float(patient_frac.iloc[0] * 100) if len(patient_frac) else np.nan,
            }
        )
    audit = pd.DataFrame(rows)
    audit["state"] = pd.Categorical(audit["state"], STATE_ORDER, ordered=True)
    audit = audit.sort_values("state")
    audit.to_csv(TABLE_DIR / "state_specific_malignancy_audit.csv", index=False)
    return audit


def expression_z_scores(adata: ad.AnnData, genes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return pd.DataFrame(index=adata.obs_names), []
    X = adata[:, present].X
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    keep = sd > 0
    present = [g for g, k in zip(present, keep) if k]
    if not present:
        return pd.DataFrame(index=adata.obs_names), []
    Z = (X[:, keep] - mean[keep]) / sd[keep]
    return pd.DataFrame(Z, index=adata.obs_names, columns=present), present


def unweighted_mean_z_score(adata: ad.AnnData, genes: list[str]) -> tuple[pd.Series, list[str]]:
    z, present = expression_z_scores(adata, genes)
    if z.empty:
        return pd.Series(np.nan, index=adata.obs_names), []
    return z.mean(axis=1), present


def weighted_dorothea_bach1_score(adata: ad.AnnData, exclude_genes: set[str]) -> tuple[pd.Series, list[str]]:
    prior = pd.read_csv(DOROTHEA)
    prior = prior[prior["source"].eq("BACH1") & prior["target"].notna()].copy()
    prior = prior.groupby("target", as_index=False).agg(weight=("weight", "mean"))
    prior["target"] = prior["target"].astype(str)
    prior = prior[~prior["target"].isin(exclude_genes)].copy()
    z, present = expression_z_scores(adata, prior["target"].tolist())
    if z.empty:
        return pd.Series(np.nan, index=adata.obs_names), []
    weights = prior.set_index("target").loc[present, "weight"].astype(float)
    score = z.mul(weights, axis=1).sum(axis=1) / weights.abs().sum()
    return score, present


def immediate_early_sensitivity(
    adata: ad.AnnData,
    df: pd.DataFrame,
    signature: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ph = pd.read_csv(PHENO_TARGETS)
    hypoxia_targets = set(ph.loc[ph["signature"].eq("HALLMARK_HYPOXIA"), "target"].astype(str))
    dor_bach1_targets = set(
        pd.read_csv(DOROTHEA)
        .loc[lambda x: x["source"].eq("BACH1") & x["target"].notna(), "target"]
        .astype(str)
    )
    stress_genes = set(signature["gene"].astype(str))
    glycolysis_targets = set(ph.loc[ph["signature"].eq("HALLMARK_GLYCOLYSIS"), "target"].astype(str))
    emt_targets = set(
        ph.loc[
            ph["signature"].eq("HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"),
            "target",
        ].astype(str)
    )
    programme_defs = {
        "Canonical immediate-early score": (IMMEDIATE_EARLY_GENES, IMMEDIATE_EARLY_GENES, set()),
        "Stress_AP1 22-gene signature": (stress_genes, stress_genes, set()),
        "Hypoxia without BACH1-target or immediate-early overlap": (
            hypoxia_targets,
            hypoxia_targets - dor_bach1_targets - IMMEDIATE_EARLY_GENES,
            (hypoxia_targets & dor_bach1_targets) | (hypoxia_targets & IMMEDIATE_EARLY_GENES),
        ),
        "Glycolysis without immediate-early overlap": (
            glycolysis_targets,
            glycolysis_targets - IMMEDIATE_EARLY_GENES,
            glycolysis_targets & IMMEDIATE_EARLY_GENES,
        ),
        "EMT without immediate-early overlap": (
            emt_targets,
            emt_targets - IMMEDIATE_EARLY_GENES,
            emt_targets & IMMEDIATE_EARLY_GENES,
        ),
    }

    score_df = pd.DataFrame(index=adata.obs_names)
    rows = []
    for label, (input_genes, score_genes, removed_genes) in programme_defs.items():
        score, present = unweighted_mean_z_score(adata, sorted(score_genes))
        score_df[label] = score
        rows.append(
            {
                "programme": label,
                "score_type": "unweighted_gene_mean_z",
                "n_input_genes": len(input_genes),
                "n_scored_genes_after_exclusion": len(score_genes),
                "n_present_genes": len(present),
                "n_removed_genes": len(removed_genes),
                "removed_genes": ";".join(sorted(removed_genes)),
                "genes_present": ";".join(present),
            }
        )

    dor_score, dor_present = weighted_dorothea_bach1_score(adata, exclude_genes=hypoxia_targets | IMMEDIATE_EARLY_GENES)
    dor_label = "DoRothEA BACH1 without hypoxia-target or immediate-early overlap"
    score_df[dor_label] = dor_score
    rows.append(
        {
            "programme": dor_label,
            "score_type": "signed_weighted_gene_mean_z",
            "n_input_genes": int(len(dor_bach1_targets)),
            "n_scored_genes_after_exclusion": int(len(dor_bach1_targets - hypoxia_targets - IMMEDIATE_EARLY_GENES)),
            "n_present_genes": int(len(dor_present)),
            "n_removed_genes": int(len((dor_bach1_targets & hypoxia_targets) | (dor_bach1_targets & IMMEDIATE_EARLY_GENES))),
            "removed_genes": ";".join(sorted((dor_bach1_targets & hypoxia_targets) | (dor_bach1_targets & IMMEDIATE_EARLY_GENES))),
            "genes_present": ";".join(dor_present),
        }
    )

    meta = df[["state", "patient", "dataset"]].copy()
    state_rows = []
    for programme in score_df.columns:
        tmp = meta.join(score_df[[programme]])
        state_mean = (
            tmp.groupby("state", observed=True)
            .agg(
                n_cells=("state", "size"),
                n_patients=("patient", "nunique"),
                mean_score=(programme, "mean"),
                median_score=(programme, "median"),
            )
            .reset_index()
        )
        state_mean["programme"] = programme
        state_mean["state_rank_by_mean_desc"] = state_mean["mean_score"].rank(method="min", ascending=False).astype(int)
        stress_mean = state_mean.loc[state_mean["state"].eq("Stress_AP1"), "mean_score"]
        next_highest = state_mean.loc[~state_mean["state"].eq("Stress_AP1"), "mean_score"].max()
        state_mean["stress_ap1_minus_next_highest_state_mean"] = (
            float(stress_mean.iloc[0] - next_highest) if len(stress_mean) and np.isfinite(next_highest) else np.nan
        )
        state_rows.append(state_mean)

    programme_metadata = pd.DataFrame(rows)
    state_summary = pd.concat(state_rows, ignore_index=True)
    state_summary["state"] = pd.Categorical(state_summary["state"], STATE_ORDER, ordered=True)
    state_summary = state_summary.sort_values(["programme", "state"])

    overlap = pd.DataFrame(
        [
            {
                "gene_set": label,
                "n_genes": len(genes),
                "n_overlap_with_canonical_immediate_early": len(set(genes) & IMMEDIATE_EARLY_GENES),
                "overlap_genes": ";".join(sorted(set(genes) & IMMEDIATE_EARLY_GENES)),
            }
            for label, genes in [
                ("canonical_immediate_early", IMMEDIATE_EARLY_GENES),
                ("stress_ap1_22_gene_signature", stress_genes),
                ("hallmark_hypoxia", hypoxia_targets),
                ("dorothea_bach1_targets", dor_bach1_targets),
            ]
        ]
    )

    programme_metadata.to_csv(TABLE_DIR / "stress_ap1_immediate_early_sensitivity_programmes.csv", index=False)
    state_summary.to_csv(TABLE_DIR / "stress_ap1_immediate_early_sensitivity_by_state.csv", index=False)
    overlap.to_csv(TABLE_DIR / "stress_ap1_immediate_early_gene_overlap.csv", index=False)
    return {"ie_programmes": programme_metadata, "ie_state_summary": state_summary, "ie_overlap": overlap}


def fisher_ci(rho: float, n: int) -> tuple[float, float]:
    if not np.isfinite(rho) or n <= 3 or abs(rho) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(rho)
    se = 1 / math.sqrt(n - 3)
    return tuple(np.tanh([z - 1.96 * se, z + 1.96 * se]))


def residualize(y: pd.Series, x: pd.DataFrame) -> np.ndarray:
    x = sm.add_constant(x, has_constant="add")
    fit = sm.OLS(y.astype(float), x.astype(float), missing="drop").fit()
    return fit.resid.reindex(y.index).to_numpy()


def pca_scores(matrix: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    x = matrix.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    scores = x @ vt[:n_components].T
    denom = np.sum(s**2)
    explained = (s[:n_components] ** 2 / denom) if denom > 0 else np.full(n_components, np.nan)
    out = pd.DataFrame(
        scores,
        index=matrix.index,
        columns=[f"pc{i + 1}" for i in range(n_components)],
    )
    for col in out.columns:
        out[f"{col}_z"] = zscore(out[col])
    return out, explained


def composition_pca_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    patients = pd.read_csv(PATIENT_SCORES)
    counts = pd.crosstab(df["patient"], df["state"]).reindex(columns=STATE_ORDER, fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    raw_scores, raw_explained = pca_scores(props)
    pseudo = 0.5
    comp = counts.add(pseudo).div(counts.sum(axis=1) + pseudo * len(STATE_ORDER), axis=0)
    clr = np.log(comp).sub(np.log(comp).mean(axis=1), axis=0)
    clr_scores, clr_explained = pca_scores(clr)

    base = patients.set_index("patient").copy()
    base["log_total_counts_median"] = np.log1p(base["total_counts_median"])
    base["dor_z"] = zscore(base[DOR_NO])
    base["hyp_z"] = zscore(base[HYP_NO])

    rows = []
    qc_cols = ["log_total_counts_median", "n_genes_median", "pct_mt_median"]
    for label, scores, explained in [
        ("centred_proportion_pca", raw_scores, raw_explained),
        ("clr_pca_pseudocount_0.5_cell", clr_scores, clr_explained),
    ]:
        dat = base.join(scores[["pc1_z", "pc2_z"]], how="left")
        for subset_label, sub in [
            ("all_patients", dat),
            (">=20-cell_patients", dat[dat["n_cells"] >= 20].copy()),
        ]:
            dataset = pd.get_dummies(sub["dataset"], prefix="dataset", drop_first=True, dtype=float)
            cov = pd.concat([dataset, sub[qc_cols + ["pc1_z", "pc2_z"]]], axis=1)
            tmp = sub.copy()
            tmp["dor_resid"] = residualize(tmp["dor_z"], cov)
            tmp["hyp_resid"] = residualize(tmp["hyp_z"], cov)
            g = tmp[["dor_resid", "hyp_resid"]].dropna()
            rho, p = spearmanr(g["dor_resid"], g["hyp_resid"]) if len(g) >= 4 else (np.nan, np.nan)
            lo, hi = fisher_ci(rho, len(g))
            rows.append(
                {
                    "composition_transform": label,
                    "subset": subset_label,
                    "n_patients": int(len(g)),
                    "rho": rho,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p,
                    "pc1_variance_explained": float(explained[0]),
                    "pc2_variance_explained": float(explained[1]),
                    "pseudocount_cells": pseudo if label.startswith("clr") else 0,
                    "covariates": "dataset + log_total_counts_median + n_genes_median + pct_mt_median + composition_PC1_PC2",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "state_composition_pca_transformation_sensitivity.csv", index=False)
    return out


def attenuation_table() -> pd.DataFrame:
    patients = pd.read_csv(PATIENT_SCORES)
    pcs = pd.read_csv(STATE_PCS)
    df = patients.merge(pcs, on="patient", how="left")
    df["log_total_counts_median"] = np.log1p(df["total_counts_median"])
    df["dor_z"] = zscore(df[DOR_NO])
    df["hyp_z"] = zscore(df[HYP_NO])

    rows = []

    def add(label: str, data: pd.DataFrame, x: str, y: str) -> None:
        g = data[[x, y]].dropna()
        rho, p = spearmanr(g[x], g[y])
        lo, hi = fisher_ci(rho, len(g))
        rows.append({"comparison": label, "n_patients": len(g), "rho": rho, "ci_low": lo, "ci_high": hi, "p_value": p})

    add("Original scores", df, DOR_ORIG, HYP_ORIG)
    add("Shared genes removed", df, DOR_NO, HYP_NO)

    qc_cols = ["log_total_counts_median", "n_genes_median", "pct_mt_median"]
    dataset = pd.get_dummies(df["dataset"], prefix="dataset", drop_first=True, dtype=float)
    qc = pd.concat([dataset, df[qc_cols]], axis=1)
    tmp = df.copy()
    tmp["dor_resid"] = residualize(tmp["dor_z"], qc)
    tmp["hyp_resid"] = residualize(tmp["hyp_z"], qc)
    add("Dataset and QC residuals", tmp, "dor_resid", "hyp_resid")

    state_cov = pd.concat([qc, df[["state_pc1_z", "state_pc2_z"]]], axis=1)
    tmp2 = df.copy()
    tmp2["dor_resid"] = residualize(tmp2["dor_z"], state_cov)
    tmp2["hyp_resid"] = residualize(tmp2["hyp_z"], state_cov)
    add("Dataset, QC and state-PC residuals", tmp2, "dor_resid", "hyp_resid")

    df20 = df[df["n_cells"] >= 20].copy()
    dataset20 = pd.get_dummies(df20["dataset"], prefix="dataset", drop_first=True, dtype=float)
    qc20 = pd.concat([dataset20, df20[qc_cols + ["state_pc1_z", "state_pc2_z"]]], axis=1)
    df20["dor_resid"] = residualize(zscore(df20[DOR_NO]), qc20)
    df20["hyp_resid"] = residualize(zscore(df20[HYP_NO]), qc20)
    add("Full residuals, >=20-cell patients", df20, "dor_resid", "hyp_resid")

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "figure4d_patient_level_association_attenuation.csv", index=False)
    return out


def cohort_split_tables() -> pd.DataFrame:
    patients = pd.read_csv(PATIENT_SCORES)
    rows = []
    for label, g in [("All patients", patients), *list(patients.groupby("dataset"))]:
        rho, p = spearmanr(g[DOR_NO], g[HYP_NO])
        lo, hi = fisher_ci(rho, len(g))
        rows.append({"dataset": label, "n_patients": len(g), "rho": rho, "ci_low": lo, "ci_high": hi, "p_value": p})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "scrna_cohort_split_dorothea_hypoxia_correlations.csv", index=False)
    return out


def tf_contextual_benchmark(adata: ad.AnnData, df: pd.DataFrame) -> pd.DataFrame:
    prior = pd.read_csv(DOROTHEA)
    prior = prior[prior["source"].notna() & prior["target"].notna()].copy()
    prior = prior.groupby(["source", "target"], as_index=False).agg(weight=("weight", "mean"), confidence=("confidence", "first"))

    var_index = pd.Series(np.arange(adata.n_vars), index=adata.var_names)
    prior = prior[prior["target"].isin(var_index.index)].copy()
    hypoxia_targets = set(
        pd.read_csv(PHENO_TARGETS)
        .loc[lambda x: x["signature"].eq("HALLMARK_HYPOXIA"), "target"]
        .astype(str)
    )
    overlap_counts = (
        prior.assign(hypoxia_overlap_target=prior["target"].isin(hypoxia_targets))
        .groupby("source")
        .agg(
            n_targets_present_raw=("target", "nunique"),
            n_hypoxia_overlap_targets=("hypoxia_overlap_target", "sum"),
        )
        .reset_index()
        .rename(columns={"source": "tf"})
    )

    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()
    gene_means = np.asarray(X.mean(axis=0)).ravel()
    gene_sq_means = np.asarray(X.power(2).mean(axis=0)).ravel()
    gene_stds = np.sqrt(np.maximum(gene_sq_means - gene_means**2, 0))
    gene_stds[gene_stds == 0] = np.nan

    def score_prior(prior_subset: pd.DataFrame, min_targets: int = 10) -> tuple[pd.DataFrame, dict[str, int]]:
        target_counts = prior_subset.groupby("source")["target"].nunique()
        keep_tfs = target_counts[target_counts >= min_targets].index
        prior_keep = prior_subset[prior_subset["source"].isin(keep_tfs)].copy()
        tf_order = sorted(prior_keep["source"].unique())
        tf_to_idx = {tf: i for i, tf in enumerate(tf_order)}

        row_idx = []
        col_idx = []
        data = []
        denom = np.zeros(len(tf_order))
        offset = np.zeros(len(tf_order))
        present_counts = np.zeros(len(tf_order), dtype=int)
        for r in prior_keep.itertuples(index=False):
            gi = int(var_index[r.target])
            sd = gene_stds[gi]
            if not np.isfinite(sd) or sd == 0:
                continue
            ti = tf_to_idx[r.source]
            w = float(r.weight)
            row_idx.append(gi)
            col_idx.append(ti)
            data.append(w / sd)
            denom[ti] += abs(w)
            offset[ti] += gene_means[gi] * w / sd
            present_counts[ti] += 1

        W = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(adata.n_vars, len(tf_order)))
        score_mat = X @ W
        if sparse.issparse(score_mat):
            score_mat = score_mat.toarray()
        else:
            score_mat = np.asarray(score_mat)
        score_mat = (score_mat - offset) / np.where(denom == 0, np.nan, denom)

        valid_tfs = [tf for i, tf in enumerate(tf_order) if present_counts[i] >= min_targets and denom[i] > 0]
        valid_idx = [tf_order.index(tf) for tf in valid_tfs]
        score_df = pd.DataFrame(score_mat[:, valid_idx], columns=valid_tfs, index=adata.obs_names)
        score_df["patient"] = df["patient"].values
        patient_tf = score_df.groupby("patient", observed=True)[valid_tfs].mean()
        count_map = {tf: int(present_counts[tf_order.index(tf)]) for tf in valid_tfs}
        return patient_tf, count_map

    hyp = pd.read_csv(PATIENT_SCORES).set_index("patient")[HYP_NO]

    def corr_table(patient_tf: pd.DataFrame, count_map: dict[str, int], prefix: str) -> pd.DataFrame:
        common = patient_tf.index.intersection(hyp.index)
        rows = []
        for tf in patient_tf.columns:
            vals = patient_tf.loc[common, tf]
            if vals.notna().sum() >= 4:
                rho, p = spearmanr(vals, hyp.loc[common])
                lo, hi = fisher_ci(rho, len(common))
            else:
                rho = p = lo = hi = np.nan
            rows.append(
                {
                    "tf": tf,
                    f"{prefix}_n_targets_present": int(count_map[tf]),
                    f"{prefix}_spearman_rho_with_hypoxia": rho,
                    f"{prefix}_ci_low": lo,
                    f"{prefix}_ci_high": hi,
                    f"{prefix}_p_value": p,
                }
            )
        out = pd.DataFrame(rows).dropna(subset=[f"{prefix}_spearman_rho_with_hypoxia"])
        out = add_adjusted_p(
            out,
            p_col=f"{prefix}_p_value",
            out_col=f"{prefix}_fdr_bh_across_tfs",
            method="fdr_bh",
        )
        out[f"{prefix}_rank_by_rho_desc"] = out[f"{prefix}_spearman_rho_with_hypoxia"].rank(
            method="min", ascending=False
        ).astype(int)
        out[f"{prefix}_percentile_by_rho"] = 100 * (
            1 - (out[f"{prefix}_rank_by_rho_desc"] - 1) / max(len(out) - 1, 1)
        )
        return out

    raw_patient_tf, raw_counts = score_prior(prior)
    deoverlap_prior = prior[~prior["target"].isin(hypoxia_targets)].copy()
    de_patient_tf, de_counts = score_prior(deoverlap_prior)
    raw = corr_table(raw_patient_tf, raw_counts, "raw")
    de = corr_table(de_patient_tf, de_counts, "deoverlap")

    out = raw.merge(de, on="tf", how="outer").merge(overlap_counts, on="tf", how="left")
    out["n_targets_present"] = out["deoverlap_n_targets_present"]
    out["spearman_rho_with_hypoxia"] = out["deoverlap_spearman_rho_with_hypoxia"]
    out["ci_low"] = out["deoverlap_ci_low"]
    out["ci_high"] = out["deoverlap_ci_high"]
    out["p_value"] = out["deoverlap_p_value"]
    out["fdr_bh_across_tfs"] = out["deoverlap_fdr_bh_across_tfs"]
    out["rank_by_rho_desc"] = out["deoverlap_rank_by_rho_desc"]
    out["percentile_by_rho"] = out["deoverlap_percentile_by_rho"]
    out = out.dropna(subset=["spearman_rho_with_hypoxia"]).sort_values("rank_by_rho_desc")
    front_cols = [
        "tf",
        "n_targets_present",
        "spearman_rho_with_hypoxia",
        "ci_low",
        "ci_high",
        "p_value",
        "fdr_bh_across_tfs",
        "rank_by_rho_desc",
        "percentile_by_rho",
        "n_targets_present_raw",
        "n_hypoxia_overlap_targets",
    ]
    out = out[front_cols + [c for c in out.columns if c not in front_cols]]
    out.to_csv(TABLE_DIR / "dorothea_all_tf_hypoxia_contextual_benchmark.csv", index=False)
    top_tfs = out.head(10)["tf"].tolist()
    reference_tfs = ["JUN", "ATF4", "NFE2L2", "MYC"]
    display_tfs = list(dict.fromkeys(top_tfs + reference_tfs))
    selected = out[out["tf"].isin(display_tfs)].copy().sort_values("rank_by_rho_desc")
    selected.to_csv(TABLE_DIR / "dorothea_selected_tf_hypoxia_contextual_benchmark.csv", index=False)
    return out


def clean_marker_gene(g: str, excluded: set[str]) -> bool:
    if g in excluded:
        return False
    if g.startswith(("MT-", "RPL", "RPS")):
        return False
    if g in {"MALAT1", "XIST", "BACH1", "HBB", "HBA1", "HBA2", "HBM", "HBD", "HBG1", "HBG2"}:
        return False
    if re.match(r"^AC[0-9A-Z]+\\.", g):
        return False
    return True


def stress_state_signature(adata: ad.AnnData, df: pd.DataFrame) -> pd.DataFrame:
    ph = pd.read_csv(PHENO_TARGETS)
    excluded = set(ph.loc[ph["signature"].eq("HALLMARK_HYPOXIA"), "target"].astype(str))
    excluded.add("BACH1")

    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    genes = np.asarray(adata.var_names)
    rows = []
    for dataset in sorted(df["dataset"].unique()):
        idx_dataset = np.where(df["dataset"].values == dataset)[0]
        idx_state = np.where((df["dataset"].values == dataset) & (df["state"].values == "Stress_AP1"))[0]
        idx_rest = np.setdiff1d(idx_dataset, idx_state)
        if len(idx_state) == 0 or len(idx_rest) == 0:
            continue
        mean_state = np.asarray(X[idx_state].mean(axis=0)).ravel()
        mean_rest = np.asarray(X[idx_rest].mean(axis=0)).ravel()
        pct_state = np.asarray((X[idx_state] > 0).mean(axis=0)).ravel()
        pct_rest = np.asarray((X[idx_rest] > 0).mean(axis=0)).ravel()
        for i, gene in enumerate(genes):
            rows.append(
                {
                    "dataset": dataset,
                    "gene": gene,
                    "mean_stress_ap1": mean_state[i],
                    "mean_reference": mean_rest[i],
                    "mean_delta": mean_state[i] - mean_rest[i],
                    "pct_stress_ap1": pct_state[i],
                    "pct_reference": pct_rest[i],
                    "pct_delta": pct_state[i] - pct_rest[i],
                }
            )
    per_dataset = pd.DataFrame(rows)
    per_dataset = per_dataset[per_dataset["gene"].map(lambda g: clean_marker_gene(str(g), excluded))]
    per_dataset.to_csv(TABLE_DIR / "stress_ap1_marker_candidates_by_dataset.csv.gz", index=False)

    summary = (
        per_dataset.groupby("gene")
        .agg(
            n_datasets=("dataset", "nunique"),
            min_delta=("mean_delta", "min"),
            mean_delta=("mean_delta", "mean"),
            min_pct_delta=("pct_delta", "min"),
            mean_pct_stress=("pct_stress_ap1", "mean"),
        )
        .reset_index()
    )
    summary = summary[
        (summary["n_datasets"] == df["dataset"].nunique())
        & (summary["min_delta"] > 0.15)
        & (summary["min_pct_delta"] > 0.05)
        & (summary["mean_pct_stress"] > 0.15)
    ].copy()
    summary = summary.sort_values(["min_delta", "mean_delta", "min_pct_delta"], ascending=False)
    signature = summary.head(30).copy()
    signature["signature"] = "Stress_AP1_state_marker_without_BACH1_or_Hallmark_hypoxia"
    signature.to_csv(TABLE_DIR / "stress_ap1_state_signature_genes.csv", index=False)
    return signature


def parse_stage_group(x: object) -> str | float:
    s = str(x)
    if not s or s in {"nan", "[Not Available]", "[Discrepancy]"}:
        return np.nan
    if "III" in s or "IV" in s:
        return "III-IV"
    if "I" in s or "II" in s:
        return "I-II"
    return np.nan


def parse_t_group(x: object) -> str | float:
    s = str(x)
    if s.startswith(("T3", "T4")):
        return "T3-T4"
    if s.startswith(("T1", "T2")):
        return "T1-T2"
    return np.nan


def parse_n_group(x: object) -> str | float:
    s = str(x)
    if s == "N0":
        return "N0"
    if s.startswith("N") and s not in {"NX", "nan"}:
        return "N+"
    return np.nan


def parse_m_group(x: object) -> str | float:
    s = str(x)
    if s == "M0":
        return "M0"
    if s.startswith("M1"):
        return "M1"
    return np.nan


def read_tcga_cohort(cohort: str, signature_genes: list[str]) -> pd.DataFrame:
    expr = pd.read_csv(ROOT / f"data/tcga_xena/TCGA.{cohort}.HiSeqV2.gz", sep="\t")
    expr = expr.rename(columns={expr.columns[0]: "gene"}).set_index("gene")
    sample_cols = [c for c in expr.columns if c.endswith("-01")]
    expr = expr[sample_cols]
    genes = [g for g in signature_genes if g in expr.index]
    z = expr.loc[genes].T.apply(zscore, axis=0)
    score = z.mean(axis=1)
    score_df = pd.DataFrame({"sampleID": score.index, "state_signature_score": score.values})
    score_df["patient"] = score_df["sampleID"].str[:12]
    score_df = score_df.groupby("patient", as_index=False).agg(
        sampleID=("sampleID", "first"), state_signature_score=("state_signature_score", "mean")
    )

    clin = pd.read_csv(ROOT / f"data/tcga_xena/TCGA.{cohort}.clinicalMatrix", sep="\t")
    clin = clin[clin["sampleID"].astype(str).str.endswith("-01")].copy()
    clin["patient"] = clin["sampleID"].astype(str).str[:12]
    clin = clin.drop_duplicates("patient")
    out = score_df.merge(clin, on="patient", how="left", suffixes=("", "_clinical"))
    out["cohort"] = cohort
    out["stage_group"] = out["pathologic_stage"].map(parse_stage_group) if "pathologic_stage" in out else np.nan
    out["t_group"] = out["pathologic_T"].map(parse_t_group) if "pathologic_T" in out else np.nan
    out["n_group"] = out["pathologic_N"].map(parse_n_group) if "pathologic_N" in out else np.nan
    out["m_group"] = out["pathologic_M"].map(parse_m_group) if "pathologic_M" in out else np.nan
    out["os_event"] = out["vital_status"].astype(str).str.upper().eq("DECEASED") if "vital_status" in out else np.nan
    death = pd.to_numeric(out.get("days_to_death", pd.Series(index=out.index)), errors="coerce")
    follow = pd.to_numeric(out.get("days_to_last_followup", pd.Series(index=out.index)), errors="coerce")
    out["os_time"] = np.where(out["os_event"], death, follow)
    out["os_time"] = pd.to_numeric(out["os_time"], errors="coerce")
    return out


def tcga_clinical_projection(signature: pd.DataFrame) -> dict[str, pd.DataFrame]:
    genes = signature["gene"].astype(str).tolist()
    rng = np.random.default_rng(20260914)
    tcga = pd.concat([read_tcga_cohort("LUAD", genes), read_tcga_cohort("LUSC", genes)], ignore_index=True)
    tcga["state_signature_z"] = zscore(tcga["state_signature_score"])
    tcga.to_csv(TABLE_DIR / "tcga_stress_ap1_state_signature_scores_clinical.csv", index=False)

    assoc_rows = []
    endpoints = [
        ("Stage III-IV vs I-II", "stage_group", "III-IV", "I-II"),
        ("T3-T4 vs T1-T2", "t_group", "T3-T4", "T1-T2"),
        ("N+ vs N0", "n_group", "N+", "N0"),
        ("M1 vs M0", "m_group", "M1", "M0"),
    ]
    for cohort_label, cohort_df in [("Combined", tcga), *list(tcga.groupby("cohort"))]:
        for label, col, high, low in endpoints:
            g = cohort_df[cohort_df[col].isin([high, low])].copy()
            if g[col].nunique() == 2 and len(g) >= 20:
                x1 = g.loc[g[col] == high, "state_signature_z"].dropna()
                x0 = g.loc[g[col] == low, "state_signature_z"].dropna()
                if len(x1) > 1 and len(x0) > 1:
                    stat, p = mannwhitneyu(x1, x0, alternative="two-sided")
                    delta = float(x1.median() - x0.median())
                    median_high = float(x1.median())
                    median_low = float(x0.median())
                    ci_low, ci_high = bootstrap_median_difference(x1, x0, rng)
                else:
                    p = delta = median_high = median_low = ci_low = ci_high = np.nan
                assoc_rows.append(
                    {
                        "cohort": cohort_label,
                        "endpoint": label,
                        "n_total": int(len(g)),
                        "n_high": int((g[col] == high).sum()),
                        "n_low": int((g[col] == low).sum()),
                        "median_z_high": median_high,
                        "median_z_low": median_low,
                        "median_z_high_minus_low": delta,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "bootstrap_iterations": 10000,
                        "mannwhitney_p": p,
                    }
                )
    assoc = pd.DataFrame(assoc_rows)
    assoc = add_adjusted_p(
        assoc,
        p_col="mannwhitney_p",
        out_col="holm_adjusted_p_12_tests",
        method="holm",
    )
    assoc = add_adjusted_p(
        assoc,
        p_col="mannwhitney_p",
        out_col="bh_fdr_12_tests",
        method="fdr_bh",
    )
    assoc["multiplicity_family"] = "Stage, T, N and M comparisons in combined, LUAD and LUSC cohorts (12 tests)"
    assoc.to_csv(TABLE_DIR / "tcga_stress_ap1_state_signature_clinical_associations.csv", index=False)

    cox_rows = []
    for label, g0 in [("Combined", tcga), *list(tcga.groupby("cohort"))]:
        g = g0[["os_time", "os_event", "state_signature_z", "stage_group", "cohort"]].dropna().copy()
        g = g[g["os_time"] > 0].copy()
        if len(g) < 50 or g["os_event"].sum() < 10:
            continue
        cov = pd.DataFrame({"state_signature_z": g["state_signature_z"].astype(float)})
        if label == "Combined":
            cov = pd.concat([cov, pd.get_dummies(g["cohort"], prefix="cohort", drop_first=True, dtype=float)], axis=1)
        cov = pd.concat([cov, pd.get_dummies(g["stage_group"], prefix="stage", drop_first=True, dtype=float)], axis=1)
        try:
            fit = PHReg(g["os_time"].astype(float), cov.astype(float), status=g["os_event"].astype(int)).fit(disp=False)
            beta = float(fit.params[0])
            se = float(fit.bse[0])
            p = float(fit.pvalues[0])
            schoenfeld = np.asarray(fit.schoenfeld_residuals)[:, 0]
            log_time = np.log(g["os_time"].astype(float).to_numpy())
            ph_mask = np.isfinite(schoenfeld) & np.isfinite(log_time)
            if ph_mask.sum() >= 10:
                ph_rho, ph_p = spearmanr(schoenfeld[ph_mask], log_time[ph_mask])
            else:
                ph_rho = ph_p = np.nan
            cox_rows.append(
                {
                    "cohort": label,
                    "n": int(len(g)),
                    "events": int(g["os_event"].sum()),
                    "hr_per_sd": math.exp(beta),
                    "ci_low": math.exp(beta - 1.96 * se),
                    "ci_high": math.exp(beta + 1.96 * se),
                    "p_value": p,
                    "state_score_schoenfeld_logtime_rho": ph_rho,
                    "state_score_schoenfeld_logtime_p": ph_p,
                    "state_score_schoenfeld_events_used": int(ph_mask.sum()),
                    "adjustment": "stage_group" + (" + cohort" if label == "Combined" else ""),
                }
            )
        except Exception as exc:
            cox_rows.append({"cohort": label, "n": int(len(g)), "events": int(g["os_event"].sum()), "error": str(exc)})
    cox = pd.DataFrame(cox_rows)
    if "p_value" in cox.columns:
        cox = add_adjusted_p(
            cox,
            p_col="p_value",
            out_col="holm_adjusted_p_across_os_models",
            method="holm",
        )
        cox["multiplicity_family"] = "Exploratory OS Cox models in combined, LUAD and LUSC cohorts"
    cox.to_csv(TABLE_DIR / "tcga_stress_ap1_state_signature_os_cox.csv", index=False)

    mut_rows = []
    for gene in ["EGFR", "KRAS", "STK11", "BRAF"]:
        if gene not in tcga.columns:
            continue
        g = tcga[tcga["cohort"].eq("LUAD") & tcga[gene].notna()].copy()
        if g.empty:
            continue
        g["altered"] = ~g[gene].astype(str).str.lower().isin(["none", "nan", "[not available]"])
        if g["altered"].nunique() == 2:
            x1 = g.loc[g["altered"], "state_signature_z"].dropna()
            x0 = g.loc[~g["altered"], "state_signature_z"].dropna()
            if len(x1) >= 3 and len(x0) >= 3:
                _, p = mannwhitneyu(x1, x0)
                mut_rows.append(
                    {
                        "cohort": "LUAD",
                        "gene": gene,
                        "n_altered": int(len(x1)),
                        "n_wildtype_or_none": int(len(x0)),
                        "median_z_altered_minus_wildtype": float(x1.median() - x0.median()),
                        "mannwhitney_p": p,
                    }
                )
    muts = pd.DataFrame(mut_rows)
    muts.to_csv(TABLE_DIR / "tcga_luad_state_signature_mutation_associations.csv", index=False)
    return {"tcga": tcga, "assoc": assoc, "cox": cox, "muts": muts}


def candidate_evidence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rec = pd.read_csv(RECURRENT_TARGETS)
    cand = pd.read_csv(CONSENSUS_CANDIDATES)
    atac = pd.read_csv(ATAC_SUPPORT)
    stress = []
    for gene in rec["gene"].astype(str):
        if gene in df.columns:
            stress.append({"gene": gene, "stress_ap1_expression_delta": np.nan})
    merged = rec.merge(cand, on="gene", how="left", suffixes=("", "_expr"))
    merged = merged.merge(
        atac[
            [
                "gene",
                "n_ucsc_score_ge_300_motif_peak_links_in_window",
                "n_samples_with_motif_peak_in_window",
                "promoter_support",
                "recurrent_ge3_samples",
            ]
        ],
        on="gene",
        how="left",
    )
    # Add expression enrichment from the primary state table directly from AnnData when genes are present.
    adata = ad.read_h5ad(PRIMARY_H5AD)
    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    obs = adata.obs.copy()
    obs["state"] = obs["epi_marker_only_broad_class"].map(simplify_state)
    var_index = pd.Series(np.arange(adata.n_vars), index=adata.var_names)
    deltas = {}
    idx_stress = np.where(obs["state"].values == "Stress_AP1")[0]
    idx_rest = np.where(obs["state"].values != "Stress_AP1")[0]
    for gene in merged["gene"].astype(str):
        if gene in var_index.index:
            gi = int(var_index[gene])
            deltas[gene] = float(np.asarray(X[idx_stress, gi].mean()).ravel()[0] - np.asarray(X[idx_rest, gi].mean()).ravel()[0])
    merged["stress_ap1_expression_delta"] = merged["gene"].map(deltas)
    merged["atac_ge300_context"] = merged["n_ucsc_score_ge_300_motif_peak_links_in_window"].fillna(0) > 0
    merged["candidate_display_score"] = (
        merged["n_runs_detected"].fillna(0)
        + merged["atac_ge300_context"].astype(int)
        + (merged["stress_ap1_expression_delta"].fillna(-999) > 0).astype(int)
        + (merged["partial_corr_significant"].fillna(False).astype(bool)).astype(int)
    )
    top = merged.sort_values(
        ["candidate_display_score", "n_runs_detected", "weighted_importance"], ascending=False
    ).head(10)
    out = top[
        [
            "gene",
            "n_runs_detected",
            "weighted_importance",
            "partial_pearson_r",
            "stress_ap1_expression_delta",
            "n_ucsc_score_ge_300_motif_peak_links_in_window",
            "n_samples_with_motif_peak_in_window",
            "promoter_support",
            "candidate_display_score",
        ]
    ].copy()
    out.to_csv(TABLE_DIR / "figure4e_candidate_evidence_matrix.csv", index=False)
    return out


def plot_figure4(tables: dict[str, pd.DataFrame], attenuation: pd.DataFrame, candidates: pd.DataFrame) -> None:
    sns.set_theme(style="white", font="Arial")
    fig = plt.figure(figsize=(13.2, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.0], width_ratios=[1.1, 1.05])

    ax = fig.add_subplot(gs[0, 0])
    umap = tables["umap"].copy()
    for st in STATE_ORDER:
        sub = umap[umap["state"] == st]
        if not sub.empty:
            ax.scatter(
                sub["UMAP1"],
                sub["UMAP2"],
                s=3,
                alpha=0.65,
                c=COLORS[st],
                label=STATE_DISPLAY.get(st, st),
                rasterized=True,
            )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(frameon=False, markerscale=4, fontsize=7, ncol=2, loc="best")
    ax.text(-0.08, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=13)

    ax = fig.add_subplot(gs[0, 1])
    heat_df = tables["heat_z"].loc[[s for s in STATE_ORDER if s in tables["heat_z"].index]]
    n_by_state = tables["state_summary"].set_index("state")["n_cells"].to_dict()
    p_by_state = tables["state_summary"].set_index("state")["n_patients"].to_dict()
    sns.heatmap(
        heat_df,
        cmap="vlag",
        center=0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Column z"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.set_yticklabels(
        [
            f"{STATE_DISPLAY.get(s, s)}{' (descriptive)' if s == 'Other' else ''} | {int(n_by_state.get(s, 0)):,} cells | {int(p_by_state.get(s, 0))} patients"
            for s in heat_df.index
        ],
        rotation=0,
    )
    ax.tick_params(axis="y", labelsize=8)
    for tick, st in zip(ax.get_yticklabels(), heat_df.index):
        if st == "Other":
            tick.set_color("0.45")
    ax.text(-0.08, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=13)

    sub = gs[1, 0].subgridspec(1, 2, width_ratios=[0.67, 0.33], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    att = attenuation.copy()
    y = np.arange(len(att))
    ax.axvline(0, color="0.75", lw=0.8)
    ax.errorbar(att["rho"], y, xerr=[att["rho"] - att["ci_low"], att["ci_high"] - att["rho"]], fmt="o", color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(att["comparison"], fontsize=8)
    ax.set_xlabel(r"Spearman $\rho$")
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    for i, r in att.iterrows():
        label = rf"$\rho$={r.rho:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" + "\n" + f"n={int(r.n_patients)}; raw P={format_p(r.p_value)}"
        ax_ann.text(0.0, i, label, ha="left", va="center", fontsize=8.0)
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    ax.text(-0.08, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=13)

    sub = gs[1, 1].subgridspec(1, 2, width_ratios=[0.62, 0.38], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    ws = tables["within_state"].dropna(subset=["spearman_rho"]).copy()
    ws["state"] = pd.Categorical(ws["state"].astype(str), WITHIN_STATE_DISPLAY_ORDER, ordered=True)
    ws = ws.dropna(subset=["state"]).sort_values("state")
    y = np.arange(len(ws))
    ax.axvline(0, color="0.75", lw=0.8)
    ax.errorbar(ws["spearman_rho"], y, xerr=[ws["spearman_rho"] - ws["ci_low"], ws["ci_high"] - ws["spearman_rho"]], fmt="o", color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels([STATE_DISPLAY.get(str(s), str(s)) for s in ws["state"]], fontsize=8)
    ax.set_xlabel(r"Within-state patient-level $\rho$ (exploratory)")
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    for i, r in enumerate(ws.itertuples(index=False)):
        label = (
            rf"$\rho$={r.spearman_rho:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]"
            + "\n"
            + f"n={int(r.n_patient_state_units)}; Holm P={format_p(r.holm_adjusted_p_across_states)}"
        )
        ax_ann.text(0.0, i, label, ha="left", va="center", fontsize=8.0)
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    ax.text(-0.08, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=13)

    ax = fig.add_subplot(gs[2, :])
    mat = candidates.set_index("gene")[
        [
            "n_runs_detected",
            "weighted_importance",
            "partial_pearson_r",
            "stress_ap1_expression_delta",
            "n_ucsc_score_ge_300_motif_peak_links_in_window",
            "n_samples_with_motif_peak_in_window",
        ]
    ].copy()
    display = mat.apply(zscore, axis=0)
    display.columns = ["Seeds", "Importance", "Partial r", "Stress delta", "Motif links", "ATAC samples"]
    sns.heatmap(display, cmap="vlag", center=0, linewidths=0.35, linecolor="white", cbar_kws={"label": "Column z (display)"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.text(-0.035, 1.04, "e", transform=ax.transAxes, fontweight="bold", fontsize=13)

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"Figure4_epithelial_state_context.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure5(tfbench: pd.DataFrame, cohort_split: pd.DataFrame, clin: dict[str, pd.DataFrame]) -> None:
    tcga = clin["tcga"].copy()
    assoc = clin["assoc"].copy()
    cox = clin["cox"].copy()
    top_tfs = tfbench.sort_values("rank_by_rho_desc").head(10)["tf"].tolist()
    reference_tfs = ["JUN", "ATF4", "NFE2L2", "MYC"]
    display_tfs = list(dict.fromkeys(top_tfs + reference_tfs))
    selected_tfs = tfbench[tfbench["tf"].isin(display_tfs)].copy()
    selected_tfs = selected_tfs.sort_values("rank_by_rho_desc")

    sns.set_theme(style="white", font="Arial")
    fig = plt.figure(figsize=(12.3, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    sub = gs[0, 0].subgridspec(1, 2, width_ratios=[0.82, 0.18], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_rank = fig.add_subplot(sub[0, 1], sharey=ax)
    colors = ["#C44E52" if tf == "BACH1" else "#4C72B0" for tf in selected_tfs["tf"]]
    y = np.arange(len(selected_tfs))
    ax.barh(y, selected_tfs["spearman_rho_with_hypoxia"], color=colors)
    ax.axvline(0, color="0.75", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(selected_tfs["tf"], fontsize=7)
    for tick in ax.get_yticklabels():
        tick.set_fontstyle("italic")
    ax.invert_yaxis()
    ax.set_xlabel("De-overlapped rho with hypoxia")
    ax.set_xlim(0, 1)
    ax.set_ylabel("")
    for i, r in enumerate(selected_tfs.itertuples(index=False)):
        ax_rank.text(0.0, i, f"rank {int(r.rank_by_rho_desc)}", ha="left", va="center", fontsize=6.5)
    ax_rank.set_xlim(0, 1)
    ax_rank.set_ylim(ax.get_ylim())
    ax_rank.axis("off")
    ax.text(-0.1, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=13)

    sub = gs[0, 1].subgridspec(1, 2, width_ratios=[0.64, 0.36], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    cs = cohort_split.sort_values("rho")
    y = np.arange(len(cs))
    ax.axvline(0, color="0.75", lw=0.8)
    ax.errorbar(cs["rho"], y, xerr=[cs["rho"] - cs["ci_low"], cs["ci_high"] - cs["rho"]], fmt="o", color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(cs["dataset"], fontsize=8)
    ax.set_xlabel("scRNA cohort rho")
    ax.set_xlim(-1, 1)
    for i, r in enumerate(cs.itertuples(index=False)):
        label = rf"$\rho$={r.rho:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" + "\n" + f"n={int(r.n_patients)}; P={format_p(r.p_value)}"
        ax_ann.text(0.0, i, label, ha="left", va="center", fontsize=6.2)
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    ax.text(-0.1, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=13)

    sub = gs[1, 0].subgridspec(1, 2, width_ratios=[0.72, 0.28], wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    ax_ann = fig.add_subplot(sub[0, 1], sharey=ax)
    clinical = assoc[assoc["endpoint"].isin(["Stage III-IV vs I-II", "N+ vs N0"])].copy()
    clinical["label"] = clinical["cohort"] + ": " + clinical["endpoint"].str.replace(" vs ", " vs\n", regex=False)
    endpoint_order = {"Stage III-IV vs I-II": 0, "N+ vs N0": 1}
    cohort_order = {"Combined": 0, "LUAD": 1, "LUSC": 2}
    clinical["_endpoint_order"] = clinical["endpoint"].map(endpoint_order)
    clinical["_cohort_order"] = clinical["cohort"].map(cohort_order)
    clinical = clinical.sort_values(["_endpoint_order", "_cohort_order"])
    ax.axvline(0, color="0.75", lw=0.8)
    y = np.arange(len(clinical))
    ax.errorbar(
        clinical["median_z_high_minus_low"],
        y,
        xerr=[
            clinical["median_z_high_minus_low"] - clinical["bootstrap_ci_low"],
            clinical["bootstrap_ci_high"] - clinical["median_z_high_minus_low"],
        ],
        fmt="o",
        color="#55A868",
        ecolor="#55A868",
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(clinical["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("State-signature median z difference")
    for i, r in enumerate(clinical.itertuples(index=False)):
        ax_ann.text(0.0, i, f"Holm P={format_p(r.holm_adjusted_p_12_tests)}", ha="left", va="center", fontsize=7)
    finite_ci = clinical[["bootstrap_ci_low", "bootstrap_ci_high"]].to_numpy().ravel()
    finite_ci = finite_ci[np.isfinite(finite_ci)]
    if len(finite_ci):
        lo = min(-0.45, float(finite_ci.min()) - 0.08)
        hi = max(0.45, float(finite_ci.max()) + 0.08)
        ax.set_xlim(lo, hi)
    ax_ann.set_xlim(0, 1)
    ax_ann.set_ylim(ax.get_ylim())
    ax_ann.axis("off")
    ax.text(-0.1, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=13)

    ax = fig.add_subplot(gs[1, 1])
    c = cox.dropna(subset=["hr_per_sd"]).copy()
    if not c.empty:
        c["_cohort_order"] = c["cohort"].map({"Combined": 0, "LUAD": 1, "LUSC": 2}).fillna(9)
        c = c.sort_values("_cohort_order")
        y = np.arange(len(c))
        ax.axvline(1, color="0.75", lw=0.8)
        ax.errorbar(c["hr_per_sd"], y, xerr=[c["hr_per_sd"] - c["ci_low"], c["ci_high"] - c["hr_per_sd"]], fmt="o", color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels(c["cohort"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("OS hazard ratio per SD")
        hi = max(2.5, float(c["ci_high"].max()) * 1.15)
        ax.set_xlim(0.2, hi)
        for i, r in enumerate(c.itertuples(index=False)):
            p_show = getattr(r, "holm_adjusted_p_across_os_models", np.nan)
            ax.text(hi * 0.98, i, f"Holm P={format_p(p_show)}", ha="right", va="center", fontsize=7)
    ax.text(-0.1, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=13)

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"Figure5_pan_tf_contextual_clinical_projection.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    state_tables: dict[str, pd.DataFrame],
    attenuation: pd.DataFrame,
    tfbench: pd.DataFrame,
    cohort_split: pd.DataFrame,
    signature: pd.DataFrame,
    clin: dict[str, pd.DataFrame],
    ie_sensitivity: dict[str, pd.DataFrame],
    composition_sensitivity: pd.DataFrame,
) -> None:
    bach1 = tfbench[tfbench["tf"].eq("BACH1")].iloc[0].to_dict()
    selected = pd.read_csv(TABLE_DIR / "dorothea_selected_tf_hypoxia_contextual_benchmark.csv")
    state_top = state_tables["state_summary"].sort_values(HYP_NO, ascending=False).iloc[0].to_dict()
    clinical_stage = clin["assoc"][
        (clin["assoc"]["cohort"].eq("Combined")) & (clin["assoc"]["endpoint"].eq("Stage III-IV vs I-II"))
    ]
    clinical_n = clin["assoc"][
        (clin["assoc"]["cohort"].eq("Combined")) & (clin["assoc"]["endpoint"].eq("N+ vs N0"))
    ]
    cox_combined = clin["cox"][clin["cox"]["cohort"].eq("Combined")]
    summary = {
        "top_hypoxia_state": state_top,
        "bach1_tf_benchmark": bach1,
        "selected_tf_benchmark": selected.to_dict(orient="records"),
        "cohort_split": cohort_split.to_dict(orient="records"),
        "attenuation": attenuation.to_dict(orient="records"),
        "signature_genes": signature["gene"].tolist(),
        "immediate_early_sensitivity_by_state": ie_sensitivity["ie_state_summary"].to_dict(orient="records"),
        "immediate_early_gene_overlap": ie_sensitivity["ie_overlap"].to_dict(orient="records"),
        "composition_pca_transformation_sensitivity": composition_sensitivity.to_dict(orient="records"),
        "clinical_stage_combined": clinical_stage.to_dict(orient="records"),
        "clinical_n_combined": clinical_n.to_dict(orient="records"),
        "cox_combined": cox_combined.to_dict(orient="records"),
    }
    (REPORT_DIR / "state_contextual_clinical_revision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    adata, cell_df = load_cell_data()
    state_tables = state_context_tables(adata, cell_df)
    state_malignancy_audit(cell_df)
    attenuation = attenuation_table()
    composition_sensitivity = composition_pca_sensitivity(cell_df)
    cohort_split = cohort_split_tables()
    tfbench = tf_contextual_benchmark(adata, cell_df)
    signature = stress_state_signature(adata, cell_df)
    ie_sensitivity = immediate_early_sensitivity(adata, cell_df, signature)
    clin = tcga_clinical_projection(signature)
    candidates = candidate_evidence_matrix(cell_df)
    plot_figure4(state_tables, attenuation, candidates)
    plot_figure5(tfbench, cohort_split, clin)
    write_summary(state_tables, attenuation, tfbench, cohort_split, signature, clin, ie_sensitivity, composition_sensitivity)
    print(f"Wrote revision outputs to {OUT}")


if __name__ == "__main__":
    main()
