"""Revision-focused spatial statistics for BACH1/NOD overlap analyses.

This script keeps the original spot-level scoring intact and adds reviewer-facing
statistics that avoid treating spots as independent patients.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
SPATIAL = ROOT / "out" / "emtab13530_spatial_bach1_nod"
TABLE = SPATIAL / "tables"
SUPP_TABLE = SPATIAL / "spatial_skill_supplement" / "tables"
OUT = SPATIAL / "revision_statistics"
OUT.mkdir(parents=True, exist_ok=True)


def bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x
    return x.astype(str).str.lower().isin(["true", "1", "yes"])


def add_holm(df: pd.DataFrame, p_col: str = "wilcoxon_p", out_col: str = "holm_p") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    mask = np.isfinite(pd.to_numeric(out[p_col], errors="coerce"))
    if mask.any():
        out.loc[mask, out_col] = multipletests(out.loc[mask, p_col].astype(float), method="holm")[1]
    return out


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.sum(a | b))
    if union == 0:
        return np.nan
    return float(np.sum(a & b) / union)


def section_jaccard_permutation(spot: pd.DataFrame, n_perm: int = 1000, seed: int = 2309) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sample, sub in spot.groupby("sample", sort=False):
        bach = bool_series(sub["BACH1_high"]).to_numpy()
        nod = bool_series(sub["NOD_like_high"]).to_numpy()
        co = bach & nod
        obs = jaccard(bach, nod)
        perms = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perms[i] = jaccard(bach, rng.permutation(nod))
        expected = float(np.nanmean(perms))
        sd = float(np.nanstd(perms, ddof=1))
        p_greater = float((np.sum(perms >= obs) + 1) / (np.sum(np.isfinite(perms)) + 1))
        rows.append(
            {
                "sample": sample,
                "patient_id": sub["patient_id"].iloc[0],
                "tissue_group": sub["tissue_group"].iloc[0],
                "n_spots": int(len(sub)),
                "bach1_high_n": int(bach.sum()),
                "nod_high_n": int(nod.sum()),
                "cohigh_n": int(co.sum()),
                "cohigh_fraction": float(co.mean()),
                "observed_jaccard": obs,
                "expected_jaccard": expected,
                "expected_jaccard_sd": sd,
                "observed_expected_ratio": float(obs / expected) if expected > 0 else np.nan,
                "empirical_p_greater": p_greater,
                "permutation_n": int(n_perm),
            }
        )
    return pd.DataFrame(rows)


def paired_wilcoxon(patient_summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        wide = patient_summary.pivot(index="patient_id", columns="tissue_group", values=metric)
        wide = wide.dropna(subset=["Adjacent", "Tumor"])
        delta = wide["Tumor"] - wide["Adjacent"]
        try:
            _, p = stats.wilcoxon(wide["Tumor"], wide["Adjacent"], zero_method="wilcox")
        except ValueError:
            p = np.nan
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(len(delta)),
                "median_adjacent": float(np.nanmedian(wide["Adjacent"])) if len(delta) else np.nan,
                "median_tumor": float(np.nanmedian(wide["Tumor"])) if len(delta) else np.nan,
                "median_delta_tumor_minus_adjacent": float(np.nanmedian(delta)) if len(delta) else np.nan,
                "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
            }
        )
    return add_holm(pd.DataFrame(rows), "wilcoxon_p", "holm_p")


def qc_adjusted_bach1_detection(spot: pd.DataFrame) -> pd.DataFrame:
    df = spot[spot["tissue_group"].isin(["Adjacent", "Tumor"])].copy()
    for col in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["BACH1_detected_binary"] = bool_series(df["BACH1_detected"]).astype(int)
    df["tumor_vs_adjacent"] = df["tissue_group"].eq("Tumor").astype(int)
    df["log_total_counts"] = np.log1p(df["total_counts"])
    df["log_n_genes_by_counts"] = np.log1p(df["n_genes_by_counts"])
    model_df = df.dropna(
        subset=[
            "BACH1_detected_binary",
            "tumor_vs_adjacent",
            "log_total_counts",
            "log_n_genes_by_counts",
            "pct_counts_mt",
            "patient_id",
            "sample",
        ]
    ).copy()
    model = smf.glm(
        "BACH1_detected_binary ~ tumor_vs_adjacent + log_total_counts + log_n_genes_by_counts + pct_counts_mt + C(patient_id)",
        data=model_df,
        family=sm.families.Binomial(),
    )
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})
    ci = fit.conf_int()
    rows = []
    for term in fit.params.index:
        rows.append(
            {
                "term": term,
                "coef_log_odds": float(fit.params[term]),
                "se_cluster_by_section": float(fit.bse[term]),
                "odds_ratio": float(np.exp(fit.params[term])),
                "ci_low_or": float(np.exp(ci.loc[term, 0])),
                "ci_high_or": float(np.exp(ci.loc[term, 1])),
                "p_value": float(fit.pvalues[term]),
                "n_spots": int(model_df.shape[0]),
                "n_sections": int(model_df["sample"].nunique()),
                "n_patients": int(model_df["patient_id"].nunique()),
                "model": "cluster-robust logistic GLM with patient fixed effects; clustered by Visium section",
            }
        )
    return pd.DataFrame(rows)


def patient_level_signature_tests() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = SUPP_TABLE / "spatial_skill_all_spot_annotations_deconv_domains.csv.gz"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["tissue_group"].isin(["Adjacent", "Tumor"])].copy()
    df["BACH1_NOD_cohigh"] = bool_series(df["BACH1_NOD_cohigh"])
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    rows = []
    for (patient, tissue, sample), sub in df.groupby(["patient_id", "tissue_group", "sample"], sort=False):
        co = sub["BACH1_NOD_cohigh"].to_numpy(bool)
        for col in sig_cols:
            if co.sum() < 3 or (~co).sum() < 3:
                delta = np.nan
                co_mean = np.nan
                non_mean = np.nan
            else:
                co_mean = float(np.nanmean(sub.loc[co, col]))
                non_mean = float(np.nanmean(sub.loc[~co, col]))
                delta = co_mean - non_mean
            rows.append(
                {
                    "patient_id": patient,
                    "tissue_group": tissue,
                    "sample": sample,
                    "signature": col.replace("sig_", ""),
                    "cohigh_mean": co_mean,
                    "non_cohigh_mean": non_mean,
                    "delta": delta,
                    "n_cohigh_spots": int(co.sum()),
                    "n_non_cohigh_spots": int((~co).sum()),
                }
            )
    section = pd.DataFrame(rows)
    patient = (
        section.groupby(["patient_id", "tissue_group", "signature"], as_index=False)
        .agg(
            n_sections=("sample", "nunique"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            total_cohigh_spots=("n_cohigh_spots", "sum"),
            total_non_cohigh_spots=("n_non_cohigh_spots", "sum"),
        )
    )
    tests = []
    for sig, sub in patient.groupby("signature", sort=True):
        wide = sub.pivot(index="patient_id", columns="tissue_group", values="mean_delta").dropna(
            subset=["Adjacent", "Tumor"]
        )
        paired_delta = wide["Tumor"] - wide["Adjacent"]
        tumor = wide["Tumor"].dropna()
        try:
            _, paired_p = stats.wilcoxon(wide["Tumor"], wide["Adjacent"], zero_method="wilcox")
        except ValueError:
            paired_p = np.nan
        try:
            _, tumor_one_sample_p = stats.wilcoxon(tumor, zero_method="wilcox")
        except ValueError:
            tumor_one_sample_p = np.nan
        tests.append(
            {
                "signature": sig,
                "n_pairs": int(len(paired_delta)),
                "tumor_median_delta_cohigh_minus_non": float(np.nanmedian(wide["Tumor"])) if len(wide) else np.nan,
                "adjacent_median_delta_cohigh_minus_non": float(np.nanmedian(wide["Adjacent"])) if len(wide) else np.nan,
                "paired_median_delta_tumor_minus_adjacent": float(np.nanmedian(paired_delta)) if len(paired_delta) else np.nan,
                "paired_wilcoxon_p": float(paired_p) if np.isfinite(paired_p) else np.nan,
                "tumor_one_sample_wilcoxon_p": float(tumor_one_sample_p)
                if np.isfinite(tumor_one_sample_p)
                else np.nan,
            }
        )
    tests = pd.DataFrame(tests)
    tests = add_holm(tests, "paired_wilcoxon_p", "paired_holm_p")
    tests = add_holm(tests, "tumor_one_sample_wilcoxon_p", "tumor_one_sample_holm_p")
    return patient, tests


def main() -> None:
    spot = pd.read_csv(TABLE / "E-MTAB-13530_bach1_nod_all_spot_scores.csv.gz")
    orig_stats = pd.read_csv(TABLE / "E-MTAB-13530_tumor_vs_adjacent_paired_wilcoxon.csv")
    orig_stats_holm = add_holm(orig_stats, "wilcoxon_p", "holm_p")
    orig_stats_holm.to_csv(OUT / "E-MTAB-13530_tumor_vs_adjacent_paired_wilcoxon_holm.csv", index=False)

    section_perm = section_jaccard_permutation(spot)
    section_perm.to_csv(OUT / "spatial_jaccard_section_permutation.csv.gz", index=False, compression="gzip")
    patient_perm = (
        section_perm[section_perm["tissue_group"].isin(["Adjacent", "Tumor"])]
        .groupby(["patient_id", "tissue_group"], as_index=False)
        .agg(
            n_sections=("sample", "nunique"),
            observed_jaccard=("observed_jaccard", "mean"),
            expected_jaccard=("expected_jaccard", "mean"),
            observed_expected_ratio=("observed_expected_ratio", "mean"),
            cohigh_fraction=("cohigh_fraction", "mean"),
        )
    )
    patient_perm.to_csv(OUT / "spatial_jaccard_patient_means.csv", index=False)
    jaccard_tests = paired_wilcoxon(
        patient_perm,
        ["observed_jaccard", "expected_jaccard", "observed_expected_ratio", "cohigh_fraction"],
    )
    jaccard_tests.to_csv(OUT / "spatial_jaccard_patient_paired_tests.csv", index=False)

    qc_model = qc_adjusted_bach1_detection(spot)
    qc_model.to_csv(OUT / "spatial_bach1_detection_qc_adjusted_glm.csv", index=False)

    patient_sig, sig_tests = patient_level_signature_tests()
    if not patient_sig.empty:
        patient_sig.to_csv(OUT / "spatial_cohigh_signature_patient_means.csv", index=False)
        sig_tests.to_csv(OUT / "spatial_cohigh_signature_patient_tests.csv", index=False)

    summary = {
        "n_spots": int(spot.shape[0]),
        "n_sections": int(spot["sample"].nunique()),
        "n_patients_tumor_adjacent": int(
            spot.loc[spot["tissue_group"].isin(["Adjacent", "Tumor"]), "patient_id"].nunique()
        ),
        "holm_corrected_metrics": orig_stats_holm.to_dict(orient="records"),
        "jaccard_patient_tests": jaccard_tests.to_dict(orient="records"),
        "bach1_detection_qc_adjusted_tumor_term": qc_model.loc[
            qc_model["term"].eq("tumor_vs_adjacent")
        ].to_dict(orient="records"),
        "signature_patient_tests": sig_tests.to_dict(orient="records") if not sig_tests.empty else [],
        "outputs": sorted(p.name for p in OUT.glob("*")),
    }
    (OUT / "spatial_revision_statistics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
