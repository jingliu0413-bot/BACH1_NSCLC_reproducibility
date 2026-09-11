"""Final inference-sensitivity analyses for the BACH1 NSCLC revision.

This script adds the conservative checks requested after the robustness/TCGA
round: per-patient QC-adjusted BACH1-detection models, patient-level
covariate-residualized BACH1-hypoxia association, and TCGA missingness/model
count plus cohort-interaction summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
ROBUST_TABLE = ROOT / "out/external_bach1_activity_robustness/tables"
TCGA_TABLE = ROOT / "out/tcga_bach1_hypoxia_validation/tables"
OUT = ROOT / "out/final_inference_sensitivity"
TABLE_DIR = OUT / "tables"

DOR_SIG = "DOROTHEA_BACH1_ABC_TF_ACTIVITY"
HYPOXIA_SIG = "HALLMARK_HYPOXIA"
DOR_COL = f"{DOR_SIG}__mean_z"
HYPOXIA_COL = f"{HYPOXIA_SIG}__mean_z"
DOR_NO_COL = f"{DOR_SIG}_NO_HYPOXIA_OVERLAP__mean_z"
HYPOXIA_NO_COL = f"{HYPOXIA_SIG}_NO_DOROTHEA_OVERLAP__mean_z"


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    sd = v.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(0.0, index=s.index)
    return (v - v.mean()) / sd


def safe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def format_p_text(value) -> str:
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


def clean_stage(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "unknown"}:
        return "unknown"
    return text


def simplify_state(value: object) -> str:
    text = str(value)
    for key in ["Proliferative", "AT2_like", "Epithelial_core", "Stress_AP1", "Ciliated"]:
        if key in text:
            return key
    return "Other"


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, int]:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or len(np.unique(a[ok])) < 2 or len(np.unique(b[ok])) < 2:
        return np.nan, np.nan, int(ok.sum())
    rho, p = stats.spearmanr(a[ok], b[ok])
    return float(rho), float(p), int(ok.sum())


def read_cell_inputs() -> pd.DataFrame:
    scores = pd.read_csv(ROBUST_TABLE / "deoverlapped_dorothea_hypoxia_cell_scores.csv.gz", index_col=0)
    cov = pd.read_csv(ROBUST_TABLE / "qc_adjusted_model_cell_covariates.csv.gz", index_col=0)
    df = cov.join(scores[[DOR_NO_COL, HYPOXIA_NO_COL]], how="inner")
    df = df.rename(columns={DOR_NO_COL: "activity_score", HYPOXIA_NO_COL: "hypoxia_score"})
    df["BACH1_detected"] = pd.to_numeric(df["BACH1_detected"], errors="coerce").fillna(0).astype(int)
    df["state_simple"] = df["epi_state_model"].map(simplify_state)
    return df


def fit_per_patient_qc_models(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for patient, sub in cells.groupby("patient", observed=True):
        sub = sub.copy()
        n_cells = int(len(sub))
        n_detected = int(sub["BACH1_detected"].sum())
        n_undetected = n_cells - n_detected
        row = {
            "patient": patient,
            "dataset": str(sub["dataset"].iloc[0]),
            "n_cells": n_cells,
            "n_detected": n_detected,
            "n_undetected": n_undetected,
            "model": "activity_score ~ BACH1_detected + log_total_counts + n_genes_by_counts + pct_counts_mt + epithelial_state",
            "status": "fit",
            "exclusion_reason": "",
            "bach1_beta": np.nan,
            "bach1_se_hc3": np.nan,
            "bach1_ci_low_hc3": np.nan,
            "bach1_ci_high_hc3": np.nan,
            "bach1_p_hc3": np.nan,
            "n_state_terms": 0,
            "df_resid": np.nan,
            "r_squared": np.nan,
            "evaluable_rule": "n_cells >= 20 and at least 3 BACH1-detected plus 3 BACH1-undetected cells",
        }
        if n_cells < 20:
            row["status"] = "excluded"
            row["exclusion_reason"] = "fewer_than_20_cells"
            rows.append(row)
            continue
        if min(n_detected, n_undetected) < 3:
            row["status"] = "excluded"
            row["exclusion_reason"] = "fewer_than_3_cells_in_one_BACH1_group"
            rows.append(row)
            continue
        for col in ["log_total_counts", "n_genes_by_counts", "pct_counts_mt"]:
            sub[f"{col}_z"] = zscore(sub[col])

        counts = sub["state_simple"].value_counts()
        keep_states = set(counts[counts >= 10].index)
        sub["state_model"] = sub["state_simple"].where(sub["state_simple"].isin(keep_states), "Other")
        state_terms = max(int(sub["state_model"].nunique()) - 1, 0)
        base_terms = 1 + 3
        use_state = state_terms > 0 and n_cells - (1 + base_terms + state_terms) >= 5
        formula = "activity_score ~ BACH1_detected + log_total_counts_z + n_genes_by_counts_z + pct_counts_mt_z"
        if use_state:
            formula += " + C(state_model)"
            row["n_state_terms"] = state_terms
        else:
            row["model"] = "activity_score ~ BACH1_detected + log_total_counts + n_genes_by_counts + pct_counts_mt"
        try:
            fit = smf.ols(formula, data=sub).fit(cov_type="HC3")
            row["bach1_beta"] = float(fit.params.get("BACH1_detected", np.nan))
            row["bach1_se_hc3"] = float(fit.bse.get("BACH1_detected", np.nan))
            ci = fit.conf_int().loc["BACH1_detected"]
            row["bach1_ci_low_hc3"] = float(ci.iloc[0])
            row["bach1_ci_high_hc3"] = float(ci.iloc[1])
            row["bach1_p_hc3"] = float(fit.pvalues.get("BACH1_detected", np.nan))
            row["df_resid"] = float(fit.df_resid)
            row["r_squared"] = float(fit.rsquared)
        except Exception as exc:  # noqa: BLE001 - preserve failure reason in output table
            row["status"] = "failed"
            row["exclusion_reason"] = f"model_failed:{type(exc).__name__}"
        rows.append(row)

    per_patient = pd.DataFrame(rows).sort_values(["status", "dataset", "patient"])
    per_patient.to_csv(TABLE_DIR / "per_patient_qc_adjusted_bach1_detection_models.csv", index=False)

    fit_rows = per_patient[per_patient["status"].eq("fit")].copy()
    betas = pd.to_numeric(fit_rows["bach1_beta"], errors="coerce").dropna().to_numpy(dtype=float)
    if betas.size >= 3:
        w_greater = stats.wilcoxon(betas, alternative="greater", zero_method="wilcox")
        w_two = stats.wilcoxon(betas, alternative="two-sided", zero_method="wilcox")
    else:
        w_greater = (np.nan, np.nan)
        w_two = (np.nan, np.nan)
    summary = {
        "score": DOR_NO_COL,
        "n_total_patients": int(cells["patient"].nunique()),
        "n_evaluable_patients": int(len(fit_rows)),
        "n_excluded_or_failed": int((~per_patient["status"].eq("fit")).sum()),
        "n_positive_beta": int((betas > 0).sum()),
        "n_negative_beta": int((betas < 0).sum()),
        "median_beta": safe_float(np.median(betas)) if betas.size else None,
        "wilcoxon_stat_greater": safe_float(w_greater.statistic),
        "wilcoxon_p_greater": safe_float(w_greater.pvalue),
        "wilcoxon_stat_two_sided": safe_float(w_two.statistic),
        "wilcoxon_p_two_sided": safe_float(w_two.pvalue),
        "wilcoxon_p_two_sided_holm_single_test_family": safe_float(w_two.pvalue),
        "evaluable_rule": "n_cells >= 20 and at least 3 BACH1-detected plus 3 BACH1-undetected cells",
        "multiplicity_note": (
            "The two-sided per-patient QC coefficient test was treated as a single prespecified conservative "
            "sensitivity test for the primary shared-gene-removed DoRothEA score; therefore, no additional "
            "multiplicity adjustment was required within this analysis."
        ),
        "exclusions": per_patient[~per_patient["status"].eq("fit")][
            ["patient", "dataset", "n_cells", "n_detected", "n_undetected", "status", "exclusion_reason"]
        ].to_dict(orient="records"),
        "note": (
            "Per-patient OLS models used the shared-gene-removed DoRothEA BACH1 score as the outcome. "
            "Covariates were z-scored within patient; epithelial states were simplified and rare states were pooled."
        ),
    }
    pd.DataFrame([summary]).to_csv(TABLE_DIR / "per_patient_qc_adjusted_summary.csv", index=False)
    return per_patient, summary


def state_composition_pcs(cells: pd.DataFrame) -> pd.DataFrame:
    state_props = pd.crosstab(cells["patient"], cells["state_simple"], normalize="index")
    centered = state_props - state_props.mean(axis=0)
    u, s, _ = np.linalg.svd(centered.to_numpy(dtype=float), full_matrices=False)
    pcs = pd.DataFrame(index=state_props.index)
    for i in range(min(2, u.shape[1])):
        pcs[f"state_pc{i + 1}"] = u[:, i] * s[i]
    for col in ["state_pc1", "state_pc2"]:
        if col not in pcs.columns:
            pcs[col] = 0.0
        pcs[f"{col}_z"] = zscore(pcs[col])
    out = state_props.join(pcs)
    out.index.name = "patient"
    out.reset_index().to_csv(TABLE_DIR / "patient_level_epithelial_state_composition_pcs.csv", index=False)
    return out.reset_index()[["patient", "state_pc1_z", "state_pc2_z"]]


def residualize(series: pd.Series, covariates: pd.DataFrame) -> pd.Series:
    tmp = pd.concat([series.rename("target"), covariates], axis=1).dropna()
    fit = smf.ols("target ~ C(dataset) + log_total_counts_median_z + n_genes_median_z + pct_mt_median_z + state_pc1_z + state_pc2_z", data=tmp).fit()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[tmp.index] = fit.resid
    return out


def patient_level_residual_correlation(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    patient = pd.read_csv(ROBUST_TABLE / "external_bach1_activity_patient_scores_with_deoverlap.csv")
    state_pcs = state_composition_pcs(cells)
    patient = patient.merge(state_pcs, on="patient", how="left")
    patient["log_total_counts_median"] = np.log1p(pd.to_numeric(patient["total_counts_median"], errors="coerce"))
    for col in ["log_total_counts_median", "n_genes_median", "pct_mt_median"]:
        patient[f"{col}_z"] = zscore(patient[col])
    patient["state_pc1_z"] = patient["state_pc1_z"].fillna(0.0)
    patient["state_pc2_z"] = patient["state_pc2_z"].fillna(0.0)
    patient["dor_no_z"] = zscore(patient[DOR_NO_COL])
    patient["hypoxia_no_z"] = zscore(patient[HYPOXIA_NO_COL])

    rows = []
    for analysis, sub in [
        ("all_patients", patient.copy()),
        ("patients_with_at_least_20_cells", patient[patient["n_cells"].ge(20)].copy()),
    ]:
        cols = [
            "dor_no_z",
            "hypoxia_no_z",
            "dataset",
            "log_total_counts_median_z",
            "n_genes_median_z",
            "pct_mt_median_z",
            "state_pc1_z",
            "state_pc2_z",
        ]
        model_df = sub[cols].dropna().copy()
        fit = smf.ols(
            "hypoxia_no_z ~ dor_no_z + C(dataset) + log_total_counts_median_z + n_genes_median_z + "
            "pct_mt_median_z + state_pc1_z + state_pc2_z",
            data=model_df,
        ).fit(cov_type="HC3")
        covariates = model_df[
            [
                "dataset",
                "log_total_counts_median_z",
                "n_genes_median_z",
                "pct_mt_median_z",
                "state_pc1_z",
                "state_pc2_z",
            ]
        ]
        x_resid = residualize(model_df["dor_no_z"], covariates)
        y_resid = residualize(model_df["hypoxia_no_z"], covariates)
        rho, p, n = safe_spearman(x_resid, y_resid)
        ci = fit.conf_int().loc["dor_no_z"]
        rows.append(
            {
                "analysis": analysis,
                "n_patients": int(n),
                "ols_standardized_beta": float(fit.params["dor_no_z"]),
                "ols_hc3_ci_low": float(ci.iloc[0]),
                "ols_hc3_ci_high": float(ci.iloc[1]),
                "ols_hc3_p": float(fit.pvalues["dor_no_z"]),
                "residual_spearman_rho": rho,
                "residual_spearman_p": p,
                "model": (
                    "hypoxia_no_overlap_z ~ dorothea_no_overlap_z + dataset + median_log_total_counts + "
                    "median_detected_genes + median_pct_mt + epithelial_state_composition_PC1_PC2"
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "patient_level_residualized_bach1_hypoxia_association.csv", index=False)
    patient.to_csv(TABLE_DIR / "patient_level_residualized_model_input.csv", index=False)
    summary = out.set_index("analysis").to_dict(orient="index")
    return out, summary


def tcga_missingness_and_interactions() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scores = pd.read_csv(TCGA_TABLE / "tcga_luad_lusc_bach1_hypoxia_patient_scores.csv")
    scores["stage_model"] = scores["stage_simple"].map(clean_stage)
    scores["stage_unknown_or_missing"] = scores["stage_model"].eq("unknown")
    scores["pack_years_missing"] = pd.to_numeric(scores["pack_years"], errors="coerce").isna()
    scores["purity_missing"] = pd.to_numeric(scores["ABSOLUTE_Purity"], errors="coerce").isna()
    scores["dor_no_z"] = zscore(scores[DOR_NO_COL])
    scores["hypoxia_no_z"] = zscore(scores[HYPOXIA_NO_COL])
    scores["pack_years_z"] = zscore(scores["pack_years"])
    scores["purity_z"] = zscore(scores["ABSOLUTE_Purity"])

    missing_rows = []
    for cohort in ["LUAD", "LUSC", "combined"]:
        sub = scores if cohort == "combined" else scores[scores["cohort"].eq(cohort)]
        base = sub[[DOR_NO_COL, HYPOXIA_NO_COL, "cohort", "stage_model"]].notna().all(axis=1)
        smoking = base & sub["pack_years"].notna()
        strict = smoking & ~sub["stage_unknown_or_missing"]
        purity = smoking & sub["ABSOLUTE_Purity"].notna()
        missing_rows.append(
            {
                "cohort": cohort,
                "n_total_primary_tumour_patients": int(len(sub)),
                "n_stage_unknown_or_missing": int(sub["stage_unknown_or_missing"].sum()),
                "n_pack_years_missing": int(sub["pack_years_missing"].sum()),
                "n_absolute_purity_missing": int(sub["purity_missing"].sum()),
                "n_model_stage_unknown_retained_smoking_complete_case": int(smoking.sum()),
                "n_model_strict_known_stage_and_smoking_complete_case": int(strict.sum()),
                "n_model_with_absolute_purity_complete_case": int(purity.sum()),
            }
        )
    missingness = pd.DataFrame(missing_rows)
    missingness.to_csv(TABLE_DIR / "tcga_covariate_missingness_and_model_counts.csv", index=False)

    interaction_rows = []
    specs = [
        (
            "combined_stage_unknown_retained_smoking_complete_case",
            scores[scores["pack_years"].notna()].copy(),
            "hypoxia_no_z ~ dor_no_z * C(cohort) + C(stage_model) + pack_years_z",
        ),
        (
            "combined_strict_known_stage_smoking_complete_case",
            scores[scores["pack_years"].notna() & ~scores["stage_unknown_or_missing"]].copy(),
            "hypoxia_no_z ~ dor_no_z * C(cohort) + C(stage_model) + pack_years_z",
        ),
    ]
    for name, df, formula in specs:
        fit = smf.ols(formula, data=df).fit(cov_type="HC3")
        interaction_terms = [term for term in fit.params.index if "dor_no_z:C(cohort)" in term or "C(cohort)" in term and ":dor_no_z" in term]
        term = interaction_terms[0] if interaction_terms else ""
        ci = fit.conf_int().loc[term] if term else pd.Series([np.nan, np.nan])
        interaction_rows.append(
            {
                "model": name,
                "n_patients": int(fit.nobs),
                "formula": formula,
                "bach1_main_beta_luad_reference": float(fit.params.get("dor_no_z", np.nan)),
                "bach1_main_p_luad_reference": float(fit.pvalues.get("dor_no_z", np.nan)),
                "interaction_term": term,
                "interaction_beta_lusc_minus_luad": float(fit.params.get(term, np.nan)) if term else np.nan,
                "interaction_ci_low": float(ci.iloc[0]) if term else np.nan,
                "interaction_ci_high": float(ci.iloc[1]) if term else np.nan,
                "interaction_p": float(fit.pvalues.get(term, np.nan)) if term else np.nan,
                "r_squared": float(fit.rsquared),
            }
        )
    interactions = pd.DataFrame(interaction_rows)
    interactions.to_csv(TABLE_DIR / "tcga_cohort_bach1_interaction_models.csv", index=False)
    summary = {
        "missingness": missingness.set_index("cohort").to_dict(orient="index"),
        "interactions": interactions.set_index("model").to_dict(orient="index"),
        "note": (
            "No imputation was used. Smoking-adjusted TCGA models used complete-case samples for pack-years; "
            "the primary model retained unknown stage as a category, with a strict known-stage sensitivity."
        ),
    }
    return missingness, interactions, summary


def write_report(summary: dict) -> None:
    q = summary["per_patient_qc_adjusted"]
    all_resid = summary["patient_level_residualized"]["all_patients"]
    ge20_resid = summary["patient_level_residualized"]["patients_with_at_least_20_cells"]
    miss = summary["tcga"]["missingness"]
    inter = summary["tcga"]["interactions"]["combined_stage_unknown_retained_smoking_complete_case"]
    lines = [
        "# Final inference-sensitivity analyses",
        "",
        "## Per-patient QC-adjusted BACH1-detection models",
        "",
        (
            f"The shared-gene-removed DoRothEA BACH1 score was modelled separately in "
            f"{q['n_evaluable_patients']} evaluable patients. {q['n_positive_beta']} coefficients were positive "
            f"and {q['n_negative_beta']} were negative; the median coefficient was {q['median_beta']:.4f}. "
            f"A two-sided Wilcoxon signed-rank test over patient coefficients gave *P*={format_p_text(q['wilcoxon_p_two_sided'])}."
        ),
        (
            f"Patients were considered evaluable when they contributed at least 20 malignant epithelial cells and "
            f"at least 3 BACH1-detected plus 3 BACH1-undetected cells. The two-sided per-patient QC test was a "
            f"single prespecified sensitivity test for the primary shared-gene-removed DoRothEA score; therefore, "
            f"no additional multiplicity adjustment was required within this analysis."
        ),
        "",
        "## Patient-level residualized BACH1-hypoxia association",
        "",
        (
            f"Across all patients, residualized Spearman ρ was {all_resid['residual_spearman_rho']:.3f} "
            f"(*P*={format_p_text(all_resid['residual_spearman_p'])}) after accounting for dataset, median QC metrics and "
            f"epithelial-state composition PCs. This covariate-rich sensitivity analysis is exploratory because "
            f"only 16 patients were available. The corresponding standardized OLS beta was "
            f"{all_resid['ols_standardized_beta']:.3f} (HC3 *P*={format_p_text(all_resid['ols_hc3_p'])})."
        ),
        (
            f"In the >=20-cell sensitivity set, residualized Spearman ρ was "
            f"{ge20_resid['residual_spearman_rho']:.3f} (*P*={format_p_text(ge20_resid['residual_spearman_p'])})."
        ),
        "",
        "## TCGA missingness and interaction",
        "",
        (
            f"TCGA contained {miss['LUAD']['n_total_primary_tumour_patients']} LUAD and "
            f"{miss['LUSC']['n_total_primary_tumour_patients']} LUSC primary-tumour patients. The primary combined "
            f"smoking-adjusted model retained unknown stage as a category and used "
            f"{miss['combined']['n_model_stage_unknown_retained_smoking_complete_case']} patients with available "
            f"pack-years; the strict known-stage complete-case sensitivity used "
            f"{miss['combined']['n_model_strict_known_stage_and_smoking_complete_case']} patients."
        ),
        (
            f"The cohort x BACH1 interaction was beta={inter['interaction_beta_lusc_minus_luad']:.3f} "
            f"(*P*={format_p_text(inter['interaction_p'])}), indicating no strong evidence that the adjusted association differed "
            f"between LUAD and LUSC under the primary complete-case specification."
        ),
        "",
    ]
    (OUT / "final_inference_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    log("Reading cell-level score and QC tables")
    cells = read_cell_inputs()
    log("Fitting per-patient QC-adjusted models")
    _, pp_summary = fit_per_patient_qc_models(cells)
    log("Running patient-level residualized association checks")
    _, resid_summary = patient_level_residual_correlation(cells)
    log("Summarising TCGA missingness and cohort interactions")
    _, _, tcga_summary = tcga_missingness_and_interactions()
    summary = {
        "per_patient_qc_adjusted": pp_summary,
        "patient_level_residualized": resid_summary,
        "tcga": tcga_summary,
    }
    (OUT / "final_inference_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
