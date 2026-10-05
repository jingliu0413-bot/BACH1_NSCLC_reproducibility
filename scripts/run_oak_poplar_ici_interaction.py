#!/usr/bin/env python3
"""Run the frozen OAK/POPLAR BACH1-score treatment-interaction analysis.

This script is intentionally gated on locally supplied EGA-derived files. It
does not download controlled data and it never chooses a score using outcomes.
The EGA clinical column names vary by release, so they are explicit CLI
arguments rather than guessed silently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "source_data" / "manuscript_submission_20260911" / "external_bach1_signature_targets.csv"
OVERLAP = {"ALDOA", "HMOX1", "IL6"}
SIGNATURE = "DOROTHEA_BACH1_ABC_TF_ACTIVITY"
TREATMENT_REFERENCE = "chemotherapy"
TREATMENT_EXPOSED = "atezolizumab"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--poplar-expression", type=Path, required=True)
    p.add_argument("--oak-expression", type=Path, required=True)
    p.add_argument("--poplar-clinical", type=Path, required=True)
    p.add_argument("--oak-clinical", type=Path, required=True)
    p.add_argument("--sample-col", required=True)
    p.add_argument("--treatment-col", required=True)
    p.add_argument("--histology-col", required=True)
    p.add_argument("--os-time-col", required=True)
    p.add_argument("--os-event-col", required=True)
    p.add_argument("--pfs-time-col", required=True)
    p.add_argument("--pfs-event-col", required=True)
    p.add_argument("--response-col")
    p.add_argument("--out", type=Path, default=ROOT / "out/clinical_ici_feasibility/OAK_POPLAR")
    return p.parse_args()


def load_targets() -> tuple[list[str], list[float]]:
    table = pd.read_csv(TARGETS)
    sub = table[table["signature"].eq(SIGNATURE) & table["present_in_primary"].eq(True)].copy()
    sub["gene"] = sub["target"].astype(str).str.upper()
    sub = sub[~sub["gene"].isin(OVERLAP)].drop_duplicates("gene")
    return sub["gene"].tolist(), sub["weight"].astype(float).tolist()


def read_expression(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    gene_col = raw.columns[0]
    raw[gene_col] = raw[gene_col].astype(str).str.upper()
    expr = raw.set_index(gene_col)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby(level=0, sort=False).mean()
    return expr


def weighted_mean_z(expr: pd.DataFrame, genes: list[str], weights: list[float]) -> tuple[pd.Series, list[str]]:
    present = [gene for gene in genes if gene in expr.index]
    if not present:
        raise ValueError("No frozen BACH1 targets were found in the expression matrix")
    z = expr.loc[present].T
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=0).replace(0, np.nan)
    weight_map = dict(zip(genes, weights))
    w = np.array([weight_map[gene] for gene in present], dtype=float)
    score = pd.Series(z.fillna(0).to_numpy().dot(w) / np.abs(w).sum(), index=expr.columns, name="bach1_score")
    return score, present


def treatment_arm(value: object) -> str:
    s = str(value).strip().lower()
    if "atezo" in s or "mpdl" in s:
        return "atezolizumab"
    if "docetaxel" in s or "chemotherapy" in s or s in {"chemo", "control"}:
        return "chemotherapy"
    raise ValueError(f"Unrecognized treatment-arm value: {value!r}")


def event_flag(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value > 0)
    s = str(value).strip().lower()
    if s in {"1", "yes", "true", "event", "dead", "death", "progressed", "progression"}:
        return 1.0
    if s in {"0", "no", "false", "censored", "alive", "none"}:
        return 0.0
    return np.nan


def build_study(expression_path: Path, clinical_path: Path, trial: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    expr = read_expression(expression_path)
    genes, weights = load_targets()
    score, present = weighted_mean_z(expr, genes, weights)
    clinical = pd.read_csv(clinical_path)
    clinical["sample"] = clinical[args.sample_col].astype(str)
    clinical["treatment_arm"] = clinical[args.treatment_col].map(treatment_arm)
    clinical["histology"] = clinical[args.histology_col].astype(str)
    clinical["os_time"] = pd.to_numeric(clinical[args.os_time_col], errors="coerce")
    clinical["os_event"] = clinical[args.os_event_col].map(event_flag)
    clinical["pfs_time"] = pd.to_numeric(clinical[args.pfs_time_col], errors="coerce")
    clinical["pfs_event"] = clinical[args.pfs_event_col].map(event_flag)
    if args.response_col:
        clinical["response"] = clinical[args.response_col].astype(str)
    out = clinical.merge(score.rename_axis("sample").reset_index(), on="sample", how="inner")
    out["trial"] = trial
    # Freeze the interaction direction: chemotherapy=0 is the reference and
    # atezolizumab=1 is the exposed treatment arm.
    out["treatment_atezo"] = (out["treatment_arm"] == TREATMENT_EXPOSED).astype(int)
    out["score_z"] = (out["bach1_score"] - out["bach1_score"].mean()) / out["bach1_score"].std(ddof=0)
    out["score_treatment_interaction"] = out["score_z"] * out["treatment_atezo"]
    audit = {
        "trial": trial,
        "n_expression_samples": int(expr.shape[1]),
        "n_matched_samples": int(len(out)),
        "n_targets_requested": len(genes),
        "n_targets_present": len(present),
        "missing_targets": sorted(set(genes) - set(present)),
        "treatment_reference": TREATMENT_REFERENCE,
        "treatment_exposed": TREATMENT_EXPOSED,
        "score_standardization": "within_trial_1SD",
    }
    return out, audit


def design(
    df: pd.DataFrame,
    time_col: str | None = None,
    include_trial_covariate: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    x = pd.DataFrame({"intercept": 1.0, "score_z": df["score_z"], "treatment_atezo": df["treatment_atezo"], "score_treatment_interaction": df["score_treatment_interaction"]}, index=df.index)
    if time_col is not None:
        log_time = np.log(np.maximum(pd.to_numeric(df[time_col], errors="coerce"), np.finfo(float).tiny))
        x["score_treatment_log_time"] = df["score_treatment_interaction"] * log_time
    covariates = [x]
    if include_trial_covariate:
        covariates.append(pd.get_dummies(df["trial"], prefix="trial", drop_first=True, dtype=float))
    covariates.append(pd.get_dummies(df["histology"], prefix="histology", drop_first=True, dtype=float))
    x = pd.concat(covariates, axis=1)
    return x.astype(float), x.columns.tolist()


def ph_diagnostics(fit: object, keep: pd.DataFrame, names: list[str], time_col: str, event_col: str) -> dict:
    """Approximate PH diagnostics from event-time Schoenfeld residuals."""
    try:
        residuals = np.asarray(fit.schoenfeld_residuals, dtype=float)
        if residuals.ndim != 2 or residuals.shape[1] != len(names):
            raise ValueError("unexpected Schoenfeld residual shape")
        event = keep[event_col].astype(bool).to_numpy()
        log_time = np.log(np.maximum(keep[time_col].astype(float).to_numpy(), np.finfo(float).tiny))
        terms = []
        for index, name in enumerate(names):
            values = residuals[:, index]
            mask = event & np.isfinite(values) & np.isfinite(log_time)
            if int(mask.sum()) < 4 or np.unique(log_time[mask]).size < 3 or np.unique(values[mask]).size < 3:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(log_time[mask], values[mask])
            terms.append({"term": name, "n_events_used": int(mask.sum()), "rho": None if not np.isfinite(rho) else float(rho), "p_value": None if not np.isfinite(p_value) else float(p_value)})
        interaction = next((item for item in terms if item["term"] == "score_treatment_interaction"), None)
        return {"method": "Spearman correlation of log event time with Schoenfeld residuals", "terms": terms, "interaction_p": None if interaction is None else interaction["p_value"]}
    except Exception as exc:  # pragma: no cover - depends on statsmodels internals
        return {"method": "Spearman correlation of log event time with Schoenfeld residuals", "status": "unavailable", "reason": str(exc)}


def fit_time_varying_interaction(keep: pd.DataFrame, time_col: str, event_col: str) -> dict:
    """Sensitivity model adding score:treatment x log(time)."""
    x, names = design(keep, time_col=time_col)
    fit = PHReg(keep[time_col].astype(float), x.to_numpy(), status=keep[event_col].astype(int), ties="breslow").fit(disp=False)
    term = "score_treatment_log_time"
    index = names.index(term)
    return {"status": "fit", "terms": names, "coef": fit.params.tolist(), "hazard_ratio": np.exp(fit.params).tolist(), "p_value": fit.pvalues.tolist(), "time_varying_interaction_p": float(fit.pvalues[index])}


def interaction_summary(fit: object, names: list[str]) -> dict:
    index = names.index("score_treatment_interaction")
    confidence = np.asarray(fit.conf_int(), dtype=float)
    return {
        "interaction_hr": float(np.exp(fit.params[index])),
        "interaction_ci_lower": float(np.exp(confidence[index, 0])),
        "interaction_ci_upper": float(np.exp(confidence[index, 1])),
        "interaction_p": float(fit.pvalues[index]),
        "interpretation": "HR<1 indicates relatively greater atezolizumab benefit per 1-SD higher score; HR>1 indicates relatively less benefit",
    }


def fit_cox(df: pd.DataFrame, time_col: str, event_col: str) -> dict:
    keep = df[[time_col, event_col, "score_z", "treatment_atezo", "score_treatment_interaction", "trial", "histology"]].dropna()
    if len(keep) < 20 or keep[event_col].sum() < 5:
        return {"endpoint": time_col, "n": int(len(keep)), "events": int(keep[event_col].sum()), "status": "insufficient_events", "complete_case_rule": True}
    x, names = design(keep)
    fit = PHReg(keep[time_col].astype(float), x.to_numpy(), status=keep[event_col].astype(int), ties="breslow").fit(disp=False)
    ph = ph_diagnostics(fit, keep, names, time_col, event_col)
    result = {"endpoint": time_col, "n": int(len(keep)), "events": int(keep[event_col].sum()), "status": "fit", "complete_case_rule": True, "no_outcome_informed_imputation": True, "terms": names, "coef": fit.params.tolist(), "hazard_ratio": np.exp(fit.params).tolist(), "p_value": fit.pvalues.tolist(), "ph_check": ph, "time_varying_sensitivity": {"status": "not_triggered"}}
    result.update(interaction_summary(fit, names))
    if ph.get("interaction_p") is not None and ph["interaction_p"] < 0.05:
        try:
            result["time_varying_sensitivity"] = fit_time_varying_interaction(keep, time_col, event_col)
        except Exception as exc:  # pragma: no cover - numerical failure is data-dependent
            result["time_varying_sensitivity"] = {"status": "failed", "reason": str(exc)}
    return result


def fit_trial_stratified_cox(df: pd.DataFrame, time_col: str, event_col: str) -> dict:
    columns = [time_col, event_col, "score_z", "treatment_atezo", "score_treatment_interaction", "trial", "histology"]
    keep = df[columns].dropna()
    if len(keep) < 20 or keep[event_col].sum() < 5 or keep["trial"].nunique() < 2:
        return {"endpoint": time_col, "n": int(len(keep)), "events": int(keep[event_col].sum()), "status": "insufficient_events_or_strata", "complete_case_rule": True}
    x, names = design(keep, include_trial_covariate=False)
    fit = PHReg(keep[time_col].astype(float), x.to_numpy(), status=keep[event_col].astype(int), strata=keep["trial"].astype(str).to_numpy(), ties="breslow").fit(disp=False)
    result = {"endpoint": time_col, "n": int(len(keep)), "events": int(keep[event_col].sum()), "status": "fit", "model": "trial-stratified baseline hazard", "complete_case_rule": True, "no_outcome_informed_imputation": True, "terms": names, "coef": fit.params.tolist(), "hazard_ratio": np.exp(fit.params).tolist(), "p_value": fit.pvalues.tolist(), "ph_check": ph_diagnostics(fit, keep, names, time_col, event_col)}
    result.update(interaction_summary(fit, names))
    return result


def fit_response(df: pd.DataFrame) -> dict:
    if "response" not in df.columns:
        return {"status": "not_requested"}
    keep = df[df["response"].isin(["MPR", "pCR", "non-MPR"])].copy()
    keep["response_binary"] = keep["response"].isin(["MPR", "pCR"]).astype(int)
    if len(keep) < 20 or keep["response_binary"].nunique() < 2:
        return {"status": "insufficient_response_data", "n": int(len(keep))}
    x, names = design(keep)
    fit = sm.Logit(keep["response_binary"], x).fit(disp=False)
    return {"status": "fit", "n": int(len(keep)), "terms": names, "coef": fit.params.tolist(), "odds_ratio": np.exp(fit.params).tolist(), "p_value": fit.pvalues.tolist(), "interaction_p": float(fit.pvalues["score_treatment_interaction"])}


def write_interaction_forest(results: dict, output_dir: Path) -> None:
    records = []

    def append_result(endpoint: str, label: str, model: str, result: dict) -> None:
        if result.get("status") != "fit":
            return
        records.append({
            "endpoint": endpoint,
            "label": label,
            "model": model,
            "interaction_hr": result["interaction_hr"],
            "ci_lower": result["interaction_ci_lower"],
            "ci_upper": result["interaction_ci_upper"],
            "p_value": result["interaction_p"],
            "n": result["n"],
            "events": result["events"],
        })

    for endpoint in ("PFS", "OS"):
        append_result(endpoint, "Pooled primary", "trial covariate", results[endpoint])
        append_result(endpoint, "Pooled sensitivity", "trial-stratified baseline hazard", results["trial_stratified_sensitivity"][endpoint])
        for trial in ("POPLAR", "OAK"):
            append_result(endpoint, trial, "trial-specific", results["trial_specific_interactions"][endpoint][trial])

    table = pd.DataFrame.from_records(records)
    table.to_csv(output_dir / "OAK_POPLAR_trial_specific_interaction_estimates.csv", index=False)
    if table.empty:
        return

    label_order = ["Pooled primary", "Pooled sensitivity", "POPLAR", "OAK"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
    for axis, endpoint in zip(axes, ("PFS", "OS")):
        subset = table[table["endpoint"].eq(endpoint)].set_index("label").reindex(label_order).dropna(subset=["interaction_hr"])
        y = np.arange(len(subset))
        lower = subset["interaction_hr"].to_numpy() - subset["ci_lower"].to_numpy()
        upper = subset["ci_upper"].to_numpy() - subset["interaction_hr"].to_numpy()
        axis.errorbar(subset["interaction_hr"], y, xerr=np.vstack([lower, upper]), fmt="o", color="#B43C39", ecolor="#4A4A4A", capsize=3)
        axis.axvline(1.0, color="#777777", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_yticks(y)
        axis.set_yticklabels(subset.index)
        axis.invert_yaxis()
        axis.set_title(endpoint)
        axis.set_xlabel("Score x treatment HR (95% CI)")
        axis.grid(axis="x", color="#E1E1E1", linewidth=0.7)
    figure.suptitle("OAK/POPLAR trial-specific treatment-interaction sensitivity")
    figure.text(0.5, 0.005, "HR < 1 indicates relatively greater atezolizumab benefit per 1-SD higher BACH1 score", ha="center", fontsize=8)
    figure.savefig(output_dir / "OAK_POPLAR_trial_specific_interaction_forest.pdf", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "OAK_POPLAR_trial_specific_interaction_forest.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    poplar, poplar_audit = build_study(args.poplar_expression, args.poplar_clinical, "POPLAR", args)
    oak, oak_audit = build_study(args.oak_expression, args.oak_clinical, "OAK", args)
    scores = pd.concat([poplar, oak], ignore_index=True)
    scores.to_csv(args.out / "OAK_POPLAR_patient_scores.csv", index=False)
    results = {
        "PFS": fit_cox(scores, "pfs_time", "pfs_event"),
        "OS": fit_cox(scores, "os_time", "os_event"),
        "response": fit_response(scores),
        "trial_stratified_sensitivity": {
            "PFS": fit_trial_stratified_cox(scores, "pfs_time", "pfs_event"),
            "OS": fit_trial_stratified_cox(scores, "os_time", "os_event"),
        },
        "trial_specific_interactions": {
            "PFS": {
                trial: fit_cox(scores[scores["trial"].eq(trial)], "pfs_time", "pfs_event")
                for trial in ("POPLAR", "OAK")
            },
            "OS": {
                trial: fit_cox(scores[scores["trial"].eq(trial)], "os_time", "os_event")
                for trial in ("POPLAR", "OAK")
            },
        },
    }
    with (args.out / "OAK_POPLAR_interaction_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    write_interaction_forest(results, args.out)
    with (args.out / "OAK_POPLAR_data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"score_definition": "81-gene de-overlapped DoRothEA BACH1 weighted mean-z", "poplar": poplar_audit, "oak": oak_audit, "treatment_reference": TREATMENT_REFERENCE, "treatment_exposed": TREATMENT_EXPOSED, "interaction_interpretation": "HR<1 indicates relatively greater atezolizumab benefit per 1-SD higher score; HR>1 indicates relatively less benefit", "score_standardization": "within_trial_1SD", "primary_model": "endpoint ~ score + treatment + score:treatment + trial + histology", "trial_stratified_sensitivity": "endpoint ~ score + treatment + score:treatment + histology, with baseline hazard stratified by trial", "interaction_is_primary_estimand": True, "cox_complete_case_rule": True, "outcome_informed_imputation": False, "ph_check": "Schoenfeld residual diagnostics; time-varying interaction only as sensitivity if interaction PH is violated"}, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
