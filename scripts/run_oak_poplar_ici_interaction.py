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

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "source_data" / "manuscript_submission_20260911" / "external_bach1_signature_targets.csv"
OVERLAP = {"ALDOA", "HMOX1", "IL6"}
SIGNATURE = "DOROTHEA_BACH1_ABC_TF_ACTIVITY"


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
    out["treatment_atezo"] = (out["treatment_arm"] == "atezolizumab").astype(int)
    out["score_z"] = (out["bach1_score"] - out["bach1_score"].mean()) / out["bach1_score"].std(ddof=0)
    out["score_treatment_interaction"] = out["score_z"] * out["treatment_atezo"]
    audit = {"trial": trial, "n_expression_samples": int(expr.shape[1]), "n_matched_samples": int(len(out)), "n_targets_requested": len(genes), "n_targets_present": len(present), "missing_targets": sorted(set(genes) - set(present))}
    return out, audit


def design(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = pd.DataFrame({"intercept": 1.0, "score_z": df["score_z"], "treatment_atezo": df["treatment_atezo"], "score_treatment_interaction": df["score_treatment_interaction"]}, index=df.index)
    x = pd.concat([x, pd.get_dummies(df["trial"], prefix="trial", drop_first=True, dtype=float), pd.get_dummies(df["histology"], prefix="histology", drop_first=True, dtype=float)], axis=1)
    return x.astype(float), x.columns.tolist()


def fit_cox(df: pd.DataFrame, time_col: str, event_col: str) -> dict:
    keep = df[[time_col, event_col, "score_z", "treatment_atezo", "score_treatment_interaction", "trial", "histology"]].dropna()
    if len(keep) < 20 or keep[event_col].sum() < 5:
        return {"endpoint": time_col, "n": int(len(keep)), "events": int(keep[event_col].sum()), "status": "insufficient_events"}
    x, names = design(keep)
    fit = PHReg(keep[time_col].astype(float), x.to_numpy(), status=keep[event_col].astype(int), ties="breslow").fit(disp=False)
    return {"endpoint": time_col, "n": int(len(keep)), "events": int(keep[event_col].sum()), "status": "fit", "terms": names, "coef": fit.params.tolist(), "hazard_ratio": np.exp(fit.params).tolist(), "p_value": fit.pvalues.tolist(), "interaction_p": float(fit.pvalues[names.index("score_treatment_interaction")])}


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


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    poplar, poplar_audit = build_study(args.poplar_expression, args.poplar_clinical, "POPLAR", args)
    oak, oak_audit = build_study(args.oak_expression, args.oak_clinical, "OAK", args)
    scores = pd.concat([poplar, oak], ignore_index=True)
    scores.to_csv(args.out / "OAK_POPLAR_patient_scores.csv", index=False)
    results = {"PFS": fit_cox(scores, "pfs_time", "pfs_event"), "OS": fit_cox(scores, "os_time", "os_event"), "response": fit_response(scores)}
    with (args.out / "OAK_POPLAR_interaction_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (args.out / "OAK_POPLAR_data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"score_definition": "81-gene de-overlapped DoRothEA BACH1 weighted mean-z", "poplar": poplar_audit, "oak": oak_audit, "model": "endpoint ~ score + treatment + score:treatment + trial + histology", "interaction_is_primary_estimand": True}, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
