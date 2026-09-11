"""Summarize spacexr/RCTD cell-type weights for spatial BACH1/NOD analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from project_paths import OUTPUT_DIR


INPUTS = OUTPUT_DIR / "emtab13530_spatial_bach1_nod" / "rctd_spacexr_inputs"
RESULTS = OUTPUT_DIR / "emtab13530_spatial_bach1_nod" / "rctd_spacexr_results"
DEFAULT_OUT = OUTPUT_DIR / "emtab13530_spatial_bach1_nod" / "rctd_spacexr_summary"


def holm(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    mask = np.isfinite(values)
    if mask.any():
        out.loc[mask] = multipletests(values.loc[mask], method="holm")[1]
    return out


def paired_test(df: pd.DataFrame, value_col: str, metric_name: str) -> dict:
    wide = df.pivot(index="patient_id", columns="tissue_group", values=value_col).dropna(
        subset=["Adjacent", "Tumor"]
    )
    delta = wide["Tumor"] - wide["Adjacent"]
    try:
        _, p = stats.wilcoxon(wide["Tumor"], wide["Adjacent"], zero_method="wilcox")
    except ValueError:
        p = np.nan
    return {
        "metric": metric_name,
        "n_pairs": int(delta.shape[0]),
        "median_adjacent": float(np.nanmedian(wide["Adjacent"])) if len(delta) else np.nan,
        "median_tumor": float(np.nanmedian(wide["Tumor"])) if len(delta) else np.nan,
        "median_delta_tumor_minus_adjacent": float(np.nanmedian(delta)) if len(delta) else np.nan,
        "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
    }


def read_section(input_dir: Path, result_dir: Path, sample: str) -> pd.DataFrame | None:
    meta_path = input_dir / "spatial" / sample / "spot_metadata.csv"
    weight_path = result_dir / "weights" / f"{sample}_rctd_weights.csv.gz"
    if not meta_path.exists() or not weight_path.exists() or weight_path.stat().st_size == 0:
        return None
    meta = pd.read_csv(meta_path)
    weights = pd.read_csv(weight_path)
    out = meta.merge(weights, on="spot", how="inner")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUTS)
    parser.add_argument("--result-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.input_dir / "spatial_sample_manifest.csv")
    status_path = args.result_dir / "rctd_run_status.csv"
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    completed = set(status.loc[status["status"].isin(["completed", "skipped_existing"]), "sample"]) if not status.empty else set()

    sections = []
    for sample in manifest.loc[manifest["status"].eq("ready"), "sample"].astype(str):
        if completed and sample not in completed:
            continue
        section = read_section(args.input_dir, args.result_dir, sample)
        if section is not None and not section.empty:
            sections.append(section)
    if not sections:
        summary = {
            "status": "skipped",
            "reason": "No completed RCTD weight files were found.",
            "n_sections": 0,
            "n_spots": 0,
            "outputs": [],
        }
        (args.output_dir / "spatial_rctd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    all_spots = pd.concat(sections, ignore_index=True)
    cell_type_cols = [
        col
        for col in all_spots.columns
        if col
        not in {
            "spot",
            "x",
            "y",
            "sample",
            "patient_id",
            "tissue_group",
            "total_counts",
            "n_genes_by_counts",
            "BACH1_log_norm",
            "BACH1_detected",
            "BACH1_high",
            "NOD_like_score_scanpy",
            "NOD_like_high",
            "BACH1_NOD_cohigh",
        }
        and pd.api.types.is_numeric_dtype(all_spots[col])
    ]
    for col in cell_type_cols:
        all_spots[col] = pd.to_numeric(all_spots[col], errors="coerce")
    all_spots["log_total_counts"] = np.log1p(pd.to_numeric(all_spots["total_counts"], errors="coerce"))
    all_spots.to_csv(args.output_dir / "spatial_rctd_spot_weights_with_metadata.csv.gz", index=False, compression="gzip")

    sample_summary = (
        all_spots.groupby(["sample", "patient_id", "tissue_group"], as_index=False)
        .agg(n_spots=("spot", "nunique"), **{f"{ct}_mean": (ct, "mean") for ct in cell_type_cols})
    )
    sample_summary.to_csv(args.output_dir / "spatial_rctd_sample_mean_weights.csv", index=False)

    patient_summary = (
        sample_summary[sample_summary["tissue_group"].isin(["Adjacent", "Tumor"])]
        .groupby(["patient_id", "tissue_group"], as_index=False)
        .agg(n_sections=("sample", "nunique"), n_spots=("n_spots", "sum"), **{f"{ct}_mean": (f"{ct}_mean", "mean") for ct in cell_type_cols})
    )
    patient_summary.to_csv(args.output_dir / "spatial_rctd_patient_mean_weights.csv", index=False)

    paired_rows = [paired_test(patient_summary, f"{ct}_mean", f"{ct}_mean") for ct in cell_type_cols]
    paired = pd.DataFrame(paired_rows)
    paired["holm_p"] = holm(paired["wilcoxon_p"])
    paired.to_csv(args.output_dir / "spatial_rctd_patient_paired_tests.csv", index=False)

    cohigh_rows = []
    for sample, sub in all_spots.groupby("sample"):
        meta = sub[["patient_id", "tissue_group"]].iloc[0].to_dict()
        cohigh = sub["BACH1_NOD_cohigh"].astype(bool)
        for ct in cell_type_cols:
            if cohigh.sum() == 0 or (~cohigh).sum() == 0:
                delta = np.nan
            else:
                delta = float(sub.loc[cohigh, ct].mean() - sub.loc[~cohigh, ct].mean())
            cohigh_rows.append({"sample": sample, **meta, "cell_type": ct, "cohigh_minus_non_mean_delta": delta})
    cohigh_sample = pd.DataFrame(cohigh_rows)
    cohigh_sample.to_csv(args.output_dir / "spatial_rctd_cohigh_sample_deltas.csv", index=False)
    cohigh_patient = (
        cohigh_sample[cohigh_sample["tissue_group"].isin(["Adjacent", "Tumor"])]
        .groupby(["patient_id", "tissue_group", "cell_type"], as_index=False)
        .agg(cohigh_minus_non_mean_delta=("cohigh_minus_non_mean_delta", "mean"))
    )
    cohigh_patient.to_csv(args.output_dir / "spatial_rctd_cohigh_patient_deltas.csv", index=False)
    cohigh_test_rows = []
    for ct, sub in cohigh_patient.groupby("cell_type"):
        cohigh_test_rows.append(paired_test(sub, "cohigh_minus_non_mean_delta", f"{ct}_cohigh_delta"))
    cohigh_tests = pd.DataFrame(cohigh_test_rows)
    cohigh_tests["holm_p"] = holm(cohigh_tests["wilcoxon_p"])
    cohigh_tests.to_csv(args.output_dir / "spatial_rctd_cohigh_patient_paired_tests.csv", index=False)

    model_terms = [ct for ct in ["Malignant_epithelial", "Myeloid", "Fibroblast", "Endothelial", "T_NK"] if ct in cell_type_cols]
    model_df = all_spots[
        all_spots["tissue_group"].isin(["Adjacent", "Tumor"])
        & np.isfinite(all_spots["NOD_like_score_scanpy"])
        & np.isfinite(all_spots["BACH1_log_norm"])
    ].copy()
    formula = None
    model_rows = []
    if model_terms and model_df["patient_id"].nunique() >= 2 and model_df["sample"].nunique() >= 2:
        formula = "NOD_like_score_scanpy ~ BACH1_log_norm + log_total_counts + n_genes_by_counts + " + " + ".join(model_terms) + " + C(patient_id)"
        try:
            fit = smf.ols(formula, data=model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})
            for term in ["BACH1_log_norm", *model_terms, "log_total_counts", "n_genes_by_counts"]:
                if term in fit.params:
                    ci_low, ci_high = fit.conf_int().loc[term]
                    model_rows.append(
                        {
                            "term": term,
                            "coef": float(fit.params[term]),
                            "se_cluster_by_section": float(fit.bse[term]),
                            "ci_low": float(ci_low),
                            "ci_high": float(ci_high),
                            "p_value": float(fit.pvalues[term]),
                            "n_spots": int(fit.nobs),
                            "n_sections": int(model_df["sample"].nunique()),
                            "n_patients": int(model_df["patient_id"].nunique()),
                            "model": "OLS with patient fixed effects and section-clustered SE",
                        }
                    )
        except Exception as err:
            model_rows.append({"term": "model_failed", "coef": np.nan, "p_value": np.nan, "model": str(err)})
    model_table = pd.DataFrame(model_rows)
    model_table.to_csv(args.output_dir / "spatial_rctd_nod_model.csv", index=False)

    summary = {
        "status": "completed",
        "n_sections": int(all_spots["sample"].nunique()),
        "n_spots": int(all_spots.shape[0]),
        "cell_type_columns": cell_type_cols,
        "n_patient_pairs": int(patient_summary["patient_id"].nunique()),
        "paired_tests": paired.to_dict(orient="records"),
        "cohigh_paired_tests": cohigh_tests.to_dict(orient="records"),
        "nod_model_formula": formula,
        "nod_model_terms": model_rows,
        "outputs": sorted(p.name for p in args.output_dir.glob("*")),
    }
    (args.output_dir / "spatial_rctd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
