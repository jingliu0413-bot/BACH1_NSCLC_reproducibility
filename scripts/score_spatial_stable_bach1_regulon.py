"""Score a stable BACH1 regulon signature in E-MTAB-13530 spatial sections."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests

from project_paths import OUTPUT_DIR, PROJECT_ROOT


SPATIAL = OUTPUT_DIR / "emtab13530_spatial_bach1_nod"
H5AD = SPATIAL / "h5ad_by_sample"
DEFAULT_TARGETS = (
    OUTPUT_DIR
    / "revision_diagnostics"
    / "pyscenic_bach1_consensus_seed_stability_v2"
    / "pyscenic_bach1_stable_ge60pct_targets.csv"
)
DEFAULT_OUT = SPATIAL / "stable_bach1_regulon_spatial"


def read_h5ad_compat(path: Path) -> ad.AnnData:
    try:
        return ad.read_h5ad(path)
    except Exception as err:
        if "encoding_type='null'" not in str(err) and "uns/log1p" not in str(err):
            raise
    with tempfile.TemporaryDirectory(prefix="h5ad_compat_") as tmpdir:
        tmp_path = Path(tmpdir) / path.name
        shutil.copy2(path, tmp_path)
        with h5py.File(tmp_path, "a") as handle:
            if "uns/log1p/base" in handle:
                del handle["uns/log1p/base"]
        return ad.read_h5ad(tmp_path)


def high_mask(values: np.ndarray, q: float = 0.75) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    cutoff = np.nanquantile(values, q)
    return values >= cutoff


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 5 or np.nanstd(x[good]) == 0 or np.nanstd(y[good]) == 0:
        return np.nan, np.nan
    res = stats.spearmanr(x[good], y[good])
    return float(res.statistic), float(res.pvalue)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else np.nan


def paired_tests(patient_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "stable_BACH1_regulon_score_mean",
        "stable_BACH1_regulon_high_fraction",
        "stable_BACH1_regulon_NOD_spearman_rho",
        "stable_BACH1_regulon_NOD_high_jaccard",
    ]
    rows = []
    for metric in metrics:
        wide = patient_summary.pivot(index="patient_id", columns="tissue_group", values=metric).dropna(
            subset=["Adjacent", "Tumor"]
        )
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
    out = pd.DataFrame(rows)
    out["holm_p"] = np.nan
    mask = np.isfinite(out["wilcoxon_p"])
    if mask.any():
        out.loc[mask, "holm_p"] = multipletests(out.loc[mask, "wilcoxon_p"], method="holm")[1]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--score-name", default="stable_BACH1_regulon_score")
    parser.add_argument("--random-state", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_dir = args.output_dir / "tables"
    h5ad_dir = args.output_dir / "h5ad_by_sample"
    table_dir.mkdir(parents=True, exist_ok=True)
    h5ad_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(args.target_table)
    genes = [g for g in targets["gene"].dropna().astype(str).drop_duplicates().tolist() if g != "BACH1"]
    if len(genes) < 3:
        summary = {
            "target_table": str(args.target_table),
            "n_stable_targets_input_excluding_BACH1": int(len(genes)),
            "n_sections": 0,
            "n_spots": 0,
            "n_tumor_adjacent_patients": 0,
            "paired_tests": [],
            "status": "skipped",
            "reason": "Need at least 3 non-BACH1 stable regulon target genes for spatial signature scoring.",
            "outputs": [],
        }
        (args.output_dir / "stable_bach1_regulon_spatial_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    spot_tables = []
    sample_rows = []
    for path in sorted(H5AD.glob("*_bach1_nod_scored.h5ad")):
        sample = path.name.replace("_bach1_nod_scored.h5ad", "")
        adata = read_h5ad_compat(path)
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 3:
            score = np.full(adata.n_obs, np.nan)
        else:
            sc.tl.score_genes(
                adata,
                gene_list=present,
                score_name=args.score_name,
                ctrl_size=min(50, len(present)),
                random_state=args.random_state,
                use_raw=False,
            )
            score = adata.obs[args.score_name].to_numpy(float)
        high = high_mask(score)
        nod_high = adata.obs["NOD_like_high"].astype(bool).to_numpy()
        nod_score = adata.obs["NOD_like_score_scanpy"].to_numpy(float)
        rho, pvalue = safe_spearman(score, nod_score)
        adata.obs[f"{args.score_name}_high"] = high
        adata.obs[f"{args.score_name}_NOD_high_overlap"] = high & nod_high
        adata.write_h5ad(h5ad_dir / f"{sample}_stable_bach1_regulon_scored.h5ad", compression="gzip")

        obs = adata.obs[
            [
                "sample",
                "patient_id",
                "tissue_group",
                "total_counts",
                "n_genes_by_counts",
                "pct_counts_mt",
                "NOD_like_score_scanpy",
                "NOD_like_high",
            ]
        ].copy()
        obs["barcode"] = obs.index
        obs[args.score_name] = score
        obs[f"{args.score_name}_high"] = high
        obs[f"{args.score_name}_NOD_high_overlap"] = high & nod_high
        spot_tables.append(obs)
        sample_rows.append(
            {
                "sample": sample,
                "patient_id": adata.obs["patient_id"].iloc[0],
                "tissue_group": adata.obs["tissue_group"].iloc[0],
                "n_spots": int(adata.n_obs),
                "n_stable_targets_input": int(len(genes)),
                "n_stable_targets_present": int(len(present)),
                "stable_BACH1_regulon_score_mean": float(np.nanmean(score)),
                "stable_BACH1_regulon_score_median": float(np.nanmedian(score)),
                "stable_BACH1_regulon_high_fraction": float(np.nanmean(high)),
                "stable_BACH1_regulon_NOD_spearman_rho": rho,
                "stable_BACH1_regulon_NOD_spearman_p": pvalue,
                "stable_BACH1_regulon_NOD_high_jaccard": jaccard(high, nod_high),
                "stable_BACH1_regulon_NOD_overlap_fraction": float(np.nanmean(high & nod_high)),
            }
        )

    sample_metrics = pd.DataFrame(sample_rows)
    sample_metrics.to_csv(table_dir / "stable_bach1_regulon_spatial_sample_metrics.csv", index=False)
    all_spots = pd.concat(spot_tables, ignore_index=True)
    all_spots.to_csv(table_dir / "stable_bach1_regulon_spatial_all_spot_scores.csv.gz", index=False, compression="gzip")

    patient_summary = (
        sample_metrics[sample_metrics["tissue_group"].isin(["Adjacent", "Tumor"])]
        .groupby(["patient_id", "tissue_group"], as_index=False)
        .agg(
            n_sections=("sample", "nunique"),
            n_spots=("n_spots", "sum"),
            stable_BACH1_regulon_score_mean=("stable_BACH1_regulon_score_mean", "mean"),
            stable_BACH1_regulon_high_fraction=("stable_BACH1_regulon_high_fraction", "mean"),
            stable_BACH1_regulon_NOD_spearman_rho=("stable_BACH1_regulon_NOD_spearman_rho", "mean"),
            stable_BACH1_regulon_NOD_high_jaccard=("stable_BACH1_regulon_NOD_high_jaccard", "mean"),
            stable_BACH1_regulon_NOD_overlap_fraction=("stable_BACH1_regulon_NOD_overlap_fraction", "mean"),
        )
    )
    patient_summary.to_csv(table_dir / "stable_bach1_regulon_spatial_patient_means.csv", index=False)
    tests = paired_tests(patient_summary)
    tests.to_csv(table_dir / "stable_bach1_regulon_spatial_patient_paired_tests.csv", index=False)

    summary = {
        "target_table": str(args.target_table),
        "n_stable_targets_input_excluding_BACH1": int(len(genes)),
        "n_sections": int(sample_metrics["sample"].nunique()),
        "n_spots": int(all_spots.shape[0]),
        "n_tumor_adjacent_patients": int(patient_summary["patient_id"].nunique()),
        "paired_tests": tests.to_dict(orient="records"),
        "outputs": sorted(p.name for p in table_dir.glob("*")),
    }
    (args.output_dir / "stable_bach1_regulon_spatial_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
