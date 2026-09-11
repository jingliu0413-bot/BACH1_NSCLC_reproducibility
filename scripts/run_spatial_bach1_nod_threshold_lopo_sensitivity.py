"""Threshold and leave-one-patient-out sensitivity for spatial BACH1/NOD overlap.

The input is the already scored E-MTAB-13530 spot table. The output is intended
to keep the spatial result exploratory: patient-level pairing is the inferential
unit, threshold changes are made explicit, and no spot-level P values are used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out/emtab13530_spatial_bach1_nod"
INPUT = BASE / "tables/E-MTAB-13530_bach1_nod_all_spot_scores.csv.gz"
OUT = BASE / "revision_statistics/spatial_threshold_lopo_sensitivity"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"

BACH1_QUANTILES = [0.25, 0.50, 0.75]
NOD_QUANTILES = [0.50, 0.75, 0.90]
DEFAULT_SETTING = "bach1_detected_q50__nod_q75"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


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


def tissue_key(value: str) -> str:
    text = str(value).lower()
    if "tumor" in text or "tumour" in text:
        return "tumor"
    if "adjacent" in text:
        return "adjacent"
    if "healthy" in text or "normal" in text:
        return "healthy"
    return text


def high_status_for_sample(sub: pd.DataFrame, bach1_q: float, nod_q: float) -> tuple[pd.Series, pd.Series]:
    bach1 = pd.to_numeric(sub["BACH1_log_norm"], errors="coerce")
    nod = pd.to_numeric(sub["NOD_like_score_scanpy"], errors="coerce")
    detected = sub["BACH1_detected"].astype(bool)
    detected_values = bach1[detected & bach1.notna()]
    if detected_values.empty:
        bach1_high = pd.Series(False, index=sub.index)
    else:
        threshold = float(detected_values.quantile(bach1_q))
        bach1_high = detected & bach1.ge(threshold)
    nod_threshold = float(nod.quantile(nod_q)) if nod.notna().any() else np.nan
    nod_high = nod.ge(nod_threshold) if np.isfinite(nod_threshold) else pd.Series(False, index=sub.index)
    return bach1_high.fillna(False), nod_high.fillna(False)


def sample_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bach1_q in BACH1_QUANTILES:
        for nod_q in NOD_QUANTILES:
            setting = f"bach1_detected_q{int(bach1_q * 100):02d}__nod_q{int(nod_q * 100):02d}"
            for sample, sub in df.groupby("sample", observed=True):
                bach1_high, nod_high = high_status_for_sample(sub, bach1_q, nod_q)
                cohigh = bach1_high & nod_high
                union = bach1_high | nod_high
                b_frac = float(bach1_high.mean()) if len(bach1_high) else np.nan
                n_frac = float(nod_high.mean()) if len(nod_high) else np.nan
                co_frac = float(cohigh.mean()) if len(cohigh) else np.nan
                expected = b_frac * n_frac if np.isfinite(b_frac) and np.isfinite(n_frac) else np.nan
                rows.append(
                    {
                        "setting": setting,
                        "bach1_detected_quantile": bach1_q,
                        "nod_score_quantile": nod_q,
                        "sample": sample,
                        "patient_id": str(sub["patient_id"].iloc[0]),
                        "tissue_group": tissue_key(sub["tissue_group"].iloc[0]),
                        "n_spots": int(sub.shape[0]),
                        "bach1_detected_fraction": float(sub["BACH1_detected"].astype(bool).mean()),
                        "bach1_high_fraction": b_frac,
                        "nod_high_fraction": n_frac,
                        "cohigh_fraction": co_frac,
                        "expected_cohigh_fraction": expected,
                        "observed_expected_ratio": float(co_frac / expected) if expected and expected > 0 else np.nan,
                        "high_high_jaccard": float(cohigh.sum() / union.sum()) if union.sum() else 0.0,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "spatial_threshold_lopo_sample_metrics.csv", index=False)
    return out


def patient_means(sample_df: pd.DataFrame) -> pd.DataFrame:
    keep = sample_df[sample_df["tissue_group"].isin(["tumor", "adjacent"])].copy()
    metrics = [
        "bach1_detected_fraction",
        "bach1_high_fraction",
        "nod_high_fraction",
        "cohigh_fraction",
        "observed_expected_ratio",
        "high_high_jaccard",
    ]
    out = (
        keep.groupby(["setting", "bach1_detected_quantile", "nod_score_quantile", "patient_id", "tissue_group"], observed=True)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    out.to_csv(TABLE_DIR / "spatial_threshold_lopo_patient_means.csv", index=False)
    return out


def paired_tests(patient_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "bach1_high_fraction",
        "nod_high_fraction",
        "cohigh_fraction",
        "observed_expected_ratio",
        "high_high_jaccard",
    ]
    delta_rows = []
    test_rows = []
    for setting, sub in patient_df.groupby("setting", observed=True):
        wide = sub.pivot(index="patient_id", columns="tissue_group", values=metrics)
        patients = sorted(set(wide.index[wide.xs("tumor", level=1, axis=1).notna().any(axis=1)]) & set(wide.index[wide.xs("adjacent", level=1, axis=1).notna().any(axis=1)]))
        for metric in metrics:
            values = []
            for patient in patients:
                tumor = wide.loc[patient, (metric, "tumor")]
                adjacent = wide.loc[patient, (metric, "adjacent")]
                delta = float(tumor - adjacent) if pd.notna(tumor) and pd.notna(adjacent) else np.nan
                values.append(delta)
                delta_rows.append(
                    {
                        "setting": setting,
                        "patient_id": patient,
                        "metric": metric,
                        "tumor_minus_adjacent": delta,
                    }
                )
            stat, pvalue, n = safe_wilcoxon(values)
            test_rows.append(
                {
                    "setting": setting,
                    "metric": metric,
                    "n_pairs": n,
                    "median_delta_tumor_minus_adjacent": float(pd.Series(values).median(skipna=True)),
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": pvalue,
                }
            )
    deltas = pd.DataFrame(delta_rows)
    tests = pd.DataFrame(test_rows)
    tests["holm_p"] = holm_adjust(tests["wilcoxon_p"])
    tests = tests.sort_values(["metric", "holm_p", "wilcoxon_p", "setting"])
    deltas.to_csv(TABLE_DIR / "spatial_threshold_lopo_patient_deltas.csv", index=False)
    tests.to_csv(TABLE_DIR / "spatial_threshold_lopo_paired_tests.csv", index=False)

    lopo_rows = []
    for (setting, metric), sub in deltas.groupby(["setting", "metric"], observed=True):
        patients = sorted(sub["patient_id"].dropna().astype(str).unique())
        for held in patients:
            keep = sub[sub["patient_id"].astype(str).ne(held)]
            stat, pvalue, n = safe_wilcoxon(keep["tumor_minus_adjacent"])
            med = pd.to_numeric(keep["tumor_minus_adjacent"], errors="coerce").median()
            lopo_rows.append(
                {
                    "setting": setting,
                    "metric": metric,
                    "held_out_patient": held,
                    "n_pairs": n,
                    "median_delta": float(med) if pd.notna(med) else np.nan,
                    "wilcoxon_p": pvalue,
                    "sign": "positive" if pd.notna(med) and med > 0 else ("negative" if pd.notna(med) and med < 0 else "zero_or_nan"),
                }
            )
    lopo = pd.DataFrame(lopo_rows)
    lopo.to_csv(TABLE_DIR / "spatial_threshold_lopo_leave_one_patient_out.csv", index=False)
    lopo_summary = (
        lopo.groupby(["setting", "metric"], observed=True)
        .agg(
            n_lopo=("held_out_patient", "nunique"),
            min_median_delta=("median_delta", "min"),
            max_median_delta=("median_delta", "max"),
            max_wilcoxon_p=("wilcoxon_p", "max"),
            n_positive=("sign", lambda x: int((x == "positive").sum())),
            n_negative=("sign", lambda x: int((x == "negative").sum())),
        )
        .reset_index()
    )
    lopo_summary["sign_stability"] = np.where(
        lopo_summary["n_positive"].eq(lopo_summary["n_lopo"]),
        "all_positive",
        np.where(lopo_summary["n_negative"].eq(lopo_summary["n_lopo"]), "all_negative", "mixed_or_zero"),
    )
    lopo_summary.to_csv(TABLE_DIR / "spatial_threshold_lopo_summary.csv", index=False)
    return deltas, tests, lopo_summary


def make_figure(deltas: pd.DataFrame, tests: pd.DataFrame, lopo_summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]})

    metric = "high_high_jaccard"
    default_delta = deltas[(deltas["setting"].eq(DEFAULT_SETTING)) & (deltas["metric"].eq(metric))].copy()
    default_delta = default_delta.sort_values("tumor_minus_adjacent")
    axes[0].axvline(0, color="black", lw=0.7)
    axes[0].scatter(default_delta["tumor_minus_adjacent"], np.arange(default_delta.shape[0]), color="#4C78A8", s=28)
    axes[0].axvline(default_delta["tumor_minus_adjacent"].median(), color="#E45756", lw=1.2)
    axes[0].set_yticks(np.arange(default_delta.shape[0]))
    axes[0].set_yticklabels(default_delta["patient_id"], fontsize=7)
    axes[0].set_xlabel("tumour - adjacent Jaccard")
    add_panel_label(axes[0], "a")
    axes[0].set_title("Default threshold", loc="left", fontsize=8, pad=2)

    grid = tests[tests["metric"].eq(metric)].copy()
    grid["bach1_q"] = grid["setting"].str.extract(r"bach1_detected_q(\d+)")[0].astype(float) / 100
    grid["nod_q"] = grid["setting"].str.extract(r"nod_q(\d+)")[0].astype(float) / 100
    pivot = grid.pivot(index="bach1_q", columns="nod_q", values="median_delta_tumor_minus_adjacent")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="vlag", center=0, cbar_kws={"label": ""}, ax=axes[1])
    axes[1].set_xlabel("NOD score quantile")
    axes[1].set_ylabel("BACH1-detected quantile")
    add_panel_label(axes[1], "b")
    axes[1].set_title("Threshold sensitivity", loc="left", fontsize=8, pad=2)

    lopo = lopo_summary[lopo_summary["metric"].eq(metric)].copy()
    lopo = lopo.merge(grid[["setting", "median_delta_tumor_minus_adjacent", "wilcoxon_p", "holm_p"]], on="setting", how="left")
    lopo = lopo.sort_values("median_delta_tumor_minus_adjacent")
    y = np.arange(lopo.shape[0])
    axes[2].errorbar(
        lopo["median_delta_tumor_minus_adjacent"],
        y,
        xerr=[
            lopo["median_delta_tumor_minus_adjacent"] - lopo["min_median_delta"],
            lopo["max_median_delta"] - lopo["median_delta_tumor_minus_adjacent"],
        ],
        fmt="o",
        color="#333333",
        ecolor="#999999",
        lw=1,
    )
    axes[2].axvline(0, color="black", lw=0.7)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(
        lopo["setting"].str.replace("bach1_detected_q", "", regex=False).str.replace("__nod_q", "/", regex=False),
        fontsize=6,
    )
    axes[2].set_xlabel("median delta with LOPO range")
    axes[2].set_ylabel("")
    axes[2].yaxis.label.set_text("")
    axes[2].yaxis.label.set_visible(False)
    add_panel_label(axes[2], "c")
    axes[2].set_title("LOPO", loc="left", fontsize=8, pad=2)
    fig.subplots_adjust(top=0.88)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"spatial_bach1_nod_threshold_lopo_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(INPUT)
    sample_df = sample_metrics(df)
    patient_df = patient_means(sample_df)
    deltas, tests, lopo_summary = paired_tests(patient_df)
    make_figure(deltas, tests, lopo_summary)

    default = tests[(tests["setting"].eq(DEFAULT_SETTING)) & (tests["metric"].eq("high_high_jaccard"))]
    best = tests[tests["metric"].eq("high_high_jaccard")].sort_values(["holm_p", "wilcoxon_p"]).head(1)
    summary = {
        "n_spots": int(df.shape[0]),
        "n_samples": int(df["sample"].nunique()),
        "n_patient_pairs": int(default["n_pairs"].iloc[0]) if not default.empty else 0,
        "threshold_settings": int(tests["setting"].nunique()),
        "default_high_high_jaccard": default.iloc[0].to_dict() if not default.empty else {},
        "best_high_high_jaccard_across_thresholds": best.iloc[0].to_dict() if not best.empty else {},
        "interpretation": "Threshold/LOPO sensitivity is exploratory; spot-level P values are not used as inferential evidence.",
    }
    with open(OUT / "spatial_threshold_lopo_sensitivity_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
