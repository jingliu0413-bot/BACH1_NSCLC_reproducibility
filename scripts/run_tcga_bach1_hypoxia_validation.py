"""Independent TCGA-LUAD/LUSC bulk validation of BACH1-hypoxia scores.

Inputs are UCSC Xena TCGA sampleMap HiSeqV2 expression matrices and
clinicalMatrix files downloaded into data/tcga_xena.
"""

from __future__ import annotations

import gzip
import json
import re
import urllib.request
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tcga_xena"
EXT_TABLE = ROOT / "out/external_bach1_activity_primary/tables"
ROBUST_TABLE = ROOT / "out/external_bach1_activity_robustness/tables"
OUT = ROOT / "out/tcga_bach1_hypoxia_validation"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"

DOR_SIG = "DOROTHEA_BACH1_ABC_TF_ACTIVITY"
COL_SIG = "COLLECTRI_BACH1_TF_ACTIVITY"
KLENJA_SIG = "KLENJA2025_BACH1_INVERSE_ACTIVITY"
HYPOXIA_SIG = "HALLMARK_HYPOXIA"

URLS = {
    "LUAD_expression": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap/HiSeqV2.gz",
    "LUSC_expression": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUSC.sampleMap/HiSeqV2.gz",
    "LUAD_clinical": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap/LUAD_clinicalMatrix",
    "LUSC_clinical": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUSC.sampleMap/LUSC_clinicalMatrix",
}


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


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


def download_if_missing() -> None:
    paths = {
        "LUAD_expression": DATA / "TCGA.LUAD.HiSeqV2.gz",
        "LUSC_expression": DATA / "TCGA.LUSC.HiSeqV2.gz",
        "LUAD_clinical": DATA / "TCGA.LUAD.clinicalMatrix",
        "LUSC_clinical": DATA / "TCGA.LUSC.clinicalMatrix",
    }
    for key, path in paths.items():
        if path.exists() and path.stat().st_size > 1000:
            continue
        log(f"Downloading {key}")
        urllib.request.urlretrieve(URLS[key], path)


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
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or len(np.unique(a[ok])) < 2 or len(np.unique(b[ok])) < 2:
        return np.nan, np.nan, int(ok.sum())
    rho, p = stats.spearmanr(a[ok], b[ok])
    return float(rho), float(p), int(ok.sum())


def read_expression(cohort: str) -> pd.DataFrame:
    path = DATA / f"TCGA.{cohort}.HiSeqV2.gz"
    log(f"Reading TCGA-{cohort} expression")
    expr = pd.read_csv(path, sep="\t", index_col=0)
    expr.index = expr.index.astype(str).str.upper()
    expr = expr.groupby(expr.index).mean(numeric_only=True)
    tumor_cols = [c for c in expr.columns.astype(str) if re.search(r"-01[A-Z]?$|-01$", c)]
    expr = expr[tumor_cols]
    return expr


def clean_stage(value: object) -> str:
    text = str(value).upper().strip()
    if not text or text in {"NAN", "NA", "NONE", "[NOT AVAILABLE]", "[NOT APPLICABLE]"}:
        return "unknown"
    if "IV" in text:
        return "IV"
    if "III" in text:
        return "III"
    if re.search(r"\bII\b|STAGE II", text):
        return "II"
    if re.search(r"\bI\b|STAGE I", text):
        return "I"
    return "unknown"


def read_clinical(cohort: str) -> pd.DataFrame:
    path = DATA / f"TCGA.{cohort}.clinicalMatrix"
    clin = pd.read_csv(path, sep="\t", dtype=str)
    clin = clin.rename(columns={"sampleID": "sample"})
    clin["sample"] = clin["sample"].astype(str)
    clin["patient"] = clin["sample"].str.slice(0, 12)
    clin["cohort"] = cohort
    clin["stage_simple"] = clin.get("pathologic_stage", pd.Series(index=clin.index, dtype=str)).map(clean_stage)
    clin["pack_years"] = pd.to_numeric(clin.get("number_pack_years_smoked", np.nan), errors="coerce")
    if "ABSOLUTE_Purity" in clin.columns:
        clin["ABSOLUTE_Purity"] = pd.to_numeric(clin["ABSOLUTE_Purity"], errors="coerce")
    else:
        clin["ABSOLUTE_Purity"] = np.nan
    clin["histology"] = clin.get("histological_type", pd.Series(index=clin.index, dtype=str)).astype(str)
    clin["sample_type"] = clin.get("sample_type", pd.Series(index=clin.index, dtype=str)).astype(str)
    keep = ["sample", "patient", "cohort", "stage_simple", "pack_years", "ABSOLUTE_Purity", "histology", "sample_type"]
    return clin[keep].drop_duplicates("sample")


def present_targets(table: pd.DataFrame, signature: str) -> pd.DataFrame:
    sub = table[table["signature"].astype(str).eq(signature)].copy()
    gene_col = "target_in_adata" if "target_in_adata" in sub.columns else "target"
    sub = sub[sub[gene_col].notna()].copy()
    sub["gene"] = sub[gene_col].astype(str).str.upper()
    if "weight" not in sub.columns:
        sub["weight"] = 1.0
    sub["weight"] = pd.to_numeric(sub["weight"], errors="coerce").fillna(1.0)
    sub = sub[~sub["gene"].isin(["", "NAN", "NONE", "NA"])]
    return sub.drop_duplicates("gene")


def weighted_mean_z(expr: pd.DataFrame, genes: list[str], weights: list[float]) -> pd.Series:
    genes2 = [g for g in genes if g in expr.index]
    if not genes2:
        return pd.Series(np.nan, index=expr.columns)
    weight_map = {g: float(w) for g, w in zip(genes, weights)}
    w = np.asarray([weight_map[g] for g in genes2], dtype=float)
    sub = expr.loc[genes2].T.apply(pd.to_numeric, errors="coerce")
    sd = sub.std(axis=0, ddof=0).replace(0, np.nan)
    z = (sub - sub.mean(axis=0)) / sd
    denom = np.sum(np.abs(w[np.isfinite(sd.to_numpy())]))
    if denom <= 0:
        return pd.Series(np.nan, index=expr.columns)
    z = z.loc[:, sd.notna()]
    w = np.asarray([weight_map[g] for g in z.columns], dtype=float)
    return pd.Series(z.to_numpy(float).dot(w) / np.sum(np.abs(w)), index=z.index)


def score_cohort(cohort: str, targets: pd.DataFrame, pathways: pd.DataFrame, overlap: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr = read_expression(cohort)
    clin = read_clinical(cohort)
    scores = pd.DataFrame(index=expr.columns.astype(str))
    score_rows = []
    score_specs = []
    for sig in [DOR_SIG, COL_SIG, KLENJA_SIG]:
        sub = present_targets(targets, sig)
        score_specs.append((f"{sig}__mean_z", sub["gene"].tolist(), sub["weight"].tolist()))
    hyp = present_targets(pathways, HYPOXIA_SIG)
    score_specs.append((f"{HYPOXIA_SIG}__mean_z", hyp["gene"].tolist(), hyp["weight"].tolist()))
    dor_no = present_targets(targets, DOR_SIG)
    dor_no = dor_no[~dor_no["gene"].isin(overlap)]
    hyp_no = hyp[~hyp["gene"].isin(overlap)]
    score_specs.append((f"{DOR_SIG}_NO_HYPOXIA_OVERLAP__mean_z", dor_no["gene"].tolist(), dor_no["weight"].tolist()))
    score_specs.append((f"{HYPOXIA_SIG}_NO_DOROTHEA_OVERLAP__mean_z", hyp_no["gene"].tolist(), hyp_no["weight"].tolist()))

    for name, genes, weights in score_specs:
        present = [g for g in genes if g in expr.index]
        scores[name] = weighted_mean_z(expr, genes, weights)
        score_rows.append({"cohort": cohort, "score": name, "n_genes_requested": len(genes), "n_genes_present": len(present), "missing_genes": ";".join(sorted(set(genes) - set(present)))})

    scores = scores.reset_index().rename(columns={"index": "sample"})
    scores["patient"] = scores["sample"].str.slice(0, 12)
    scores = scores.merge(clin, on=["sample", "patient"], how="left")
    scores["cohort"] = cohort
    numeric_cols = [c for c in scores.columns if c.endswith("__mean_z")]
    grouped = scores.groupby("patient", observed=True)
    patient_scores = grouped[numeric_cols].mean().join(grouped[["cohort", "stage_simple", "histology", "sample_type"]].first())
    patient_scores["pack_years"] = grouped["pack_years"].first()
    patient_scores["ABSOLUTE_Purity"] = grouped["ABSOLUTE_Purity"].first()
    patient_scores["n_primary_tumour_samples"] = grouped.size()
    patient_scores = patient_scores.reset_index()
    patient_scores.to_csv(TABLE_DIR / f"TCGA_{cohort}_bach1_hypoxia_scores.csv", index=False)
    return patient_scores, pd.DataFrame(score_rows)


def standardize(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    sd = v.std(ddof=0)
    if not sd or not np.isfinite(sd):
        return pd.Series(np.nan, index=s.index)
    return (v - v.mean()) / sd


def correlations(scores: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("original", f"{DOR_SIG}__mean_z", f"{HYPOXIA_SIG}__mean_z"),
        ("both_no_shared", f"{DOR_SIG}_NO_HYPOXIA_OVERLAP__mean_z", f"{HYPOXIA_SIG}_NO_DOROTHEA_OVERLAP__mean_z"),
        ("collectri_original", f"{COL_SIG}__mean_z", f"{HYPOXIA_SIG}__mean_z"),
        ("klenja_original", f"{KLENJA_SIG}__mean_z", f"{HYPOXIA_SIG}__mean_z"),
    ]
    rows = []
    for cohort in ["LUAD", "LUSC", "combined"]:
        sub = scores if cohort == "combined" else scores[scores["cohort"].eq(cohort)]
        for label, x, y in pairs:
            rho, p, n = safe_spearman(sub[x], sub[y])
            rows.append({"cohort": cohort, "comparison": label, "x": x, "y": y, "n_patients": n, "spearman_rho": rho, "spearman_p": p})
    out = pd.DataFrame(rows)
    out["spearman_holm_p"] = out.groupby("cohort")["spearman_p"].transform(holm_adjust)
    out["spearman_bh_q"] = out.groupby("cohort")["spearman_p"].transform(bh_adjust)
    out.to_csv(TABLE_DIR / "tcga_bach1_hypoxia_correlations.csv", index=False)
    return out


def fit_models(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    df["dor_no_z"] = standardize(df[f"{DOR_SIG}_NO_HYPOXIA_OVERLAP__mean_z"])
    df["hyp_no_z"] = standardize(df[f"{HYPOXIA_SIG}_NO_DOROTHEA_OVERLAP__mean_z"])
    df["dor_orig_z"] = standardize(df[f"{DOR_SIG}__mean_z"])
    df["hyp_orig_z"] = standardize(df[f"{HYPOXIA_SIG}__mean_z"])
    df["pack_years_z"] = standardize(df["pack_years"])
    df["purity_z"] = standardize(df["ABSOLUTE_Purity"])
    df["stage_model"] = df["stage_simple"].fillna("unknown").replace("", "unknown")
    rows = []
    specs = [
        (
            "combined_deoverlap_cohort_stage_smoking",
            df,
            "hyp_no_z ~ dor_no_z + C(cohort) + C(stage_model) + pack_years_z",
            "dor_no_z",
            ["hyp_no_z", "dor_no_z", "cohort", "stage_model", "pack_years_z"],
        ),
        (
            "combined_original_cohort_stage_smoking",
            df,
            "hyp_orig_z ~ dor_orig_z + C(cohort) + C(stage_model) + pack_years_z",
            "dor_orig_z",
            ["hyp_orig_z", "dor_orig_z", "cohort", "stage_model", "pack_years_z"],
        ),
        (
            "combined_deoverlap_cohort_stage",
            df,
            "hyp_no_z ~ dor_no_z + C(cohort) + C(stage_model)",
            "dor_no_z",
            ["hyp_no_z", "dor_no_z", "cohort", "stage_model"],
        ),
        (
            "LUAD_deoverlap_purity_stage_smoking",
            df[df["cohort"].eq("LUAD")],
            "hyp_no_z ~ dor_no_z + purity_z + C(stage_model) + pack_years_z",
            "dor_no_z",
            ["hyp_no_z", "dor_no_z", "purity_z", "stage_model", "pack_years_z"],
        ),
        (
            "LUAD_deoverlap_purity_stage",
            df[df["cohort"].eq("LUAD")],
            "hyp_no_z ~ dor_no_z + purity_z + C(stage_model)",
            "dor_no_z",
            ["hyp_no_z", "dor_no_z", "purity_z", "stage_model"],
        ),
        (
            "LUSC_deoverlap_stage_smoking",
            df[df["cohort"].eq("LUSC")],
            "hyp_no_z ~ dor_no_z + C(stage_model) + pack_years_z",
            "dor_no_z",
            ["hyp_no_z", "dor_no_z", "stage_model", "pack_years_z"],
        ),
        (
            "LUSC_deoverlap_stage",
            df[df["cohort"].eq("LUSC")],
            "hyp_no_z ~ dor_no_z + C(stage_model)",
            "dor_no_z",
            ["hyp_no_z", "dor_no_z", "stage_model"],
        ),
    ]
    for name, sub, formula, term, cols in specs:
        d = sub[cols].replace([np.inf, -np.inf], np.nan).dropna()
        try:
            fit = smf.ols(formula, data=d).fit(cov_type="HC3")
            ci = fit.conf_int()
            rows.append(
                {
                    "model": name,
                    "formula": formula,
                    "n_patients": int(d.shape[0]),
                    "beta": float(fit.params.get(term, np.nan)),
                    "se_HC3": float(fit.bse.get(term, np.nan)),
                    "ci_low": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                    "ci_high": float(ci.loc[term, 1]) if term in ci.index else np.nan,
                    "p_value": float(fit.pvalues.get(term, np.nan)),
                    "r_squared": float(fit.rsquared),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": name,
                    "formula": formula,
                    "n_patients": int(d.shape[0]),
                    "beta": np.nan,
                    "se_HC3": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    "note": f"{type(exc).__name__}: {exc}",
                }
            )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"])
    out["p_bh_q"] = bh_adjust(out["p_value"])
    out.to_csv(TABLE_DIR / "tcga_bach1_hypoxia_adjusted_models.csv", index=False)
    return out


def make_figure(scores: pd.DataFrame, cor: pd.DataFrame, models: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.85)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.18]})
    palette = {"LUAD": "#4C78A8", "LUSC": "#F58518"}
    for ax, comparison, x, y, panel in [
        (
            axes[0],
            "original",
            f"{DOR_SIG}__mean_z",
            f"{HYPOXIA_SIG}__mean_z",
            "a",
        ),
        (
            axes[1],
            "both_no_shared",
            f"{DOR_SIG}_NO_HYPOXIA_OVERLAP__mean_z",
            f"{HYPOXIA_SIG}_NO_DOROTHEA_OVERLAP__mean_z",
            "b",
        ),
    ]:
        sns.scatterplot(data=scores, x=x, y=y, hue="cohort", palette=palette, s=18, alpha=0.78, ax=ax, linewidth=0)
        sns.regplot(data=scores, x=x, y=y, scatter=False, ax=ax, color="#333333", line_kws={"lw": 1})
        add_panel_label(ax, panel)
        ax.set_xlabel("TCGA DoRothEA BACH1 mean-z")
        ax.set_ylabel("TCGA Hallmark hypoxia mean-z")
        row = cor[cor["cohort"].eq("combined") & cor["comparison"].eq(comparison)]
        if not row.empty:
            r = row.iloc[0]
            ax.text(
                0.04,
                0.96,
                f"Spearman ρ={r['spearman_rho']:.3f}\nHolm-adjusted $P={format_p_mpl(r['spearman_holm_p'])}$",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        if comparison == "both_no_shared":
            ax.text(0.04, 0.82, "shared genes\nexcluded", transform=ax.transAxes, va="top", fontsize=7, color="#555555")
        ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[2]
    plot_models = models[models["model"].isin(["combined_deoverlap_cohort_stage_smoking", "LUAD_deoverlap_purity_stage_smoking", "LUSC_deoverlap_stage_smoking"])].copy()
    order = ["combined_deoverlap_cohort_stage_smoking", "LUAD_deoverlap_purity_stage_smoking", "LUSC_deoverlap_stage_smoking"]
    plot_models["model"] = pd.Categorical(plot_models["model"], categories=order, ordered=True)
    plot_models = plot_models.sort_values("model")
    y = np.arange(plot_models.shape[0])
    ax.errorbar(plot_models["beta"], y, xerr=[plot_models["beta"] - plot_models["ci_low"], plot_models["ci_high"] - plot_models["beta"]], fmt="o", color="#333333", ecolor="#777777")
    ax.axvline(0, color="black", lw=0.7)
    short_label_map = {
        "combined_deoverlap_cohort_stage_smoking": "Combined",
        "LUAD_deoverlap_purity_stage_smoking": "LUAD",
        "LUSC_deoverlap_stage_smoking": "LUSC",
    }
    ax.set_xlim(-0.05, max(0.78, float(plot_models["ci_high"].max()) + 0.34))
    labels = []
    for _, row in plot_models.iterrows():
        model_key = str(row["model"])
        labels.append(short_label_map.get(model_key, model_key))
        ax.text(
            float(row["ci_high"]) + 0.005,
            int(np.where(plot_models["model"].astype(str).to_numpy() == model_key)[0][0]),
            f"beta={row['beta']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]\nn={int(row['n_patients'])}, $P={format_p_mpl(row['p_value'])}$",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized beta for de-overlapped DoRothEA score")
    add_panel_label(ax, "c")
    fig.subplots_adjust(top=0.88, wspace=0.42)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"tcga_bach1_hypoxia_validation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict, cor: pd.DataFrame, models: pd.DataFrame, counts: pd.DataFrame) -> None:
    lines = [
        "# TCGA BACH1-Hypoxia Validation Report",
        "",
        "## Sources",
        "",
        "- TCGA-LUAD HiSeqV2 and clinicalMatrix from UCSC Xena TCGA sampleMap.",
        "- TCGA-LUSC HiSeqV2 and clinicalMatrix from UCSC Xena TCGA sampleMap.",
        "",
        "## Key Results",
        "",
        f"- Primary tumour samples/patients: LUAD n={summary['n_luad']}, LUSC n={summary['n_lusc']}, combined n={summary['n_combined']}.",
        f"- Combined original DoRothEA-hypoxia Spearman ρ={summary['combined_original_rho']:.4f}, Holm-adjusted *P*={format_p_text(summary['combined_original_holm_p'])}.",
        f"- Combined shared-gene-removed Spearman ρ={summary['combined_deoverlap_rho']:.4f}, Holm-adjusted *P*={format_p_text(summary['combined_deoverlap_holm_p'])}.",
        f"- Combined adjusted de-overlapped model beta={summary['combined_adjusted_beta']:.4f}, 95% CI {summary['combined_adjusted_ci_low']:.4f} to {summary['combined_adjusted_ci_high']:.4f}, *P*={format_p_text(summary['combined_adjusted_p'])}.",
        f"- LUAD purity-adjusted de-overlapped model beta={summary['luad_purity_adjusted_beta']:.4f}, *P*={format_p_text(summary['luad_purity_adjusted_p'])}.",
        "",
        "## Gene Coverage",
        "",
        "```csv",
        counts.to_csv(index=False).strip(),
        "```",
        "",
        "## Correlations",
        "",
        "```csv",
        cor.to_csv(index=False).strip(),
        "```",
        "",
        "## Adjusted Models",
        "",
        "```csv",
        models.to_csv(index=False).strip(),
        "```",
        "",
    ]
    (OUT / "tcga_bach1_hypoxia_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    download_if_missing()
    targets = pd.read_csv(EXT_TABLE / "external_bach1_signature_targets.csv")
    pathways = pd.read_csv(EXT_TABLE / "phenotype_pathway_signature_targets.csv")
    overlap_path = ROBUST_TABLE / "bach1_phenotype_gene_set_overlap_summary.csv"
    if overlap_path.exists():
        overlap_row = pd.read_csv(overlap_path)
        overlap_genes = overlap_row[
            overlap_row["bach1_signature"].eq(DOR_SIG) & overlap_row["phenotype_signature"].eq(HYPOXIA_SIG)
        ]["overlap_genes"].iloc[0]
        overlap = set(str(overlap_genes).split(";")) if pd.notna(overlap_genes) and str(overlap_genes) else set()
    else:
        dor = set(present_targets(targets, DOR_SIG)["gene"])
        hyp = set(present_targets(pathways, HYPOXIA_SIG)["gene"])
        overlap = dor & hyp

    all_scores = []
    count_rows = []
    for cohort in ["LUAD", "LUSC"]:
        scores, counts = score_cohort(cohort, targets, pathways, overlap)
        all_scores.append(scores)
        count_rows.append(counts)
    scores = pd.concat(all_scores, ignore_index=True)
    counts = pd.concat(count_rows, ignore_index=True)
    scores.to_csv(TABLE_DIR / "tcga_luad_lusc_bach1_hypoxia_patient_scores.csv", index=False)
    counts.to_csv(TABLE_DIR / "tcga_signature_gene_coverage.csv", index=False)

    cor = correlations(scores)
    models = fit_models(scores)
    make_figure(scores, cor, models)

    cor_idx = cor.set_index(["cohort", "comparison"])
    model_idx = models.set_index("model")
    summary = {
        "n_luad": int(scores["cohort"].eq("LUAD").sum()),
        "n_lusc": int(scores["cohort"].eq("LUSC").sum()),
        "n_combined": int(scores.shape[0]),
        "combined_original_rho": float(cor_idx.loc[("combined", "original"), "spearman_rho"]),
        "combined_original_holm_p": float(cor_idx.loc[("combined", "original"), "spearman_holm_p"]),
        "combined_deoverlap_rho": float(cor_idx.loc[("combined", "both_no_shared"), "spearman_rho"]),
        "combined_deoverlap_holm_p": float(cor_idx.loc[("combined", "both_no_shared"), "spearman_holm_p"]),
        "luad_deoverlap_rho": float(cor_idx.loc[("LUAD", "both_no_shared"), "spearman_rho"]),
        "lusc_deoverlap_rho": float(cor_idx.loc[("LUSC", "both_no_shared"), "spearman_rho"]),
        "combined_adjusted_beta": float(model_idx.loc["combined_deoverlap_cohort_stage_smoking", "beta"]),
        "combined_adjusted_ci_low": float(model_idx.loc["combined_deoverlap_cohort_stage_smoking", "ci_low"]),
        "combined_adjusted_ci_high": float(model_idx.loc["combined_deoverlap_cohort_stage_smoking", "ci_high"]),
        "combined_adjusted_p": float(model_idx.loc["combined_deoverlap_cohort_stage_smoking", "p_value"]),
        "luad_purity_adjusted_beta": float(model_idx.loc["LUAD_deoverlap_purity_stage_smoking", "beta"]),
        "luad_purity_adjusted_p": float(model_idx.loc["LUAD_deoverlap_purity_stage_smoking", "p_value"]),
        "notes": [
            "Primary tumour samples were selected by TCGA sample type code 01.",
            "LUSC clinicalMatrix did not include ABSOLUTE_Purity; purity-adjusted modelling was therefore limited to LUAD.",
            "Scores were recomputed as sample-wise weighted mean-z values in TCGA expression space using the same signature definitions.",
        ],
    }
    with (OUT / "tcga_bach1_hypoxia_validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_report(summary, cor, models, counts)
    log(f"Wrote outputs under {OUT}")


if __name__ == "__main__":
    main()
