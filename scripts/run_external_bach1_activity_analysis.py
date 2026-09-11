"""External BACH1 activity signatures in the primary malignant epithelial set.

This script implements the revised manuscript direction: prioritize the
16-patient primary malignant epithelial set, use externally defined BACH1
activity signatures, keep patient-level statistics as the inferential unit,
and treat de novo pySCENIC/ATAC/spatial analyses as sensitivity context.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import decoupler as dc
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[1]
INPUT_H5AD = ROOT / "out/bach1_malignant_epithelial_primary/nsclc_malignant_epithelial_primary_bach1_analysis_object.h5ad"
OUT = ROOT / "out/external_bach1_activity_primary"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
RESOURCE_DIR = ROOT / "resources/bach1_external_signatures"

PATIENT_COL = "patient"
BACH1_GROUP_COL = "BACH1_group"
BACH1_EXPR_COL = "BACH1_expr"

KLENJA_SOURCE = {
    "source_id": "Klenja_Skudrinja_2025_RedoxBiol",
    "pmid": "40716152",
    "pmcid": "PMC12314328",
    "doi": "10.1016/j.redox.2025.103789",
    "geo_chipseq": "GSE288627",
    "geo_rnaseq": "GSE288626",
    "note": (
        "Validated lung-cancer BACH1 signature genes HMOX1, ZNF469 and HTRA3 were reported "
        "from RNA-seq plus ChIP-seq in BACH1-proficient and BACH1-deficient lung cancer cells. "
        "They are induced by BACH1 depletion; inverse scores are interpreted as higher BACH1-like activity."
    ),
}

NRF2_RESPONSE_GENES = [
    "NQO1",
    "GCLC",
    "GCLM",
    "SLC7A11",
    "TXNRD1",
    "HMOX1",
    "FTH1",
    "FTL",
    "AKR1B1",
    "AKR1C1",
    "AKR1C2",
    "AKR1C3",
    "GSR",
    "GSS",
    "SRXN1",
    "GPX2",
    "ME1",
    "ABCC1",
    "ABCC2",
]

LUNG_BACH1_EFFECTOR_GENES = [
    "HK2",
    "GAPDH",
    "ITGA2",
    "MMP1",
    "CXCR4",
]

HALLMARK_PATHWAYS = [
    "REACTIVE_OXYGEN_SPECIES_PATHWAY",
    "HEME_METABOLISM",
    "GLYCOLYSIS",
    "HYPOXIA",
    "EPITHELIAL_MESENCHYMAL_TRANSITION",
    "OXIDATIVE_PHOSPHORYLATION",
]


def is_bach1_activity_column(column: str) -> bool:
    """Return True for interpretable BACH1-activity estimates."""
    col = str(column)
    if "_p" in col:
        return False
    if col.startswith("DOROTHEA_BACH1") or col.startswith("COLLECTRI_BACH1"):
        return True
    if col.startswith("KLENJA2025_BACH1_ACTIVITY_INVERTED"):
        return True
    # The weighted mean-z score already uses negative weights for the depletion-induced genes.
    return col.startswith("KLENJA2025_BACH1_INVERSE_ACTIVITY") and col.endswith("__mean_z")


def is_phenotype_column(column: str) -> bool:
    col = str(column)
    return (
        (
            col.startswith("HALLMARK_")
            or col.startswith("CURATED_NRF2")
            or col.startswith("LITERATURE_LUNG_BACH1_EFFECTOR")
        )
        and "_p" not in col
    )


def activity_vs_phenotype_correlations(cor: pd.DataFrame) -> pd.DataFrame:
    if cor.empty:
        return cor.copy()
    subset = cor[cor["x"].map(is_bach1_activity_column) & cor["y"].map(is_phenotype_column)].copy()
    if subset.empty:
        return subset
    subset["abs_rho"] = pd.to_numeric(subset["spearman_rho"], errors="coerce").abs()
    subset = subset.sort_values(["spearman_holm_p", "spearman_p", "abs_rho", "x", "y"], ascending=[True, True, False, True, True])
    return subset


def bach1_expression_vs_phenotype_correlations(cor: pd.DataFrame) -> pd.DataFrame:
    if cor.empty:
        return cor.copy()
    subset = cor[
        cor["x"].isin(["BACH1_detected_fraction", "BACH1_expression_mean"])
        & cor["y"].map(is_phenotype_column)
    ].copy()
    if subset.empty:
        return subset
    subset["abs_rho"] = pd.to_numeric(subset["spearman_rho"], errors="coerce").abs()
    subset = subset.sort_values(["spearman_holm_p", "spearman_p", "abs_rho", "x", "y"], ascending=[True, True, False, True, True])
    return subset


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)


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


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or len(np.unique(a[ok])) < 2 or len(np.unique(b[ok])) < 2:
        return np.nan, np.nan
    rho, pvalue = stats.spearmanr(a[ok], b[ok])
    return float(rho), float(pvalue)


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


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def gene_lookup(var_names: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for gene in var_names:
        key = str(gene).upper()
        lookup.setdefault(key, str(gene))
    return lookup


def map_gene_list(genes: Iterable[str], lookup: dict[str, str]) -> list[str]:
    seen: set[str] = set()
    mapped: list[str] = []
    for gene in genes:
        mapped_gene = lookup.get(str(gene).upper())
        if mapped_gene and mapped_gene not in seen:
            seen.add(mapped_gene)
            mapped.append(mapped_gene)
    return mapped


def download_database_regulons() -> tuple[pd.DataFrame, pd.DataFrame]:
    dor_path = RESOURCE_DIR / "dorothea_human_ABC_decoupler.csv"
    col_path = RESOURCE_DIR / "collectri_human_decoupler.csv"

    if dor_path.exists():
        dor = pd.read_csv(dor_path)
    else:
        log("Fetching DoRothEA A-C regulons through decoupler/OmniPath")
        dor = dc.op.dorothea(organism="human", levels=["A", "B", "C"], verbose=True)
        dor.to_csv(dor_path, index=False)

    if col_path.exists():
        col = pd.read_csv(col_path)
    else:
        log("Fetching CollecTRI regulons through decoupler/OmniPath")
        col = dc.op.collectri(organism="human", verbose=True)
        col.to_csv(col_path, index=False)

    return dor, col


def build_networks(lookup: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dor, col = download_database_regulons()
    klenja_rows = []
    for gene in ["HMOX1", "ZNF469", "HTRA3"]:
        klenja_rows.append(
            {
                "signature": "KLENJA2025_BACH1_INVERSE_ACTIVITY",
                "target": gene,
                "weight": -1.0,
                "source_type": "published_RNAseq_ChIPseq_lung_cancer",
                **KLENJA_SOURCE,
            }
        )

    effector_rows = []
    for gene in LUNG_BACH1_EFFECTOR_GENES:
        effector_rows.append(
            {
                "signature": "LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY",
                "target": gene,
                "weight": 1.0,
                "source_type": "published_lung_cancer_effectors",
                "source_id": "Wiel_2019_Lignitto_2019_and_related_lung_cancer_reports",
                "pmid": "31257027;31257028",
                "doi": "10.1016/j.cell.2019.06.005;10.1016/j.cell.2019.06.003",
                "geo_chipseq": "",
                "geo_rnaseq": "",
                "note": "Small literature effector set used as a descriptive lung cancer BACH1 phenotype score, not as an independent regulon.",
            }
        )

    dor_bach1 = dor[dor["source"].astype(str).str.upper().eq("BACH1")].copy()
    dor_bach1["signature"] = "DOROTHEA_BACH1_ABC_TF_ACTIVITY"
    dor_bach1["source_type"] = "database_regulon"
    dor_bach1["source_id"] = "DoRothEA_A-C_via_decoupler_OmniPath"
    dor_bach1["pmid"] = ""
    dor_bach1["pmcid"] = ""
    dor_bach1["doi"] = ""
    dor_bach1["geo_chipseq"] = ""
    dor_bach1["geo_rnaseq"] = ""
    dor_bach1["note"] = "DoRothEA A-C human BACH1 regulon fetched through decoupler/OmniPath."

    col_bach1 = col[col["source"].astype(str).str.upper().eq("BACH1")].copy()
    col_bach1["signature"] = "COLLECTRI_BACH1_TF_ACTIVITY"
    col_bach1["source_type"] = "database_regulon"
    col_bach1["source_id"] = "CollecTRI_via_decoupler_OmniPath"
    col_bach1["pmid"] = col_bach1.get("references", "")
    col_bach1["pmcid"] = ""
    col_bach1["doi"] = ""
    col_bach1["geo_chipseq"] = ""
    col_bach1["geo_rnaseq"] = ""
    col_bach1["note"] = "CollecTRI human BACH1 regulon fetched through decoupler/OmniPath."

    harmonized = []
    for frame in [pd.DataFrame(klenja_rows), pd.DataFrame(effector_rows), dor_bach1, col_bach1]:
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            target = lookup.get(str(row["target"]).upper())
            harmonized.append(
                {
                    "signature": row["signature"],
                    "target": str(row["target"]),
                    "target_in_adata": target if target else "",
                    "weight": float(row.get("weight", 1.0)),
                    "source_type": row.get("source_type", ""),
                    "source_id": row.get("source_id", ""),
                    "confidence": row.get("confidence", ""),
                    "resources": row.get("resources", ""),
                    "references": row.get("references", row.get("pmid", "")),
                    "pmid": row.get("pmid", ""),
                    "pmcid": row.get("pmcid", ""),
                    "doi": row.get("doi", ""),
                    "geo_chipseq": row.get("geo_chipseq", ""),
                    "geo_rnaseq": row.get("geo_rnaseq", ""),
                    "note": row.get("note", ""),
                    "present_in_primary": bool(target),
                }
            )
    targets = pd.DataFrame(harmonized).drop_duplicates(["signature", "target_in_adata", "target", "weight"])
    targets.to_csv(TABLE_DIR / "external_bach1_signature_targets.csv", index=False)

    main_bach1 = targets[
        targets["signature"].isin(
            [
                "KLENJA2025_BACH1_INVERSE_ACTIVITY",
                "DOROTHEA_BACH1_ABC_TF_ACTIVITY",
                "COLLECTRI_BACH1_TF_ACTIVITY",
            ]
        )
        & targets["present_in_primary"]
    ].copy()
    effectors = targets[
        targets["signature"].eq("LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY") & targets["present_in_primary"]
    ].copy()
    return targets, main_bach1, effectors


def build_pathway_sets(lookup: dict[str, str]) -> pd.DataFrame:
    hallmark = dc.op.hallmark(organism="human")
    hallmark = hallmark[hallmark["source"].isin(HALLMARK_PATHWAYS)].copy()
    hallmark["signature"] = "HALLMARK_" + hallmark["source"].astype(str)
    hallmark["weight"] = 1.0
    hallmark["source_type"] = "MSigDB_Hallmark_via_decoupler"

    custom_rows = []
    for gene in NRF2_RESPONSE_GENES:
        custom_rows.append(
            {
                "signature": "CURATED_NRF2_ANTIOXIDANT_RESPONSE",
                "target": gene,
                "weight": 1.0,
                "source_type": "curated_NRF2_antioxidant_response",
            }
        )
    for gene in LUNG_BACH1_EFFECTOR_GENES:
        custom_rows.append(
            {
                "signature": "LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY",
                "target": gene,
                "weight": 1.0,
                "source_type": "published_lung_cancer_effectors",
            }
        )
    custom = pd.DataFrame(custom_rows)

    frames = []
    h = hallmark.rename(columns={"target": "target"})[["signature", "target", "weight", "source_type"]]
    frames.append(h)
    frames.append(custom)
    pathways = pd.concat(frames, ignore_index=True)
    pathways["target_in_adata"] = pathways["target"].map(lambda g: lookup.get(str(g).upper(), ""))
    pathways["present_in_primary"] = pathways["target_in_adata"].ne("")
    pathways.to_csv(TABLE_DIR / "phenotype_pathway_signature_targets.csv", index=False)
    return pathways[pathways["present_in_primary"]].copy()


def selected_gene_matrix(ad: sc.AnnData, genes: Iterable[str]) -> pd.DataFrame:
    mapped = [g for g in genes if g in ad.var_names]
    if not mapped:
        return pd.DataFrame(index=ad.obs_names)
    sub = ad[:, mapped]
    x = to_dense(sub.X).astype(np.float32)
    return pd.DataFrame(x, index=ad.obs_names.astype(str), columns=mapped)


def score_weighted_mean_z(expr: pd.DataFrame, net: pd.DataFrame, min_targets: int = 2) -> pd.DataFrame:
    z = expr.copy()
    for col in z.columns:
        values = pd.to_numeric(z[col], errors="coerce").to_numpy(dtype=float)
        sd = np.nanstd(values, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            z[col] = 0.0
        else:
            z[col] = (values - np.nanmean(values)) / sd

    scores: dict[str, np.ndarray] = {}
    rows = []
    for sig, sub in net.groupby("signature"):
        genes = [g for g in sub["target_in_adata"].astype(str).tolist() if g in z.columns]
        if len(genes) < min_targets:
            continue
        weights = []
        for gene in genes:
            w = sub.loc[sub["target_in_adata"].astype(str).eq(gene), "weight"].astype(float).iloc[0]
            weights.append(float(w))
        w = np.asarray(weights, dtype=float)
        denom = np.sum(np.abs(w))
        if denom == 0:
            continue
        score = (z[genes].to_numpy(dtype=float) @ w) / denom
        scores[f"{sig}__mean_z"] = score
        rows.append({"signature": sig, "method": "directional_mean_z", "n_targets_present": len(genes)})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "activity_score_target_counts.csv", index=False)
    return pd.DataFrame(scores, index=expr.index)


def patient_pseudobulk(expr: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    tmp = expr.copy()
    tmp[PATIENT_COL] = obs[PATIENT_COL].astype(str).to_numpy()
    return tmp.groupby(PATIENT_COL, observed=True).mean(numeric_only=True)


def run_ssgsea(pb_expr: pd.DataFrame, pathways: pd.DataFrame, bach1_targets: pd.DataFrame) -> pd.DataFrame:
    gene_sets: dict[str, list[str]] = {}
    combined = pd.concat([pathways, bach1_targets], ignore_index=True, sort=False)
    for sig, sub in combined.groupby("signature"):
        genes = sorted(set([g for g in sub["target_in_adata"].astype(str) if g in pb_expr.columns]))
        if len(genes) >= 2:
            gene_sets[sig] = genes
    if not gene_sets:
        return pd.DataFrame(index=pb_expr.index)

    log("Running patient-pseudobulk ssGSEA")
    res = gp.ssgsea(
        data=pb_expr.T,
        gene_sets=gene_sets,
        sample_norm_method="rank",
        min_size=2,
        max_size=5000,
        outdir=None,
        permutation_num=0,
        threads=4,
        seed=7,
        no_plot=True,
        verbose=False,
    )
    r = res.res2d.copy()
    # gseapy versions use either Name/Term/NES or sample/Term/NES.
    sample_col = "Name" if "Name" in r.columns else "sample"
    value_col = "NES" if "NES" in r.columns else "ES"
    wide = r.pivot(index=sample_col, columns="Term", values=value_col)
    wide.index = wide.index.astype(str)
    wide = wide.reindex(pb_expr.index)
    wide = wide.add_suffix("__ssgsea_NES")
    # The Klenja three-gene signature is induced when BACH1 is depleted, so invert it for activity.
    source_col = "KLENJA2025_BACH1_INVERSE_ACTIVITY__ssgsea_NES"
    if source_col in wide.columns:
        wide["KLENJA2025_BACH1_ACTIVITY_INVERTED__ssgsea_NES"] = -pd.to_numeric(wide[source_col], errors="coerce")
    wide.to_csv(TABLE_DIR / "patient_pseudobulk_ssgsea_scores.csv")
    return wide


def run_decoupler_activity(pb_expr: pd.DataFrame, bach1_targets: pd.DataFrame) -> pd.DataFrame:
    net = bach1_targets[["signature", "target_in_adata", "weight"]].rename(
        columns={"signature": "source", "target_in_adata": "target"}
    )
    net = net[net["target"].isin(pb_expr.columns)].copy()
    net = net.drop_duplicates(["source", "target"])
    if net.empty:
        return pd.DataFrame(index=pb_expr.index)

    score_frames = []
    pval_frames = []
    for method_name in ["ulm", "mlm"]:
        method = getattr(dc.mt, method_name)
        log(f"Running patient-pseudobulk decoupler {method_name.upper()}")
        est, pvals = method(pb_expr, net, tmin=2, verbose=False)
        est = est.add_suffix(f"__decoupler_{method_name}")
        pvals = pvals.add_suffix(f"__decoupler_{method_name}_p")
        score_frames.append(est)
        pval_frames.append(pvals)
    scores = pd.concat(score_frames, axis=1)
    pvals = pd.concat(pval_frames, axis=1)
    scores.to_csv(TABLE_DIR / "patient_pseudobulk_decoupler_bach1_activity_scores.csv")
    pvals.to_csv(TABLE_DIR / "patient_pseudobulk_decoupler_bach1_activity_pvalues.csv")
    return scores


def patient_metadata(ad: sc.AnnData) -> pd.DataFrame:
    obs = ad.obs.copy()
    obs["BACH1_detected"] = obs[BACH1_GROUP_COL].astype(str).eq("BACH1_high")
    rows = []
    for (dataset, patient, sample_id), sub in obs.groupby(["dataset", PATIENT_COL, "sample_id"], observed=True):
        if sub.empty:
            continue
        source_logic = []
        if str(dataset) == "GSE131907":
            source_logic.append("author tS1/tS2/tS3 tumour epithelial annotation")
        if str(dataset) == "GSE274934":
            source_logic.append("tumour epithelial cell with >=1 expression-derived CNV support")
        rows.append(
            {
                "dataset": dataset,
                "patient": patient,
                "sample_id": sample_id,
                "sample": ";".join(sorted(set(sub.get("sample", pd.Series(dtype=str)).astype(str)))),
                "tumour_type": ";".join(sorted(set(sub.get("tumor_type", pd.Series(dtype=str)).astype(str)))),
                "tissue_status": ";".join(sorted(set(sub.get("tissue_status", pd.Series(dtype=str)).astype(str)))),
                "n_primary_malignant_epithelial_cells": int(sub.shape[0]),
                "n_BACH1_detected": int(sub["BACH1_detected"].sum()),
                "BACH1_detected_fraction": float(sub["BACH1_detected"].mean()),
                "mean_BACH1_expression": float(pd.to_numeric(sub[BACH1_EXPR_COL], errors="coerce").mean()),
                "malignant_call_logic": " + ".join(source_logic),
            }
        )
    table = pd.DataFrame(rows).sort_values(["dataset", "patient", "sample_id"])
    table.to_csv(TABLE_DIR / "primary_malignant_epithelial_sample_information.csv", index=False)
    return table


def aggregate_patient_scores(ad: sc.AnnData, cell_scores: pd.DataFrame, pb_scores: pd.DataFrame) -> pd.DataFrame:
    obs = ad.obs.copy()
    obs["BACH1_detected"] = obs[BACH1_GROUP_COL].astype(str).eq("BACH1_high")
    base = (
        obs.groupby(PATIENT_COL, observed=True)
        .agg(
            dataset=("dataset", lambda x: ";".join(sorted(set(map(str, x))))),
            sample_id=("sample_id", lambda x: ";".join(sorted(set(map(str, x))))),
            tumour_type=("tumor_type", lambda x: ";".join(sorted(set(map(str, x))))),
            n_cells=(PATIENT_COL, "size"),
            n_BACH1_detected=("BACH1_detected", "sum"),
            BACH1_detected_fraction=("BACH1_detected", "mean"),
            BACH1_expression_mean=(BACH1_EXPR_COL, "mean"),
            total_counts_median=("total_counts", "median"),
            n_genes_median=("n_genes_by_counts", "median"),
            pct_mt_median=("pct_counts_mt", "median"),
        )
        .reset_index()
        .set_index(PATIENT_COL)
    )
    cell_tmp = cell_scores.copy()
    cell_tmp[PATIENT_COL] = obs[PATIENT_COL].astype(str).to_numpy()
    cell_mean = cell_tmp.groupby(PATIENT_COL, observed=True).mean(numeric_only=True)
    patient = base.join(cell_mean, how="left").join(pb_scores, how="left")
    patient.to_csv(TABLE_DIR / "external_bach1_activity_patient_scores.csv")
    return patient


def within_patient_deltas(ad: sc.AnnData, cell_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = ad.obs.copy()
    obs["BACH1_detected"] = obs[BACH1_GROUP_COL].astype(str).eq("BACH1_high")
    tmp = cell_scores.copy()
    tmp[PATIENT_COL] = obs[PATIENT_COL].astype(str).to_numpy()
    tmp["dataset"] = obs["dataset"].astype(str).to_numpy()
    tmp["BACH1_detected"] = obs["BACH1_detected"].to_numpy()

    rows = []
    for (patient, dataset), sub in tmp.groupby([PATIENT_COL, "dataset"], observed=True):
        high = sub[sub["BACH1_detected"]]
        low = sub[~sub["BACH1_detected"]]
        for score_col in cell_scores.columns:
            rows.append(
                {
                    "patient": patient,
                    "dataset": dataset,
                    "score": score_col,
                    "n_detected": int(high.shape[0]),
                    "n_undetected": int(low.shape[0]),
                    "mean_detected": float(pd.to_numeric(high[score_col], errors="coerce").mean()) if not high.empty else np.nan,
                    "mean_undetected": float(pd.to_numeric(low[score_col], errors="coerce").mean()) if not low.empty else np.nan,
                    "delta_detected_minus_undetected": (
                        float(pd.to_numeric(high[score_col], errors="coerce").mean() - pd.to_numeric(low[score_col], errors="coerce").mean())
                        if not high.empty and not low.empty
                        else np.nan
                    ),
                }
            )
    deltas = pd.DataFrame(rows)
    deltas.to_csv(TABLE_DIR / "within_patient_bach1_detected_activity_deltas.csv", index=False)

    tests = []
    for score, sub in deltas.groupby("score"):
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
    tests_df = tests_df.sort_values(["wilcoxon_holm_p", "wilcoxon_p", "score"])
    tests_df.to_csv(TABLE_DIR / "within_patient_bach1_detected_activity_delta_tests.csv", index=False)

    lopo_rows = []
    for score, sub in deltas.groupby("score"):
        patients = sorted(sub["patient"].dropna().astype(str).unique())
        for held in patients:
            keep = sub[sub["patient"].astype(str).ne(held)]
            stat, pvalue, n = safe_wilcoxon(keep["delta_detected_minus_undetected"])
            med = pd.to_numeric(keep["delta_detected_minus_undetected"], errors="coerce").median()
            lopo_rows.append(
                {
                    "score": score,
                    "held_out_patient": held,
                    "n_patients": n,
                    "median_delta": float(med) if pd.notna(med) else np.nan,
                    "sign": "positive" if pd.notna(med) and med > 0 else ("negative" if pd.notna(med) and med < 0 else "zero_or_nan"),
                    "wilcoxon_p": pvalue,
                }
            )
    lopo = pd.DataFrame(lopo_rows)
    lopo.to_csv(TABLE_DIR / "within_patient_bach1_detected_activity_lopo.csv", index=False)
    lopo_summary = (
        lopo.groupby("score")
        .agg(
            n_lopo=("held_out_patient", "nunique"),
            min_median_delta=("median_delta", "min"),
            max_median_delta=("median_delta", "max"),
            median_of_lopo_medians=("median_delta", "median"),
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
    lopo_summary.to_csv(TABLE_DIR / "within_patient_bach1_detected_activity_lopo_summary.csv", index=False)
    return deltas, tests_df, lopo_summary


def patient_correlations(patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bach1_activity_cols = [c for c in patient.columns if is_bach1_activity_column(c)]
    phenotype_cols = [c for c in patient.columns if is_phenotype_column(c)]
    predictor_cols = ["BACH1_detected_fraction", "BACH1_expression_mean"] + bach1_activity_cols

    rows = []
    for xcol in predictor_cols:
        for ycol in phenotype_cols + bach1_activity_cols:
            if xcol == ycol:
                continue
            rho, pvalue = safe_spearman(patient[xcol], patient[ycol])
            rows.append({"x": xcol, "y": ycol, "n_patients": int(patient[[xcol, ycol]].dropna().shape[0]), "spearman_rho": rho, "spearman_p": pvalue})
    cor = pd.DataFrame(rows)
    cor["spearman_holm_p"] = holm_adjust(cor["spearman_p"])
    cor["spearman_bh_q"] = bh_adjust(cor["spearman_p"])
    cor = cor.sort_values(["spearman_holm_p", "spearman_p", "x", "y"])
    cor.to_csv(TABLE_DIR / "patient_level_activity_correlations.csv", index=False)
    activity_vs_phenotype_correlations(cor).to_csv(
        TABLE_DIR / "patient_level_bach1_activity_vs_phenotype_correlations.csv", index=False
    )
    bach1_expression_vs_phenotype_correlations(cor).to_csv(
        TABLE_DIR / "patient_level_bach1_expression_vs_phenotype_correlations.csv", index=False
    )

    lopo_rows = []
    for _, row in cor.iterrows():
        xcol = row["x"]
        ycol = row["y"]
        for held in patient.index.astype(str):
            keep = patient[patient.index.astype(str) != held]
            rho, pvalue = safe_spearman(keep[xcol], keep[ycol])
            lopo_rows.append(
                {
                    "x": xcol,
                    "y": ycol,
                    "held_out_patient": held,
                    "n_patients": int(keep[[xcol, ycol]].dropna().shape[0]),
                    "spearman_rho": rho,
                    "spearman_p": pvalue,
                    "sign": "positive" if np.isfinite(rho) and rho > 0 else ("negative" if np.isfinite(rho) and rho < 0 else "zero_or_nan"),
                }
            )
    lopo = pd.DataFrame(lopo_rows)
    lopo.to_csv(TABLE_DIR / "patient_level_activity_correlations_lopo.csv", index=False)
    lopo_summary = (
        lopo.groupby(["x", "y"])
        .agg(
            n_lopo=("held_out_patient", "nunique"),
            min_rho=("spearman_rho", "min"),
            max_rho=("spearman_rho", "max"),
            median_rho=("spearman_rho", "median"),
            max_p=("spearman_p", "max"),
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
    lopo_summary.to_csv(TABLE_DIR / "patient_level_activity_correlations_lopo_summary.csv", index=False)
    return cor, lopo_summary


def cross_dataset_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, score), sub in deltas.groupby(["dataset", "score"], observed=True):
        stat, pvalue, n = safe_wilcoxon(sub["delta_detected_minus_undetected"])
        rows.append(
            {
                "dataset": dataset,
                "score": score,
                "n_patients_with_nonzero_delta": n,
                "median_delta_detected_minus_undetected": float(pd.to_numeric(sub["delta_detected_minus_undetected"], errors="coerce").median()),
                "wilcoxon_p": pvalue,
            }
        )
    out = pd.DataFrame(rows).sort_values(["score", "dataset"])
    out.to_csv(TABLE_DIR / "within_patient_bach1_detected_activity_delta_by_dataset.csv", index=False)
    return out


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows.\n"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    view = view.fillna("")
    cols = list(view.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if math.isfinite(value):
                    value = f"{value:.4g}"
                else:
                    value = ""
            vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


SHORT_LABELS = {
    "KLENJA2025_BACH1_INVERSE_ACTIVITY__mean_z": "Klenja inverse\nmean-z",
    "KLENJA2025_BACH1_ACTIVITY_INVERTED__ssgsea_NES": "Klenja inverted\nssGSEA",
    "DOROTHEA_BACH1_ABC_TF_ACTIVITY__mean_z": "DoRothEA\nmean-z",
    "COLLECTRI_BACH1_TF_ACTIVITY__mean_z": "CollecTRI\nmean-z",
    "DOROTHEA_BACH1_ABC_TF_ACTIVITY__ssgsea_NES": "DoRothEA\nssGSEA",
    "COLLECTRI_BACH1_TF_ACTIVITY__ssgsea_NES": "CollecTRI\nssGSEA",
    "DOROTHEA_BACH1_ABC_TF_ACTIVITY__decoupler_ulm": "DoRothEA\nULM",
    "COLLECTRI_BACH1_TF_ACTIVITY__decoupler_ulm": "CollecTRI\nULM",
    "DOROTHEA_BACH1_ABC_TF_ACTIVITY__decoupler_mlm": "DoRothEA\nMLM",
    "COLLECTRI_BACH1_TF_ACTIVITY__decoupler_mlm": "CollecTRI\nMLM",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION__mean_z": "EMT",
    "HALLMARK_GLYCOLYSIS__mean_z": "Glycolysis",
    "HALLMARK_HEME_METABOLISM__mean_z": "Heme",
    "HALLMARK_HYPOXIA__mean_z": "Hypoxia",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION__mean_z": "OxPhos",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY__mean_z": "ROS",
    "CURATED_NRF2_ANTIOXIDANT_RESPONSE__mean_z": "NRF2",
    "LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY__mean_z": "Lung BACH1\neffectors",
}


def short_label(column: str) -> str:
    col = str(column)
    if col in SHORT_LABELS:
        return SHORT_LABELS[col]
    return (
        col.replace("HALLMARK_", "")
        .replace("KLENJA2025_", "Klenja ")
        .replace("__ssgsea_NES", "")
        .replace("__mean_z", "")
        .replace("__decoupler_ulm", " ULM")
        .replace("__decoupler_mlm", " MLM")
        .replace("_", " ")
    )


def make_figure(patient: pd.DataFrame, tests: pd.DataFrame, cor: pd.DataFrame, lopo_cor: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.8)
    fig = plt.figure(figsize=(13.2, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.22, 1.12], hspace=0.48, wspace=0.7)

    order = patient.sort_values("BACH1_detected_fraction").index.astype(str).tolist()
    ax_a = fig.add_subplot(gs[0, 0])
    plot_patient = patient.loc[order].reset_index()
    ax_a.bar(plot_patient[PATIENT_COL], plot_patient["BACH1_detected_fraction"], color="#4C78A8")
    ax_a.set_ylabel("BACH1-detected fraction")
    ax_a.set_xlabel("Patient")
    ax_a.set_ylim(0, max(0.35, float(plot_patient["BACH1_detected_fraction"].max()) * 1.2))
    ax_a.tick_params(axis="x", rotation=90, labelsize=6)
    ax_a.set_title("a  Primary malignant epithelial set", loc="left", fontweight="bold")

    ax_b = fig.add_subplot(gs[0, 1])
    heat_cols = [
        c
        for c in [
            "KLENJA2025_BACH1_INVERSE_ACTIVITY__mean_z",
            "KLENJA2025_BACH1_ACTIVITY_INVERTED__ssgsea_NES",
            "DOROTHEA_BACH1_ABC_TF_ACTIVITY__mean_z",
            "COLLECTRI_BACH1_TF_ACTIVITY__mean_z",
            "DOROTHEA_BACH1_ABC_TF_ACTIVITY__decoupler_ulm",
        ]
        if c in patient.columns
    ]
    heat = patient.loc[order, heat_cols].copy()
    heat = heat.apply(pd.to_numeric, errors="coerce")
    heat = (heat - heat.mean(axis=0)) / heat.std(axis=0, ddof=0).replace(0, np.nan)
    labels = [short_label(c) for c in heat.columns]
    sns.heatmap(heat.T, cmap="vlag", center=0, xticklabels=order, yticklabels=labels, cbar_kws={"label": "patient z"}, ax=ax_b)
    ax_b.tick_params(axis="x", rotation=90, labelsize=6)
    ax_b.tick_params(axis="y", labelsize=6)
    ax_b.set_title("b  External BACH1 activity estimates", loc="left", fontweight="bold")

    ax_c = fig.add_subplot(gs[0, 2])
    plot_tests = tests[tests["score"].isin([c for c in tests["score"] if c.startswith(("KLENJA2025_BACH1", "DOROTHEA_BACH1", "COLLECTRI_BACH1"))])].copy()
    plot_tests = plot_tests.sort_values("median_delta_detected_minus_undetected")
    y = np.arange(plot_tests.shape[0])
    colors = np.where(plot_tests["median_delta_detected_minus_undetected"] >= 0, "#E45756", "#4C78A8")
    ax_c.barh(y, plot_tests["median_delta_detected_minus_undetected"], color=colors)
    ax_c.axvline(0, color="black", lw=0.6)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([short_label(s).replace("\n", " ") for s in plot_tests["score"]], fontsize=6)
    ax_c.set_xlabel("patient median delta\nBACH1-detected - undetected")
    ax_c.set_title("c  Within-patient score shifts", loc="left", fontweight="bold")

    ax_d = fig.add_subplot(gs[1, 0:2])
    bach1_rows = [
        "KLENJA2025_BACH1_INVERSE_ACTIVITY__mean_z",
        "KLENJA2025_BACH1_ACTIVITY_INVERTED__ssgsea_NES",
        "DOROTHEA_BACH1_ABC_TF_ACTIVITY__mean_z",
        "COLLECTRI_BACH1_TF_ACTIVITY__mean_z",
        "DOROTHEA_BACH1_ABC_TF_ACTIVITY__decoupler_ulm",
    ]
    phenotype_cols = [c for c in patient.columns if c.startswith("HALLMARK_") and c.endswith("__mean_z")]
    phenotype_cols += [c for c in patient.columns if c.startswith("CURATED_NRF2") and c.endswith("__mean_z")]
    phenotype_cols += [c for c in patient.columns if c.startswith("LITERATURE_LUNG_BACH1_EFFECTOR") and c.endswith("__mean_z")]
    mat = pd.DataFrame(index=[c for c in bach1_rows if c in patient.columns], columns=phenotype_cols, dtype=float)
    for x in mat.index:
        for ycol in mat.columns:
            row = cor[(cor["x"].eq(x)) & (cor["y"].eq(ycol))]
            if not row.empty:
                mat.loc[x, ycol] = float(row["spearman_rho"].iloc[0])
    row_labels = [short_label(x) for x in mat.index]
    col_labels = [short_label(x).replace("\n", " ") for x in mat.columns]
    sns.heatmap(mat, cmap="vlag", center=0, vmin=-1, vmax=1, xticklabels=col_labels, yticklabels=row_labels, cbar_kws={"label": "Spearman rho"}, ax=ax_d)
    ax_d.tick_params(axis="x", rotation=35, labelsize=6)
    ax_d.tick_params(axis="y", labelsize=6)
    ax_d.set_title("d  Patient-level BACH1 activity versus phenotype signatures", loc="left", fontweight="bold")

    ax_e = fig.add_subplot(gs[1, 2])
    # Show the strongest correlations that are sign-stable under leave-one-patient-out.
    stable = lopo_cor[lopo_cor["sign_stability"].isin(["all_positive", "all_negative"])].copy()
    merged = cor.merge(stable[["x", "y", "sign_stability", "min_rho", "max_rho", "max_p"]], on=["x", "y"], how="inner")
    merged = merged[merged["y"].isin(phenotype_cols) & merged["x"].map(is_bach1_activity_column)].copy()
    merged["abs_rho"] = merged["spearman_rho"].abs()
    merged = merged.sort_values(["abs_rho", "spearman_p"], ascending=[False, True]).head(8)
    yy = np.arange(merged.shape[0])
    if not merged.empty:
        ax_e.errorbar(
            merged["spearman_rho"],
            yy,
            xerr=[merged["spearman_rho"] - merged["min_rho"], merged["max_rho"] - merged["spearman_rho"]],
            fmt="o",
            color="#333333",
            ecolor="#888888",
            lw=1,
        )
        ax_e.axvline(0, color="black", lw=0.6)
        labels = [
            f"{short_label(r['x']).replace(chr(10), ' ')}\nvs {short_label(r['y']).replace(chr(10), ' ')}"
            for _, r in merged.iterrows()
        ]
        ax_e.set_yticks(yy)
        ax_e.set_yticklabels(labels, fontsize=6)
    else:
        ax_e.text(0.5, 0.5, "No sign-stable\nLOPO correlations", ha="center", va="center", transform=ax_e.transAxes)
        ax_e.set_yticks([])
    ax_e.set_xlabel("Spearman rho with LOPO range")
    ax_e.set_title("e  Leave-one-patient-out stability", loc="left", fontweight="bold")

    fig.suptitle(
        "Externally defined BACH1 activity signatures in the 16-patient primary malignant epithelial set",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"external_bach1_activity_primary.{ext}", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"figure5_external_bach1_activity_primary.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: dict,
    tests: pd.DataFrame,
    cor: pd.DataFrame,
    activity_pheno: pd.DataFrame,
    lopo: pd.DataFrame,
    sample_info: pd.DataFrame,
) -> None:
    top_tests = tests.head(12)
    top_cor = activity_pheno.head(20)
    stable = lopo[lopo["sign_stability"].isin(["all_positive", "all_negative"])].copy()
    if not stable.empty and not activity_pheno.empty:
        stable = activity_pheno[["x", "y", "spearman_rho", "spearman_p", "spearman_holm_p", "abs_rho"]].merge(
            stable[["x", "y", "sign_stability", "min_rho", "max_rho", "max_p"]],
            on=["x", "y"],
            how="inner",
        )
    stable = stable.sort_values("abs_rho", ascending=False).head(12) if not stable.empty else stable

    with open(OUT / "external_bach1_activity_report.md", "w") as fh:
        fh.write("# External BACH1 Activity Analysis\n\n")
        fh.write("## Summary\n\n")
        for key, value in summary.items():
            fh.write(f"- {key}: {value}\n")
        fh.write("\n## Sample Information\n\n")
        fh.write(markdown_table(sample_info))
        fh.write("\n\n## Within-Patient BACH1-Detected Score Shifts\n\n")
        fh.write(markdown_table(top_tests))
        fh.write("\n\n## BACH1 Activity Versus Phenotype Correlations\n\n")
        fh.write(markdown_table(top_cor))
        fh.write("\n\n## LOPO Sign-Stable Correlations\n\n")
        if stable.empty:
            fh.write("No patient-level activity/pathway correlations retained the same sign across all leave-one-patient-out runs.\n")
        else:
            fh.write(markdown_table(stable))
        fh.write("\n\n## Interpretation Guardrails\n\n")
        fh.write(
            "- The primary inferential unit is the patient, not the cell.\n"
            "- Klenja-Skudrinja 2025 HMOX1/ZNF469/HTRA3 scores are inverted because the genes are induced by BACH1 depletion.\n"
            "- DoRothEA and CollecTRI activities are database-guided TF-activity estimates, not proof of BACH1 occupancy in these tumours.\n"
            "- Directionally consistent, leave-one-patient-out-stable results should be described as BACH1-associated transcriptional context, not BACH1-driven causality.\n"
            "- The restricted three-patient consensus pySCENIC target set should remain a sensitivity analysis.\n"
        )


def main() -> None:
    ensure_dirs()
    log(f"Loading {INPUT_H5AD}")
    ad = sc.read_h5ad(INPUT_H5AD)
    ad.obs[PATIENT_COL] = ad.obs[PATIENT_COL].astype(str)
    lookup = gene_lookup(ad.var_names)

    sample_info = patient_metadata(ad)
    all_targets, bach1_targets, effector_targets = build_networks(lookup)
    pathway_targets = build_pathway_sets(lookup)

    target_counts = (
        all_targets.groupby("signature")
        .agg(n_targets=("target", "nunique"), n_present=("present_in_primary", "sum"))
        .reset_index()
        .sort_values("signature")
    )
    target_counts.to_csv(TABLE_DIR / "external_bach1_signature_target_counts.csv", index=False)

    all_gene_names = sorted(
        set(bach1_targets["target_in_adata"].astype(str))
        | set(effector_targets["target_in_adata"].astype(str))
        | set(pathway_targets["target_in_adata"].astype(str))
    )
    all_gene_names = [g for g in all_gene_names if g]
    expr = selected_gene_matrix(ad, all_gene_names)
    log(f"Expression matrix for signature scoring: {expr.shape[0]} cells x {expr.shape[1]} genes")

    cell_score_net = pd.concat([bach1_targets, effector_targets, pathway_targets], ignore_index=True, sort=False)
    cell_scores = score_weighted_mean_z(expr, cell_score_net, min_targets=2)
    cell_scores.to_csv(TABLE_DIR / "external_bach1_activity_scores_by_cell.csv.gz", index=True, compression="gzip")

    pb_expr = patient_pseudobulk(expr, ad.obs)
    pb_expr.to_csv(TABLE_DIR / "primary_malignant_epithelial_patient_pseudobulk_mean_log_expression.csv.gz", compression="gzip")
    ss = run_ssgsea(pb_expr, pathway_targets, pd.concat([bach1_targets, effector_targets], ignore_index=True, sort=False))
    dec = run_decoupler_activity(pb_expr, bach1_targets)
    pb_scores = pd.concat([ss, dec], axis=1)

    patient = aggregate_patient_scores(ad, cell_scores, pb_scores)
    deltas, delta_tests, lopo_delta = within_patient_deltas(ad, cell_scores)
    by_dataset = cross_dataset_summary(deltas)
    cor, lopo_cor = patient_correlations(patient)
    activity_pheno = activity_vs_phenotype_correlations(cor)

    summary = {
        "n_primary_malignant_epithelial_cells": int(ad.n_obs),
        "n_patients": int(ad.obs[PATIENT_COL].nunique()),
        "n_samples": int(ad.obs["sample_id"].nunique()),
        "bach1_detected_cells": int(ad.obs[BACH1_GROUP_COL].astype(str).eq("BACH1_high").sum()),
        "bach1_detected_fraction": float(ad.obs[BACH1_GROUP_COL].astype(str).eq("BACH1_high").mean()),
        "klenja_signature_present_targets": int(
            bach1_targets[bach1_targets["signature"].eq("KLENJA2025_BACH1_INVERSE_ACTIVITY")]["target_in_adata"].nunique()
        ),
        "dorothea_bach1_present_targets": int(
            bach1_targets[bach1_targets["signature"].eq("DOROTHEA_BACH1_ABC_TF_ACTIVITY")]["target_in_adata"].nunique()
        ),
        "collectri_bach1_present_targets": int(
            bach1_targets[bach1_targets["signature"].eq("COLLECTRI_BACH1_TF_ACTIVITY")]["target_in_adata"].nunique()
        ),
        "n_phenotype_signatures_scored": int(pathway_targets["signature"].nunique()),
        "strongest_bach1_activity_phenotype_correlation": (
            activity_pheno.iloc[0].to_dict()
            if not activity_pheno.empty and np.isfinite(activity_pheno.iloc[0]["spearman_p"])
            else {}
        ),
        "most_stable_within_patient_delta": (
            delta_tests.iloc[0].to_dict() if not delta_tests.empty else {}
        ),
        "interpretation": (
            "Use external BACH1 signatures as patient-level exploratory activity estimates. "
            "Report database-guided TF activity, lung perturbation signatures, and phenotype associations separately; "
            "do not interpret mixed external signatures as direct validation."
        ),
    }
    with open(OUT / "external_bach1_activity_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    make_figure(patient, delta_tests, cor, lopo_cor)
    write_report(summary, delta_tests, cor, activity_pheno, lopo_cor, sample_info)

    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
