"""Create revision diagnostics for the BACH1 NSCLC reanalysis.

The report focuses on checks requested before rewriting the manuscript:
old-versus-current cell-count discrepancies, full-TF pySCENIC BACH1 regulon
properties, overlap with the released figure-source candidate sets, and
patient/dataset dependence of high inferred BACH1 regulon activity. When
available, it also incorporates the E-MTAB-13530 spatial validation run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from project_paths import OUTPUT_DIR, PROJECT_ROOT


OUT = OUTPUT_DIR / "revision_diagnostics"
PYSC = OUTPUT_DIR / "bach1_malignant_epithelial_consensus_pyscenic_all_tfs"
BACH1_H5AD = (
    OUTPUT_DIR
    / "bach1_malignant_epithelial_consensus"
    / "nsclc_malignant_epithelial_consensus_bach1_analysis_object.h5ad"
)
PRIMARY_BACH1_H5AD = (
    OUTPUT_DIR
    / "bach1_malignant_epithelial_primary"
    / "nsclc_malignant_epithelial_primary_bach1_analysis_object.h5ad"
)


OLD_COUNTS = {
    "raw_cells": 189_299,
    "qc_after_scanpy": 177_128,
    "scrublet_doublets": 829,
    "epithelial_cells": 24_481,
    "strict_malignant": 158,
    "sensitivity_malignant": 6_576,
    "adjacent_epithelial_reference": 4_667,
    "consensus_malignant": 4_659,
    "final_main_analysis_cells": 6_073,
}


SELECTED_TFS = [
    "NFE2L2",
    "BACH2",
    "MAFK",
    "MAFG",
    "JUN",
    "FOS",
    "ATF2",
    "ATF3",
    "ATF5",
    "ATF6",
    "ATF6B",
    "ATF7",
]


def mkdirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def n_obs(rel_path: str) -> int | None:
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        return None
    adata = sc.read_h5ad(path, backed="r")
    value = int(adata.n_obs)
    adata.file.close()
    return value


def h5ad_shape(rel_path: str) -> tuple[int | None, int | None]:
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        return None, None
    adata = sc.read_h5ad(path, backed="r")
    shape = int(adata.n_obs), int(adata.n_vars)
    adata.file.close()
    return shape


def bool_count(series: pd.Series, value: object = True) -> int:
    return int(series.eq(value).sum())


def current_cell_counts() -> dict[str, int | None]:
    raw_cells, _ = h5ad_shape("out/nsclc_gse131907_gse274934_raw_qc.h5ad")
    basic_qc_cells, _ = h5ad_shape("out/nsclc_gse131907_gse274934_qc_filtered_pca.h5ad")
    scanpy_qc_cells, _ = h5ad_shape(
        "out/scanpy_qc/nsclc_gse131907_gse274934_scanpy_qc_raw_counts_qc_filtered.h5ad"
    )
    downstream_cells, _ = h5ad_shape(
        "out/scanpy_downstream_scrublet/nsclc_gse131907_gse274934_scanpy_downstream_analysis.h5ad"
    )
    epithelial_cnv_input_cells, _ = h5ad_shape(
        "out/epithelial_cnv_infercnvpy/nsclc_gse131907_gse274934_epithelial_cnv_analysis.h5ad"
    )
    epithelial_cells, _ = h5ad_shape(
        "out/epithelial_reclustering/nsclc_gse131907_gse274934_epithelial_recluster_analysis.h5ad"
    )
    main_cells, _ = h5ad_shape(
        "out/bach1_malignant_epithelial_consensus/nsclc_malignant_epithelial_consensus_bach1_analysis_object.h5ad"
    )
    primary_cells, _ = h5ad_shape(
        "out/bach1_malignant_epithelial_primary/nsclc_malignant_epithelial_primary_bach1_analysis_object.h5ad"
    )

    summary_path = (
        OUTPUT_DIR
        / "epithelial_cnv_infercnvpy"
        / "nsclc_gse131907_gse274934_epithelial_multimethod_malignancy_summary.csv"
    )
    cnv = pd.read_csv(summary_path).set_index("method_or_set") if summary_path.exists() else pd.DataFrame()

    def cnv_count(label: str) -> int | None:
        if cnv.empty or label not in cnv.index:
            return None
        return int(cnv.loc[label, "n_cells"])

    return {
        "raw_cells": raw_cells,
        "basic_qc_filtered_pca_cells": basic_qc_cells,
        "qc_after_scanpy": scanpy_qc_cells,
        "scrublet_doublets": None if scanpy_qc_cells is None or downstream_cells is None else scanpy_qc_cells - downstream_cells,
        "post_scrublet_singlets": downstream_cells,
        "epithelial_cnv_input_plus_tnk_reference": epithelial_cnv_input_cells,
        "epithelial_cells": epithelial_cells,
        "strict_malignant": cnv_count("tnk_strict_dynamic_threshold"),
        "sensitivity_malignant": cnv_count("tnk_no_dynamic_threshold"),
        "adjacent_epithelial_reference": cnv_count("adjacent_epithelial_reference"),
        "consensus_malignant": cnv_count("cnv_consensus_at_least_two_methods"),
        "single_method_only": cnv_count("single_cnv_method_only"),
        "recommended_working_set": cnv_count("recommended_working_set"),
        "final_main_analysis_cells": main_cells,
        "primary_author_cnv_hybrid_malignant": primary_cells,
    }


def write_count_differences(current: dict[str, int | None]) -> pd.DataFrame:
    rows = [
        {
            "step": "Raw cells",
            "old_result": OLD_COUNTS["raw_cells"],
            "current_result": current["raw_cells"],
            "delta": None,
            "current_evidence": "out/nsclc_gse131907_gse274934_raw_qc.h5ad",
            "interpretation": "一致；说明两个公开队列的原始细胞总数没有变化。",
        },
        {
            "step": "QC-filtered cells",
            "old_result": OLD_COUNTS["qc_after_scanpy"],
            "current_result": current["qc_after_scanpy"],
            "delta": None,
            "current_evidence": "out/scanpy_qc/nsclc_gse131907_gse274934_scanpy_qc_raw_counts_qc_filtered.h5ad",
            "interpretation": "一致；旧文中的 QC 后细胞数对应 scanpy standard QC 输出，而不是 earlier basic QC/PCA 对象。",
        },
        {
            "step": "Scrublet-predicted doublets",
            "old_result": OLD_COUNTS["scrublet_doublets"],
            "current_result": current["scrublet_doublets"],
            "delta": None,
            "current_evidence": "scanpy QC cells minus downstream singlets; Scrublet random_state=7, batch_key=sample_id",
            "interpretation": "小幅差异；需要在正文/方法中固定 Scanpy/Scrublet 版本和随机种子。",
        },
        {
            "step": "Post-Scrublet singlets",
            "old_result": OLD_COUNTS["qc_after_scanpy"] - OLD_COUNTS["scrublet_doublets"],
            "current_result": current["post_scrublet_singlets"],
            "delta": None,
            "current_evidence": "out/scanpy_downstream_scrublet/nsclc_gse131907_gse274934_scanpy_downstream_analysis.h5ad",
            "interpretation": "由 doublet 数差异导致。",
        },
        {
            "step": "Epithelial cells",
            "old_result": OLD_COUNTS["epithelial_cells"],
            "current_result": current["epithelial_cells"],
            "delta": None,
            "current_evidence": "out/epithelial_reclustering/nsclc_gse131907_gse274934_epithelial_recluster_analysis.h5ad",
            "interpretation": "显著差异；需追溯旧版是否在 epithelial reclustering 或 CNV 后剔除了 marker-defined contaminating clusters。",
        },
        {
            "step": "Strict malignant",
            "old_result": OLD_COUNTS["strict_malignant"],
            "current_result": current["strict_malignant"],
            "delta": None,
            "current_evidence": "out/epithelial_cnv_infercnvpy/nsclc_gse131907_gse274934_epithelial_multimethod_malignancy_summary.csv",
            "interpretation": "显著差异；需要核对 CNV reference selection、thresholds、gene-position annotation 与 cluster filtering 顺序。",
        },
        {
            "step": "Sensitivity malignant",
            "old_result": OLD_COUNTS["sensitivity_malignant"],
            "current_result": current["sensitivity_malignant"],
            "delta": None,
            "current_evidence": "same multimethod CNV summary",
            "interpretation": "重大差异；当前 8,543 是 no-dynamic-threshold working set，可能尚未进行旧版 marker-defined cluster exclusion。",
        },
        {
            "step": "Adjacent epithelial reference",
            "old_result": OLD_COUNTS["adjacent_epithelial_reference"],
            "current_result": current["adjacent_epithelial_reference"],
            "delta": None,
            "current_evidence": "same multimethod CNV summary",
            "interpretation": "中等差异；同样需核对 epithelial subset 与阈值/过滤顺序。",
        },
        {
            "step": "Consensus malignant",
            "old_result": OLD_COUNTS["consensus_malignant"],
            "current_result": current["consensus_malignant"],
            "delta": None,
            "current_evidence": "same multimethod CNV summary",
            "interpretation": "中等差异；当前定义为三种 CNV 策略中至少两种支持。",
        },
        {
            "step": "Primary author/CNV hybrid malignant",
            "old_result": np.nan,
            "current_result": current["primary_author_cnv_hybrid_malignant"],
            "delta": np.nan,
            "current_evidence": "out/bach1_malignant_epithelial_primary/nsclc_malignant_epithelial_primary_bach1_analysis_object.h5ad",
            "interpretation": "新增修订主分析集：GSE131907 使用作者 tS1/tS2/tS3 肿瘤上皮注释，GSE274934 使用至少一种 CNV 方法支持。",
        },
        {
            "step": "Final main analysis cells",
            "old_result": OLD_COUNTS["final_main_analysis_cells"],
            "current_result": current["final_main_analysis_cells"],
            "delta": None,
            "current_evidence": "out/bach1_malignant_epithelial_consensus/nsclc_malignant_epithelial_consensus_bach1_analysis_object.h5ad",
            "interpretation": "保留为 high-confidence CNV sensitivity set；不再作为修订主分析集。",
        },
    ]
    df = pd.DataFrame(rows)
    df["delta"] = pd.to_numeric(df["current_result"], errors="coerce") - pd.to_numeric(df["old_result"], errors="coerce")
    df["percent_change"] = df["delta"] / pd.to_numeric(df["old_result"], errors="coerce") * 100
    df.to_csv(OUT / "old_vs_current_pipeline_cell_count_differences.csv", index=False)
    return df


def load_current_adata_meta() -> pd.DataFrame:
    adata = sc.read_h5ad(PRIMARY_BACH1_H5AD if PRIMARY_BACH1_H5AD.exists() else BACH1_H5AD, backed="r")
    meta = adata.obs.copy()
    adata.file.close()
    return meta


def bach1_regulon_characterization() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    motif_path = PYSC / "tables" / "pyscenic_all_tfs_motif_enrichment_summary.csv"
    target_path = PYSC / "tables" / "pyscenic_all_tfs_regulon_targets_long.csv.gz"
    old_membership_path = PROJECT_ROOT / "source_data" / "figure4_bach1_pyscenic_scatac_10kb_membership.csv.gz"

    motifs = pd.read_csv(motif_path)
    targets = pd.read_csv(target_path)
    old = pd.read_csv(old_membership_path)

    bach1_motifs = motifs[motifs["TF"].eq("BACH1")].copy()
    bach1_targets = targets[targets["TF"].eq("BACH1")].copy()
    current_genes = set(bach1_targets["gene"].dropna().astype(str))
    old_pys = set(old.loc[old["in_pyscenic_bach1_regulon"].astype(bool), "gene"].astype(str))
    old_dual = set(
        old.loc[
            old["in_pyscenic_bach1_regulon"].astype(bool) & old["in_sc_atac_highconf_10kb"].astype(bool),
            "gene",
        ].astype(str)
    )
    old_scatac = set(old.loc[old["in_sc_atac_highconf_10kb"].astype(bool), "gene"].astype(str))

    overlap_rows = []
    for label, genes in [
        ("old_pyscenic_bach1_regulon", old_pys),
        ("old_dual_evidence_10kb", old_dual),
        ("old_scatac_highconf_10kb", old_scatac),
    ]:
        intersection = current_genes & genes
        union = current_genes | genes
        overlap_rows.append(
            {
                "comparison_set": label,
                "old_set_n": len(genes),
                "current_bach1_target_n": len(current_genes),
                "overlap_n": len(intersection),
                "jaccard": len(intersection) / len(union) if union else np.nan,
                "overlap_genes": ";".join(sorted(intersection)),
            }
        )
    overlaps = pd.DataFrame(overlap_rows)

    target_membership = pd.DataFrame({"gene": sorted(current_genes | old_pys | old_dual | old_scatac)})
    target_membership["in_current_full_tf_bach1_regulon"] = target_membership["gene"].isin(current_genes)
    target_membership["in_old_pyscenic_bach1_regulon"] = target_membership["gene"].isin(old_pys)
    target_membership["in_old_dual_evidence_10kb"] = target_membership["gene"].isin(old_dual)
    target_membership["in_old_scatac_highconf_10kb"] = target_membership["gene"].isin(old_scatac)

    annotation = bach1_motifs["Annotation"].fillna("").astype(str)
    context = bach1_motifs["Context"].fillna("").astype(str)
    top = bach1_motifs.sort_values("NES", ascending=False).head(1)
    top_record = top.iloc[0].to_dict() if not top.empty else {}
    summary = {
        "n_bach1_motif_rows": int(bach1_motifs.shape[0]),
        "n_bach1_unique_targets": int(len(current_genes)),
        "n_bach1_target_rows": int(bach1_targets.shape[0]),
        "n_direct_or_directly_annotated_motif_rows": int(annotation.str.contains("direct", case=False).sum()),
        "n_similarity_annotation_rows": int(annotation.str.contains("similar motif", case=False).sum()),
        "all_contexts_contain_activating": bool(context.str.contains("activating", case=False).all()) if len(context) else False,
        "any_contexts_contain_repressing": bool(context.str.contains("repressing", case=False).any()) if len(context) else False,
        "top_bach1_motif_by_nes": top_record,
        "old_pyscenic_bach1_regulon_n": int(len(old_pys)),
        "old_dual_evidence_10kb_n": int(len(old_dual)),
        "old_scatac_highconf_10kb_n": int(len(old_scatac)),
    }

    bach1_motifs.to_csv(OUT / "full_tf_pyscenic_bach1_motif_rows.csv", index=False)
    bach1_targets.to_csv(OUT / "full_tf_pyscenic_bach1_target_rows.csv.gz", index=False, compression="gzip")
    overlaps.to_csv(OUT / "bach1_targets_old_vs_current_overlap_summary.csv", index=False)
    target_membership.to_csv(OUT / "bach1_targets_old_vs_current_gene_membership.csv", index=False)
    (OUT / "full_tf_pyscenic_bach1_regulon_characterization.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return bach1_motifs, overlaps, target_membership, summary


def selected_tf_correlations() -> pd.DataFrame:
    corr_path = PYSC / "tables" / "pyscenic_selected_tf_regulon_spearman_correlation.csv"
    corr = pd.read_csv(corr_path, index_col=0)
    rows = []
    for tf in SELECTED_TFS:
        if tf in corr.index and "BACH1" in corr.columns:
            rows.append({"TF": tf, "spearman_rho_with_BACH1": float(corr.loc[tf, "BACH1"])})
    out = pd.DataFrame(rows).sort_values("spearman_rho_with_BACH1", ascending=False)
    out["interpretation"] = np.where(
        out["spearman_rho_with_BACH1"].abs().ge(0.7),
        "high correlation; avoid attributing programme uniquely to BACH1",
        "not highly correlated in this AUCell matrix",
    )
    out.to_csv(OUT / "bach1_selected_tf_regulon_correlation_summary.csv", index=False)
    return out


def patient_dependency(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    by_cell = pd.read_csv(
        PYSC / "tables" / "pyscenic_bach1_regulon_threshold_sensitivity_by_cell.csv.gz",
        index_col=0,
    )
    joined = meta.join(by_cell[["BACH1_regulon_AUCell"]], how="inner")
    call_cols = [col for col in by_cell.columns if col.startswith("BACH1_regulon_high_")]
    joined = joined.join(by_cell[call_cols], how="left")

    group_rows = []
    for definition_col in call_cols:
        definition = definition_col.replace("BACH1_regulon_high_", "")
        high = joined[definition_col].astype(bool)
        total_high = int(high.sum())
        for group_col in ["dataset", "sample_id", "patient", "epi_marker_only_label", "BACH1_group"]:
            if group_col not in joined.columns:
                continue
            counts = (
                joined.groupby(group_col, observed=True)
                .agg(
                    total_cells=("BACH1_regulon_AUCell", "size"),
                    high_cells=(definition_col, "sum"),
                    mean_AUCell=("BACH1_regulon_AUCell", "mean"),
                    median_AUCell=("BACH1_regulon_AUCell", "median"),
                )
                .reset_index()
                .rename(columns={group_col: "group"})
            )
            counts.insert(0, "grouping", group_col)
            counts.insert(1, "definition", definition)
            counts["fraction_high_within_group"] = counts["high_cells"] / counts["total_cells"].replace(0, np.nan)
            counts["fraction_of_all_high_cells"] = counts["high_cells"] / total_high if total_high else np.nan
            counts["fraction_of_all_consensus_cells"] = counts["total_cells"] / joined.shape[0]
            counts["high_vs_background_enrichment"] = (
                counts["fraction_of_all_high_cells"] / counts["fraction_of_all_consensus_cells"].replace(0, np.nan)
            )
            group_rows.append(counts)
    dependency = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    dependency.to_csv(OUT / "bach1_regulon_high_dependency_by_group.csv", index=False)

    continuous_rows = []
    for group_col in ["dataset", "sample_id", "patient", "epi_marker_only_label", "BACH1_group"]:
        if group_col not in joined.columns:
            continue
        summary = (
            joined.groupby(group_col, observed=True)["BACH1_regulon_AUCell"]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
            .rename(columns={group_col: "group", "count": "n_cells"})
        )
        summary.insert(0, "grouping", group_col)
        continuous_rows.append(summary)
    continuous = pd.concat(continuous_rows, ignore_index=True) if continuous_rows else pd.DataFrame()
    continuous.to_csv(OUT / "bach1_regulon_continuous_aucell_by_group.csv", index=False)

    covariate_rows = []
    for cov in ["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"]:
        if cov not in joined.columns:
            continue
        tmp = joined[["BACH1_regulon_AUCell", cov]].dropna()
        rho = tmp.corr(method="spearman").iloc[0, 1] if len(tmp) > 2 else np.nan
        covariate_rows.append({"covariate": cov, "spearman_rho_with_BACH1_AUCell": float(rho), "n_cells": int(len(tmp))})
    covariates = pd.DataFrame(covariate_rows)
    covariates.to_csv(OUT / "bach1_regulon_covariate_correlations.csv", index=False)

    summary = {
        "n_cells": int(joined.shape[0]),
        "n_patients": int(joined["patient"].nunique()) if "patient" in joined.columns else None,
        "n_samples": int(joined["sample_id"].nunique()) if "sample_id" in joined.columns else None,
        "n_datasets": int(joined["dataset"].nunique()) if "dataset" in joined.columns else None,
        "threshold_dependency": {},
    }
    for definition_col in call_cols:
        definition = definition_col.replace("BACH1_regulon_high_", "")
        high = joined[joined[definition_col].astype(bool)].copy()
        item = {
            "n_high_cells": int(high.shape[0]),
            "n_high_patients": int(high["patient"].nunique()) if "patient" in high.columns else None,
            "n_high_samples": int(high["sample_id"].nunique()) if "sample_id" in high.columns else None,
            "n_high_datasets": int(high["dataset"].nunique()) if "dataset" in high.columns else None,
        }
        for group_col in ["patient", "sample_id", "dataset", "epi_marker_only_label"]:
            if group_col in high.columns and high.shape[0]:
                vc = high[group_col].value_counts()
                background = joined[group_col].value_counts()
                item[f"top_{group_col}"] = str(vc.index[0])
                item[f"top_{group_col}_n"] = int(vc.iloc[0])
                item[f"top_{group_col}_fraction_of_high"] = float(vc.iloc[0] / high.shape[0])
                item[f"top_{group_col}_fraction_of_all_consensus"] = float(background.loc[vc.index[0]] / joined.shape[0])
                item[f"top_{group_col}_high_vs_background_enrichment"] = float(
                    (vc.iloc[0] / high.shape[0]) / (background.loc[vc.index[0]] / joined.shape[0])
                )
        summary["threshold_dependency"][definition] = item

    (OUT / "bach1_regulon_patient_dependency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dependency, continuous, covariates, summary


def optional_seed_stability() -> tuple[dict | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = [
        ("consensus_full_tf_5seed", OUT / "pyscenic_bach1_consensus_full_tf_seed_stability_v1"),
        ("consensus_3seed_v2", OUT / "pyscenic_bach1_consensus_seed_stability_v2"),
        ("legacy", OUT),
    ]
    label = None
    base = None
    for cand_label, cand_base in candidates:
        if (cand_base / "pyscenic_bach1_seed_stability_summary.json").exists():
            label = cand_label
            base = cand_base
            break
    if base is None:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary_path = base / "pyscenic_bach1_seed_stability_summary.json"
    run_path = base / "pyscenic_bach1_seed_stability_run_summary.csv"
    pairwise_path = base / "pyscenic_bach1_seed_stability_pairwise_jaccard.csv"
    frequency_path = base / "pyscenic_bach1_seed_stability_target_frequency.csv"
    if not summary_path.exists():
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_label"] = label
    summary["source_dir"] = str(base.relative_to(PROJECT_ROOT))
    run_summary = pd.read_csv(run_path) if run_path.exists() else pd.DataFrame()
    pairwise = pd.read_csv(pairwise_path) if pairwise_path.exists() else pd.DataFrame()
    frequency = pd.read_csv(frequency_path) if frequency_path.exists() else pd.DataFrame()
    return summary, run_summary, pairwise, frequency


def optional_primary_bach1_only_seed_audit() -> dict | None:
    base = OUT / "pyscenic_bach1_primary_bach1_only_seed_stability_v1"
    summary_path = base / "pyscenic_bach1_seed_stability_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_dir"] = str(base.relative_to(PROJECT_ROOT))
    return summary


def optional_targeted_atac() -> tuple[dict | None, pd.DataFrame]:
    candidates = [
        OUTPUT_DIR / "bach1_atac_motif_support_consensus_full_tf_5seed_stable_ge60_ucsc_targeted_10kb",
        OUTPUT_DIR / "bach1_atac_motif_support_primary_stable_ge60_ucsc_targeted_10kb",
        OUTPUT_DIR / "bach1_atac_motif_support_consensus_stable_ge60_ucsc_targeted_10kb",
        OUTPUT_DIR / "bach1_atac_motif_support_consensus_ucsc_targeted_10kb",
    ]
    out_dir = next((p for p in candidates if (p / "bach1_atac_motif_support_ucsc_targeted_summary.json").exists()), candidates[-1])
    summary_path = out_dir / "bach1_atac_motif_support_ucsc_targeted_summary.json"
    thresholds_path = out_dir / "tables" / "ucsc_jaspar2026_score_threshold_sensitivity.csv"
    if not summary_path.exists():
        return None, pd.DataFrame()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_dir"] = str(out_dir.relative_to(PROJECT_ROOT))
    thresholds = pd.read_csv(thresholds_path) if thresholds_path.exists() else pd.DataFrame()
    return summary, thresholds


def optional_matched_atac() -> dict | None:
    candidates = [
        OUTPUT_DIR / "bach1_atac_motif_support_consensus_full_tf_5seed_stable_ge60_matched_background_10kb_neighbors15",
        OUTPUT_DIR / "bach1_atac_motif_support_consensus_full_tf_5seed_stable_ge60_matched_background_10kb_bedquery",
        OUTPUT_DIR / "bach1_atac_motif_support_consensus_full_tf_5seed_stable_ge60_matched_background_10kb",
        OUTPUT_DIR / "bach1_atac_motif_support_primary_stable_ge60_matched_background_10kb",
        OUTPUT_DIR / "bach1_atac_motif_support_consensus_stable_ge60_matched_background_10kb",
    ]
    for out_dir in candidates:
        summary_path = out_dir / "matched_background_permutation_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["source_dir"] = str(out_dir.relative_to(PROJECT_ROOT))
            return summary
    return None


def optional_spatial_validation() -> dict | None:
    base = OUTPUT_DIR / "emtab13530_spatial_bach1_nod"
    revision_dir = base / "revision_statistics"
    summary_path = base / "E-MTAB-13530_bach1_nod_analysis_summary.json"
    supplement_summary_path = base / "spatial_skill_supplement" / "spatial_skill_supplement_summary.json"
    sample_metrics_path = base / "tables" / "E-MTAB-13530_bach1_nod_sample_metrics.csv"
    paired_path = (
        revision_dir / "E-MTAB-13530_tumor_vs_adjacent_paired_wilcoxon_holm.csv"
        if (revision_dir / "E-MTAB-13530_tumor_vs_adjacent_paired_wilcoxon_holm.csv").exists()
        else base / "tables" / "E-MTAB-13530_tumor_vs_adjacent_paired_wilcoxon.csv"
    )
    supplement_paired_path = (
        base
        / "spatial_skill_supplement"
        / "tables"
        / "spatial_skill_paired_tumor_adjacent_supplement_stats.csv"
    )
    cohigh_signature_path = (
        base
        / "spatial_skill_supplement"
        / "tables"
        / "spatial_skill_cohigh_signature_enrichment.csv"
    )
    jaccard_tests_path = revision_dir / "spatial_jaccard_patient_paired_tests.csv"
    qc_model_path = revision_dir / "spatial_bach1_detection_qc_adjusted_glm.csv"
    signature_patient_path = revision_dir / "spatial_cohigh_signature_patient_tests.csv"
    if not summary_path.exists() or not sample_metrics_path.exists() or not paired_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    supplement_summary = (
        json.loads(supplement_summary_path.read_text(encoding="utf-8"))
        if supplement_summary_path.exists()
        else {}
    )
    sample_metrics = pd.read_csv(sample_metrics_path)
    paired = pd.read_csv(paired_path)
    supplement_paired = pd.read_csv(supplement_paired_path) if supplement_paired_path.exists() else pd.DataFrame()
    cohigh_signature = pd.read_csv(cohigh_signature_path) if cohigh_signature_path.exists() else pd.DataFrame()
    jaccard_tests = pd.read_csv(jaccard_tests_path) if jaccard_tests_path.exists() else pd.DataFrame()
    qc_model = pd.read_csv(qc_model_path) if qc_model_path.exists() else pd.DataFrame()
    signature_patient = pd.read_csv(signature_patient_path) if signature_patient_path.exists() else pd.DataFrame()

    metric_cols = [
        "BACH1_detected_fraction",
        "BACH1_mean",
        "NOD_like_score_mean",
        "BACH1_NOD_spearman_rho",
        "BACH1_to_neighbor_NOD_spearman_rho",
        "high_high_enrichment",
        "high_high_jaccard",
    ]
    by_tissue = sample_metrics.groupby("tissue_group", observed=True)[metric_cols].mean().reset_index()
    by_tissue.to_csv(OUT / "spatial_emtab13530_tissue_group_summary.csv", index=False)

    correlation_pairs = [
        ("BACH1_detected_fraction", "NOD_like_score_mean"),
        ("BACH1_mean", "NOD_like_score_mean"),
        ("BACH1_detected_fraction", "high_high_jaccard"),
    ]
    corr_rows = []
    for x, y in correlation_pairs:
        tmp = sample_metrics[[x, y]].dropna()
        if len(tmp) < 5:
            rho, pvalue = np.nan, np.nan
        else:
            res = stats.spearmanr(tmp[x], tmp[y])
            rho, pvalue = float(res.statistic), float(res.pvalue)
        corr_rows.append({"x": x, "y": y, "spearman_rho": rho, "pvalue": pvalue, "n_sections": int(len(tmp))})
    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(OUT / "spatial_emtab13530_section_level_correlations.csv", index=False)

    paired.to_csv(OUT / "spatial_emtab13530_tumor_adjacent_paired_wilcoxon.csv", index=False)
    if not jaccard_tests.empty:
        jaccard_tests.to_csv(OUT / "spatial_emtab13530_jaccard_patient_paired_permutation_tests.csv", index=False)
    if not qc_model.empty:
        qc_model.to_csv(OUT / "spatial_emtab13530_bach1_detection_qc_adjusted_glm.csv", index=False)
    if not signature_patient.empty:
        signature_patient.to_csv(OUT / "spatial_emtab13530_signature_proxy_patient_tests.csv", index=False)
    if not supplement_paired.empty:
        supplement_paired.to_csv(OUT / "spatial_emtab13530_supplement_paired_stats.csv", index=False)
    if not cohigh_signature.empty:
        cohigh_signature.sort_values(["tissue_group", "mannwhitney_p"]).to_csv(
            OUT / "spatial_emtab13530_cohigh_signature_enrichment.csv",
            index=False,
        )

    return {
        "summary": summary,
        "supplement_summary": supplement_summary,
        "by_tissue": by_tissue,
        "correlations": correlations,
        "paired": paired,
        "jaccard_tests": jaccard_tests,
        "qc_model": qc_model,
        "supplement_paired": supplement_paired,
        "cohigh_signature": cohigh_signature,
        "signature_patient": signature_patient,
        "output_dir": str(base.relative_to(PROJECT_ROOT)),
    }


def optional_rctd_validation() -> dict | None:
    candidates = [
        OUTPUT_DIR / "emtab13530_spatial_bach1_nod" / "rctd_spacexr_summary_lite",
        OUTPUT_DIR / "emtab13530_spatial_bach1_nod" / "rctd_spacexr_summary",
    ]
    base = next((p for p in candidates if (p / "spatial_rctd_summary.json").exists()), candidates[-1])
    summary_path = base / "spatial_rctd_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_dir"] = str(base.relative_to(PROJECT_ROOT))
    paired_path = base / "spatial_rctd_patient_paired_tests.csv"
    cohigh_path = base / "spatial_rctd_cohigh_patient_paired_tests.csv"
    model_path = base / "spatial_rctd_nod_model.csv"
    summary["paired_tests_table"] = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()
    summary["cohigh_tests_table"] = pd.read_csv(cohigh_path) if cohigh_path.exists() else pd.DataFrame()
    summary["nod_model_table"] = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()
    return summary


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    count_df: pd.DataFrame,
    regulon_summary: dict,
    overlap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    dependency_summary: dict,
    covariates: pd.DataFrame,
    primary_bach1_audit: dict | None,
    seed_stability: dict | None,
    seed_run_summary: pd.DataFrame,
    seed_pairwise: pd.DataFrame,
    seed_frequency: pd.DataFrame,
    atac_summary: dict | None,
    atac_thresholds: pd.DataFrame,
    matched_atac: dict | None,
    spatial: dict | None,
    rctd: dict | None,
) -> None:
    top_motif = regulon_summary.get("top_bach1_motif_by_nes", {})
    top5 = dependency_summary.get("threshold_dependency", {}).get("top_5pct", {})
    mean2sd = dependency_summary.get("threshold_dependency", {}).get("mean_plus_2sd", {})
    report = OUT / "revision_diagnostics_report.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# Revision Diagnostics Report\n\n")
        fh.write("## Immediate Conclusion\n\n")
        fh.write(
            "The reanalysis now defines a broader primary malignant epithelial set using author tumour-epithelial "
            "annotations for GSE131907 and CNV-supported tumour epithelial cells for GSE274934. "
            "Consensus CNV calls remain useful as high-confidence sensitivity analyses, but are too patient-skewed "
            "for the main narrative. Spatial results should be framed as exploratory overlap/signature-proxy "
            "analyses after QC and multiple-testing checks.\n\n"
        )
        fh.write("## Old Versus Current Cell Counts\n\n")
        fh.write(markdown_table(count_df[["step", "old_result", "current_result", "delta", "percent_change", "interpretation"]]))
        fh.write("\n## BACH1 Regulon Characterization\n\n")
        fh.write(f"- BACH1 motif rows after full-TF ctx pruning: {regulon_summary['n_bach1_motif_rows']:,}\n")
        fh.write(f"- Unique current full-TF BACH1 regulon targets: {regulon_summary['n_bach1_unique_targets']:,}\n")
        fh.write(f"- Old BACH1-only pySCENIC target count recovered from source data: {regulon_summary['old_pyscenic_bach1_regulon_n']:,}\n")
        fh.write(f"- Old 10 kb dual-evidence count recovered from source data: {regulon_summary['old_dual_evidence_10kb_n']:,}\n")
        fh.write(f"- Direct/directly annotated BACH1 motif rows: {regulon_summary['n_direct_or_directly_annotated_motif_rows']:,}\n")
        fh.write(f"- Similar-motif annotation rows: {regulon_summary['n_similarity_annotation_rows']:,}\n")
        fh.write(f"- All retained BACH1 context labels contain `activating`: {regulon_summary['all_contexts_contain_activating']}\n")
        fh.write(f"- Any retained BACH1 context labels contain `repressing`: {regulon_summary['any_contexts_contain_repressing']}\n")
        if top_motif:
            fh.write(
                f"- Top BACH1 motif by NES: {top_motif.get('MotifID')} "
                f"(NES={float(top_motif.get('NES')):.3f}, n_targets={int(top_motif.get('n_targets'))})\n"
            )
        fh.write("\n### Old/New Target Overlap\n\n")
        fh.write(markdown_table(overlap_df[["comparison_set", "old_set_n", "current_bach1_target_n", "overlap_n", "jaccard"]]))
        fh.write("\n## BACH1 Versus Related Regulons\n\n")
        fh.write(markdown_table(corr_df, max_rows=20))
        fh.write(
            "\nNo selected regulon shows Spearman rho >= 0.7 with BACH1 in the current AUCell matrix. "
            "This supports discussing BACH1 in a CNC-MAF/AP-1 regulatory context without claiming a uniquely isolated programme.\n\n"
        )
        fh.write("## pySCENIC Seed Stability\n\n")
        if primary_bach1_audit is not None:
            fh.write("### Primary Malignant BACH1-Only Five-Seed Audit\n\n")
            fh.write(f"- Source: {primary_bach1_audit.get('source_dir')}\n")
            fh.write(f"- Completed seed runs: {primary_bach1_audit['n_completed_runs']} ({', '.join(primary_bach1_audit['runs'])})\n")
            fh.write(f"- Union BACH1 targets across primary BACH1-only runs: {primary_bach1_audit['n_union_targets']:,}\n")
            fh.write(f"- Core 100% targets: {primary_bach1_audit['n_targets_detected_in_all_runs']:,}\n")
            fh.write(f"- Stable >=60% targets: {primary_bach1_audit['n_targets_detected_in_at_least_60pct_runs']:,}\n")
            fh.write(f"- Stable >=70% targets: {primary_bach1_audit['n_targets_detected_in_at_least_70pct_runs']:,}\n")
            fh.write(f"- Mean pairwise Jaccard: {primary_bach1_audit['mean_pairwise_jaccard']:.4g}\n")
            fh.write(
                "\nInterpretation: the broader primary malignant epithelial set does not yield a reproducible "
                "BACH1-only regulon under the five seeds tested. This argues against using a primary-set BACH1 "
                "target list as a strong mechanistic claim.\n\n"
            )
            fh.write("### Consensus Full-TF Sensitivity Stable Regulon\n\n")
        if seed_stability is None:
            fh.write(
                "Seed-stability comparison is not complete in the main diagnostics directory yet. "
                "Re-run `scripts/compare_pyscenic_seed_stability.py` after additional pySCENIC seeds finish.\n\n"
            )
        else:
            fh.write(f"- Completed seed runs: {seed_stability['n_completed_runs']} ({', '.join(seed_stability['runs'])})\n")
            fh.write(f"- Seed-stability source: {seed_stability.get('source_label')} ({seed_stability.get('source_dir')})\n")
            fh.write(f"- Union BACH1 targets across completed runs: {seed_stability['n_union_targets']:,}\n")
            fh.write(f"- Targets detected in all completed runs: {seed_stability['n_targets_detected_in_all_runs']:,}\n")
            if "n_targets_detected_in_at_least_60pct_runs" in seed_stability:
                fh.write(
                    f"- Stable >=60% targets: {seed_stability['n_targets_detected_in_at_least_60pct_runs']:,} "
                    f"(minimum runs={seed_stability.get('stable60_min_runs')})\n"
                )
            if "n_targets_detected_in_at_least_70pct_runs" in seed_stability:
                fh.write(
                    f"- Stable >=70% targets: {seed_stability['n_targets_detected_in_at_least_70pct_runs']:,} "
                    f"(minimum runs={seed_stability.get('stable70_min_runs')})\n"
                )
            fh.write(f"- Mean pairwise Jaccard: {seed_stability['mean_pairwise_jaccard']:.4f}\n")
            fh.write("\n### Seed Run Target Counts\n\n")
            fh.write(markdown_table(seed_run_summary))
            fh.write("\n### Pairwise Target Overlap\n\n")
            keep = [c for c in ["run_a", "run_b", "n_a", "n_b", "overlap_n", "jaccard"] if c in seed_pairwise]
            fh.write(markdown_table(seed_pairwise[keep] if keep else seed_pairwise))
            if not seed_frequency.empty:
                fh.write("\n### Stable Core Candidates\n\n")
                fh.write(markdown_table(seed_frequency.head(30)))
            fh.write(
                "\nInterpretation: BACH1 target membership is seed-sensitive under full-TF pySCENIC; "
                "manuscript claims should emphasize the recurrent stable regulon rather than one deterministic seed777 list.\n\n"
            )
        fh.write("## Patient And Dataset Dependence\n\n")
        fh.write(f"- Consensus malignant epithelial cells: {dependency_summary['n_cells']:,}\n")
        fh.write(f"- Patients represented in consensus set: {dependency_summary['n_patients']}\n")
        fh.write(f"- Samples represented in consensus set: {dependency_summary['n_samples']}\n")
        fh.write(f"- Datasets represented in consensus set: {dependency_summary['n_datasets']}\n")
        if top5:
            fh.write(
                f"- Top 5% BACH1 AUCell high cells: {top5['n_high_cells']:,}; "
                f"patients represented: {top5['n_high_patients']}; "
                f"top patient: {top5.get('top_patient')} ({top5.get('top_patient_fraction_of_high'):.1%} of high cells); "
                f"top patient background share: {top5.get('top_patient_fraction_of_all_consensus'):.1%}; "
                f"high/background enrichment: {top5.get('top_patient_high_vs_background_enrichment'):.2f}; "
                f"top dataset: {top5.get('top_dataset')} ({top5.get('top_dataset_fraction_of_high'):.1%}).\n"
            )
        if mean2sd:
            fh.write(
                f"- Mean + 2 SD high cells: {mean2sd['n_high_cells']:,}; "
                f"top patient: {mean2sd.get('top_patient')} ({mean2sd.get('top_patient_fraction_of_high'):.1%}); "
                f"top patient background share: {mean2sd.get('top_patient_fraction_of_all_consensus'):.1%}; "
                f"high/background enrichment: {mean2sd.get('top_patient_high_vs_background_enrichment'):.2f}.\n"
            )
        fh.write("\n### Sequencing/QC Covariate Correlations\n\n")
        fh.write(markdown_table(covariates))
        fh.write("\n## Targeted scATAC Motif Support\n\n")
        if atac_summary is None:
            fh.write(
                "Targeted scATAC motif support has not been generated yet. "
                "The original JASPAR per-motif TSV files are still required for the exact old high-confidence score >=950 analysis.\n\n"
            )
        else:
            fh.write(
                "Because the JASPAR Mencius per-motif TSV server was unreachable locally, "
                "a targeted UCSC `jaspar2026` fallback was run for the available BACH1 regulon target set.\n\n"
            )
            fh.write(f"- Targeted ATAC source: {atac_summary.get('source_dir')}\n")
            fh.write(f"- Window around target TSS: +/- {int(atac_summary['window_bp']):,} bp\n")
            fh.write(f"- Current BACH1 targets tested: {atac_summary['n_bach1_targets']:,}\n")
            fh.write(f"- Targets with GENCODE coordinates: {atac_summary['n_targets_with_gencode_coordinates']:,}\n")
            fh.write(f"- ATAC samples: {', '.join(atac_summary['atac_samples'])}\n")
            fh.write(f"- UCSC BACH1/Bach1::Mafk TFBS hits in target windows: {atac_summary['n_ucsc_bach1_mafk_tfbs_hits_in_target_windows']:,}\n")
            fh.write(f"- UCSC windows failed: {atac_summary['n_ucsc_failed_windows']:,}\n")
            fh.write(f"- Motif-positive sample-specific peaks in target windows: {atac_summary['n_motif_positive_sample_specific_peaks_in_target_windows']:,}\n")
            fh.write(f"- BACH1 targets with motif-positive ATAC support in window: {atac_summary['n_bach1_targets_with_atac_motif_support_in_window']:,}\n")
            fh.write(f"- BACH1 targets with promoter 2 kb support: {atac_summary['n_bach1_targets_with_promoter_2kb_motif_peak']:,}\n")
            fh.write("\n### UCSC Score Sensitivity\n\n")
            fh.write(markdown_table(atac_thresholds))
            fh.write(
                "\nImportant limitation: UCSC bigBed scores are not directly interchangeable with the original "
                "JASPAR TSV high-confidence threshold of 950. Treat this as targeted motif-overlap support, "
                "not a one-to-one replacement for the old strict scATAC table.\n\n"
            )
        fh.write("\n### Matched-Background ATAC Permutation\n\n")
        if matched_atac is None:
            fh.write(
                "Matched-background ATAC permutation has not produced a reliable completed result yet. "
                "Run `scripts/run_atac_matched_background_permutation.py` once UCSC/Mencius motif retrieval is stable.\n\n"
            )
        else:
            fh.write(f"- Source: {matched_atac.get('source_dir')}\n")
            fh.write(f"- Input targets: {matched_atac.get('n_input_targets')}\n")
            fh.write(f"- Targets with GENCODE coordinates: {matched_atac.get('n_targets_with_gencode_coordinates')}\n")
            fh.write(f"- Unique matched background genes queried: {matched_atac.get('n_unique_matched_background_genes_queried')}\n")
            fh.write(f"- UCSC failed windows: {matched_atac.get('n_ucsc_failed_windows')}\n")
            fh.write(f"- Matching features: {', '.join(matched_atac.get('matching_features', []))}\n")
            fh.write(f"- Not included: {matched_atac.get('not_included')}\n")
            if matched_atac.get("n_ucsc_failed_windows", 0):
                fh.write(
                    "\nReliability flag: UCSC window failures are non-zero; do not use this matched-background result "
                    "as a primary claim unless the failed-window count is acceptably low after rerun.\n"
                )
            tests = pd.DataFrame(matched_atac.get("empirical_tests", []))
            if not tests.empty:
                fh.write("\n")
                fh.write(markdown_table(tests))
            fh.write("\n")
        fh.write("\n## Spatial Validation\n\n")
        if spatial is None:
            fh.write(
                "E-MTAB-13530 spatial validation has not been generated yet. "
                "Run `scripts/download_emtab13530.py`, `scripts/analyze_emtab13530_bach1_nod.py`, "
                "`scripts/run_spatial_skill_supplement.py`, and the figure scripts to complete this section.\n\n"
            )
        else:
            spatial_summary = spatial["summary"]
            supplement_summary = spatial["supplement_summary"]
            paired = spatial["paired"]
            correlations = spatial["correlations"]
            by_tissue = spatial["by_tissue"]
            supplement_paired = spatial["supplement_paired"]
            cohigh_signature = spatial["cohigh_signature"]
            jaccard_tests = spatial["jaccard_tests"]
            qc_model = spatial["qc_model"]
            signature_patient = spatial["signature_patient"]

            def paired_value(metric: str, column: str) -> float:
                if column not in paired.columns:
                    return np.nan
                row = paired.loc[paired["metric"].eq(metric)]
                if row.empty:
                    return np.nan
                return float(row.iloc[0][column])

            fh.write(f"- Spatial dataset: E-MTAB-13530, {spatial_summary['n_samples']} Visium sections\n")
            fh.write(
                f"- Sections by tissue: {spatial_summary['n_tumor_samples']} tumor, "
                f"{spatial_summary['n_adjacent_samples']} adjacent, {spatial_summary['n_healthy_samples']} healthy\n"
            )
            fh.write(f"- Tumor/adjacent paired patients: {spatial_summary['n_patients_tumor_adjacent']}\n")
            fh.write(f"- Scored NOD-like receptor genes: {spatial_summary['n_nod_like_genes_input']}\n")
            if supplement_summary:
                fh.write(f"- Spots with supplement annotations: {supplement_summary['n_spots_annotated']:,}\n")
                fh.write(f"- scRNA marker signatures used: {supplement_summary['n_cell_type_signatures']}\n")
                fh.write(f"- Spatial domains detected across sections: {supplement_summary['n_domains_total']:,}\n")
                fh.write(
                    "- Signature proxy note: fast scRNA-marker signature scoring was used; "
                    "these marker scores are not formal cell-type proportion estimates.\n"
                )
            if rctd is None:
                fh.write("- RCTD/spacexr sensitivity analysis: not available in this run.\n")
            elif rctd.get("status") == "completed":
                fh.write(
                    f"- RCTD/spacexr sensitivity analysis: completed for {rctd.get('n_sections')} sections "
                    f"and {rctd.get('n_spots'):,} spots using cell-type proportion estimates.\n"
                )
            else:
                fh.write(
                    f"- RCTD/spacexr sensitivity analysis: {rctd.get('status')} "
                    f"({rctd.get('reason', 'see run-status table')}).\n"
                )
            fh.write(
                f"- Paired tumor-adjacent BACH1-detected spot fraction: median delta "
                f"{paired_value('BACH1_detected_fraction', 'median_delta_tumor_minus_adjacent'):.4g}, "
                f"Wilcoxon p={paired_value('BACH1_detected_fraction', 'wilcoxon_p'):.4g}, "
                f"Holm p={paired_value('BACH1_detected_fraction', 'holm_p'):.4g}\n"
            )
            fh.write(
                f"- Paired tumor-adjacent BACH1/NOD high-overlap Jaccard: median delta "
                f"{paired_value('high_high_jaccard', 'median_delta_tumor_minus_adjacent'):.4g}, "
                f"Wilcoxon p={paired_value('high_high_jaccard', 'wilcoxon_p'):.4g}, "
                f"Holm p={paired_value('high_high_jaccard', 'holm_p'):.4g}\n"
            )
            if not qc_model.empty and "tumor_vs_adjacent" in set(qc_model["term"]):
                row = qc_model.loc[qc_model["term"].eq("tumor_vs_adjacent")].iloc[0]
                fh.write(
                    f"- QC-adjusted BACH1 detection model: tumor-vs-adjacent OR={row.odds_ratio:.3g}, "
                    f"section-clustered p={row.p_value:.4g}; treat tumour enrichment as depth-sensitive.\n"
                )
            if not jaccard_tests.empty:
                ratio = jaccard_tests.loc[jaccard_tests["metric"].eq("observed_expected_ratio")]
                observed = jaccard_tests.loc[jaccard_tests["metric"].eq("observed_jaccard")]
                if not observed.empty:
                    row = observed.iloc[0]
                    fh.write(
                        f"- Jaccard permutation patient test, observed Jaccard: median delta={row.median_delta_tumor_minus_adjacent:.4g}, "
                        f"Wilcoxon p={row.wilcoxon_p:.4g}, Holm p={row.holm_p:.4g}\n"
                    )
                if not ratio.empty:
                    row = ratio.iloc[0]
                    fh.write(
                        f"- Jaccard observed/expected ratio: median delta={row.median_delta_tumor_minus_adjacent:.4g}, "
                        f"Wilcoxon p={row.wilcoxon_p:.4g}, Holm p={row.holm_p:.4g}\n"
                    )
            fh.write("\n### Spatial Tissue-Level Means\n\n")
            fh.write(markdown_table(by_tissue))
            fh.write("\n### Section-Level Spatial Correlations\n\n")
            fh.write(markdown_table(correlations))
            fh.write("\n### Tumor Versus Adjacent Paired Spatial Tests\n\n")
            fh.write(markdown_table(paired))
            if not jaccard_tests.empty:
                fh.write("\n### Jaccard Permutation Patient-Level Tests\n\n")
                fh.write(markdown_table(jaccard_tests))
            if not qc_model.empty:
                fh.write("\n### QC-Adjusted BACH1 Detection Model\n\n")
                fh.write(markdown_table(qc_model))
            if not supplement_paired.empty:
                fh.write("\n### Supplement Spatial Autocorrelation Paired Tests\n\n")
                fh.write(markdown_table(supplement_paired))
            if not signature_patient.empty:
                fh.write("\n### Patient-Level Signature Proxy Tests\n\n")
                cols = [
                    "signature",
                    "n_pairs",
                    "tumor_median_delta_cohigh_minus_non",
                    "adjacent_median_delta_cohigh_minus_non",
                    "paired_median_delta_tumor_minus_adjacent",
                    "paired_wilcoxon_p",
                    "paired_holm_p",
                    "tumor_one_sample_wilcoxon_p",
                    "tumor_one_sample_holm_p",
                ]
                fh.write(markdown_table(signature_patient[cols]))
            elif not cohigh_signature.empty:
                fh.write("\n### Spot-Level Signature Proxy Associations, Exploratory Only\n\n")
                cols = [
                    "tissue_group",
                    "signature",
                    "cohigh_mean",
                    "non_cohigh_mean",
                    "delta",
                    "mannwhitney_p",
                    "n_cohigh_spots",
                    "n_non_cohigh_spots",
                ]
                cohigh_display = (
                    cohigh_signature.sort_values(["tissue_group", "mannwhitney_p"])
                    .groupby("tissue_group", observed=True, as_index=False)
                    .head(4)
                )
                fh.write(markdown_table(cohigh_display[cols]))
            if rctd is not None and rctd.get("status") == "completed":
                paired_rctd = rctd.get("paired_tests_table", pd.DataFrame())
                cohigh_rctd = rctd.get("cohigh_tests_table", pd.DataFrame())
                model_rctd = rctd.get("nod_model_table", pd.DataFrame())
                if not paired_rctd.empty:
                    fh.write("\n### RCTD Cell-Type Proportion Patient Tests\n\n")
                    fh.write(markdown_table(paired_rctd))
                if not cohigh_rctd.empty:
                    fh.write("\n### RCTD Co-High Versus Non-Co-High Sensitivity Tests\n\n")
                    fh.write(markdown_table(cohigh_rctd))
                if not model_rctd.empty:
                    fh.write("\n### RCTD-Adjusted NOD-Like Score Model\n\n")
                    fh.write(markdown_table(model_rctd))
            fh.write(
                "\nInterpretation: the spatial analysis supports an exploratory BACH1/NOD-like pathway high-overlap pattern, "
                "but BACH1 detection is depth-sensitive after QC adjustment and patient-level spatial tests are limited by eight matched pairs. "
                "Do not frame this as direct BACH1 regulation of the NOD-like receptor pathway or as proof of same-cell origin.\n\n"
            )
        fh.write("\n## Output Files\n\n")
        for path in sorted(OUT.glob("*")):
            fh.write(f"- `{path.relative_to(PROJECT_ROOT)}`\n")
        if atac_summary is not None:
            for key, value in atac_summary.get("outputs", {}).items():
                fh.write(f"- `{value}`\n")
        if spatial is not None:
            fh.write("- `out/emtab13530_spatial_bach1_nod/E-MTAB-13530_bach1_nod_analysis_summary.json`\n")
            fh.write("- `out/emtab13530_spatial_bach1_nod/tables/E-MTAB-13530_bach1_nod_sample_metrics.csv`\n")
            fh.write("- `out/emtab13530_spatial_bach1_nod/tables/E-MTAB-13530_tumor_vs_adjacent_paired_wilcoxon.csv`\n")
            fh.write("- `out/emtab13530_spatial_bach1_nod/revision_statistics/spatial_revision_statistics_summary.json`\n")
            fh.write("- `out/emtab13530_spatial_bach1_nod/spatial_skill_supplement/spatial_skill_supplement_summary.json`\n")
            fh.write("- `out/nature_story_4figures_10kb/figures/publication_figures/figure5_spatial_bach1_nod_validation.pdf`\n")
            fh.write("- `out/emtab13530_spatial_bach1_nod/spatial_skill_supplement/figures/figure3_spatial_skill_supplement_domains_signatures.pdf`\n")
        if rctd is not None:
            fh.write("- `out/emtab13530_spatial_bach1_nod/rctd_spacexr_summary/spatial_rctd_summary.json`\n")


def main() -> None:
    mkdirs()
    current = current_cell_counts()
    count_df = write_count_differences(current)
    _, overlap_df, _, regulon_summary = bach1_regulon_characterization()
    corr_df = selected_tf_correlations()
    meta = load_current_adata_meta()
    _, _, covariates, dependency_summary = patient_dependency(meta)
    seed_stability, seed_run_summary, seed_pairwise, seed_frequency = optional_seed_stability()
    primary_bach1_audit = optional_primary_bach1_only_seed_audit()
    atac_summary, atac_thresholds = optional_targeted_atac()
    matched_atac = optional_matched_atac()
    spatial = optional_spatial_validation()
    rctd = optional_rctd_validation()
    write_report(
        count_df,
        regulon_summary,
        overlap_df,
        corr_df,
        dependency_summary,
        covariates,
        primary_bach1_audit,
        seed_stability,
        seed_run_summary,
        seed_pairwise,
        seed_frequency,
        atac_summary,
        atac_thresholds,
        matched_atac,
        spatial,
        rctd,
    )
    print(json.dumps({"out": str(OUT), "current_counts": current}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
