#!/usr/bin/env python3
"""Synchronize final submission materials after Translational Oncology benchmark review.

The script freezes the candidate-prioritisation layer to the latest ATAC
target-support table, fixes the state CNV audit, removes an orphan mutation
analysis sheet, and replaces legacy supplementary figures with current-source
versions. It does not rerun pySCENIC, TCGA, or ATAC primary analyses.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "manuscript_submission_20260914"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
PRIMARY_H5AD = ROOT / "out/bach1_malignant_epithelial_primary/nsclc_malignant_epithelial_primary_bach1_analysis_object.h5ad"
EPITHELIAL_UMAP = ROOT / "out/nature_story_4figures_10kb/source_data/figure_s1_panel_b_epithelial_cnv_burden_umap.csv.gz"
CONSENSUS_CANDIDATES = ROOT / "out/bach1_malignant_epithelial_consensus/tables/nsclc_malignant_epithelial_consensus_bach1_candidate_targets_integrated_evidence.csv"
ATAC_SUPPORT = ROOT / "out/bach1_atac_motif_support_consensus_full_tf_5seed_stable_ge60_ucsc_bedmode_10kb/tables/pyscenic_bach1_targets_with_ucsc_jaspar2026_atac_support.csv"
WORKBOOKS = [
    OUT / "Supplementary_Tables.xlsx",
    OUT / "Supplementary Tables.xlsx",
]

STATE_ORDER = ["Stress_AP1", "Ciliated", "Epithelial_core", "Proliferative", "AT2_like", "Other"]
STATE_DISPLAY = {
    "Stress_AP1": "Stress-associated",
    "Ciliated": "Ciliated-like",
    "Epithelial_core": "Epithelial_core",
    "Proliferative": "Proliferative",
    "AT2_like": "AT2_like",
    "Other": "Other",
}
CNV_COLORS = {
    "Consensus CNV malignant": "#4C72B0",
    "Single-method CNV candidate": "#55A868",
    "Author malignant / no CNV support": "#C44E52",
    "Not malignant by CNV": "#B8B8B8",
}


def set_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")


def simplify_state(x: object) -> str:
    s = str(x)
    for key in ["Stress_AP1", "Proliferative", "Epithelial_core", "Ciliated", "AT2_like"]:
        if key in s:
            return key
    return "Other"


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    for col in out.columns:
        sd = out[col].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out[col] = 0.0
        else:
            out[col] = (out[col] - out[col].mean()) / sd
    return out


def load_primary_obs_and_matrix(genes: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    adata = ad.read_h5ad(PRIMARY_H5AD)
    obs = adata.obs.copy()
    obs["state"] = obs["epi_marker_only_broad_class"].map(simplify_state)
    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    var_index = pd.Series(np.arange(adata.n_vars), index=adata.var_names.astype(str))
    idx_stress = np.where(obs["state"].values == "Stress_AP1")[0]
    idx_rest = np.where(obs["state"].values != "Stress_AP1")[0]
    deltas: dict[str, float] = {}
    for gene in genes:
        if gene in var_index.index:
            gi = int(var_index[gene])
            deltas[gene] = float(np.asarray(X[idx_stress, gi].mean()).ravel()[0] - np.asarray(X[idx_rest, gi].mean()).ravel()[0])
        else:
            deltas[gene] = np.nan
    return obs, deltas


def build_candidate_tier_matrix(obs: pd.DataFrame | None = None) -> pd.DataFrame:
    support = pd.read_csv(ATAC_SUPPORT)
    consensus = pd.read_csv(CONSENSUS_CANDIDATES)
    recurrent = support.loc[pd.to_numeric(support["n_runs_detected"], errors="coerce") >= 4].copy()

    if obs is None:
        obs, stress_delta = load_primary_obs_and_matrix(recurrent["gene"].astype(str).tolist())
    else:
        _, stress_delta = load_primary_obs_and_matrix(recurrent["gene"].astype(str).tolist())

    merged = recurrent.merge(
        consensus[
            [
                "gene",
                "logfoldchanges",
                "pvals_adj",
                "pearson_r",
                "partial_pearson_r",
                "partial_padj",
                "candidate_score",
            ]
        ],
        on="gene",
        how="left",
    )
    merged["stress_ap1_expression_delta"] = merged["gene"].map(stress_delta)
    merged["promoter_support"] = pd.to_numeric(merged["n_promoter_2kb_motif_peak_links"], errors="coerce").fillna(0) > 0
    merged["atac_ge300_context"] = pd.to_numeric(
        merged["n_ucsc_score_ge_300_motif_peak_links_in_window"], errors="coerce"
    ).fillna(0) > 0

    def tier(row: pd.Series) -> str:
        n_runs = int(row["n_runs_detected"])
        n_samples = float(row["n_samples_with_motif_peak_in_window"])
        has_promoter = bool(row["promoter_support"])
        if n_runs == 5 and n_samples >= 5:
            return "Tier 1: 5/5 seeds + 5-sample ATAC"
        if n_runs >= 4 and n_samples >= 5 and has_promoter:
            return "Tier 2: 4/5 seeds + promoter ATAC"
        if n_runs >= 4 and n_samples > 0:
            return "Context-limited: recurrent with limited ATAC"
        return "Context-limited: recurrent without mapped ATAC"

    merged["candidate_tier"] = merged.apply(tier, axis=1)
    tier_order = {
        "Tier 1: 5/5 seeds + 5-sample ATAC": 1,
        "Tier 2: 4/5 seeds + promoter ATAC": 2,
        "Context-limited: recurrent with limited ATAC": 3,
        "Context-limited: recurrent without mapped ATAC": 4,
    }
    merged["_tier_order"] = merged["candidate_tier"].map(tier_order)
    merged = merged.sort_values(
        [
            "_tier_order",
            "n_runs_detected",
            "weighted_importance",
            "n_ucsc_score_ge_300_motif_peak_links_in_window",
        ],
        ascending=[True, False, False, False],
    )
    cols = [
        "gene",
        "candidate_tier",
        "n_runs_detected",
        "weighted_importance",
        "partial_pearson_r",
        "stress_ap1_expression_delta",
        "logfoldchanges",
        "n_ucsc_score_ge_300_motif_peak_links_in_window",
        "n_samples_with_motif_peak_in_window",
        "n_promoter_2kb_motif_peak_links",
        "max_ucsc_motif_score",
        "samples_with_motif_peak_in_window",
        "promoter_support",
    ]
    out = merged[cols].copy()
    out.to_csv(TABLE_DIR / "figure5_candidate_tier_matrix.csv", index=False)
    out.to_csv(TABLE_DIR / "figure4e_candidate_evidence_matrix.csv", index=False)
    return out


def build_state_malignancy_audit(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    obs = obs.copy()
    for state in STATE_ORDER:
        sub = obs.loc[obs["state"].eq(state)].copy()
        if sub.empty:
            continue
        patient_counts = sub["patient"].astype(str).value_counts()
        n = len(sub)
        n_methods = pd.to_numeric(sub["n_cnv_methods_malignant"], errors="coerce").fillna(0)
        rows.append(
            {
                "state": state,
                "display_state": STATE_DISPLAY[state],
                "n_cells": int(n),
                "n_patients": int(sub["patient"].astype(str).nunique()),
                "pct_from_gse131907_author_malignant": 100.0 * float(sub["dataset"].astype(str).eq("GSE131907").mean()),
                "pct_from_gse274934_cnv_supported": 100.0 * float(sub["dataset"].astype(str).eq("GSE274934").mean()),
                "pct_with_any_cnv_method_malignant": 100.0 * float((n_methods >= 1).mean()),
                "pct_with_cnv_consensus_call": 100.0 * float(
                    sub["cnv_consensus_call"].astype(str).eq("consensus_malignant_epithelial").mean()
                ),
                "median_sensitive_cnv_burden": float(pd.to_numeric(sub["sensitive_cnv_burden"], errors="coerce").median()),
                "median_strict_cnv_burden": float(pd.to_numeric(sub["strict_cnv_burden"], errors="coerce").median()),
                "median_adjacent_ref_cnv_burden": float(pd.to_numeric(sub["cnv_burden_adjacent_ref"], errors="coerce").median()),
                "largest_patient": str(patient_counts.index[0]),
                "largest_patient_fraction_pct": 100.0 * float(patient_counts.iloc[0] / n),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "state_specific_malignancy_audit.csv", index=False)
    return out


def plot_supplementary_figure1(state_audit: pd.DataFrame) -> None:
    set_style()
    epi = pd.read_csv(EPITHELIAL_UMAP)
    epi["display_cnv"] = np.select(
        [
            epi["cnv_consensus_call"].astype(str).eq("consensus_malignant_epithelial"),
            epi["cnv_consensus_call"].astype(str).eq("single_cnv_method_malignant_candidate"),
            epi["dataset"].astype(str).eq("GSE131907") & epi["sample_id"].astype(str).str.contains("LUNG_T", na=False),
        ],
        ["Consensus CNV malignant", "Single-method CNV candidate", "Author malignant / no CNV support"],
        default="Not malignant by CNV",
    )

    fig = plt.figure(figsize=(12.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.0], width_ratios=[1.0, 1.25])

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.06, 0.58, 0.22, 0.22, "GSE131907\nAuthor malignant\nn=5,139", "#E8EEF7"),
        (0.06, 0.18, 0.22, 0.22, "GSE274934\nCNV-supported/retained\nn=8,555", "#E7F3EA"),
        (0.42, 0.42, 0.24, 0.24, "Primary malignant\nepithelial set\nn=13,694", "#F8E9E7"),
        (0.75, 0.42, 0.20, 0.24, "Restricted CNV\nconsensus set\nn=4,906", "#F7E9C8"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=color,
                edgecolor="#666666",
                linewidth=0.8,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)
    for start, end in [((0.28, 0.69), (0.42, 0.54)), ((0.28, 0.29), (0.42, 0.50)), ((0.66, 0.54), (0.75, 0.54))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#666666"))
    ax.text(0.50, 0.12, "Counts clarify the final 13,694-cell primary set and the 4,906-cell restricted sensitivity set.", ha="center", va="center", fontsize=7, color="#555555")
    panel_label(ax, "a", x=0.0, y=0.96)

    ax = fig.add_subplot(gs[1, 0])
    for label, color in CNV_COLORS.items():
        sub = epi.loc[epi["display_cnv"].eq(label)]
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=0.8, alpha=0.55, color=color, linewidths=0, label=label, rasterized=True)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=1, handletextpad=0.3)
    sns.despine(ax=ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    plot = state_audit.copy()
    plot["non_consensus_pct"] = 100 - plot["pct_with_cnv_consensus_call"]
    y = np.arange(len(plot))
    ax.barh(y, plot["pct_with_cnv_consensus_call"], color="#4C72B0", label="CNV consensus")
    ax.barh(y, plot["non_consensus_pct"], left=plot["pct_with_cnv_consensus_call"], color="#D0D0D0", label="Other primary-set cells")
    labels = [f"{r.display_state} (n={int(r.n_cells):,})" for r in plot.itertuples(index=False)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cells in state (%)")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax, left=True)
    panel_label(ax, "c")
    save_all(fig, "Supplementary_Figure1_CNV_malignant_epithelial_selection")


def plot_supplementary_figure3(candidate_matrix: pd.DataFrame) -> None:
    set_style()
    support = pd.read_csv(ATAC_SUPPORT)
    thresholds = pd.read_excel(WORKBOOKS[0], sheet_name="S3C_score_thresholds")
    recurrent_counts = support["n_runs_detected"].value_counts().sort_index()
    mapped = int(support["max_ucsc_motif_score"].notna().sum())
    with_window = int((pd.to_numeric(support["n_ucsc_score_ge_300_motif_peak_links_in_window"], errors="coerce").fillna(0) > 0).sum())
    promoter = int((pd.to_numeric(support["n_promoter_2kb_motif_peak_links"], errors="coerce").fillna(0) > 0).sum())

    fig = plt.figure(figsize=(12.2, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.88, 1.15], height_ratios=[0.9, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    labels = [">=3/5\nrecurrent", "Coordinate\nmapped", "Window\nmotif context", "Promoter\ncontext"]
    values = [len(support), mapped, with_window, promoter]
    ax.bar(np.arange(4), values, color=["#CFE6D8", "#DCE6F2", "#55A868", "#4C72B0"])
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Genes")
    for i, v in enumerate(values):
        ax.text(i, v + 1, str(v), ha="center", va="bottom", fontsize=8)
    sns.despine(ax=ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(len(candidate_matrix))
    ax.scatter(
        candidate_matrix["weighted_importance"],
        y,
        s=20 + candidate_matrix["n_ucsc_score_ge_300_motif_peak_links_in_window"].fillna(0) * 2.2,
        c=candidate_matrix["n_runs_detected"],
        cmap="viridis",
        vmin=3,
        vmax=5,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(candidate_matrix["gene"], fontstyle="italic")
    ax.invert_yaxis()
    ax.set_xlabel("Weighted GRNBoost2 importance")
    ax.set_ylabel("")
    ax.text(0.98, 0.04, "Point area: motif links\nColour: seed recurrence", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    sns.despine(ax=ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(thresholds))
    ax.plot(x, thresholds["n_target_genes_supported_in_window"], marker="o", label="Window")
    ax.plot(x, thresholds["n_target_genes_supported_promoter_2kb"], marker="o", label="Promoter")
    ax.set_xticks(x)
    ax.set_xticklabels([f">={int(v)}" for v in thresholds["ucsc_motif_score_threshold"]])
    ax.set_xlabel("UCSC motif-score sensitivity threshold")
    ax.set_ylabel("Supported genes")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    mat = candidate_matrix.set_index("gene")[
        [
            "n_runs_detected",
            "weighted_importance",
            "partial_pearson_r",
            "stress_ap1_expression_delta",
            "n_ucsc_score_ge_300_motif_peak_links_in_window",
            "n_promoter_2kb_motif_peak_links",
            "n_samples_with_motif_peak_in_window",
        ]
    ]
    disp = zscore_columns(mat)
    disp.columns = ["Seeds", "Importance", "Partial r\n(relative z)", "Stress delta", "Motif links", "Promoter", "Samples"]
    sns.heatmap(disp, cmap="vlag", center=0, linewidths=0.3, linecolor="white", cbar_kws={"label": "Column z"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    for tick in ax.get_yticklabels():
        tick.set_fontstyle("italic")
    panel_label(ax, "d")
    save_all(fig, "Supplementary_Figure3_current_candidate_integration")


def update_workbooks(candidate_matrix: pd.DataFrame, state_audit: pd.DataFrame) -> None:
    source = WORKBOOKS[0]
    backup = OUT / "Supplementary_Tables_before_translon2026_sync_20260915.xlsx"
    if not backup.exists():
        shutil.copy2(source, backup)

    all_sheets = pd.read_excel(source, sheet_name=None)
    if "S1B_score_summary" in all_sheets:
        s1b = all_sheets["S1B_score_summary"].copy()
        if "manuscript_role" in s1b.columns:
            s1b["manuscript_role"] = s1b["manuscript_role"].astype(str).str.replace(
                "TCGA Figure 3", "TCGA Figure 4c-d", regex=False
            )
        all_sheets["S1B_score_summary"] = s1b
    if "S4D_resources" in all_sheets:
        resources = all_sheets["S4D_resources"].copy()
        row_text = resources.astype(str).agg(" ".join, axis=1)
        all_sheets["S4D_resources"] = resources.loc[
            ~row_text.str.contains("GO/KEGG|Enrichr|GSEApy|over-representation", case=False, regex=True, na=False)
        ].copy()
    if "S3A_atac_summary" in all_sheets:
        s3a = all_sheets["S3A_atac_summary"].copy()
        if {"parameter", "details"}.issubset(s3a.columns):
            s3a.loc[
                s3a["parameter"].eq("targets_with_window_support"),
                "details",
            ] = "Coordinate-mapped recurrent candidates with at least one motif-overlapping accessible peak in the TSS +/- 10-kb window at any UCSC motif score; thresholded summaries are reported separately in S3C."
            s3a.loc[
                s3a["parameter"].eq("targets_with_promoter_2kb_support"),
                "details",
            ] = "Coordinate-mapped recurrent candidates with at least one motif-overlapping accessible peak within promoter +/- 2 kb at any UCSC motif score; score >=300 promoter support is reported separately in S3C."
        all_sheets["S3A_atac_summary"] = s3a
    all_sheets["S4_candidate_matrix"] = candidate_matrix
    all_sheets["S4_state_malignancy_audit"] = state_audit
    all_sheets.pop("S5_LUAD_mut_context", None)
    for obsolete in ["S5_state_signature", "S5_TCGA_clinical", "S5_TCGA_OS_Cox"]:
        all_sheets.pop(obsolete, None)

    for workbook in WORKBOOKS:
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            for name, df in all_sheets.items():
                safe_name = name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)

    flat = []
    for name, df in all_sheets.items():
        x = df.copy()
        x.insert(0, "source_sheet", name)
        flat.append(x)
    flat_df = pd.concat(flat, ignore_index=True, sort=False)
    flat_df.to_csv(OUT / "Supplementary_Tables.csv", index=False)
    flat_df.to_csv(OUT / "Supplementary_Tables_revision_source_data.csv", index=False)


def main() -> None:
    obs, _ = load_primary_obs_and_matrix([])
    candidate_matrix = build_candidate_tier_matrix(obs)
    state_audit = build_state_malignancy_audit(obs)
    update_workbooks(candidate_matrix, state_audit)
    plot_supplementary_figure1(state_audit)
    print("Candidate tiers:")
    print(candidate_matrix[["gene", "candidate_tier", "n_runs_detected", "n_ucsc_score_ge_300_motif_peak_links_in_window", "n_samples_with_motif_peak_in_window", "n_promoter_2kb_motif_peak_links"]].to_string(index=False))
    print("Updated workbooks and supplementary figures in", OUT)


if __name__ == "__main__":
    main()
