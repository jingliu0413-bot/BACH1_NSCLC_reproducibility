from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from project_paths import DATA_DIR, OUTPUT_DIR

BASE_DIR = OUTPUT_DIR
INPUT_H5AD = (
    BASE_DIR
    / "scanpy_downstream_scrublet"
    / "nsclc_gse131907_gse274934_scanpy_downstream_analysis.h5ad"
)
CNV_CALLS = (
    BASE_DIR
    / "epithelial_cnv_infercnvpy"
    / "nsclc_gse131907_gse274934_epithelial_multimethod_malignancy_calls.csv.gz"
)
GSE131907_AUTHOR_ANNOTATION = (
    DATA_DIR / "GSE131907" / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"
)
OUT_DIR = BASE_DIR / "epithelial_reclustering"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PREFIX = "nsclc_gse131907_gse274934_epithelial_recluster"
RANDOM_STATE = 7


PROGRAM_MARKERS = {
    "Epithelial_core": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "CLDN4", "KRT7"],
    "AT1_like": ["AGER", "PDPN", "CAV1", "CAV2", "HOPX", "RTKN2", "CLIC5"],
    "AT2_like": ["SFTPA1", "SFTPA2", "SFTPB", "SFTPC", "SLC34A2", "ABCA3", "NAPSA", "LAMP3"],
    "Club_secretory": ["SCGB1A1", "SCGB3A1", "SCGB3A2", "CYP2F1", "BPIFB1"],
    "Ciliated": ["FOXJ1", "TPPP3", "PIFO", "DNAH5", "DNAI1", "CAPS"],
    "Basal_squamous": ["KRT5", "KRT14", "KRT15", "KRT6A", "KRT6B", "TP63"],
    "Goblet_mucous": ["MUC5AC", "MUC5B", "SPDEF", "AGR2", "TFF3"],
    "EMT_stromal_like": ["VIM", "FN1", "ITGA5", "ZEB1", "ZEB2", "SNAI2", "COL1A1"],
    "Proliferative": ["MKI67", "TOP2A", "CENPF", "UBE2C", "STMN1", "PCLAF", "PCNA"],
    "Stress_AP1": ["FOS", "JUN", "JUNB", "FOSB", "HSPA1A", "HSPA1B", "HSPB1"],
    "Immune_contamination": ["PTPRC", "LYZ", "CD3D", "MS4A1", "NKG7", "LST1"],
}

CELL_CYCLE_GENES = {
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
    "CDCA7", "DTL", "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1",
    "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
    "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8", "HMGB2", "CDK1", "NUSAP1",
    "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B",
    "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L",
    "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B",
    "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA",
    "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3",
    "GAS2L3", "CBX5", "CENPA",
}

S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP",
    "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2",
    "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1",
    "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
]

G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A",
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK",
    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
    "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
    "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]

EXCLUDE_HVG_PREFIXES = ("MT-", "RPL", "RPS", "HB")
EXCLUDE_HVG_GENES = CELL_CYCLE_GENES | {
    "FOS", "JUN", "JUNB", "FOSB", "HSPA1A", "HSPA1B", "HSPB1", "MALAT1", "XIST",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False


def save_pub(fig: plt.Figure, stem: Path, dpi: int = 450) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def load_epithelial_subset() -> ad.AnnData:
    log(f"Reading epithelial subset from {INPUT_H5AD}")
    backed = sc.read_h5ad(INPUT_H5AD, backed="r")
    mask = backed.obs["major_celltype_auto"].astype(str).to_numpy() == "Epithelial"
    epi = backed[mask, :].to_memory()
    backed.file.close()
    if "counts" in epi.layers:
        epi.X = epi.layers["counts"].copy()
    epi.layers["counts"] = epi.X.copy()
    return epi


def attach_cnv_calls(adata: ad.AnnData) -> None:
    if not CNV_CALLS.exists():
        log(f"CNV calls not found: {CNV_CALLS}")
        return
    cnv = pd.read_csv(CNV_CALLS, low_memory=False)
    cnv = cnv.set_index("cell_id")
    keep_cols = [
        "strict_cnv_malignancy_call",
        "sensitive_cnv_malignancy_call",
        "recommended_cnv_malignancy_call",
        "recommended_working_malignant",
        "strict_cnv_burden",
        "sensitive_cnv_burden",
        "cnv_adjacent_ref_malignancy_call",
        "cnv_burden_adjacent_ref",
        "cnv_consensus_call",
        "n_cnv_methods_malignant",
    ]
    keep_cols = [c for c in keep_cols if c in cnv.columns]
    adata.obs = adata.obs.join(cnv[keep_cols], how="left")
    if "recommended_working_malignant" in adata.obs:
        adata.obs["recommended_working_malignant"] = adata.obs["recommended_working_malignant"].fillna(False).astype(bool)


def attach_gse131907_author_annotation(adata: ad.AnnData) -> None:
    if not GSE131907_AUTHOR_ANNOTATION.exists():
        log(f"GSE131907 author annotation not found: {GSE131907_AUTHOR_ANNOTATION}")
        return
    anno = pd.read_csv(GSE131907_AUTHOR_ANNOTATION, sep="\t")
    anno = anno.set_index("Index")
    target = adata.obs.index.to_series().str.replace("^GSE131907_", "", regex=True)
    author = pd.DataFrame(index=adata.obs.index)
    for col in ["Cell_type", "Cell_type.refined", "Cell_subtype", "Sample_Origin"]:
        author[f"gse131907_author_{col}"] = target.map(anno[col])
    author.loc[adata.obs["dataset"].astype(str) != "GSE131907", :] = pd.NA
    author["gse131907_author_tS_state"] = author["gse131907_author_Cell_subtype"].where(
        author["gse131907_author_Cell_subtype"].isin(["tS1", "tS2", "tS3"]),
        other="not_tS1_tS2_tS3_or_unavailable",
    )
    adata.obs = adata.obs.join(author)


def filter_genes(adata: ad.AnnData) -> ad.AnnData:
    log("Filtering genes detected in fewer than 10 epithelial cells")
    sc.pp.filter_genes(adata, min_cells=10)
    return adata


def score_programs(adata: ad.AnnData) -> None:
    for program, genes in PROGRAM_MARKERS.items():
        available = [g for g in genes if g in adata.var_names]
        if len(available) >= 2:
            sc.tl.score_genes(
                adata,
                gene_list=available,
                score_name=f"score_{program}",
                random_state=RANDOM_STATE,
                use_raw=False,
            )
        else:
            adata.obs[f"score_{program}"] = np.nan
    s_available = [g for g in S_GENES if g in adata.var_names]
    g_available = [g for g in G2M_GENES if g in adata.var_names]
    if len(s_available) >= 5 and len(g_available) >= 5:
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_available, g2m_genes=g_available, use_raw=False)


def select_hvgs(adata: ad.AnnData) -> None:
    log("Selecting epithelial-subset HVGs")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=4000,
        flavor="seurat",
        batch_key="dataset",
        subset=False,
    )
    var = adata.var.copy()
    symbols = pd.Index(var.index.astype(str))
    exclude = symbols.str.startswith(EXCLUDE_HVG_PREFIXES) | symbols.isin(EXCLUDE_HVG_GENES)
    var["exclude_from_epithelial_hvg"] = exclude
    candidates = var.index[var["highly_variable"] & ~var["exclude_from_epithelial_hvg"]]
    if len(candidates) < 1500:
        log(f"WARNING: only {len(candidates)} HVGs after exclusion; keeping original HVGs")
        var["highly_variable_epithelial_main"] = var["highly_variable"]
    else:
        var["highly_variable_epithelial_main"] = False
        var.loc[candidates, "highly_variable_epithelial_main"] = True
    adata.var = var
    pd.DataFrame(
        [
            {"metric": "initial_hvg_n", "value": int(adata.var["highly_variable"].sum())},
            {"metric": "excluded_hvg_n", "value": int((adata.var["highly_variable"] & adata.var["exclude_from_epithelial_hvg"]).sum())},
            {"metric": "main_hvg_n", "value": int(adata.var["highly_variable_epithelial_main"].sum())},
        ]
    ).to_csv(TABLE_DIR / f"{PREFIX}_hvg_summary.csv", index=False)


def run_dimensionality_and_clustering(adata: ad.AnnData) -> str:
    log("Running PCA on epithelial HVGs")
    sc.tl.pca(
        adata,
        n_comps=50,
        mask_var="highly_variable_epithelial_main",
        svd_solver="arpack",
        random_state=RANDOM_STATE,
    )
    use_rep = "X_pca"
    try:
        import scanpy.external as sce

        log("Running Harmony integration on epithelial PCA using dataset only")
        sce.pp.harmony_integrate(
            adata,
            key="dataset",
            basis="X_pca",
            adjusted_basis="X_pca_harmony_dataset",
            max_iter_harmony=30,
        )
        use_rep = "X_pca_harmony_dataset"
        adata.uns["epithelial_recluster_integration_note"] = (
            "Main epithelial reclustering uses Harmony correction by dataset only, "
            "not sample_id, to avoid erasing patient/sample-specific tumor states."
        )
    except Exception as exc:
        log(f"WARNING: Harmony failed: {type(exc).__name__}: {exc}")
        adata.uns["epithelial_recluster_integration_note"] = (
            f"Harmony failed; main epithelial reclustering uses X_pca. Error: {type(exc).__name__}: {exc}"
        )

    log(f"Computing neighbors and UMAP using {use_rep}")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40, use_rep=use_rep, key_added="epi_neighbors")
    sc.tl.umap(adata, neighbors_key="epi_neighbors", min_dist=0.30, spread=1.0, random_state=RANDOM_STATE)
    log("Running epithelial Leiden clustering at multiple resolutions")
    for res in [0.2, 0.4, 0.6, 0.8, 1.0]:
        sc.tl.leiden(
            adata,
            resolution=res,
            key_added=f"epi_leiden_r{str(res).replace('.', '_')}",
            random_state=RANDOM_STATE,
            neighbors_key="epi_neighbors",
        )
    adata.obs["epi_leiden"] = adata.obs["epi_leiden_r0_6"].astype("category")
    return use_rep


def annotate_clusters(adata: ad.AnnData) -> pd.DataFrame:
    score_cols = [f"score_{k}" for k in PROGRAM_MARKERS]
    cluster_score = adata.obs.groupby("epi_leiden", observed=True)[score_cols].mean()
    rows = []
    cluster_to_label = {}
    for cluster, row in cluster_score.iterrows():
        marker_scores = {c.replace("score_", ""): float(row[c]) for c in score_cols if pd.notna(row[c])}
        best_program = max(marker_scores, key=marker_scores.get)
        sub = adata.obs.loc[adata.obs["epi_leiden"].astype(str) == str(cluster)]
        malignant_fraction = float(sub.get("recommended_working_malignant", pd.Series(False, index=sub.index)).fillna(False).mean())
        consensus_fraction = float((sub.get("cnv_consensus_call", pd.Series("", index=sub.index)).astype(str) == "consensus_malignant_epithelial").mean())
        author_ts_fraction = float(sub.get("gse131907_author_tS_state", pd.Series("", index=sub.index)).astype(str).isin(["tS1", "tS2", "tS3"]).mean())
        label = best_program
        if consensus_fraction >= 0.50:
            label = f"CNV_consensus_{best_program}"
        elif malignant_fraction >= 0.50:
            label = f"CNV_working_{best_program}"
        elif author_ts_fraction >= 0.50:
            label = f"GSE131907_tS_{best_program}"
        cluster_to_label[str(cluster)] = label
        rows.append(
            {
                "epi_leiden": str(cluster),
                "n_cells": int(len(sub)),
                "auto_label": label,
                "best_program": best_program,
                "malignant_working_fraction": malignant_fraction,
                "cnv_consensus_fraction": consensus_fraction,
                "gse131907_tS_fraction": author_ts_fraction,
                "dataset_GSE131907_fraction": float((sub["dataset"].astype(str) == "GSE131907").mean()),
                "dataset_GSE274934_fraction": float((sub["dataset"].astype(str) == "GSE274934").mean()),
                "tumor_fraction": float((sub["tissue_status"].astype(str) == "tumor").mean()),
                **marker_scores,
            }
        )
    summary = pd.DataFrame(rows).sort_values("epi_leiden")
    summary.to_csv(TABLE_DIR / f"{PREFIX}_epi_leiden_cluster_program_summary.csv", index=False)
    adata.obs["epi_subtype_auto"] = adata.obs["epi_leiden"].astype(str).map(cluster_to_label).astype("category")
    return summary


def rank_markers(adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Ranking epithelial subcluster markers")
    adata.raw = adata
    sc.tl.rank_genes_groups(
        adata,
        groupby="epi_leiden",
        method="wilcoxon",
        use_raw=True,
        pts=True,
        key_added="rank_genes_epi_leiden",
    )
    epi_df = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_epi_leiden")
    epi_df = epi_df.groupby("group", observed=True).head(200).reset_index(drop=True)
    epi_df.to_csv(TABLE_DIR / f"{PREFIX}_epi_leiden_markers_top200.csv", index=False)

    log("Ranking automatic epithelial subtype markers")
    sc.tl.rank_genes_groups(
        adata,
        groupby="epi_subtype_auto",
        method="wilcoxon",
        use_raw=True,
        pts=True,
        key_added="rank_genes_epi_subtype_auto",
    )
    subtype_df = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_epi_subtype_auto")
    subtype_df = subtype_df.groupby("group", observed=True).head(200).reset_index(drop=True)
    subtype_df.to_csv(TABLE_DIR / f"{PREFIX}_epi_subtype_auto_markers_top200.csv", index=False)
    with pd.ExcelWriter(TABLE_DIR / f"{PREFIX}_marker_tables_top200.xlsx", engine="openpyxl") as writer:
        epi_df.to_excel(writer, sheet_name="epi_leiden", index=False)
        subtype_df.to_excel(writer, sheet_name="epi_subtype_auto", index=False)
    return epi_df, subtype_df


def write_tables(adata: ad.AnnData, marker_df: pd.DataFrame) -> None:
    obs = adata.obs.copy()
    keys = [
        "epi_leiden",
        "epi_subtype_auto",
        "dataset",
        "sample_id",
        "tissue_status",
        "recommended_working_malignant",
        "cnv_consensus_call",
        "gse131907_author_tS_state",
    ]
    obs_cols = [c for c in keys if c in obs.columns]
    cell_meta = obs[obs_cols].copy()
    cell_meta.insert(0, "cell_id", cell_meta.index)
    cell_meta.to_csv(TABLE_DIR / f"{PREFIX}_cell_metadata.csv.gz", index=False)

    for key in ["epi_leiden", "epi_subtype_auto"]:
        counts = (
            obs.groupby([key, "dataset", "tissue_status"], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        counts.to_csv(TABLE_DIR / f"{PREFIX}_{key}_cell_counts_by_dataset_status.csv", index=False)
    if "sample_id" in obs:
        sample_counts = (
            obs.groupby(["epi_leiden", "dataset", "sample_id", "tissue_status"], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        sample_counts.to_csv(TABLE_DIR / f"{PREFIX}_epi_leiden_cell_counts_by_sample.csv", index=False)

    top10 = (
        marker_df.sort_values(["group", "scores"], ascending=[True, False])
        .groupby("group", observed=True)
        .head(10)
        .groupby("group", observed=True)["names"]
        .apply(lambda x: ", ".join(x.astype(str)))
        .reset_index(name="top10_marker_genes")
    )
    top10.to_csv(TABLE_DIR / f"{PREFIX}_epi_leiden_top10_markers_by_cluster.csv", index=False)


def plot_umap(adata: ad.AnnData, color: str, title: str, stem: str, point_size: float = 2.0, alpha: float = 0.78) -> None:
    fig = sc.pl.umap(
        adata,
        color=color,
        frameon=False,
        size=point_size,
        alpha=alpha,
        title=title,
        show=False,
        return_fig=True,
    )
    save_pub(fig, FIG_DIR / stem)


def plot_marker_dotplot(adata: ad.AnnData) -> None:
    markers = {
        "Core": ["EPCAM", "KRT8", "KRT18", "KRT19"],
        "AT1": ["AGER", "PDPN", "CAV1", "HOPX"],
        "AT2": ["SFTPA1", "SFTPB", "SLC34A2", "NAPSA"],
        "Secretory": ["SCGB1A1", "SCGB3A1", "SCGB3A2"],
        "Ciliated": ["FOXJ1", "TPPP3", "PIFO"],
        "Basal/Squamous": ["KRT5", "KRT14", "TP63"],
        "Mucous": ["MUC5AC", "MUC5B", "SPDEF"],
        "EMT": ["VIM", "FN1", "ITGA5"],
        "Cycling": ["MKI67", "TOP2A", "CENPF"],
        "Immune check": ["PTPRC", "LYZ", "CD3D"],
    }
    markers = {k: [g for g in v if g in adata.var_names] for k, v in markers.items()}
    markers = {k: v for k, v in markers.items() if v}
    dp = sc.pl.dotplot(
        adata,
        markers,
        groupby="epi_subtype_auto",
        use_raw=True,
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    dp.savefig(FIG_DIR / f"{PREFIX}_epithelial_marker_dotplot.png", dpi=450)
    dp.savefig(FIG_DIR / f"{PREFIX}_epithelial_marker_dotplot.pdf")
    dp.savefig(FIG_DIR / f"{PREFIX}_epithelial_marker_dotplot.svg")
    plt.close("all")


def plot_score_heatmap(summary: pd.DataFrame) -> None:
    score_cols = [k for k in PROGRAM_MARKERS if k in summary.columns]
    data = summary.set_index("epi_leiden")[score_cols]
    z = (data - data.mean(axis=0)) / data.std(axis=0).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(6.2, max(2.5, 0.18 * len(z))))
    im = ax.imshow(z.fillna(0).to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels(score_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels(z.index)
    ax.set_xlabel("Marker program")
    ax.set_ylabel("Epithelial Leiden")
    ax.set_title("Epithelial subcluster program scores", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("z-scored mean score")
    save_pub(fig, FIG_DIR / f"{PREFIX}_epi_leiden_program_score_heatmap")


def write_report(adata: ad.AnnData, use_rep: str, cluster_summary: pd.DataFrame) -> None:
    n_cells, n_genes = adata.shape
    report = [
        "# Epithelial Reclustering Report",
        "",
        "## Literature-informed workflow",
        "",
        "- Re-subset analyses should start again from raw/count data for the selected lineage, then rerun normalization, feature selection, PCA, neighbors, UMAP and clustering.",
        "- For tumor epithelial reclustering, CNV/malignancy status is retained as annotation rather than used as the only classifier.",
        "- The main reclustering corrects dataset-level batch effects, but avoids sample_id-level correction to preserve patient-specific malignant programs and clone/state differences.",
        "- Cell-cycle/stress/mitochondrial/ribosomal genes are excluded from the main HVG set; cell-cycle and stress scores are retained for interpretation.",
        "",
        "## Parameters",
        "",
        f"- Input object: `{INPUT_H5AD}`",
        f"- Epithelial cells: {n_cells:,}",
        f"- Genes after epithelial min-cell filtering: {n_genes:,}",
        "- Gene filter: detected in >=10 epithelial cells.",
        "- Normalization: `normalize_total(target_sum=1e4)` then `log1p`.",
        "- HVG: 4,000 epithelial-subset HVGs with `batch_key='dataset'`, then cell-cycle/stress/MT/RP/HB genes excluded for main PCA.",
        f"- Main representation: `{use_rep}`.",
        "- Main clustering: Leiden resolution 0.6 stored as `epi_leiden`; additional resolutions 0.2/0.4/0.8/1.0 saved.",
        "",
        "## Results",
        "",
        f"- Epithelial Leiden clusters: {adata.obs['epi_leiden'].nunique()}",
        f"- Automatic epithelial subtype labels: {adata.obs['epi_subtype_auto'].nunique()}",
        f"- Recommended CNV-working malignant cells retained in this object: {int(adata.obs.get('recommended_working_malignant', pd.Series(False, index=adata.obs.index)).fillna(False).sum()):,}",
        f"- CNV consensus malignant cells retained in this object: {int((adata.obs.get('cnv_consensus_call', pd.Series('', index=adata.obs.index)).astype(str) == 'consensus_malignant_epithelial').sum()):,}",
        "",
        "## Main outputs",
        "",
        f"- Analysis object: `{PREFIX}_analysis.h5ad`",
        f"- Cell metadata: `tables/{PREFIX}_cell_metadata.csv.gz`",
        f"- Cluster summary: `tables/{PREFIX}_epi_leiden_cluster_program_summary.csv`",
        f"- Marker tables: `tables/{PREFIX}_marker_tables_top200.xlsx`",
        f"- UMAPs and dotplot: `figures/`",
        "",
    ]
    (OUT_DIR / f"{PREFIX}_report.md").write_text("\n".join(report), encoding="utf-8")
    cluster_summary.to_csv(TABLE_DIR / f"{PREFIX}_epi_leiden_cluster_summary_for_report.csv", index=False)


def main() -> None:
    ensure_dirs()
    apply_publication_style()
    params = {
        "input_h5ad": str(INPUT_H5AD),
        "subset": "major_celltype_auto == Epithelial",
        "gene_min_cells": 10,
        "normalization": "normalize_total(target_sum=1e4) + log1p",
        "hvg": "n_top_genes=4000, flavor=seurat, batch_key=dataset; remove cell-cycle/stress/MT/RP/HB genes for main PCA",
        "batch_correction": "Harmony by dataset only",
        "main_resolution": 0.6,
        "random_state": RANDOM_STATE,
    }
    with open(OUT_DIR / f"{PREFIX}_params.json", "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)

    adata = load_epithelial_subset()
    attach_cnv_calls(adata)
    attach_gse131907_author_annotation(adata)
    adata = filter_genes(adata)

    log("Normalizing epithelial counts and log-transforming")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    score_programs(adata)
    select_hvgs(adata)
    use_rep = run_dimensionality_and_clustering(adata)
    cluster_summary = annotate_clusters(adata)
    marker_df, _ = rank_markers(adata)
    write_tables(adata, marker_df)

    log("Saving epithelial reclustering figures")
    plot_umap(adata, "epi_leiden", "Epithelial subclusters", f"{PREFIX}_umap_epi_leiden")
    plot_umap(adata, "epi_subtype_auto", "Automatic epithelial subtype labels", f"{PREFIX}_umap_epi_subtype_auto", point_size=2.0)
    plot_umap(adata, "dataset", "Dataset", f"{PREFIX}_umap_dataset", point_size=2.0)
    plot_umap(adata, "tissue_status", "Tissue status", f"{PREFIX}_umap_tissue_status", point_size=2.0)
    if "recommended_working_malignant" in adata.obs:
        plot_umap(adata, "recommended_working_malignant", "CNV working malignant label", f"{PREFIX}_umap_cnv_working_malignant", point_size=2.0)
    if "cnv_consensus_call" in adata.obs:
        plot_umap(adata, "cnv_consensus_call", "CNV consensus call", f"{PREFIX}_umap_cnv_consensus_call", point_size=2.0)
    if "gse131907_author_tS_state" in adata.obs:
        plot_umap(adata, "gse131907_author_tS_state", "GSE131907 author tS states", f"{PREFIX}_umap_gse131907_author_tS_state", point_size=2.0)
    for color in ["score_AT2_like", "score_AT1_like", "score_Proliferative", "score_EMT_stromal_like"]:
        if color in adata.obs:
            plot_umap(adata, color, color.replace("score_", ""), f"{PREFIX}_umap_{color}", point_size=2.0)
    plot_marker_dotplot(adata)
    plot_score_heatmap(cluster_summary)

    out_h5ad = OUT_DIR / f"{PREFIX}_analysis.h5ad"
    log(f"Writing final epithelial reclustering object: {out_h5ad}")
    adata.write_h5ad(out_h5ad)
    write_report(adata, use_rep, cluster_summary)
    log("Done")


if __name__ == "__main__":
    main()
