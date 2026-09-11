"""Summarize BACH1 and related regulons from the full-TF pySCENIC run."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from project_paths import OUTPUT_DIR


PYSC = OUTPUT_DIR / "bach1_malignant_epithelial_pyscenic_all_tfs"
BACH1 = OUTPUT_DIR / "bach1_malignant_epithelial"
H5AD = BACH1 / "nsclc_malignant_epithelial_bach1_analysis_object.h5ad"
REG = PYSC / "pyscenic_all_tfs_regulons.csv"
AUC = PYSC / "pyscenic_all_tfs_aucell.csv"
FIG = PYSC / "figures"
TABLE = PYSC / "tables"

DEFAULT_TFS = [
    "BACH1",
    "NFE2L2",
    "BACH2",
    "MAFK",
    "MAFG",
    "JUN",
    "FOS",
    "ATF1",
    "ATF2",
    "ATF3",
    "ATF4",
    "ATF5",
    "ATF6",
    "ATF6B",
    "ATF7",
]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def savefig(fig: mpl.figure.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def auc_column_to_tf(column: str) -> str:
    match = re.search(r"Regulon\(([^+(]+)", column)
    if match:
        return match.group(1)
    return column.split("_", 1)[0].split("(", 1)[0]


def parse_regulons(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    reg.index.names = ["TF", "MotifID"]
    motif_rows = []
    target_rows = []
    for (tf, motif), row in reg.iterrows():
        tf = str(tf)
        motif = str(motif)
        auc = float(row[("Enrichment", "AUC")])
        nes = float(row[("Enrichment", "NES")])
        annotation = row[("Enrichment", "Annotation")]
        context = row[("Enrichment", "Context")]
        targets = ast.literal_eval(row[("Enrichment", "TargetGenes")])
        motif_rows.append(
            {
                "TF": tf,
                "MotifID": motif,
                "AUC": auc,
                "NES": nes,
                "Annotation": annotation,
                "Context": context,
                "n_targets": len(targets),
            }
        )
        for gene, importance in targets:
            target_rows.append(
                {
                    "TF": tf,
                    "MotifID": motif,
                    "gene": gene,
                    "importance_in_regulon": float(importance),
                    "NES": nes,
                    "AUC": auc,
                    "Annotation": annotation,
                    "Context": context,
                }
            )
    motifs = pd.DataFrame(motif_rows).sort_values(["TF", "NES"], ascending=[True, False])
    targets = pd.DataFrame(target_rows)
    return motifs, targets


def write_tf_target_summary(targets: pd.DataFrame, tf: str) -> pd.DataFrame:
    tf_targets = targets.loc[targets["TF"].eq(tf)].copy()
    if tf_targets.empty:
        out = pd.DataFrame(
            columns=[
                "gene",
                "TF",
                "n_motif_rows",
                "max_importance",
                "mean_importance",
                "max_NES",
                "max_AUC",
                "motif_ids",
                "contexts",
                "annotations",
                "candidate_score",
            ]
        )
    else:
        out = (
            tf_targets.groupby("gene", sort=False)
            .agg(
                TF=("TF", "first"),
                n_motif_rows=("MotifID", "nunique"),
                max_importance=("importance_in_regulon", "max"),
                mean_importance=("importance_in_regulon", "mean"),
                max_NES=("NES", "max"),
                max_AUC=("AUC", "max"),
                motif_ids=("MotifID", lambda x: ";".join(sorted(set(map(str, x))))),
                contexts=("Context", lambda x: " | ".join(sorted(set(map(str, x))))),
                annotations=("Annotation", lambda x: " | ".join(sorted(set(map(str, x))))),
            )
            .reset_index()
        )
        out["candidate_score"] = out["max_importance"] * out["max_NES"]
        out = out.sort_values(["candidate_score", "max_importance", "max_NES"], ascending=False).reset_index(drop=True)
    out.to_csv(TABLE / f"pyscenic_{tf.lower()}_regulon_targets_integrated.csv", index=False)
    return out


def selected_auc_matrix(path: Path, selected_tfs: list[str]) -> pd.DataFrame:
    auc = pd.read_csv(path, index_col=0)
    tf_for_col = {column: auc_column_to_tf(column) for column in auc.columns}
    selected_set = set(selected_tfs)
    keep = [column for column, tf in tf_for_col.items() if tf in selected_set]
    if not keep:
        raise ValueError("None of the requested TF regulons were found in the AUCell table.")
    out = auc[keep].copy()
    out = out.rename(columns={column: tf_for_col[column] for column in keep})
    out = out.loc[:, ~out.columns.duplicated()]
    out.index.name = "cell_id"
    return out


def plot_matrix(matrix: pd.DataFrame, stem: Path, title: str, cmap: str = "vlag") -> None:
    if matrix.empty:
        return
    fig_h = max(2.4, 0.24 * matrix.shape[0] + 0.8)
    fig_w = max(4.2, 0.35 * matrix.shape[1] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(matrix, cmap=cmap, center=0 if cmap == "vlag" else None, linewidths=0.2, linecolor="white", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    savefig(fig, stem)


def write_bach1_threshold_sensitivity(joined: pd.DataFrame, auc: pd.DataFrame) -> dict:
    if "BACH1" not in auc.columns:
        return {"BACH1_regulon_found": False}

    score = auc["BACH1"].astype(float)
    cutoffs = {
        "top_5pct": float(score.quantile(0.95)),
        "top_10pct": float(score.quantile(0.90)),
        "mean_plus_2sd": float(score.mean() + 2 * score.std(ddof=1)),
    }
    calls = pd.DataFrame({"BACH1_regulon_AUCell": score})
    for name, cutoff in cutoffs.items():
        calls[f"BACH1_regulon_high_{name}"] = score >= cutoff
    calls.index.name = "cell_id"

    meta_cols = [
        col
        for col in ["dataset", "sample_id", "patient", "tissue_status", "epi_marker_only_label", "BACH1_group"]
        if col in joined.columns
    ]
    cell_out = joined[meta_cols].join(calls, how="right")
    cell_out.to_csv(TABLE / "pyscenic_bach1_regulon_threshold_sensitivity_by_cell.csv.gz", compression="gzip")

    group_rows = []
    for col in [c for c in ["dataset", "sample_id", "patient", "epi_marker_only_label"] if c in cell_out.columns]:
        grouped = cell_out.groupby(col, observed=True)
        summary = grouped["BACH1_regulon_AUCell"].agg(["count", "mean", "median", "std"]).reset_index()
        summary.insert(0, "grouping", col)
        summary = summary.rename(columns={col: "group", "count": "n_cells"})
        for name in cutoffs:
            summary[f"fraction_high_{name}"] = grouped[f"BACH1_regulon_high_{name}"].mean().to_numpy()
        group_rows.append(summary)
    group_summary = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    group_summary.to_csv(TABLE / "pyscenic_bach1_regulon_threshold_sensitivity_by_group.csv", index=False)

    if "patient" in cell_out.columns:
        patient = group_summary[group_summary["grouping"].eq("patient")].copy()
        if not patient.empty:
            plot_df = patient.melt(
                id_vars=["group", "n_cells"],
                value_vars=[f"fraction_high_{name}" for name in cutoffs],
                var_name="definition",
                value_name="fraction_high",
            )
            plot_df["definition"] = plot_df["definition"].str.replace("fraction_high_", "", regex=False)
            fig, ax = plt.subplots(figsize=(max(4.8, 0.45 * patient.shape[0]), 2.8))
            sns.barplot(data=plot_df, x="group", y="fraction_high", hue="definition", ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("Fraction high")
            ax.set_title("BACH1 regulon-high fraction by patient")
            ax.tick_params(axis="x", rotation=45)
            savefig(fig, FIG / "pyscenic_bach1_regulon_threshold_sensitivity_by_patient")

    return {
        "BACH1_regulon_found": True,
        "BACH1_AUCell_mean": float(score.mean()),
        "BACH1_AUCell_sd": float(score.std(ddof=1)),
        "cutoffs": cutoffs,
        "n_high": {name: int((score >= cutoff).sum()) for name, cutoff in cutoffs.items()},
        "outputs": {
            "by_cell": str(TABLE / "pyscenic_bach1_regulon_threshold_sensitivity_by_cell.csv.gz"),
            "by_group": str(TABLE / "pyscenic_bach1_regulon_threshold_sensitivity_by_group.csv"),
        },
    }


def main() -> None:
    global PYSC, H5AD, REG, AUC, FIG, TABLE

    parser = argparse.ArgumentParser()
    parser.add_argument("--pyscenic-dir", type=Path, default=PYSC)
    parser.add_argument("--input-h5ad", type=Path, default=H5AD)
    parser.add_argument("--selected-tfs", nargs="+", default=DEFAULT_TFS)
    args = parser.parse_args()

    PYSC = args.pyscenic_dir
    H5AD = args.input_h5ad
    REG = PYSC / "pyscenic_all_tfs_regulons.csv"
    AUC = PYSC / "pyscenic_all_tfs_aucell.csv"
    FIG = PYSC / "figures"
    TABLE = PYSC / "tables"

    missing = [str(path) for path in [REG, AUC, H5AD] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing full-TF pySCENIC postprocess inputs:\n" + "\n".join(missing))

    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)

    selected_tfs = [tf.upper() for tf in args.selected_tfs]
    motifs, targets = parse_regulons(REG)
    motifs.to_csv(TABLE / "pyscenic_all_tfs_motif_enrichment_summary.csv", index=False)
    bach1_target_summary = write_tf_target_summary(targets, "BACH1")
    if not targets.empty:
        targets.to_csv(TABLE / "pyscenic_all_tfs_regulon_targets_long.csv.gz", index=False, compression="gzip")
        targets.loc[targets["TF"].isin(selected_tfs)].to_csv(
            TABLE / "pyscenic_selected_tf_regulon_targets_long.csv", index=False
        )

    auc = selected_auc_matrix(AUC, selected_tfs)
    auc.to_csv(TABLE / "pyscenic_selected_tf_regulon_aucell_by_cell.csv.gz", compression="gzip")

    adata = sc.read_h5ad(H5AD)
    meta = adata.obs.copy()
    joined = meta.join(auc, how="left")
    bach1_sensitivity = write_bach1_threshold_sensitivity(joined, auc)

    group_cols = [col for col in ["dataset", "sample_id", "patient", "epi_marker_only_label", "BACH1_group"] if col in joined]
    group_rows = []
    for col in group_cols:
        means = joined.groupby(col, observed=True)[auc.columns].mean().reset_index()
        means.insert(0, "grouping", col)
        means = means.rename(columns={col: "group"})
        group_rows.append(means)
    grouped = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    grouped.to_csv(TABLE / "pyscenic_selected_tf_regulon_group_means.csv", index=False)

    corr = auc.corr(method="spearman")
    corr.to_csv(TABLE / "pyscenic_selected_tf_regulon_spearman_correlation.csv")
    plot_matrix(corr, FIG / "pyscenic_selected_tf_regulon_spearman_correlation", "Selected regulon correlations")

    if "sample_id" in joined:
        sample_means = joined.groupby("sample_id", observed=True)[auc.columns].mean().T
        z = sample_means.sub(sample_means.mean(axis=1), axis=0).div(sample_means.std(axis=1).replace(0, np.nan), axis=0)
        plot_matrix(z.fillna(0), FIG / "pyscenic_selected_tf_regulon_sample_mean_zscore", "Mean regulon activity by sample")

    if "epi_marker_only_label" in joined:
        state_means = joined.groupby("epi_marker_only_label", observed=True)[auc.columns].mean().T
        z = state_means.sub(state_means.mean(axis=1), axis=0).div(state_means.std(axis=1).replace(0, np.nan), axis=0)
        plot_matrix(z.fillna(0), FIG / "pyscenic_selected_tf_regulon_epithelial_state_mean_zscore", "Mean regulon activity by epithelial state")

    summary = {
        "n_selected_tfs_requested": len(selected_tfs),
        "selected_tfs_requested": selected_tfs,
        "selected_tfs_found_in_aucell": list(auc.columns),
        "n_cells": int(auc.shape[0]),
        "n_all_motif_rows": int(motifs.shape[0]),
        "n_selected_motif_rows": int(motifs["TF"].isin(selected_tfs).sum()),
        "n_bach1_unique_targets": int(bach1_target_summary["gene"].nunique()) if "gene" in bach1_target_summary else 0,
        "bach1_threshold_sensitivity": bach1_sensitivity,
        "outputs": {
            "bach1_target_summary": str(TABLE / "pyscenic_bach1_regulon_targets_integrated.csv"),
            "selected_aucell": str(TABLE / "pyscenic_selected_tf_regulon_aucell_by_cell.csv.gz"),
            "selected_group_means": str(TABLE / "pyscenic_selected_tf_regulon_group_means.csv"),
            "selected_correlations": str(TABLE / "pyscenic_selected_tf_regulon_spearman_correlation.csv"),
            "bach1_threshold_sensitivity_by_cell": str(
                TABLE / "pyscenic_bach1_regulon_threshold_sensitivity_by_cell.csv.gz"
            ),
            "bach1_threshold_sensitivity_by_group": str(
                TABLE / "pyscenic_bach1_regulon_threshold_sensitivity_by_group.csv"
            ),
        },
    }
    (PYSC / "pyscenic_all_tfs_selected_regulon_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
