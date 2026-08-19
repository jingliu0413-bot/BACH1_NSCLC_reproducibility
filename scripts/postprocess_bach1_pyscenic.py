from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats

from project_paths import PROJECT_ROOT

BASE = PROJECT_ROOT
PYSC = BASE / "out" / "bach1_malignant_epithelial_pyscenic"
BACH1 = BASE / "out" / "bach1_malignant_epithelial"
H5AD = BACH1 / "nsclc_malignant_epithelial_bach1_analysis_object.h5ad"
FIG = PYSC / "figures"
TABLE = PYSC / "tables"
FIG.mkdir(exist_ok=True)
TABLE.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def savefig(fig, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_regulons() -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = pd.read_csv(PYSC / "pyscenic_bach1_regulons.csv", header=[0, 1], index_col=[0, 1])
    reg.index.names = ["TF", "MotifID"]
    rows = []
    target_rows = []
    for (tf, motif), row in reg.iterrows():
        auc = row[("Enrichment", "AUC")]
        nes = row[("Enrichment", "NES")]
        q = row[("Enrichment", "MotifSimilarityQvalue")]
        orth = row[("Enrichment", "OrthologousIdentity")]
        annot = row[("Enrichment", "Annotation")]
        context = row[("Enrichment", "Context")]
        rank = row[("Enrichment", "RankAtMax")]
        targets = ast.literal_eval(row[("Enrichment", "TargetGenes")])
        rows.append({
            "TF": tf,
            "MotifID": motif,
            "AUC": auc,
            "NES": nes,
            "MotifSimilarityQvalue": q,
            "OrthologousIdentity": orth,
            "Annotation": annot,
            "Context": context,
            "n_targets": len(targets),
            "RankAtMax": rank,
        })
        for gene, importance in targets:
            target_rows.append({
                "TF": tf,
                "MotifID": motif,
                "gene": gene,
                "importance_in_regulon": float(importance),
                "NES": float(nes),
                "AUC": float(auc),
                "Context": context,
                "Annotation": annot,
            })
    motif_df = pd.DataFrame(rows).sort_values("NES", ascending=False)
    targets = pd.DataFrame(target_rows)
    if targets.empty:
        return motif_df, targets
    agg = targets.groupby("gene").agg(
        n_supporting_motifs=("MotifID", "nunique"),
        max_NES=("NES", "max"),
        max_AUC=("AUC", "max"),
        max_importance=("importance_in_regulon", "max"),
        mean_importance=("importance_in_regulon", "mean"),
        motifs=("MotifID", lambda s: ";".join(sorted(set(map(str, s))))),
        annotations=("Annotation", lambda s: " | ".join(sorted(set(map(str, s))))[:1000]),
    ).reset_index()
    return motif_df, agg


def main() -> None:
    motif_df, targets = parse_regulons()
    motif_df.to_csv(TABLE / "pyscenic_bach1_motif_enrichment_summary.csv", index=False)

    adj = pd.read_csv(PYSC / "pyscenic_bach1_grnboost2_adjacencies.csv")
    adj = adj.rename(columns={"target": "gene", "importance": "grnboost2_importance"})
    if not targets.empty:
        targets = targets.merge(adj[["gene", "grnboost2_importance"]], on="gene", how="left")
    evidence_path = BACH1 / "tables" / "nsclc_malignant_epithelial_bach1_candidate_targets_integrated_evidence.csv"
    if evidence_path.exists() and not targets.empty:
        ev = pd.read_csv(evidence_path)
        keep = [
            "gene", "candidate_score", "logfoldchanges", "pvals_adj",
            "pearson_r", "partial_pearson_r", "prior_support",
        ]
        targets = targets.merge(ev[[c for c in keep if c in ev.columns]], on="gene", how="left")
    if not targets.empty:
        targets = targets.sort_values(
            ["n_supporting_motifs", "max_NES", "max_importance", "grnboost2_importance"],
            ascending=[False, False, False, False],
        )
        targets.to_csv(TABLE / "pyscenic_bach1_regulon_targets_integrated.csv", index=False)

    auc = pd.read_csv(PYSC / "pyscenic_bach1_aucell.csv", index_col=0)
    auc_col = auc.columns[0]
    auc = auc.rename(columns={auc_col: "BACH1_regulon_AUC"})
    auc.index.name = "cell_id"
    ad = sc.read_h5ad(H5AD)
    meta = ad.obs.copy()
    meta["BACH1_expr"] = meta["BACH1_expr"].astype(float)
    out = meta.join(auc, how="left")
    out.to_csv(TABLE / "pyscenic_bach1_regulon_aucell_by_cell.csv.gz", compression="gzip")
    ad.obs["BACH1_regulon_AUC"] = out["BACH1_regulon_AUC"]
    ad.write_h5ad(PYSC / "malignant_epithelial_bach1_pyscenic_aucell_annotated.h5ad", compression="gzip")

    summary = {
        "n_cells_aucell": int(auc.shape[0]),
        "aucell_column": "BACH1_regulon_AUC",
        "n_motif_enrichment_rows": int(motif_df.shape[0]),
        "n_unique_bach1_regulon_targets": int(0 if targets.empty else targets.shape[0]),
        "top_targets": [] if targets.empty else targets.head(30)["gene"].tolist(),
        "spearman_BACH1_expr_vs_regulon_AUC": {
            "rho": float(stats.spearmanr(out["BACH1_expr"], out["BACH1_regulon_AUC"], nan_policy="omit").statistic),
            "pvalue": float(stats.spearmanr(out["BACH1_expr"], out["BACH1_regulon_AUC"], nan_policy="omit").pvalue),
        },
    }
    (PYSC / "pyscenic_bach1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "X_umap" in ad.obsm:
        coords = ad.obsm["X_umap"]
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        sca = ax.scatter(coords[:, 0], coords[:, 1], c=out["BACH1_regulon_AUC"], s=3, cmap="viridis", linewidths=0, rasterized=True)
        ax.set_title("BACH1 regulon activity")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        cbar = fig.colorbar(sca, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("AUCell score")
        savefig(fig, FIG / "pyscenic_bach1_regulon_aucell_umap")

    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    order = [g for g in ["BACH1_low", "BACH1_high"] if g in set(out["BACH1_group"].astype(str))]
    sns.boxplot(data=out, x="BACH1_group", y="BACH1_regulon_AUC", order=order, width=0.55, fliersize=0.5, linewidth=0.6, ax=ax, palette={"BACH1_low": "#4C78A8", "BACH1_high": "#E45756"})
    ax.set_xlabel("")
    ax.set_ylabel("BACH1 regulon AUCell")
    ax.set_title("Regulon activity by BACH1 expression")
    savefig(fig, FIG / "pyscenic_bach1_regulon_aucell_by_bach1_group")

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.scatter(out["BACH1_expr"], out["BACH1_regulon_AUC"], s=3, alpha=0.35, linewidths=0, rasterized=True, color="#6A5ACD")
    rho = summary["spearman_BACH1_expr_vs_regulon_AUC"]["rho"]
    ax.set_title(f"BACH1 expression vs regulon activity\nSpearman rho={rho:.2f}")
    ax.set_xlabel("BACH1 log-normalized expression")
    ax.set_ylabel("BACH1 regulon AUCell")
    savefig(fig, FIG / "pyscenic_bach1_expr_vs_regulon_aucell")

    if not targets.empty:
        top = targets.head(25).copy()
        fig, ax = plt.subplots(figsize=(3.4, max(2.8, 0.13 * len(top))))
        sns.barplot(data=top, y="gene", x="max_importance", hue="n_supporting_motifs", dodge=False, palette="crest", ax=ax)
        ax.set_xlabel("GRNBoost2 importance in pruned regulon")
        ax.set_ylabel("")
        ax.set_title("Top BACH1 regulon targets")
        ax.legend(title="Motifs", loc="lower right", fontsize=6, title_fontsize=6)
        savefig(fig, FIG / "pyscenic_bach1_top_regulon_targets")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
