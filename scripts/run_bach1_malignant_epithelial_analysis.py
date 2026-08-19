from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scanpy as sc
import seaborn as sns
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

from project_paths import OUTPUT_DIR

BASE = OUTPUT_DIR
INPUT_H5AD = BASE / "epithelial_reclustering" / "nsclc_gse131907_gse274934_epithelial_recluster_analysis.h5ad"
MARKER_ONLY_ANNOT = BASE / "epithelial_reclustering" / "tables" / "nsclc_gse131907_gse274934_epithelial_recluster_epi_leiden_marker_only_annotation.csv"
OUTDIR = BASE / "bach1_malignant_epithelial"
TABLE_DIR = OUTDIR / "tables"
FIG_DIR = OUTDIR / "figures"
PREFIX = "nsclc_malignant_epithelial_bach1"


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


def log(msg: str) -> None:
    print(msg, flush=True)


def savefig(fig: mpl.figure.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def sparse_to_1d(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def safe_makedirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_and_subset() -> sc.AnnData:
    log("Reading epithelial reclustering object")
    ad = sc.read_h5ad(INPUT_H5AD)
    if "BACH1" not in ad.var_names:
        raise ValueError("BACH1 is absent from var_names.")
    if "recommended_working_malignant" not in ad.obs.columns:
        raise ValueError("recommended_working_malignant is absent from obs.")

    annot = pd.read_csv(MARKER_ONLY_ANNOT)
    exclude_clusters = set(
        annot.loc[
            annot["exclude_from_epithelial_interpretation"].astype(str).str.lower().eq("yes"),
            "epi_leiden",
        ].astype(str)
    )
    malignant = ad.obs["recommended_working_malignant"].astype(bool).to_numpy()
    epithelial_clean = ~ad.obs["epi_leiden"].astype(str).isin(exclude_clusters).to_numpy()
    sub = ad[malignant & epithelial_clean].copy()

    label_map = annot.set_index("epi_leiden")["marker_only_label"].to_dict()
    broad_map = annot.set_index("epi_leiden")["broad_marker_only_class"].to_dict()
    sub.obs["epi_marker_only_label"] = sub.obs["epi_leiden"].astype(str).map(label_map).astype("category")
    sub.obs["epi_marker_only_broad_class"] = sub.obs["epi_leiden"].astype(str).map(broad_map).astype("category")
    return sub


def assign_bach1_groups(ad: sc.AnnData) -> tuple[sc.AnnData, dict]:
    expr = sparse_to_1d(ad[:, "BACH1"].X).astype(float)
    ad.obs["BACH1_expr"] = expr
    nonzero_frac = float((expr > 0).mean())
    q = np.quantile(expr, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    groups = np.full(ad.n_obs, "BACH1_mid", dtype=object)

    if nonzero_frac < 0.30:
        groups[expr <= 0] = "BACH1_low"
        groups[expr > 0] = "BACH1_high"
        mode = "detected_vs_zero"
        high_threshold = 0.0
        low_threshold = 0.0
    else:
        low_threshold = float(q[1])
        high_threshold = float(q[3])
        groups[expr <= low_threshold] = "BACH1_low"
        groups[expr >= high_threshold] = "BACH1_high"
        mode = "quartile"

    ad.obs["BACH1_group"] = pd.Categorical(
        groups,
        categories=["BACH1_low", "BACH1_mid", "BACH1_high"],
        ordered=True,
    )
    stats_dict = {
        "n_cells": int(ad.n_obs),
        "BACH1_nonzero_fraction": nonzero_frac,
        "BACH1_quantiles_min_q25_median_q75_q90_q95_max": [float(v) for v in q],
        "grouping_mode": mode,
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "n_BACH1_low": int((groups == "BACH1_low").sum()),
        "n_BACH1_mid": int((groups == "BACH1_mid").sum()),
        "n_BACH1_high": int((groups == "BACH1_high").sum()),
    }
    return ad, stats_dict


def rank_high_low(ad: sc.AnnData) -> pd.DataFrame:
    log("Running BACH1-high vs BACH1-low Wilcoxon DE")
    de = ad[ad.obs["BACH1_group"].isin(["BACH1_low", "BACH1_high"])].copy()
    de.obs["BACH1_binary"] = de.obs["BACH1_group"].astype(str)
    sc.tl.rank_genes_groups(
        de,
        groupby="BACH1_binary",
        groups=["BACH1_high"],
        reference="BACH1_low",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        pts=True,
        use_raw=False,
    )
    df = sc.get.rank_genes_groups_df(de, group="BACH1_high")
    df = df.rename(columns={"names": "gene"})
    high_mask = de.obs["BACH1_binary"].eq("BACH1_high").to_numpy()
    low_mask = de.obs["BACH1_binary"].eq("BACH1_low").to_numpy()
    X = de.X
    pct_high = np.asarray((X[high_mask] > 0).mean(axis=0)).ravel()
    pct_low = np.asarray((X[low_mask] > 0).mean(axis=0)).ravel()
    pct = pd.DataFrame({"gene": de.var_names, "pct_expr_high": pct_high, "pct_expr_low": pct_low})
    df = df.merge(pct, on="gene", how="left")
    df["direction_high_vs_low"] = np.where(df["logfoldchanges"] > 0, "higher_in_BACH1_high", "lower_in_BACH1_high")
    return df


def pearson_sparse_matrix(X, y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    y0 = y - y.mean()
    y_ss = float(np.dot(y0, y0))
    if sparse.issparse(X):
        sums = np.asarray(X.sum(axis=0)).ravel()
        means = sums / X.shape[0]
        x2 = np.asarray(X.multiply(X).sum(axis=0)).ravel()
        x_ss = x2 - X.shape[0] * means * means
        xy = np.asarray(X.T.dot(y0)).ravel()
    else:
        means = X.mean(axis=0)
        x0 = X - means
        x_ss = np.sum(x0 * x0, axis=0)
        xy = x0.T.dot(y0)
    denom = np.sqrt(np.maximum(x_ss, 0) * y_ss)
    r = np.divide(xy, denom, out=np.zeros_like(xy, dtype=float), where=denom > 0)
    r = np.clip(r, -0.999999, 0.999999)
    return r


def pearson_pvals(r: np.ndarray, n: int) -> np.ndarray:
    df = n - 2
    t = r * np.sqrt(df / np.maximum(1.0 - r * r, 1e-12))
    return 2 * stats.t.sf(np.abs(t), df)


def correlation_tables(ad: sc.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Computing global and covariate-residualized BACH1 correlations")
    y = ad.obs["BACH1_expr"].to_numpy(dtype=float)
    X = ad.X
    r = pearson_sparse_matrix(X, y)
    p = pearson_pvals(r, ad.n_obs)
    q = multipletests(p, method="fdr_bh")[1]
    global_df = pd.DataFrame({
        "gene": ad.var_names,
        "pearson_r": r,
        "pval": p,
        "padj": q,
    }).sort_values(["padj", "pearson_r"], ascending=[True, False])

    covars = []
    for col in ["dataset", "sample_id", "epi_leiden"]:
        if col in ad.obs.columns:
            covars.append(pd.get_dummies(ad.obs[col].astype(str), prefix=col, drop_first=True, dtype=float))
    for col in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
        if col in ad.obs.columns:
            z = np.log1p(ad.obs[col].to_numpy(dtype=float)) if col != "pct_counts_mt" else ad.obs[col].to_numpy(dtype=float)
            z = (z - np.nanmean(z)) / (np.nanstd(z) + 1e-12)
            covars.append(pd.DataFrame({col: z}, index=ad.obs_names))
    if covars:
        design = pd.concat(covars, axis=1)
        design.insert(0, "intercept", 1.0)
    else:
        design = pd.DataFrame({"intercept": np.ones(ad.n_obs)}, index=ad.obs_names)
    D = design.to_numpy(dtype=float)
    qmat, _ = np.linalg.qr(D, mode="reduced")
    y_resid = y - qmat @ (qmat.T @ y)
    y_ss = float(np.dot(y_resid, y_resid))
    out = []
    chunk = 1000
    for start in range(0, ad.n_vars, chunk):
        stop = min(start + chunk, ad.n_vars)
        block = X[:, start:stop]
        if sparse.issparse(block):
            block = block.toarray()
        else:
            block = np.asarray(block)
        block = block.astype(float, copy=False)
        block_resid = block - qmat @ (qmat.T @ block)
        xy = y_resid @ block_resid
        x_ss = np.sum(block_resid * block_resid, axis=0)
        denom = np.sqrt(np.maximum(x_ss, 0) * y_ss)
        rr = np.divide(xy, denom, out=np.zeros_like(xy, dtype=float), where=denom > 0)
        out.append(np.clip(rr, -0.999999, 0.999999))
    rr = np.concatenate(out)
    pp = pearson_pvals(rr, ad.n_obs - design.shape[1])
    qq = multipletests(pp, method="fdr_bh")[1]
    resid_df = pd.DataFrame({
        "gene": ad.var_names,
        "partial_pearson_r": rr,
        "partial_pval": pp,
        "partial_padj": qq,
        "covariates": "+".join(design.columns[:20]) + ("+..." if design.shape[1] > 20 else ""),
        "n_covariates": design.shape[1],
    }).sort_values(["partial_padj", "partial_pearson_r"], ascending=[True, False])
    return global_df, resid_df


def download_omnipath_interactions(dataset: str) -> pd.DataFrame:
    url = "https://omnipathdb.org/interactions"
    params = {
        "datasets": dataset,
        "organisms": "9606",
        "genesymbols": "1",
        "fields": "sources,references,curation_effort,dorothea_level",
        "format": "tsv",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text), sep="\t")


def get_bach1_prior_targets() -> tuple[pd.DataFrame, dict]:
    log("Fetching BACH1 prior TF-target edges from OmniPath when available")
    tables = []
    status = {}
    for ds in ["collectri", "dorothea"]:
        try:
            df = download_omnipath_interactions(ds)
            df["prior_dataset"] = ds
            status[ds] = {"ok": True, "n_edges": int(df.shape[0]), "columns": list(df.columns)}
            tables.append(df)
        except Exception as e:
            status[ds] = {"ok": False, "error": repr(e)}
    if not tables:
        return pd.DataFrame(), status
    all_edges = pd.concat(tables, ignore_index=True, sort=False)
    source_col = "source_genesymbol" if "source_genesymbol" in all_edges.columns else "source"
    target_col = "target_genesymbol" if "target_genesymbol" in all_edges.columns else "target"
    bach1 = all_edges[all_edges[source_col].astype(str).str.upper().eq("BACH1")].copy()
    if bach1.empty:
        return bach1, status
    bach1 = bach1.rename(columns={source_col: "tf", target_col: "gene"})
    keep = [c for c in ["prior_dataset", "tf", "gene", "is_stimulation", "is_inhibition", "consensus_direction", "dorothea_level", "sources", "references"] if c in bach1.columns]
    bach1 = bach1[keep].drop_duplicates()
    return bach1, status


def merge_evidence(de: pd.DataFrame, corr: pd.DataFrame, pcorr: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    x = de[[
        "gene", "scores", "logfoldchanges", "pvals", "pvals_adj",
        "pct_expr_high", "pct_expr_low", "direction_high_vs_low",
    ]].copy()
    x = x.merge(corr[["gene", "pearson_r", "padj"]].rename(columns={"padj": "corr_padj"}), on="gene", how="left")
    x = x.merge(pcorr[["gene", "partial_pearson_r", "partial_padj"]], on="gene", how="left")
    if not prior.empty and "gene" in prior.columns:
        prior_genes = prior.groupby("gene").agg(
            prior_support=("prior_dataset", lambda s: ";".join(sorted(set(map(str, s))))),
            prior_n_edges=("prior_dataset", "size"),
        ).reset_index()
        x = x.merge(prior_genes, on="gene", how="left")
    else:
        x["prior_support"] = np.nan
        x["prior_n_edges"] = 0
    x["de_significant"] = (x["pvals_adj"] < 0.05) & (np.abs(x["logfoldchanges"]) >= 0.25)
    x["corr_significant"] = (x["corr_padj"] < 0.05) & (np.abs(x["pearson_r"]) >= 0.05)
    x["partial_corr_significant"] = (x["partial_padj"] < 0.05) & (np.abs(x["partial_pearson_r"]) >= 0.03)
    same_direction = np.sign(x["logfoldchanges"].fillna(0)) == np.sign(x["pearson_r"].fillna(0))
    same_partial = np.sign(x["logfoldchanges"].fillna(0)) == np.sign(x["partial_pearson_r"].fillna(0))
    x["candidate_score"] = (
        x["de_significant"].astype(int)
        + same_direction.astype(int)
        + x["corr_significant"].astype(int)
        + same_partial.astype(int)
        + x["partial_corr_significant"].astype(int)
        + x["prior_support"].notna().astype(int)
    )
    x.loc[x["gene"].eq("BACH1"), "candidate_score"] = -1
    x = x.sort_values(
        ["candidate_score", "pvals_adj", "partial_padj", "corr_padj", "logfoldchanges"],
        ascending=[False, True, True, True, False],
    )
    return x


def activity_from_prior(ad: sc.AnnData, prior: pd.DataFrame) -> pd.DataFrame:
    if prior.empty or "gene" not in prior.columns:
        return pd.DataFrame()
    targets = [g for g in sorted(set(prior["gene"])) if g in ad.var_names and g != "BACH1"]
    if len(targets) < 3:
        return pd.DataFrame()
    X = ad[:, targets].X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    score = z.mean(axis=1)
    out = pd.DataFrame({
        "cell_id": ad.obs_names,
        "BACH1_prior_target_activity": score,
        "BACH1_expr": ad.obs["BACH1_expr"].to_numpy(),
        "BACH1_group": ad.obs["BACH1_group"].astype(str).to_numpy(),
        "epi_leiden": ad.obs["epi_leiden"].astype(str).to_numpy(),
        "sample_id": ad.obs["sample_id"].astype(str).to_numpy() if "sample_id" in ad.obs.columns else "",
    })
    return out


def make_figures(ad: sc.AnnData, de: pd.DataFrame, combined: pd.DataFrame, activity: pd.DataFrame) -> None:
    log("Saving BACH1 figures")
    palette = {"BACH1_low": "#4C78A8", "BACH1_mid": "#BAB0AC", "BACH1_high": "#E45756"}
    group_order = [g for g in ["BACH1_low", "BACH1_mid", "BACH1_high"] if (ad.obs["BACH1_group"].astype(str) == g).any()]

    if "X_umap" in ad.obsm:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        coords = ad.obsm["X_umap"]
        sca = ax.scatter(coords[:, 0], coords[:, 1], c=ad.obs["BACH1_expr"], s=3, cmap="magma", linewidths=0, rasterized=True)
        ax.set_title("BACH1 expression in malignant epithelial cells")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        cbar = fig.colorbar(sca, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("log-normalized expression")
        savefig(fig, FIG_DIR / f"{PREFIX}_umap_bach1_expr")

        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        colors = ad.obs["BACH1_group"].map(palette)
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=3, linewidths=0, rasterized=True)
        ax.set_title("BACH1 expression groups")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        handles = [mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[k], markersize=4, label=k) for k in group_order]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5))
        savefig(fig, FIG_DIR / f"{PREFIX}_umap_bach1_groups")

    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    sns.violinplot(data=ad.obs, x="BACH1_group", y="BACH1_expr", order=group_order, palette=palette, inner="quartile", linewidth=0.5, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("BACH1 log-normalized expression")
    ax.set_title("BACH1 grouping")
    ax.tick_params(axis="x", rotation=35)
    savefig(fig, FIG_DIR / f"{PREFIX}_bach1_group_violin")

    plot_de = de.copy()
    plot_de["neglog10_padj"] = -np.log10(np.maximum(plot_de["pvals_adj"], 1e-300))
    plot_de["plot_logfoldchanges"] = plot_de["logfoldchanges"].clip(-3, 3)
    plot_de["class"] = "not significant"
    plot_de.loc[(plot_de["pvals_adj"] < 0.05) & (plot_de["logfoldchanges"] >= 0.25), "class"] = "higher in BACH1-high"
    plot_de.loc[(plot_de["pvals_adj"] < 0.05) & (plot_de["logfoldchanges"] <= -0.25), "class"] = "lower in BACH1-high"
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    col = {"not significant": "#B8B8B8", "higher in BACH1-high": "#D95F02", "lower in BACH1-high": "#1B9E77"}
    for cls, sub in plot_de.groupby("class"):
        ax.scatter(sub["plot_logfoldchanges"], sub["neglog10_padj"], s=4, c=col[cls], label=cls, alpha=0.65, linewidths=0, rasterized=True)
    ax.axvline(0.25, color="#666666", lw=0.6, ls="--")
    ax.axvline(-0.25, color="#666666", lw=0.6, ls="--")
    ax.axhline(-math.log10(0.05), color="#666666", lw=0.6, ls="--")
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(0, np.nanpercentile(plot_de["neglog10_padj"], 99.7) * 1.05)
    ax.set_xlabel("log2 fold-change: BACH1-high vs low (clipped)")
    ax.set_ylabel("-log10 adjusted P")
    ax.set_title("BACH1-associated differential genes", fontsize=8)
    ax.legend(markerscale=2, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
    savefig(fig, FIG_DIR / f"{PREFIX}_high_low_volcano")

    top = combined[(combined["gene"] != "BACH1") & (combined["candidate_score"] >= 4)].head(20).copy()
    if not top.empty:
        heat_genes = top["gene"].tolist()
        means = []
        for group in group_order:
            mask = ad.obs["BACH1_group"].astype(str).eq(group).to_numpy()
            Xg = ad[mask, heat_genes].X
            if sparse.issparse(Xg):
                Xg = Xg.toarray()
            means.append(np.asarray(Xg).mean(axis=0))
        mat = pd.DataFrame(means, index=[g.replace("BACH1_", "") for g in group_order], columns=heat_genes).T
        mat = (mat - mat.mean(axis=1).values[:, None]) / (mat.std(axis=1).values[:, None] + 1e-12)
        fig, ax = plt.subplots(figsize=(3.0, max(2.8, 0.15 * len(heat_genes))))
        sns.heatmap(mat, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "row z-score"}, linewidths=0.1, linecolor="white")
        ax.set_title("Top BACH1 candidate targets")
        ax.set_xlabel("BACH1 group")
        ax.set_ylabel("")
        ax.set_yticklabels(mat.index.tolist(), rotation=0, fontsize=5)
        savefig(fig, FIG_DIR / f"{PREFIX}_top_candidate_heatmap")

    if not activity.empty:
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        sns.boxplot(data=activity, x="BACH1_group", y="BACH1_prior_target_activity", order=group_order, palette=palette, width=0.6, fliersize=0.5, linewidth=0.6, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Mean z-score of prior BACH1 targets")
        ax.set_title("Prior-target activity")
        ax.tick_params(axis="x", rotation=35)
        savefig(fig, FIG_DIR / f"{PREFIX}_prior_target_activity")


def write_report(stats_dict: dict, prior_status: dict, prior: pd.DataFrame, combined: pd.DataFrame) -> None:
    top_up = combined[(combined["gene"] != "BACH1") & (combined["logfoldchanges"] > 0)].head(30)
    top_down = combined[(combined["gene"] != "BACH1") & (combined["logfoldchanges"] < 0)].head(30)
    report = OUTDIR / f"{PREFIX}_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# BACH1 malignant epithelial analysis\n\n")
        f.write("Analysis was restricted to working malignant epithelial cells after removing marker-defined immune-contaminated/doublet-like epithelial clusters.\n\n")
        f.write("## BACH1 grouping\n\n")
        for k, v in stats_dict.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Prior TF-target database status\n\n")
        f.write("```json\n")
        f.write(json.dumps(prior_status, indent=2)[:8000])
        f.write("\n```\n\n")
        f.write(f"- BACH1 prior target edges retained: {0 if prior.empty else prior.shape[0]}\n\n")
        f.write("## Top candidate genes higher in BACH1-high cells\n\n")
        f.write(top_up[["gene", "candidate_score", "logfoldchanges", "pvals_adj", "pearson_r", "partial_pearson_r", "prior_support"]].to_csv(index=False))
        f.write("\n## Top candidate genes lower in BACH1-high cells\n\n")
        f.write(top_down[["gene", "candidate_score", "logfoldchanges", "pvals_adj", "pearson_r", "partial_pearson_r", "prior_support"]].to_csv(index=False))


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    safe_makedirs()
    ad = load_and_subset()
    ad, stats_dict = assign_bach1_groups(ad)
    stats_path = TABLE_DIR / f"{PREFIX}_bach1_grouping_stats.json"
    stats_path.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")

    cell_cols = [
        "sample_id", "dataset", "tissue_status", "epi_leiden", "epi_marker_only_label",
        "epi_marker_only_broad_class", "BACH1_expr", "BACH1_group",
    ]
    if "cnv_consensus_call" in ad.obs.columns:
        cell_cols.append("cnv_consensus_call")
    ad.obs[cell_cols].to_csv(TABLE_DIR / f"{PREFIX}_cell_metadata.csv.gz", compression="gzip")

    de = rank_high_low(ad)
    de.to_csv(TABLE_DIR / f"{PREFIX}_bach1_high_vs_low_wilcoxon_all_genes.csv.gz", index=False, compression="gzip")

    corr, pcorr = correlation_tables(ad)
    corr.to_csv(TABLE_DIR / f"{PREFIX}_bach1_global_pearson_all_genes.csv.gz", index=False, compression="gzip")
    pcorr.to_csv(TABLE_DIR / f"{PREFIX}_bach1_partial_pearson_all_genes.csv.gz", index=False, compression="gzip")

    prior, prior_status = get_bach1_prior_targets()
    if not prior.empty:
        prior.to_csv(TABLE_DIR / f"{PREFIX}_bach1_prior_targets_omnipath.csv", index=False)
    else:
        (TABLE_DIR / f"{PREFIX}_bach1_prior_targets_omnipath.csv").write_text("", encoding="utf-8")
    (TABLE_DIR / f"{PREFIX}_prior_database_status.json").write_text(json.dumps(prior_status, indent=2), encoding="utf-8")

    combined = merge_evidence(de, corr, pcorr, prior)
    combined.to_csv(TABLE_DIR / f"{PREFIX}_candidate_targets_integrated_evidence.csv", index=False)

    activity = activity_from_prior(ad, prior)
    if not activity.empty:
        activity.to_csv(TABLE_DIR / f"{PREFIX}_bach1_prior_target_activity_by_cell.csv.gz", index=False, compression="gzip")

    make_figures(ad, de, combined, activity)
    write_report(stats_dict, prior_status, prior, combined)

    ad.write_h5ad(OUTDIR / f"{PREFIX}_analysis_object.h5ad", compression="gzip")
    log("Done")


if __name__ == "__main__":
    main()
