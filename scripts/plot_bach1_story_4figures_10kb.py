import json
import math
import shutil
import textwrap
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "out" / "nature_story_4figures_10kb"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "source_data"
FINAL_FIG_DIR = FIG_DIR / "publication_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_FIG_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_H5AD = ROOT / "out" / "scanpy_downstream_scrublet" / "nsclc_gse131907_gse274934_scanpy_downstream_analysis.h5ad"
EPI_H5AD = ROOT / "out" / "epithelial_reclustering" / "nsclc_gse131907_gse274934_epithelial_recluster_analysis.h5ad"
PRIMARY_H5AD = ROOT / "out" / "bach1_malignant_epithelial_primary" / "nsclc_malignant_epithelial_primary_bach1_analysis_object.h5ad"
BACH1_H5AD = ROOT / "out" / "bach1_malignant_epithelial_pyscenic" / "malignant_epithelial_bach1_pyscenic_aucell_annotated.h5ad"

PYS_TARGETS = ROOT / "out" / "bach1_malignant_epithelial_pyscenic" / "tables" / "pyscenic_bach1_regulon_targets_integrated.csv"
PYS_ATAC_TARGETS = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "pyscenic_bach1_targets_ranked_with_atac_support.csv"
PYS_MOTIFS = ROOT / "out" / "bach1_malignant_epithelial_pyscenic" / "tables" / "pyscenic_bach1_motif_enrichment_summary.csv"
PYS_SUMMARY = ROOT / "out" / "bach1_malignant_epithelial_pyscenic" / "pyscenic_bach1_summary.json"

ATAC_QC = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "gse274934_atac_sample_qc_summary.csv"
ATAC_THRESHOLDS = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "gse274934_atac_bach1_motif_score_threshold_summary.csv"
ATAC_PEAKS = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "gse274934_atac_bach1_motif_positive_peaks.csv.gz"
ATAC_LINKS = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "bach1_regulon_targets_atac_motif_peak_links.csv.gz"
ATAC_BACKGROUND = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "expressed_genes_atac_bach1_motif_support_background.csv.gz"
ATAC_SUMMARY = ROOT / "out" / "bach1_atac_motif_support" / "bach1_atac_motif_support_summary.json"
ATAC_PROX_TERMS = ROOT / "out" / "bach1_atac_proximal_go_kegg_enrichment" / "tables" / "bach1_atac_highconf_proximal_10kb_go_kegg_significant_terms.csv"
INTERSECTION_TERMS = ROOT / "out" / "bach1_intersection_go_kegg_enrichment" / "tables" / "bach1_intersection_significant_go_kegg_terms.csv"


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

COL = {
    "dark": "#272727",
    "grey": "#767676",
    "light": "#E5E7EB",
    "blue": "#4E79A7",
    "blue_dark": "#0F4D92",
    "red": "#D9544D",
    "red_soft": "#F6CFCB",
    "teal": "#42949E",
    "teal_soft": "#DCEFEF",
    "green": "#59A14F",
    "green_soft": "#EAF2E5",
    "violet": "#9A4D8E",
    "violet_soft": "#ECE7F2",
    "amber": "#E28E2C",
    "amber_soft": "#F6EFE8",
}

CELLTYPE_COLORS = {
    "Epithelial": "#D9544D",
    "T/NK": "#4E79A7",
    "Myeloid": "#59A14F",
    "B": "#76B7B2",
    "Plasma": "#B07AA1",
    "Fibroblast": "#E28E2C",
    "Mast": "#9C755F",
    "Endothelial": "#8F8F8F",
}


def add_panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")


def save_all(fig, stem, dpi=600):
    svg = FIG_DIR / f"{stem}.svg"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    shutil.copy2(pdf, FINAL_FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def wrap(s, width=32):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def true_mask(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def format_umap_axes(ax):
    ax.set_xlabel("UMAP1", labelpad=1)
    ax.set_ylabel("UMAP2", labelpad=1)
    ax.tick_params(axis="both", labelsize=5, width=0.45, length=2.2)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")


def read_umap_obs(h5ad_path, columns, out_name):
    cached = TABLE_DIR / f"{out_name}.csv.gz"
    if cached.exists():
        return pd.read_csv(cached)
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs[[c for c in columns if c in a.obs.columns]].copy()
    umap = np.asarray(a.obsm["X_umap"])
    obs.insert(0, "UMAP2", umap[:, 1])
    obs.insert(0, "UMAP1", umap[:, 0])
    obs.index.name = "cell_id"
    obs.reset_index().to_csv(cached, index=False)
    a.file.close()
    return pd.read_csv(cached)


def extract_expr(a, genes):
    valid = [g for g in genes if g in a.var_names]
    idx = [a.var_names.get_loc(g) for g in valid]
    X = a.X[:, idx]
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    return pd.DataFrame(X, index=a.obs_names, columns=valid)


def derive_auc_threshold(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fallback = float(values.mean() + 2.0 * values.std())
    try:
        from sklearn.mixture import GaussianMixture

        x = values.reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, covariance_type="full", random_state=1).fit(x)
        gmm2 = GaussianMixture(n_components=2, covariance_type="full", random_state=1).fit(x)
        if gmm2.bic(x) <= gmm1.bic(x):
            lo, hi = sorted(gmm2.means_.ravel())
            if hi > lo:
                kde = gaussian_kde(values)
                threshold = minimize_scalar(lambda z: float(kde(z)[0]), bounds=(lo, hi), method="bounded").x
                return float(threshold), "BIC two-component trough"
    except Exception:
        pass
    return fallback, "mean + 2 s.d. after unimodal BIC"


def plot_umap_categorical(ax, df, color_col, colors, title, s=1.0, alpha=0.75, label=True):
    ax.scatter(df["UMAP1"], df["UMAP2"], s=s, c="#D9D9D9", alpha=0.22, linewidths=0, rasterized=True)
    counts = df[color_col].value_counts()
    order = [x for x in colors if x in counts.index] + [x for x in counts.index if x not in colors]
    for cat in order:
        sub = df[df[color_col].astype(str) == str(cat)]
        if sub.empty:
            continue
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=s, c=colors.get(cat, COL["grey"]), alpha=alpha, linewidths=0, rasterized=True)
        if label:
            ax.text(sub["UMAP1"].median(), sub["UMAP2"].median(), str(cat), fontsize=5.5, ha="center", va="center")
    if title:
        ax.set_title(title, loc="left", fontsize=8, pad=2)
    format_umap_axes(ax)


def plot_umap_continuous(ax, df, value_col, title, cmap="magma", s=2.8):
    vals = pd.to_numeric(df[value_col], errors="coerce")
    ax.scatter(df["UMAP1"], df["UMAP2"], s=s, c="#D9D9D9", alpha=0.25, linewidths=0, rasterized=True)
    sc = ax.scatter(
        df["UMAP1"],
        df["UMAP2"],
        s=s,
        c=vals,
        cmap=cmap,
        vmin=np.nanquantile(vals, 0.02),
        vmax=np.nanquantile(vals, 0.98),
        alpha=0.86,
        linewidths=0,
        rasterized=True,
    )
    if title:
        ax.set_title(title, loc="left", fontsize=8, pad=2)
    format_umap_axes(ax)
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)
    return sc


def draw_workflow(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.05, 0.60, 0.16, 0.22, "NSCLC\nscRNA atlas", "#E8EEF7"),
        (0.28, 0.60, 0.16, 0.22, "Malignant\nepithelium", COL["red_soft"]),
        (0.51, 0.60, 0.16, 0.22, "External\nBACH1 scores", COL["violet_soft"]),
        (0.74, 0.60, 0.16, 0.22, "TCGA\nvalidation", COL["green_soft"]),
        (0.39, 0.18, 0.18, 0.20, "Regulatory\nsensitivity", COL["violet_soft"]),
        (0.66, 0.18, 0.18, 0.20, "ATAC and\nspatial context", COL["teal_soft"]),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fc, ec="#555555", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, linespacing=1.08)
    for start, end in [
        ((0.21, 0.71), (0.28, 0.71)),
        ((0.44, 0.71), (0.51, 0.71)),
        ((0.67, 0.71), (0.74, 0.71)),
        ((0.36, 0.60), (0.45, 0.38)),
        ((0.57, 0.28), (0.66, 0.28)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#555555"))


def get_10kb_sets():
    targets = pd.read_csv(PYS_ATAC_TARGETS)
    background = pd.read_csv(ATAC_BACKGROUND)
    pys = set(targets["gene"].dropna().astype(str))
    atac = set(background.loc[pd.to_numeric(background["n_high_conf_proximal_10kb_motif_peak_links"], errors="coerce").fillna(0) > 0, "gene"].dropna().astype(str))
    membership = pd.DataFrame({"gene": sorted(pys | atac)})
    membership["in_pyscenic_bach1_regulon"] = membership["gene"].isin(pys)
    membership["in_sc_atac_highconf_10kb"] = membership["gene"].isin(atac)
    membership["set"] = np.select(
        [
            membership["in_pyscenic_bach1_regulon"] & membership["in_sc_atac_highconf_10kb"],
            membership["in_pyscenic_bach1_regulon"],
            membership["in_sc_atac_highconf_10kb"],
        ],
        ["intersection_65", "pyscenic_only", "scatac_10kb_only"],
        default="outside",
    )
    return targets, background, pys, atac, membership


def compute_bach1_coexpression(a):
    cached = TABLE_DIR / "figure2_panel_a_bach1_all_gene_coexpression.csv"
    if cached.exists():
        return pd.read_csv(cached)
    y = np.asarray(a.obs["BACH1_expr"], dtype=float)
    y = np.nan_to_num(y)
    yc = y - y.mean()
    X = a.X
    n = X.shape[0]
    if sparse.issparse(X):
        sums = np.asarray(X.sum(axis=0)).ravel()
        sums_sq = np.asarray(X.multiply(X).sum(axis=0)).ravel()
        numer = np.asarray(yc @ X).ravel()
    else:
        X = np.asarray(X)
        sums = X.sum(axis=0)
        sums_sq = (X * X).sum(axis=0)
        numer = yc @ X
    denom_x = np.maximum(sums_sq - (sums * sums / n), 0)
    denom = np.sqrt(np.sum(yc * yc) * denom_x)
    corr = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom > 0)
    df = pd.DataFrame({"gene": a.var_names.astype(str), "pearson_r_with_BACH1": corr})
    df = df[df["gene"] != "BACH1"].sort_values("pearson_r_with_BACH1", ascending=False)
    df.to_csv(cached, index=False)
    return df


def plot_coexpression_heatmap(ax, a, coexpr):
    genes = coexpr.head(18)["gene"].tolist()
    expr = extract_expr(a, genes)
    bach = pd.Series(np.asarray(a.obs["BACH1_expr"], dtype=float), index=a.obs_names)
    bins = pd.qcut(bach.rank(method="first"), 6, labels=[f"Q{i}" for i in range(1, 7)])
    means = expr.groupby(bins).mean().T
    z = means.sub(means.mean(axis=1), axis=0).div(means.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(z.shape[1]))
    ax.set_xticklabels(["low", "Q2", "Q3", "Q4", "Q5", "high"], fontsize=6)
    ax.set_yticks(range(z.shape[0]))
    ax.set_yticklabels(z.index, fontsize=5.5)
    ax.set_xlabel("BACH1 expression quantile")
    ax.set_title("Co-expression", loc="left", fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("row z-score", fontsize=5)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)
    z.reset_index(names="gene").to_csv(TABLE_DIR / "figure2_panel_a_bach1_coexpression_heatmap.csv", index=False)


def motif_context_label(context):
    c = str(context)
    tags = []
    if "top50" in c:
        tags.append("top50")
    if "weight>75.0%" in c:
        tags.append("weight>75%")
    if "500bp_up_100bp_down" in c:
        tags.append("promoter")
    return ", ".join(tags) if tags else "cisTarget"


def plot_pyscenic_motif_pruning(ax, motif_df):
    df = motif_df.copy()
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    df["n_targets"] = pd.to_numeric(df["n_targets"], errors="coerce")
    df["label"] = [f"{row.MotifID} ({motif_context_label(row.Context)})" for row in df.itertuples(index=False)]
    df = df.sort_values("NES", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    ax.barh(y, df["NES"], color=COL["violet"], height=0.58)
    ax.axvline(3.0, color=COL["grey"], lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(x, 28) for x in df["label"]], fontsize=5.5)
    ax.set_xlabel("cisTarget NES")
    ax.set_title("Motif pruning", loc="left", fontsize=8)
    ax.set_xlim(2.95, max(3.65, df["NES"].max() + 0.08))
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    for yi, nes, n in zip(y, df["NES"], df["n_targets"]):
        ax.text(nes + 0.012, yi, f"{int(n)} targets", va="center", fontsize=5.4)


def plot_aucell_binarization(ax, bach1):
    auc = pd.to_numeric(bach1["BACH1_regulon_AUC"], errors="coerce").dropna().values
    threshold, method = derive_auc_threshold(auc)
    active = int((auc >= threshold).sum())
    ax.hist(auc, bins=48, color="#B4C0E4", edgecolor="white", linewidth=0.35)
    ax.axvline(threshold, color=COL["red"], lw=1.2, ls="--")
    ax.text(threshold, ax.get_ylim()[1] * 0.92, f"active cells\n{active:,}", ha="left", va="top", fontsize=5.7, color=COL["red"])
    ax.set_xlabel("BACH1 regulon AUCell")
    ax.set_ylabel("cells")
    ax.set_title("AUCell", loc="left", fontsize=8)
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    pd.DataFrame({"BACH1_regulon_AUC": auc, "threshold": threshold, "active": auc >= threshold, "method": method}).to_csv(TABLE_DIR / "figure2_panel_c_pyscenic_aucell_binarization.csv", index=False)
    return method


def plot_auc_by_bach1_group(ax, bach1):
    groups = ["BACH1_low", "BACH1_high"]
    vals = [pd.to_numeric(bach1.loc[bach1["BACH1_group"] == g, "BACH1_regulon_AUC"], errors="coerce").dropna().values for g in groups]
    parts = ax.violinplot(vals, positions=[0, 1], widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], ["#B4C0E4", COL["red"]]):
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(0.85)
    for i, v in enumerate(vals):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i - 0.18, i + 0.18], [med, med], color=COL["dark"], lw=1.0)
        ax.plot([i, i], [q1, q3], color=COL["dark"], lw=1.0)
        ax.text(i, np.nanmax(v) + 0.002, f"n={len(v):,}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BACH1-low", "BACH1-high"], rotation=15, ha="right")
    ax.set_ylabel("BACH1 regulon AUCell")
    ax.set_title("By BACH1 detection", loc="left", fontsize=8)
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)


def plot_pyscenic_target_bubble(ax, targets):
    df = targets.sort_values(["candidate_score", "max_importance", "pearson_r"], ascending=False).head(18)
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    sc = ax.scatter(
        pd.to_numeric(df["max_importance"], errors="coerce"),
        y,
        s=18 + pd.to_numeric(df["candidate_score"], errors="coerce").fillna(0) * 18,
        c=pd.to_numeric(df["pearson_r"], errors="coerce").fillna(0),
        cmap=mpl.colors.LinearSegmentedColormap.from_list("corr", ["#5B7FCA", "#F2F2F2", "#D9544D"]),
        vmin=-0.08,
        vmax=0.12,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.hlines(y, 0, pd.to_numeric(df["max_importance"], errors="coerce"), color=COL["light"], lw=1.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=6)
    ax.set_xlabel("GRNBoost2 importance")
    ax.set_title("Recurrent targets", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Pearson r\nwith BACH1", fontsize=5)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)
    df.to_csv(TABLE_DIR / "figure2_panel_e_pyscenic_target_bubble.csv", index=False)


def draw_scatac_workflow(ax, atac_summary):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.08, 0.45, 0.18, 0.26, "Tumour\nscATAC peaks", COL["teal_soft"]),
        (0.32, 0.45, 0.18, 0.26, "BACH1-family\nmotifs", COL["violet_soft"]),
        (0.56, 0.45, 0.18, 0.26, "TSS +/-10 kb\nlinkage", COL["amber_soft"]),
        (0.80, 0.45, 0.18, 0.26, "Matched-\nbackground test", COL["green_soft"]),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fc, ec="#555555", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, linespacing=1.08)
    for start, end in [((0.26, 0.58), (0.32, 0.58)), ((0.50, 0.58), (0.56, 0.58)), ((0.74, 0.58), (0.80, 0.58))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#555555"))


def plot_atac_qc(ax, qc):
    df = qc.copy()
    x = np.arange(len(df))
    cells = pd.to_numeric(df["n_cells_in_filtered_peak_matrix"], errors="coerce") / 1000
    peak_frag = pd.to_numeric(df["median_peak_region_fragments"], errors="coerce") / 1000
    ax.bar(x, cells, color="#B4C0E4", width=0.58, edgecolor="white", linewidth=0.5, label="cells")
    ax.plot(x, peak_frag, color=COL["blue_dark"], lw=1.2, marker="o", ms=4, label="peak fragments")
    for xi, c in zip(x, cells):
        ax.text(xi, c + 0.35, f"{c:.1f}", ha="center", fontsize=5.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], fontsize=6)
    ax.set_ylabel("cells / median peak\nfragments (x10^3)")
    ax.set_title("QC", loc="left", fontsize=8)
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.4, loc="upper left")


def plot_atac_threshold(ax, threshold_df, selected_threshold=950):
    df = threshold_df.sort_values("max_motif_score_threshold").copy()
    x = pd.to_numeric(df["max_motif_score_threshold"], errors="coerce")
    y = pd.to_numeric(df["n_unique_sample_specific_peaks"], errors="coerce") / 1000
    ax.plot(x, y, color=COL["blue_dark"], lw=1.4, marker="o", ms=4)
    selected = df[df["max_motif_score_threshold"] == selected_threshold]
    if not selected.empty:
        sx = float(selected["max_motif_score_threshold"].iloc[0])
        sy = float(selected["n_unique_sample_specific_peaks"].iloc[0]) / 1000
        ax.scatter([sx], [sy], s=45, color=COL["red"], edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(sx + 3, sy + 8, f"used\nscore >= {int(sx)}", fontsize=5.7, ha="left", va="bottom")
    ax.set_xlabel("motif score threshold")
    ax.set_ylabel("motif-positive\npeaks (x10^3)")
    ax.set_title("Motif threshold", loc="left", fontsize=8)
    ax.grid(color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)


def plot_motif_support_counts(ax, peaks):
    df = peaks.copy()
    high = df[pd.to_numeric(df["max_motif_score"], errors="coerce").fillna(0) >= 950]
    summary = high.groupby(["motif_model", "motif_name"]).agg(n_peaks=("peak_id", "nunique"), n_rows=("peak_id", "size")).reset_index()
    summary = summary.sort_values("n_peaks")
    y = np.arange(len(summary))
    ax.barh(y, summary["n_peaks"], color=[COL["teal"], COL["blue"]][: len(summary)], height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["motif_name"] + "\n" + summary["motif_model"], fontsize=6)
    ax.set_xlabel("high-conf motif-positive peaks")
    ax.set_title("Motif models", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    for yi, v in zip(y, summary["n_peaks"]):
        ax.text(v + max(summary["n_peaks"]) * 0.02, yi, f"{int(v):,}", va="center", fontsize=5.8)
    summary.to_csv(TABLE_DIR / "figure3_panel_d_motif_model_high_conf_counts.csv", index=False)


def plot_tss_distance(ax, links):
    high = links[pd.to_numeric(links["max_motif_score"], errors="coerce").fillna(0) >= 950].copy()
    high = high[high["distance_to_tss"].abs() <= 10000].copy()
    bins = np.arange(0, 11000, 1000)
    for motif, color in [("BACH1", COL["blue"]), ("Bach1::Mafk", COL["teal"])]:
        vals = high.loc[high["motif_name"] == motif, "distance_to_tss"].abs()
        ax.hist(vals, bins=bins, alpha=0.72, color=color, label=motif, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("|distance to TSS| (kb)")
    ax.set_ylabel("peak-gene links")
    ax.set_xticks([0, 2000, 5000, 10000])
    ax.set_xticklabels(["0", "2", "5", "10"])
    ax.set_title("TSS distance", loc="left", fontsize=8)
    ax.legend(fontsize=5.5, loc="upper right")
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    high.to_csv(TABLE_DIR / "figure3_panel_e_high_conf_10kb_peak_gene_links.csv", index=False)


def plot_scatac_target_bubble(ax, links):
    high = links[(pd.to_numeric(links["max_motif_score"], errors="coerce").fillna(0) >= 950) & (links["distance_to_tss"].abs() <= 10000)].copy()
    df = high.groupby("gene").agg(
        n_links=("gene", "size"),
        samples=("sample", "nunique"),
        max_score=("max_motif_score", "max"),
        min_abs_dist=("distance_to_tss", lambda s: s.abs().min()),
        mean_access=("frac_cells_accessible", "mean"),
    ).reset_index()
    df = df.sort_values(["n_links", "samples", "max_score"], ascending=False).head(18).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    sc = ax.scatter(df["n_links"], y, s=20 + df["samples"] * 24, c=df["min_abs_dist"] / 1000, cmap="viridis_r", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=6)
    ax.set_xlabel("high-conf 10 kb motif links")
    ax.set_title("Candidate genes", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("closest peak\n(kb to TSS)", fontsize=5)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)
    df.to_csv(TABLE_DIR / "figure3_panel_f_scatac_supported_targets.csv", index=False)


def plot_peak_lollipop(ax, links):
    high = links[(pd.to_numeric(links["max_motif_score"], errors="coerce").fillna(0) >= 950) & (links["distance_to_tss"].abs() <= 10000)].copy()
    target_genes = ["IL6ST", "RB1CC1", "ABCB6", "GSTP1", "MUL1", "PLEKHM1"]
    sub = high[high["gene"].isin(target_genes)].copy()
    sub["gene"] = pd.Categorical(sub["gene"], categories=target_genes[::-1], ordered=True)
    sub = sub.sort_values(["gene", "distance_to_tss"])
    y_map = {g: i for i, g in enumerate(target_genes[::-1])}
    colors = {"BACH1": COL["blue"], "Bach1::Mafk": COL["teal"]}
    for g, yi in y_map.items():
        ax.hlines(yi, -10, 10, color=COL["light"], lw=1.2)
        ax.plot(0, yi, marker="|", color=COL["dark"], ms=10, mew=1.0)
    for row in sub.itertuples(index=False):
        x = row.distance_to_tss / 1000
        yi = y_map[str(row.gene)]
        ax.vlines(x, yi - 0.28, yi + 0.28, color=colors.get(row.motif_name, COL["grey"]), lw=0.7, alpha=0.75)
        ax.scatter(x, yi, s=10 + float(row.frac_cells_accessible) * 120, color=colors.get(row.motif_name, COL["grey"]), edgecolor="white", linewidth=0.3, zorder=3)
    ax.axvline(0, color=COL["grey"], lw=0.8, ls="--")
    ax.set_yticks(range(len(target_genes)))
    ax.set_yticklabels(target_genes[::-1], fontsize=6)
    ax.set_xlim(-10.5, 10.5)
    ax.set_xlabel("distance from TSS (kb)")
    ax.set_title("Example loci", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[k], markeredgecolor="white", label=k, markersize=5) for k in colors]
    ax.legend(handles=handles, fontsize=5.5, loc="upper right")
    sub.to_csv(TABLE_DIR / "figure3_panel_g_representative_peak_tracks.csv", index=False)


def plot_10kb_venn(ax, pys, atac):
    counts = {
        "left_total": len(pys),
        "right_total": len(atac),
        "left_only": len(pys - atac),
        "intersection": len(pys & atac),
        "right_only": len(atac - pys),
    }
    pd.DataFrame([counts]).to_csv(TABLE_DIR / "figure4_panel_a_10kb_venn_counts.csv", index=False)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Circle((0.36, 0.50), 0.29, facecolor=COL["violet"], edgecolor="#5C2E55", lw=1.0, alpha=0.52))
    ax.add_patch(Circle((0.64, 0.50), 0.29, facecolor=COL["teal"], edgecolor="#2D6F76", lw=1.0, alpha=0.52))
    box = {"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75}
    ax.text(0.23, 0.50, f"{counts['left_only']:,}", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#5C2E55", bbox=box)
    ax.text(0.50, 0.50, f"{counts['intersection']:,}", ha="center", va="center", fontsize=12.0, fontweight="bold", color=COL["dark"], bbox=box)
    ax.text(0.78, 0.50, f"{counts['right_only']:,}", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#245A57", bbox=box)
    ax.text(0.24, 0.84, f"pySCENIC\n{counts['left_total']:,}", ha="center", va="center", fontsize=6.6)
    ax.text(0.76, 0.84, f"scATAC 10 kb\n{counts['right_total']:,}", ha="center", va="center", fontsize=6.6)
    ax.text(0.50, 0.14, "65 dual-evidence genes", ha="center", va="center", fontsize=6.2, color=COL["grey"])
    ax.set_title("Evidence overlap", loc="left", fontsize=8)


def plot_intersection_target_bubble(ax, targets, intersection):
    df = targets[targets["gene"].isin(intersection)].copy()
    df = df.sort_values(["candidate_score", "n_high_conf_proximal_10kb_motif_peak_links", "max_importance"], ascending=False).head(18)
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    sc = ax.scatter(
        pd.to_numeric(df["max_importance"], errors="coerce"),
        y,
        s=20 + pd.to_numeric(df["n_high_conf_proximal_10kb_motif_peak_links"], errors="coerce").fillna(0) * 13,
        c=pd.to_numeric(df["logfoldchanges"], errors="coerce").fillna(0),
        cmap=mpl.colors.LinearSegmentedColormap.from_list("lfc", ["#5B7FCA", "#F2F2F2", "#D9544D"]),
        vmin=-0.4,
        vmax=1.0,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.hlines(y, 0, pd.to_numeric(df["max_importance"], errors="coerce"), color=COL["light"], lw=1.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=6)
    ax.set_xlabel("GRNBoost2 importance")
    ax.set_title("Candidate genes", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("BACH1-high\nlogFC", fontsize=5)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)
    keep = ["gene", "candidate_score", "max_importance", "max_NES", "logfoldchanges", "pearson_r", "n_high_conf_proximal_10kb_motif_peak_links", "closest_high_conf_motif_peak_distance_to_tss"]
    df[[c for c in keep if c in df.columns]].to_csv(TABLE_DIR / "figure4_panel_b_top_dual_evidence_genes.csv", index=False)


def plot_terms(ax, terms, title, max_terms=10, color=COL["green"], empty_note=None):
    df = terms.copy()
    if df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, empty_note or "No significant terms", ha="center", va="center", fontsize=8)
        ax.set_title(title, loc="left", fontsize=8)
        return
    df["Adjusted_P_value"] = pd.to_numeric(df["Adjusted_P_value"], errors="coerce")
    df["Overlap"] = pd.to_numeric(df["Overlap"], errors="coerce")
    df = df.sort_values(["Adjusted_P_value", "Overlap"], ascending=[True, False]).head(max_terms)
    df = df.iloc[::-1].reset_index(drop=True)
    neglog = -np.log10(df["Adjusted_P_value"].clip(lower=np.nextafter(0, 1)))
    y = np.arange(len(df))
    ax.barh(y, neglog, color=color, height=0.62)
    ax.axvline(-math.log10(0.05), color=COL["grey"], lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(str(x).title(), 34) for x in df["Clean_term"]], fontsize=6)
    ax.set_xlabel("-log10(FDR)")
    ax.set_title(title, loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    for yi, row, xval in zip(y, df.itertuples(index=False), neglog):
        ax.text(xval + 0.03, yi, f"{int(row.Overlap)} genes", va="center", fontsize=5.4)


def select_atac_program_terms(terms):
    selected = [
        "CYTOKINE-CYTOKINE RECEPTOR INTERACTION",
        "NOD-LIKE RECEPTOR SIGNALING PATHWAY",
        "Inflammatory Response",
        "Cytokine-mediated Signaling Pathway",
        "Secretory Granule Membrane",
        "Secretory Granule Lumen",
        "Lysosome",
        "Cytoplasmic Pattern Recognition Receptor Signaling Pathway",
    ]
    df = terms[terms["Clean_term"].isin(selected)].copy()
    order = {t: i for i, t in enumerate(selected)}
    df["order"] = df["Clean_term"].map(order)
    return df.sort_values("order")


MARKER_GENES = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18"],
    "T/NK": ["CD3D", "NKG7"],
    "B": ["MS4A1", "CD79A"],
    "Plasma": ["MZB1", "JCHAIN"],
    "Myeloid": ["LYZ", "S100A8"],
    "Fibroblast": ["COL1A1", "DCN"],
    "Endothelial": ["PECAM1", "VWF"],
    "Mast": ["TPSAB1", "CPA3"],
}


def celltype_order(values=None):
    order = [x for x in CELLTYPE_COLORS]
    if values is None:
        return order
    values = [str(x) for x in values]
    return [x for x in order if x in values] + [x for x in values if x not in order]


def plot_celltype_proportions(ax, atlas):
    df = atlas.copy()
    df["tissue_status"] = df["tissue_status"].astype(str)
    df["major_celltype_auto"] = df["major_celltype_auto"].astype(str)
    counts = (
        df.groupby(["tissue_status", "major_celltype_auto"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    tissue_order = [x for x in ["adjacent_normal", "tumor"] if x in counts.index] + [x for x in counts.index if x not in ["adjacent_normal", "tumor"]]
    cols = celltype_order(counts.columns)
    counts = counts.reindex(index=tissue_order, columns=cols, fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    x = np.arange(len(props))
    bottom = np.zeros(len(props))
    for ct in cols:
        vals = props[ct].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, color=CELLTYPE_COLORS.get(ct, COL["grey"]), width=0.62, edgecolor="white", linewidth=0.45, label=ct)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(["Adjacent" if x == "adjacent_normal" else "Tumor" if x == "tumor" else x for x in props.index], fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cell fraction")
    ax.set_title("Cell fractions", loc="left", fontsize=8, pad=2)
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    for xi, tissue in enumerate(props.index):
        ax.text(
            xi,
            0.985,
            f"n={int(counts.loc[tissue].sum()):,}",
            ha="center",
            va="top",
            fontsize=5.3,
            color=COL["grey"],
            bbox={"fc": "white", "ec": "none", "alpha": 0.72, "pad": 0.5},
        )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=5.1, ncol=1, handlelength=0.8, handletextpad=0.25)
    out = counts.reset_index().melt(id_vars="tissue_status", var_name="cell_type", value_name="n_cells")
    totals = counts.sum(axis=1).rename("n_total")
    out = out.merge(totals.reset_index(), on="tissue_status")
    out["fraction"] = out["n_cells"] / out["n_total"]
    out.to_csv(TABLE_DIR / "figure1_panel_c_celltype_proportions_by_tissue.csv", index=False)


def get_marker_dotplot_source():
    out = TABLE_DIR / "figure1_panel_d_celltype_marker_dotplot_current.csv"
    if out.exists():
        return pd.read_csv(out)
    genes = [g for gs in MARKER_GENES.values() for g in gs]
    a = ad.read_h5ad(ATLAS_H5AD, backed="r")
    valid = [g for g in genes if g in a.var_names]
    X = a[:, valid].X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    groups = a.obs["major_celltype_auto"].astype(str).to_numpy()
    rows = []
    for group in celltype_order(pd.unique(groups)):
        mask = groups == group
        if not mask.any():
            continue
        sub = X[mask, :]
        for j, gene in enumerate(valid):
            vals = sub[:, j]
            rows.append(
                {
                    "cell_type": group,
                    "gene": gene,
                    "mean_expression": float(np.nanmean(vals)),
                    "fraction_expressing": float(np.mean(vals > 0) * 100.0),
                    "n_cells": int(mask.sum()),
                }
            )
    a.file.close()
    df = pd.DataFrame(rows)
    df["mean_expression_scaled"] = df.groupby("gene")["mean_expression"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)
    )
    df["mean_expression_scaled"] = df["mean_expression_scaled"].clip(-1.8, 1.8)
    df.to_csv(out, index=False)
    return df


def plot_marker_dotplot(ax):
    df = get_marker_dotplot_source()
    groups = celltype_order(df["cell_type"].unique())
    genes = [g for gs in MARKER_GENES.values() for g in gs if g in set(df["gene"])]
    x_map = {g: i for i, g in enumerate(genes)}
    y_map = {g: i for i, g in enumerate(groups[::-1])}
    plot_df = df[df["gene"].isin(genes) & df["cell_type"].isin(groups)].copy()
    sc = ax.scatter(
        plot_df["gene"].map(x_map),
        plot_df["cell_type"].map(y_map),
        s=8 + plot_df["fraction_expressing"] * 0.72,
        c=plot_df["mean_expression_scaled"],
        cmap=mpl.colors.LinearSegmentedColormap.from_list("marker_scale", ["#4E79A7", "#F3F3F3", "#D9544D"]),
        vmin=-1.8,
        vmax=1.8,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xticks(np.arange(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups[::-1], fontsize=5.8)
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_title("Markers", loc="left", fontsize=8, pad=2)
    ax.grid(color=COL["light"], lw=0.45)
    ax.set_axisbelow(True)
    for pct, x in zip([25, 50, 75], [0.08, 0.18, 0.30]):
        ax.scatter(x, -0.18, s=8 + pct * 0.72, transform=ax.transAxes, color="#AEB7C2", edgecolor="white", linewidth=0.3, clip_on=False)
        ax.text(x + 0.035, -0.18, f"{pct}%", transform=ax.transAxes, va="center", ha="left", fontsize=5.2, color=COL["grey"])
    ax.text(0.08, -0.29, "fraction expressing", transform=ax.transAxes, va="center", ha="left", fontsize=5.2, color=COL["grey"])
    cb = plt.colorbar(sc, ax=ax, fraction=0.040, pad=0.015)
    cb.set_label("scaled mean\nexpression", fontsize=5.2)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)


def make_figure1():
    atlas = read_umap_obs(ATLAS_H5AD, ["major_celltype_auto", "dataset", "tissue_status"], "figure1_panel_b_atlas_umap_current")
    epi = read_umap_obs(EPI_H5AD, ["recommended_working_malignant", "cnv_consensus_call", "epi_subtype_auto", "tissue_status"], "figure1_panel_e_epithelial_umap_current")
    bach1 = read_umap_obs(PRIMARY_H5AD, ["BACH1_expr", "BACH1_group", "sample", "patient"], "figure1_panel_f_bach1_primary_umap_current")
    primary_ids = set(bach1["cell_id"].astype(str))
    epi["malignant_call"] = np.where(epi["cell_id"].astype(str).isin(primary_ids), "malignant", "non-malignant")
    atlas.to_csv(TABLE_DIR / "figure1_panel_b_atlas_umap_source.csv.gz", index=False)
    epi.to_csv(TABLE_DIR / "figure1_panel_e_epithelial_malignancy_umap_source.csv.gz", index=False)
    bach1.to_csv(TABLE_DIR / "figure1_panel_f_bach1_expression_umap_source.csv.gz", index=False)

    fig = plt.figure(figsize=(13.4, 9.3))
    gs = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[0.58, 1.06, 1.06],
        width_ratios=[1.0, 1.0, 1.05, 1.05],
        hspace=0.44,
        wspace=0.48,
    )
    ax_a = fig.add_subplot(gs[0, :])
    draw_workflow(ax_a)
    add_panel_label(ax_a, "a", x=0.0, y=1.01)
    ax_b = fig.add_subplot(gs[1, 0])
    plot_umap_categorical(ax_b, atlas, "major_celltype_auto", CELLTYPE_COLORS, "Cell types", s=0.45, alpha=0.68)
    add_panel_label(ax_b, "b")
    ax_b.text(0.01, 0.02, f"{len(atlas):,} cells", transform=ax_b.transAxes, fontsize=6, ha="left", va="bottom", bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.0})
    ax_c = fig.add_subplot(gs[1, 1])
    plot_celltype_proportions(ax_c, atlas)
    add_panel_label(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 2:])
    plot_marker_dotplot(ax_d)
    add_panel_label(ax_d, "d", x=-0.055)
    ax_e = fig.add_subplot(gs[2, :2])
    plot_umap_categorical(ax_e, epi, "malignant_call", {"non-malignant": "#B8C0CC", "malignant": COL["red"]}, "Malignant epithelium", s=1.15, alpha=0.75)
    add_panel_label(ax_e, "e", x=-0.04)
    ax_e.text(0.01, 0.02, f"{(epi['malignant_call']=='malignant').sum():,} malignant epithelial cells", transform=ax_e.transAxes, fontsize=6, ha="left", va="bottom", bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.0})
    ax_f = fig.add_subplot(gs[2, 2:])
    plot_umap_continuous(ax_f, bach1, "BACH1_expr", "BACH1 expression", cmap="magma", s=2.8)
    add_panel_label(ax_f, "f", x=-0.04)
    fig.subplots_adjust(top=0.95)
    save_all(fig, "figure1_cellular_context")


def make_figure2():
    a = ad.read_h5ad(BACH1_H5AD)
    bach1 = read_umap_obs(BACH1_H5AD, ["BACH1_expr", "BACH1_group", "BACH1_regulon_AUC", "sample", "epi_marker_only_label"], "figure2_panel_b_bach1_umap")
    coexpr = compute_bach1_coexpression(a)
    motifs = pd.read_csv(PYS_MOTIFS)
    targets = pd.read_csv(PYS_TARGETS)
    coexpr.head(80).to_csv(TABLE_DIR / "figure2_bach1_top_coexpression_genes.csv", index=False)
    motifs.to_csv(TABLE_DIR / "figure2_panel_d_pyscenic_motif_pruning.csv", index=False)

    fig = plt.figure(figsize=(12.2, 7.5))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.06], width_ratios=[1.08, 1.0, 1.18], hspace=0.46, wspace=0.50)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_coexpression_heatmap(ax_a, a, coexpr)
    add_panel_label(ax_a, "a", x=-0.05)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_umap_continuous(ax_b, bach1, "BACH1_regulon_AUC", None, cmap="magma", s=2.7)
    add_panel_label(ax_b, "b")
    ax_c = fig.add_subplot(gs[0, 2])
    plot_aucell_binarization(ax_c, bach1)
    ax_c.set_title(None)
    add_panel_label(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 0])
    plot_auc_by_bach1_group(ax_d, bach1)
    add_panel_label(ax_d, "d")
    ax_e = fig.add_subplot(gs[1, 1])
    plot_pyscenic_motif_pruning(ax_e, motifs)
    add_panel_label(ax_e, "e")
    ax_f = fig.add_subplot(gs[1, 2])
    plot_pyscenic_target_bubble(ax_f, targets)
    add_panel_label(ax_f, "f", x=-0.05)
    fig.subplots_adjust(top=0.95)
    save_all(fig, "figure2_pyscenic_regulon")
    a.file.close() if getattr(a, "isbacked", False) else None


def make_figure3():
    with open(ATAC_SUMMARY, "r", encoding="utf-8") as fh:
        atac_summary = json.load(fh)
    qc = pd.read_csv(ATAC_QC)
    thresholds = pd.read_csv(ATAC_THRESHOLDS)
    peaks = pd.read_csv(ATAC_PEAKS)
    links = pd.read_csv(ATAC_LINKS, usecols=["gene", "sample", "distance_to_tss", "regulatory_distance_class", "motif_model", "motif_name", "max_motif_score", "frac_cells_accessible"])
    qc.to_csv(TABLE_DIR / "figure3_panel_b_scatac_sample_qc.csv", index=False)
    thresholds.to_csv(TABLE_DIR / "figure3_panel_c_motif_score_thresholds.csv", index=False)

    fig = plt.figure(figsize=(11.8, 8.6))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.72, 1.0, 1.18], width_ratios=[1.0, 1.0, 1.25], hspace=0.48, wspace=0.43)
    ax_a = fig.add_subplot(gs[0, :])
    draw_scatac_workflow(ax_a, atac_summary)
    add_panel_label(ax_a, "a", x=0.0, y=1.01)
    ax_b = fig.add_subplot(gs[1, 0])
    plot_atac_qc(ax_b, qc)
    add_panel_label(ax_b, "b")
    ax_c = fig.add_subplot(gs[1, 1])
    plot_atac_threshold(ax_c, thresholds, selected_threshold=950)
    add_panel_label(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 2])
    plot_motif_support_counts(ax_d, peaks)
    add_panel_label(ax_d, "d")
    ax_e = fig.add_subplot(gs[2, 0])
    plot_tss_distance(ax_e, links)
    add_panel_label(ax_e, "e")
    ax_f = fig.add_subplot(gs[2, 1])
    plot_scatac_target_bubble(ax_f, links)
    add_panel_label(ax_f, "f")
    ax_g = fig.add_subplot(gs[2, 2])
    plot_peak_lollipop(ax_g, links)
    add_panel_label(ax_g, "g")
    fig.subplots_adjust(top=0.95)
    save_all(fig, "figure3_scatac_motif_support")


def make_figure4():
    targets, background, pys, atac, membership = get_10kb_sets()
    intersection = pys & atac
    membership.to_csv(TABLE_DIR / "figure4_bach1_pyscenic_scatac_10kb_membership.csv.gz", index=False)
    intersection_terms = pd.read_csv(INTERSECTION_TERMS)
    intersection_10kb = intersection_terms[intersection_terms["query_set"].astype(str).str.contains("proximal_10kb", regex=False)].copy()
    atac_terms = select_atac_program_terms(pd.read_csv(ATAC_PROX_TERMS))
    intersection_10kb.to_csv(TABLE_DIR / "figure4_panel_c_intersection_10kb_go_kegg_terms.csv", index=False)
    atac_terms.to_csv(TABLE_DIR / "figure4_panel_d_context_10kb_atac_program_terms.csv", index=False)

    fig = plt.figure(figsize=(11.6, 8.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.15], width_ratios=[0.92, 1.28], hspace=0.42, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_10kb_venn(ax_a, pys, atac)
    add_panel_label(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1])
    plot_intersection_target_bubble(ax_b, targets, intersection)
    add_panel_label(ax_b, "b", x=-0.05)
    ax_c = fig.add_subplot(gs[1, 0])
    plot_terms(ax_c, intersection_10kb, "GO/KEGG", max_terms=6, color=COL["green"], empty_note="Only two significant 10 kb-intersection terms")
    add_panel_label(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 1])
    plot_terms(ax_d, atac_terms, "Motif-context terms", max_terms=8, color=COL["red"])
    add_panel_label(ax_d, "d", x=-0.05)
    fig.subplots_adjust(top=0.95)
    save_all(fig, "figure4_integration_go_kegg")


def write_legend():
    with open(PYS_SUMMARY, "r", encoding="utf-8") as fh:
        pys_summary = json.load(fh)
    targets, _, pys, atac, _ = get_10kb_sets()
    legend = OUT / "bach1_story_4figures_10kb_legend.md"
    with open(legend, "w", encoding="utf-8") as fh:
        fh.write("# BACH1 four-figure story, 10 kb scATAC criterion\n\n")
        fh.write("## Figure 1. Cellular context\n\n")
        fh.write("Workflow, integrated scRNA-seq atlas UMAP, malignant epithelial UMAP and BACH1 expression UMAP in malignant epithelial cells.\n\n")
        fh.write("## Figure 2. BACH1 co-expression and pySCENIC inference\n\n")
        fh.write("BACH1 co-expression heatmap across expression quantiles, BACH1 regulon AUCell UMAP, AUCell binarization, regulon activity by BACH1 group, cisTarget motif pruning and top pySCENIC BACH1 regulon targets.\n\n")
        fh.write("## Figure 3. scATAC motif support\n\n")
        fh.write("scATAC workflow, sample QC, motif score threshold, BACH1/Bach1::Mafk high-confidence motif support, TSS-distance distribution, scATAC-supported targets and representative motif-bearing peak tracks. Only TSS +/-10 kb support is used for integration.\n\n")
        fh.write("## Figure 4. Integration and GO/KEGG\n\n")
        fh.write("Venn-style 10 kb intersection, top dual-evidence genes, GO/KEGG terms for the 65 strict dual-evidence genes, and the broader functional context of the 10 kb BACH1 motif-associated chromatin program.\n\n")
        fh.write("## Key numbers\n\n")
        fh.write(f"- Malignant epithelial cells used for pySCENIC AUCell: {int(pys_summary['n_cells_aucell']):,}\n")
        fh.write(f"- pySCENIC BACH1 regulon targets: {len(pys):,}\n")
        fh.write(f"- scATAC high-confidence TSS +/-10 kb motif-associated genes: {len(atac):,}\n")
        fh.write(f"- Final pySCENIC and scATAC 10 kb intersection genes: {len(pys & atac):,}\n")
    return legend


def main():
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    legend = write_legend()
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote source data to {TABLE_DIR}")
    print(f"Wrote legends to {legend}")


if __name__ == "__main__":
    main()
