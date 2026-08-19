import json
import shutil
import textwrap
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import sparse

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "out" / "nature_story_4figures_10kb"
FIG_DIR = OUT / "figures"
FINAL_DIR = FIG_DIR / "publication_figures"
TABLE_DIR = OUT / "source_data"
for path in [FIG_DIR, FINAL_DIR, TABLE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

EPI_H5AD = ROOT / "out" / "epithelial_reclustering" / "nsclc_gse131907_gse274934_epithelial_recluster_analysis.h5ad"
CNV_H5AD = ROOT / "out" / "epithelial_cnv_infercnvpy" / "nsclc_gse131907_gse274934_epithelial_cnv_analysis.h5ad"
CNV_DIR = ROOT / "out" / "epithelial_cnv_infercnvpy"
CNV_TABLE = CNV_DIR / "tables"
SENS_DIR = ROOT / "out" / "epithelial_cnv_infercnvpy_sensitivity_no_dynamic_threshold"
SENS_TABLE = SENS_DIR / "tables"

MULTIMETHOD_CALLS = CNV_DIR / "nsclc_gse131907_gse274934_epithelial_multimethod_malignancy_calls.csv.gz"
MULTIMETHOD_SUMMARY = CNV_DIR / "nsclc_gse131907_gse274934_epithelial_multimethod_malignancy_summary.csv"
STRICT_THRESHOLDS = CNV_TABLE / "nsclc_gse131907_gse274934_epithelial_cnv_cnv_thresholds.json"
SENSITIVE_THRESHOLDS = SENS_TABLE / "nsclc_gse131907_gse274934_epithelial_cnv_no_dynamic_threshold_cnv_thresholds.json"
SENSITIVE_BY_SAMPLE = SENS_TABLE / "nsclc_gse131907_gse274934_epithelial_cnv_no_dynamic_threshold_epithelial_cnv_calls_by_sample.csv"
SENSITIVE_CLUSTER = SENS_TABLE / "nsclc_gse131907_gse274934_epithelial_cnv_no_dynamic_threshold_cnv_cluster_summary.csv"


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
    "blue_soft": "#D8E4F2",
    "red": "#D9544D",
    "red_dark": "#9E2F2C",
    "red_soft": "#F6CFCB",
    "teal": "#42949E",
    "teal_soft": "#DCEFEF",
    "green": "#59A14F",
    "violet": "#9A4D8E",
    "amber": "#E28E2C",
    "amber_soft": "#F6EFE8",
}

EVIDENCE_COLORS = {
    "Not malignant by CNV": "#C7CED8",
    "Single-method working call": COL["amber"],
    "Consensus malignant": COL["red"],
    "Strict high-confidence": COL["red_dark"],
}


def add_panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")


def save_all(fig, stem):
    svg = FIG_DIR / f"{stem}.svg"
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    tiff = FIG_DIR / f"{stem}.tiff"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight")
    shutil.copy2(pdf, FINAL_DIR / "Figure S1.pdf")
    plt.close(fig)


def wrap(text, width=30):
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


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


def derive_evidence_class(df):
    strict = df["strict_cnv_malignancy_call"].astype(str).eq("malignant_epithelial")
    consensus = df["cnv_consensus_call"].astype(str).eq("consensus_malignant_epithelial")
    single = pd.to_numeric(df["n_cnv_methods_malignant"], errors="coerce").fillna(0).eq(1)
    out = np.full(len(df), "Not malignant by CNV", dtype=object)
    out[single.to_numpy()] = "Single-method working call"
    out[consensus.to_numpy()] = "Consensus malignant"
    out[strict.to_numpy()] = "Strict high-confidence"
    return out


def read_epi_umap():
    cached = TABLE_DIR / "figure_s1_panel_b_epithelial_cnv_burden_umap.csv.gz"
    if cached.exists():
        return pd.read_csv(cached)
    cols = [
        "strict_cnv_burden",
        "sensitive_cnv_burden",
        "cnv_burden_adjacent_ref",
        "strict_cnv_malignancy_call",
        "recommended_working_malignant",
        "cnv_consensus_call",
        "n_cnv_methods_malignant",
        "dataset",
        "sample_id",
        "tissue_status",
        "epi_subtype_auto",
    ]
    a = ad.read_h5ad(EPI_H5AD, backed="r")
    obs = a.obs[cols].copy()
    umap = np.asarray(a.obsm["X_umap"])
    obs.insert(0, "UMAP2", umap[:, 1])
    obs.insert(0, "UMAP1", umap[:, 0])
    obs.index.name = "cell_id"
    obs.reset_index(inplace=True)
    a.file.close()
    obs["cnv_evidence_class"] = derive_evidence_class(obs)
    obs["recommended_working_malignant"] = true_mask(obs["recommended_working_malignant"])
    obs.to_csv(cached, index=False)
    return obs


def draw_workflow(ax, counts):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.04, 0.56, 0.16, 0.26, "Epithelial\nreclustering\n24,481 cells", COL["red_soft"]),
        (0.26, 0.56, 0.16, 0.26, "5% T/NK\nreference cells\nby sample", COL["blue_soft"]),
        (0.48, 0.56, 0.16, 0.26, "inferCNV-style\nsmoothed CNV\nprofiles", COL["teal_soft"]),
        (0.70, 0.56, 0.18, 0.26, "CNV burden and\nmethod agreement", COL["amber_soft"]),
        (0.37, 0.16, 0.20, 0.22, f"Consensus CNV\nmalignant cells\n{counts['consensus']:,}", COL["red_soft"]),
        (0.63, 0.16, 0.20, 0.22, f"Working malignant\nset for BACH1\n{counts['working']:,}", COL["red_soft"]),
    ]
    for x, y, w, h, txt, fc in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fc, ec="#BFC3C9", lw=0.7)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.6, color=COL["dark"])
    arrows = [
        ((0.20, 0.69), (0.26, 0.69)),
        ((0.42, 0.69), (0.48, 0.69)),
        ((0.64, 0.69), (0.70, 0.69)),
        ((0.57, 0.56), (0.47, 0.38)),
        ((0.79, 0.56), (0.73, 0.38)),
        ((0.57, 0.27), (0.63, 0.27)),
    ]
    for p0, p1 in arrows:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=0.7, color=COL["grey"]))
    ax.text(
        0.04,
        0.07,
        f"Strict high-confidence set: {counts['strict']:,} cells; non-strict single-method working calls: {counts['single_class']:,} cells.",
        fontsize=6,
        color=COL["grey"],
        ha="left",
        va="center",
    )
    ax.set_title("CNV-based malignant epithelial cell definition", loc="left", fontsize=8, pad=2)


def plot_umap_continuous(ax, df):
    vals = pd.to_numeric(df["sensitive_cnv_burden"], errors="coerce")
    ax.scatter(df["UMAP1"], df["UMAP2"], s=0.9, c="#D9D9D9", alpha=0.23, linewidths=0, rasterized=True)
    sc = ax.scatter(
        df["UMAP1"],
        df["UMAP2"],
        s=1.25,
        c=vals,
        cmap="magma",
        vmin=np.nanquantile(vals, 0.02),
        vmax=np.nanquantile(vals, 0.985),
        alpha=0.86,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("Sensitive CNV burden on epithelial UMAP", loc="left", fontsize=8, pad=2)
    format_umap_axes(ax)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("CNV burden", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)


def plot_umap_evidence(ax, df):
    ax.scatter(df["UMAP1"], df["UMAP2"], s=0.75, c="#D9D9D9", alpha=0.20, linewidths=0, rasterized=True)
    order = list(EVIDENCE_COLORS)
    for cat in order:
        sub = df[df["cnv_evidence_class"].eq(cat)]
        if sub.empty:
            continue
        s = 0.95 if cat == "Not malignant by CNV" else 1.9
        alpha = 0.52 if cat == "Not malignant by CNV" else 0.86
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=s, c=EVIDENCE_COLORS[cat], alpha=alpha, linewidths=0, rasterized=True, label=cat)
    ax.set_title("CNV evidence classes", loc="left", fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1", labelpad=0)
    ax.set_ylabel("UMAP2", labelpad=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [f"{lab} ({(df['cnv_evidence_class']==lab).sum():,})" for lab in labels], loc="lower left", fontsize=5.4, markerscale=3.0, handletextpad=0.2)


def plot_method_counts(ax, summary, epi_df):
    rows = [
        ("Strict\nT/NK q99", int(summary.loc[summary["method_or_set"].eq("tnk_strict_dynamic_threshold"), "n_cells"].iloc[0])),
        ("Consensus\n>=2 methods", int(summary.loc[summary["method_or_set"].eq("cnv_consensus_at_least_two_methods"), "n_cells"].iloc[0])),
        ("Working\nmalignant set", int(summary.loc[summary["method_or_set"].eq("recommended_working_set"), "n_cells"].iloc[0])),
        ("pySCENIC\ninput", 6073),
    ]
    df = pd.DataFrame(rows, columns=["set", "n_cells"])
    x = np.arange(len(df))
    colors = [COL["red_dark"], COL["red"], COL["amber"], COL["violet"]]
    ax.bar(x, df["n_cells"], color=colors, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(df["set"], fontsize=6)
    ax.set_ylabel("Cells")
    ax.set_title("CNV-derived malignant sets", loc="left", fontsize=8)
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    for xi, row in enumerate(df.itertuples(index=False)):
        ax.text(xi, row.n_cells + 170, f"{row.n_cells:,}", ha="center", va="bottom", fontsize=5.7)
    ax.text(
        0.02,
        0.94,
        f"{int(true_mask(epi_df['recommended_working_malignant']).sum()):,}/{len(epi_df):,} epithelial cells\nentered the working set",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color=COL["grey"],
    )
    df.to_csv(TABLE_DIR / "figure_s1_panel_c_cnv_set_sizes.csv", index=False)


def plot_sample_fraction(ax, by_sample):
    pivot = by_sample.pivot_table(index=["dataset", "sample_id", "tissue_status"], columns="cnv_malignancy_call", values="n_cells", aggfunc="sum", fill_value=0).reset_index()
    for col in ["malignant_epithelial", "nonmalignant_epithelial"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot["malignant_epithelial"] + pivot["nonmalignant_epithelial"]
    pivot["working_fraction"] = pivot["malignant_epithelial"] / pivot["total"].replace(0, np.nan)
    pivot = pivot.sort_values(["working_fraction", "malignant_epithelial"], ascending=[True, True])
    y = np.arange(len(pivot))
    ax.barh(y, pivot["working_fraction"], color=COL["red"], height=0.65)
    ax.barh(y, 1 - pivot["working_fraction"], left=pivot["working_fraction"], color=COL["light"], height=0.65)
    labels = pivot["sample_id"].str.replace("GSE131907_", "", regex=False).str.replace("GSE274934_", "", regex=False)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=4.9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Working malignant fraction")
    ax.set_title("Working malignant cells by sample", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    for yi, row in enumerate(pivot.itertuples(index=False)):
        if row.malignant_epithelial > 0:
            ax.text(min(row.working_fraction + 0.015, 0.98), yi, f"{int(row.malignant_epithelial):,}", va="center", ha="left", fontsize=4.8, color=COL["dark"])
    pivot.to_csv(TABLE_DIR / "figure_s1_panel_e_working_malignant_fraction_by_sample.csv", index=False)


def group_cnv_heatmap_source():
    out_csv = TABLE_DIR / "figure_s1_panel_e_group_mean_cnv_profile.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)
    calls = pd.read_csv(
        MULTIMETHOD_CALLS,
        usecols=["cell_id", "tnk_strict_malignant", "recommended_working_malignant", "cnv_consensus_call", "n_cnv_methods_malignant"],
    )
    calls["cnv_evidence_class"] = derive_evidence_class(
        pd.DataFrame(
            {
                "strict_cnv_malignancy_call": np.where(true_mask(calls["tnk_strict_malignant"]), "malignant_epithelial", "nonmalignant_epithelial"),
                "cnv_consensus_call": calls["cnv_consensus_call"],
                "n_cnv_methods_malignant": calls["n_cnv_methods_malignant"],
            }
        )
    )
    call_map = calls.set_index("cell_id")["cnv_evidence_class"]
    a = ad.read_h5ad(CNV_H5AD, backed="r")
    obs = a.obs[["cnv_input_group"]].copy()
    obs["cnv_profile_group"] = obs.index.map(call_map).fillna("T/NK reference")
    X = a.obsm["X_cnv"]
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)
    chr_pos = {str(k): int(v) for k, v in a.uns["cnv"]["chr_pos"].items()}
    chrom_order = [f"chr{i}" for i in range(1, 23)]
    starts = {chrom: chr_pos[chrom] for chrom in chrom_order if chrom in chr_pos}
    bounds = []
    for i, chrom in enumerate(chrom_order):
        if chrom not in starts:
            continue
        start = starts[chrom]
        following = [starts[c] for c in chrom_order[i + 1 :] if c in starts]
        end = min(following) if following else X.shape[1]
        bounds.append((chrom, start, end))
    groups = ["T/NK reference", "Not malignant by CNV", "Single-method working call", "Consensus malignant", "Strict high-confidence"]
    rows = []
    for group in groups:
        mask = obs["cnv_profile_group"].eq(group).to_numpy()
        if not mask.any():
            continue
        means = np.asarray(X[mask].mean(axis=0)).ravel()
        for chrom, start, end in bounds:
            segment = means[start:end]
            if segment.size == 0:
                continue
            n_bins = 18 if chrom not in ["chr21", "chr22"] else 8
            edges = np.linspace(0, segment.size, n_bins + 1, dtype=int)
            for b in range(n_bins):
                lo, hi = edges[b], edges[b + 1]
                if hi <= lo:
                    continue
                rows.append(
                    {
                        "cnv_profile_group": group,
                        "chromosome": chrom,
                        "bin_in_chromosome": b,
                        "x_order": len(rows),
                        "mean_cnv_signal": float(np.nanmean(segment[lo:hi])),
                        "n_cells": int(mask.sum()),
                    }
                )
    a.file.close()
    df = pd.DataFrame(rows)
    order_map = {}
    i = 0
    for chrom in chrom_order:
        bins = sorted(df.loc[df["chromosome"].eq(chrom), "bin_in_chromosome"].unique())
        for b in bins:
            order_map[(chrom, b)] = i
            i += 1
    df["x_order"] = [order_map[(r.chromosome, r.bin_in_chromosome)] for r in df.itertuples(index=False)]
    df.to_csv(out_csv, index=False)
    return df


def plot_group_cnv_heatmap(ax):
    df = group_cnv_heatmap_source()
    groups = ["T/NK reference", "Not malignant by CNV", "Single-method working call", "Consensus malignant", "Strict high-confidence"]
    pivot = df.pivot(index="cnv_profile_group", columns="x_order", values="mean_cnv_signal").reindex(groups)
    centered = pivot.sub(pivot.median(axis=1), axis=0)
    vmax = np.nanquantile(np.abs(centered.to_numpy()), 0.985)
    im = ax.imshow(centered, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels([wrap(g, 18) for g in groups], fontsize=5.6)
    ax.set_xticks([])
    chrom_mid = df.groupby("chromosome")["x_order"].agg(["min", "max"]).reset_index()
    chrom_mid["mid"] = (chrom_mid["min"] + chrom_mid["max"]) / 2
    chrom_order = [f"chr{i}" for i in range(1, 23)]
    chrom_mid["order"] = chrom_mid["chromosome"].map({c: i for i, c in enumerate(chrom_order)})
    chrom_mid = chrom_mid.sort_values("order")
    for _, row in chrom_mid.iterrows():
        ax.axvline(row["min"] - 0.5, color="white", lw=0.4)
    ax.set_xticks(chrom_mid["mid"].to_numpy())
    ax.set_xticklabels([c.replace("chr", "") for c in chrom_mid["chromosome"]], fontsize=4.8)
    ax.set_xlabel("Chromosome")
    ax.set_title("Genome-wide mean CNV profile by evidence class", loc="left", fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("Centered CNV signal", fontsize=5.4)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)


def plot_cluster_threshold(ax):
    clusters = pd.read_csv(SENSITIVE_CLUSTER)
    with open(SENSITIVE_THRESHOLDS, "r", encoding="utf-8") as fh:
        threshold = json.load(fh)["primary_threshold"]
    clusters["is_malignant_cluster"] = clusters["cnv_cluster_call"].eq("malignant_candidate_cluster")
    colors = np.where(clusters["is_malignant_cluster"], COL["red"], "#B8C0CC")
    sizes = np.sqrt(clusters["n_epithelial"].clip(lower=1)) * 8
    ax.scatter(clusters["median_cnv_burden"], clusters["fraction_cellwise_high"], s=sizes, c=colors, edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.axvline(threshold, color=COL["grey"], lw=0.8, ls="--")
    for row in clusters[clusters["is_malignant_cluster"]].itertuples(index=False):
        ax.text(row.median_cnv_burden, row.fraction_cellwise_high + 0.035, str(row.cnv_leiden), ha="center", va="bottom", fontsize=5.6, color=COL["dark"])
    ax.set_xlabel("Cluster median CNV burden")
    ax.set_ylabel("Fraction CNV-high cells")
    ax.set_title("CNV clusters passing the sensitive threshold", loc="left", fontsize=8)
    ax.grid(color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.05, f"threshold={threshold:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=COL["grey"])
    clusters.to_csv(TABLE_DIR / "figure_s1_panel_c_sensitive_cnv_cluster_thresholds.csv", index=False)


def main():
    epi = read_epi_umap()
    summary = pd.read_csv(MULTIMETHOD_SUMMARY)
    by_sample = pd.read_csv(SENSITIVE_BY_SAMPLE)
    counts = {
        "strict": int(summary.loc[summary["method_or_set"].eq("tnk_strict_dynamic_threshold"), "n_cells"].iloc[0]),
        "consensus": int(summary.loc[summary["method_or_set"].eq("cnv_consensus_at_least_two_methods"), "n_cells"].iloc[0]),
        "single": int(summary.loc[summary["method_or_set"].eq("single_cnv_method_only"), "n_cells"].iloc[0]),
        "single_class": int(epi["cnv_evidence_class"].eq("Single-method working call").sum()),
        "working": int(summary.loc[summary["method_or_set"].eq("recommended_working_set"), "n_cells"].iloc[0]),
    }

    fig = plt.figure(figsize=(11.6, 6.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.72, 1.08], width_ratios=[1.0, 1.16], hspace=0.46, wspace=0.36)

    ax_a = fig.add_subplot(gs[0, :])
    draw_workflow(ax_a, counts)
    add_panel_label(ax_a, "a", x=-0.035, y=1.02)

    ax_b = fig.add_subplot(gs[1, 0])
    plot_umap_continuous(ax_b, epi)
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 1])
    plot_cluster_threshold(ax_c)
    add_panel_label(ax_c, "c")

    fig.suptitle("CNV inference supports malignant epithelial cell selection for downstream BACH1 analysis", x=0.02, y=0.995, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.93, left=0.055, right=0.985, bottom=0.07)
    save_all(fig, "figure_s1_cnv_malignant_epithelial_support")
    print(f"Wrote Figure S1 to {FIG_DIR}")
    print(f"Copied publication PDF to {FINAL_DIR / 'Figure S1.pdf'}")
    print(f"Wrote source data to {TABLE_DIR}")


if __name__ == "__main__":
    main()
