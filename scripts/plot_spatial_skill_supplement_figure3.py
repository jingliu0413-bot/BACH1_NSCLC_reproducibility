from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


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


ROOT = PROJECT_ROOT
BASE = ROOT / "out" / "emtab13530_spatial_bach1_nod"
SUP = BASE / "spatial_skill_supplement"
TABLE = SUP / "tables"
H5AD = SUP / "h5ad_by_sample"
FIG = SUP / "figures"
SRC = SUP / "source_data"
for path in [FIG, SRC]:
    path.mkdir(parents=True, exist_ok=True)


COL = {
    "dark": "#272727",
    "grey": "#767676",
    "light": "#E5E7EB",
    "tumor": "#D9544D",
    "adjacent": "#4E79A7",
    "healthy": "#59A14F",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "amber": "#E28E2C",
    "soft_blue": "#DCE7F3",
    "soft_teal": "#DCEFEF",
    "soft_red": "#F6CFCB",
}
GROUP_COL = {"Tumor": COL["tumor"], "Adjacent": COL["adjacent"], "Healthy": COL["healthy"]}
DOMAIN_COL = {
    "D0": "#7A7A7A",
    "D1": "#D9544D",
    "D2": "#4E79A7",
    "D3": "#59A14F",
    "D4": "#9A4D8E",
    "D5": "#E28E2C",
    "D6": "#8C6D31",
    "low_QC_or_unassigned": "#D0D0D0",
}


def add_panel_label(ax, label, x=-0.08, y=1.03):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")


def save_all(fig, stem):
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def get_image_coords(adata):
    key = next(iter(adata.uns["spatial"].keys()))
    image = adata.uns["spatial"][key]["images"]["lowres"]
    scale = adata.uns["spatial"][key]["scalefactors"]["tissue_lowres_scalef"]
    return image, adata.obsm["spatial"] * scale


def draw_workflow(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.01, 0.93, "Spatial skill audit and completion", fontsize=8.5, fontweight="bold", ha="left", va="top")
    boxes = [
        (0.03, 0.60, 0.22, 0.20, "preprocessing\nQC flags", COL["soft_blue"]),
        (0.29, 0.60, 0.22, 0.20, "neighbors\nVisium grid", COL["soft_teal"]),
        (0.55, 0.60, 0.22, 0.20, "statistics\nMoran's I", "#F0E0D0"),
        (0.16, 0.25, 0.25, 0.20, "domains\ncombined graph", "#ECE7F2"),
        (0.47, 0.25, 0.30, 0.20, "deconvolution\nscRNA signature proxy", COL["soft_red"]),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fc, ec="#606060", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5, linespacing=1.05)
    for start, end in [((0.25, 0.70), (0.29, 0.70)), ((0.51, 0.70), (0.55, 0.70)), ((0.66, 0.60), (0.38, 0.45)), ((0.66, 0.60), (0.62, 0.45))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=0.7, color="#606060"))
    ax.text(
        0.03,
        0.03,
        "40 Visium sections; 88,520 spots.\nFull cell2location was not trained here;\nproxy uses scRNA-derived marker signatures.",
        fontsize=5.7,
        color=COL["grey"],
        ha="left",
        va="bottom",
    )


def plot_qc_graph(ax, qc):
    order = ["Adjacent", "Tumor", "Healthy"]
    x = np.arange(len(order))
    vals = [qc.loc[qc["tissue_group"].eq(g), "qc_pass_fraction"].mean() for g in order]
    deg = [qc.loc[qc["tissue_group"].eq(g), "spatial_graph_mean_degree"].mean() for g in order]
    ax.bar(x - 0.18, vals, width=0.36, color=[GROUP_COL[g] for g in order], edgecolor="white", lw=0.7)
    ax.set_ylabel("QC pass fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax2 = ax.twinx()
    ax2.plot(x + 0.18, deg, color=COL["dark"], marker="o", ms=3.5, lw=1.1)
    ax2.set_ylim(0, 6.2)
    ax2.set_ylabel("mean spatial degree")
    ax.set_title("Preprocessing and neighbor graph QC", loc="left", fontsize=8)
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    for xi, val in zip(x, vals):
        ax.text(xi - 0.18, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=5.5)


def plot_moran(ax, moran, qc):
    df = moran.merge(qc[["sample", "tissue_group"]], on="sample", how="left")
    features = ["BACH1_log_norm", "NOD_like_score_scanpy", "BACH1_NOD_cohigh"]
    labels = ["BACH1", "NOD score", "co-high"]
    groups = ["Adjacent", "Tumor"]
    width = 0.32
    for i, feat in enumerate(features):
        for j, group in enumerate(groups):
            sub = df[(df["feature"].eq(feat)) & (df["tissue_group"].eq(group))]
            xpos = i + (j - 0.5) * width
            rng = np.random.default_rng(100 + i * 10 + j)
            ax.scatter(
                np.full(len(sub), xpos) + rng.normal(0, 0.025, len(sub)),
                sub["moran_i"],
                s=14,
                color=GROUP_COL[group],
                edgecolor="white",
                lw=0.35,
                alpha=0.9,
                label=group if i == 0 else None,
            )
            med = sub["moran_i"].median()
            ax.plot([xpos - 0.09, xpos + 0.09], [med, med], color=COL["dark"], lw=1)
    ax.axhline(0, color=COL["grey"], ls="--", lw=0.7)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Moran's I")
    ax.set_title("Spatial autocorrelation of BACH1/NOD features", loc="left", fontsize=8)
    ax.legend(fontsize=5.8, loc="upper left")
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)


def plot_domain_map(ax, adata, sample):
    image, coords = get_image_coords(adata)
    ax.imshow(image, origin="upper")
    labels = adata.obs["spatial_domain"].astype(str).to_numpy()
    display = {"low_QC_or_unassigned": "low QC"}
    for lab in sorted(set(labels)):
        mask = labels == lab
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, c=DOMAIN_COL.get(lab, "#999999"), lw=0, alpha=0.9, label=display.get(lab, lab))
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{sample} spatial domains", loc="left", fontsize=8)
    ax.legend(fontsize=5.2, loc="lower left", ncol=2, handletextpad=0.2, columnspacing=0.7)


def plot_domain_heatmap(ax, domain):
    sub = domain[domain["sample"].eq("P10_T1")].copy()
    sub = sub.sort_values("cohigh_fraction", ascending=False)
    metrics = [
        ("BACH1+ frac", "BACH1_detected_fraction"),
        ("NOD score", "NOD_like_score_mean"),
        ("co-high frac", "cohigh_fraction"),
        ("Epithelial sig", "Epithelial_signature_mean"),
        ("Myeloid sig", "Myeloid_signature_mean"),
    ]
    mat = sub[[m[1] for m in metrics]].to_numpy(float)
    z = (mat - np.nanmean(mat, axis=0)) / np.nanstd(mat, axis=0)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-1.8, vmax=1.8, aspect="auto")
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["spatial_domain"])
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], rotation=35, ha="right")
    ax.set_title("P10_T1 domain-level BACH1/NOD and signatures", loc="left", fontsize=8)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.2, color="white" if abs(z[i, j]) > 1 else COL["dark"])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("column z-score", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)


def plot_signature_enrichment(ax, enrich):
    sub = enrich[enrich["tissue_group"].eq("Tumor")].copy()
    sub = sub.sort_values("delta")
    colors = [COL["tumor"] if v > 0 else COL["adjacent"] for v in sub["delta"]]
    ax.barh(np.arange(len(sub)), sub["delta"], color=colors, edgecolor="white", lw=0.6)
    ax.axvline(0, color=COL["grey"], ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["signature"], fontsize=6)
    ax.set_xlabel("signature fraction delta\nco-high - non-co-high")
    ax.set_title("Tumor co-high spots: signature proxy enrichment", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.034, 0.019)
    for yi, row in enumerate(sub.itertuples(index=False)):
        ax.text(0.012, yi, f"p={row.mannwhitney_p:.1e}", va="center", ha="left", fontsize=5.0, color=COL["grey"])


def write_legend(qc, moran, paired, enrich):
    moran_with_group = moran.merge(qc[["sample", "tissue_group"]], on="sample", how="left")
    tumor_nod_moran = moran_with_group[
        moran_with_group["tissue_group"].eq("Tumor") & moran_with_group["feature"].eq("NOD_like_score_scanpy")
    ]["moran_i"].mean()
    adjacent_nod_moran = moran_with_group[
        moran_with_group["tissue_group"].eq("Adjacent") & moran_with_group["feature"].eq("NOD_like_score_scanpy")
    ]["moran_i"].mean()
    with open(SUP / "spatial_skill_supplement_figure3_legend.md", "w", encoding="utf-8") as fh:
        fh.write("# Figure 3 spatial skill supplement\n\n")
        fh.write("## Core conclusion\n\n")
        fh.write(
            "The spatial-transcriptomics skill audit confirms that the E-MTAB-13530 Visium analysis includes preprocessing QC, Squidpy spatial neighbor graphs, Moran's I spatial statistics, expression-spatial domain detection, and a fast scRNA-marker signature deconvolution proxy. "
            "NOD-like receptor pathway score is spatially autocorrelated in both tumor and adjacent sections, whereas BACH1 itself is sparse with weak global Moran's I. Tumor BACH1/NOD co-high spots show modest enrichment for Myeloid and Epithelial signature proxies.\n\n"
        )
        fh.write("## Key numbers\n\n")
        fh.write(f"- Mean tumor QC pass fraction: {qc.loc[qc.tissue_group.eq('Tumor'), 'qc_pass_fraction'].mean():.3f}\n")
        fh.write(f"- Mean adjacent QC pass fraction: {qc.loc[qc.tissue_group.eq('Adjacent'), 'qc_pass_fraction'].mean():.3f}\n")
        fh.write(f"- Mean tumor NOD score Moran's I: {tumor_nod_moran:.3f}\n")
        fh.write(f"- Mean adjacent NOD score Moran's I: {adjacent_nod_moran:.3f}\n")
        top = enrich[enrich.tissue_group.eq("Tumor")].sort_values("delta", ascending=False).head(2)
        for row in top.itertuples(index=False):
            fh.write(f"- Tumor co-high signature delta, {row.signature}: {row.delta:.4f}, Mann-Whitney p={row.mannwhitney_p:.3g}\n")
        fh.write("\nFull source data are saved in `spatial_skill_supplement/source_data` and `spatial_skill_supplement/tables`.\n")


def main():
    qc = pd.read_csv(TABLE / "spatial_skill_sample_qc_graph_summary.csv")
    moran = pd.read_csv(TABLE / "spatial_skill_moran_statistics.csv")
    paired = pd.read_csv(TABLE / "spatial_skill_paired_tumor_adjacent_supplement_stats.csv")
    domain = pd.read_csv(TABLE / "spatial_skill_domain_summary.csv")
    enrich = pd.read_csv(TABLE / "spatial_skill_cohigh_signature_enrichment.csv")
    adata = ad.read_h5ad(H5AD / "P10_T1_skill_supplement.h5ad")

    qc.to_csv(SRC / "figure3_qc_graph_summary.csv", index=False)
    moran.to_csv(SRC / "figure3_moran_statistics.csv", index=False)
    paired.to_csv(SRC / "figure3_paired_supplement_stats.csv", index=False)
    domain[domain["sample"].eq("P10_T1")].to_csv(SRC / "figure3_P10_T1_domain_summary.csv", index=False)
    enrich.to_csv(SRC / "figure3_cohigh_signature_enrichment.csv", index=False)

    fig = plt.figure(figsize=(12.2, 7.4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.05, 1.05], height_ratios=[0.9, 1.2], hspace=0.45, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    draw_workflow(ax_a)
    add_panel_label(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1])
    plot_qc_graph(ax_b, qc)
    add_panel_label(ax_b, "b")
    ax_c = fig.add_subplot(gs[0, 2])
    plot_moran(ax_c, moran, qc)
    add_panel_label(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 0])
    plot_domain_map(ax_d, adata, "P10_T1")
    add_panel_label(ax_d, "d")
    ax_e = fig.add_subplot(gs[1, 1])
    plot_domain_heatmap(ax_e, domain)
    add_panel_label(ax_e, "e")
    ax_f = fig.add_subplot(gs[1, 2])
    plot_signature_enrichment(ax_f, enrich)
    add_panel_label(ax_f, "f")
    fig.suptitle(
        "Spatial-transcriptomics skill audit adds domain and signature context to BACH1/NOD co-localization",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.90, left=0.06, right=0.98, bottom=0.08)
    save_all(fig, "figure3_spatial_skill_supplement_domains_signatures")
    write_legend(qc, moran, paired, enrich)
    print(f"Wrote figure 3 to {FIG}")
    print(f"Wrote source data to {SRC}")


if __name__ == "__main__":
    main()
