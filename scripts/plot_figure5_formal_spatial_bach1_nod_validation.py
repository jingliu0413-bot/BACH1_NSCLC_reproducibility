from pathlib import Path
import shutil
import tempfile

import anndata as ad
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy import stats

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
SPATIAL = ROOT / "out" / "emtab13530_spatial_bach1_nod"
H5AD = SPATIAL / "h5ad_by_sample"
TABLE = SPATIAL / "tables"
OUT = ROOT / "out" / "nature_story_4figures_10kb"
FIG = OUT / "figures"
FINAL = FIG / "publication_figures"
SRC = OUT / "source_data"
for path in [FIG, FINAL, SRC]:
    path.mkdir(parents=True, exist_ok=True)


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
    "tumor": "#D9544D",
    "adjacent": "#4E79A7",
    "healthy": "#59A14F",
    "bach1": "#D9544D",
    "nod": "#42949E",
    "cohigh": "#2D1A3A",
    "neutral": "#C9CED6",
}
GROUP_COL = {"Tumor": COL["tumor"], "Adjacent": COL["adjacent"], "Healthy": COL["healthy"]}


def add_panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")


def save_all(fig, stem):
    svg = FIG / f"{stem}.svg"
    pdf = FIG / f"{stem}.pdf"
    png = FIG / f"{stem}.png"
    tiff = FIG / f"{stem}.tiff"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight")
    shutil.copy2(pdf, FINAL / "figure5_spatial_bach1_nod_validation.pdf")
    shutil.copy2(pdf, FINAL / "figure5_spatial_skill_supplement_domains_signatures.pdf")
    plt.close(fig)


def read_h5ad_compat(path):
    try:
        return ad.read_h5ad(path)
    except Exception as err:
        if "encoding_type='null'" not in str(err) and "uns/log1p" not in str(err):
            raise
    with tempfile.TemporaryDirectory(prefix="h5ad_compat_") as tmpdir:
        tmp_path = Path(tmpdir) / path.name
        shutil.copy2(path, tmp_path)
        with h5py.File(tmp_path, "a") as handle:
            if "uns/log1p/base" in handle:
                del handle["uns/log1p/base"]
        return ad.read_h5ad(tmp_path)


def load_sample(sample):
    return read_h5ad_compat(H5AD / f"{sample}_bach1_nod_scored.h5ad")


def get_image_coords(adata):
    key = next(iter(adata.uns["spatial"].keys()))
    image = adata.uns["spatial"][key]["images"]["lowres"]
    scale = adata.uns["spatial"][key]["scalefactors"]["tissue_lowres_scalef"]
    return image, adata.obsm["spatial"] * scale


def style_spatial_axis(ax, image):
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_spatial_metric(ax, adata, metric, title, cmap, vmin=None, vmax=None, s=6):
    image, coords = get_image_coords(adata)
    ax.imshow(image, origin="upper")
    vals = adata.obs[metric].to_numpy(float)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, alpha=0.92)
    ax.set_title(title, loc="left", fontsize=8, pad=2)
    style_spatial_axis(ax, image)
    return sc


def plot_cohigh(ax, adata, title, s=6.2):
    image, coords = get_image_coords(adata)
    ax.imshow(image, origin="upper")
    obs = adata.obs
    bach = obs["BACH1_high"].to_numpy(bool)
    nod = obs["NOD_like_high"].to_numpy(bool)
    co = obs["BACH1_NOD_cohigh"].to_numpy(bool)
    ax.scatter(coords[:, 0], coords[:, 1], s=s * 0.55, c="#D3D3D3", linewidth=0, alpha=0.45)
    masks = [
        ("NOD high", nod & ~bach, COL["nod"]),
        ("BACH1 high", bach & ~nod, COL["bach1"]),
        ("co-high", co, COL["cohigh"]),
    ]
    for _, mask, color in masks:
        ax.scatter(coords[mask, 0], coords[mask, 1], s=s, c=color, linewidth=0, alpha=0.95)
    ax.set_title(title, loc="left", fontsize=8, pad=2)
    style_spatial_axis(ax, image)


def add_spatial_annotation(ax, row):
    ax.text(
        0.02,
        0.04,
        f"BACH1+ {row.BACH1_detected_fraction:.1%}; co-high J={row.high_high_jaccard:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.6,
        color=COL["dark"],
        bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.0},
    )


def paired_metric(ax, patient_summary, stats_table, metric, ylabel, title):
    wide = patient_summary.pivot(index="patient_id", columns="tissue_group", values=metric).dropna(subset=["Adjacent", "Tumor"])
    x = np.array([0, 1])
    for _, row in wide.iterrows():
        ax.plot(x, [row["Adjacent"], row["Tumor"]], color="#B8B8B8", lw=0.75, zorder=1)
        ax.scatter(0, row["Adjacent"], s=20, color=COL["adjacent"], edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(1, row["Tumor"], s=20, color=COL["tumor"], edgecolor="white", linewidth=0.45, zorder=3)
    ax.plot(x, [wide["Adjacent"].median(), wide["Tumor"].median()], color=COL["dark"], lw=1.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Adjacent", "Tumor"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=8)
    stat = stats_table.loc[stats_table["metric"].eq(metric)].iloc[0]
    ax.text(
        0.5,
        0.98,
        f"n={int(stat.n_pairs)} pairs; p={stat.wilcoxon_p:.3g}\nmedian delta={stat.median_delta_tumor_minus_adjacent:.3g}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.6,
        color=COL["grey"],
    )
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)


def plot_section_correlation(ax, sample_metrics):
    for group in ["Adjacent", "Tumor", "Healthy"]:
        sub = sample_metrics[sample_metrics["tissue_group"].eq(group)]
        if sub.empty:
            continue
        ax.scatter(
            sub["BACH1_detected_fraction"],
            sub["NOD_like_score_mean"],
            s=28,
            color=GROUP_COL[group],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
            label=group,
        )
    rho, p = stats.spearmanr(sample_metrics["BACH1_detected_fraction"], sample_metrics["NOD_like_score_mean"])
    ax.text(0.04, 0.96, f"Spearman rho={rho:.2f}\np={p:.3g}", transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=COL["grey"])
    ax.set_xlabel("BACH1+ spot fraction")
    ax.set_ylabel("mean NOD-like score")
    ax.set_title("Section-level BACH1/NOD relationship", loc="left", fontsize=8)
    ax.grid(color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.6, loc="lower right")


def bootstrap_ci(values, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(rng.choice(values, size=len(values), replace=True))
    return np.percentile(out, [2.5, 97.5])


def plot_effect_summary(ax, patient_summary, stats_table):
    metrics = [
        ("BACH1+ fraction", "BACH1_detected_fraction"),
        ("NOD score", "NOD_like_score_mean"),
        ("same-spot rho", "BACH1_NOD_spearman_rho"),
        ("neighbor rho", "BACH1_to_neighbor_NOD_spearman_rho"),
        ("co-high Jaccard", "high_high_jaccard"),
    ]
    rows = []
    for label, metric in metrics:
        wide = patient_summary.pivot(index="patient_id", columns="tissue_group", values=metric).dropna(subset=["Adjacent", "Tumor"])
        delta = wide["Tumor"] - wide["Adjacent"]
        lo, hi = bootstrap_ci(delta)
        stat = stats_table.loc[stats_table["metric"].eq(metric)].iloc[0]
        rows.append(
            {
                "label": label,
                "metric": metric,
                "median_delta": float(np.median(delta)),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "wilcoxon_p": float(stat.wilcoxon_p),
                "n_pairs": int(len(delta)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(SRC / "figure5_panel_g_patient_paired_effects.csv", index=False)
    y = np.arange(len(df))[::-1]
    xmin = df["ci_low"].min() - 0.018
    xmax = df["ci_high"].max() + 0.035
    for row, yi in zip(df.itertuples(index=False), y):
        color = COL["tumor"] if row.median_delta >= 0 else COL["adjacent"]
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=color, lw=1.1)
        ax.scatter(row.median_delta, yi, s=24, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(xmax - 0.001, yi, f"p={row.wilcoxon_p:.3g}", ha="right", va="center", fontsize=5.4, color=COL["grey"])
    ax.axvline(0, color=COL["grey"], lw=0.8, ls="--")
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=6)
    ax.set_xlabel("paired median delta\nTumor - Adjacent")
    ax.set_title("Patient-paired spatial effects", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)


def main():
    sample_metrics = pd.read_csv(TABLE / "E-MTAB-13530_bach1_nod_sample_metrics.csv")
    patient_summary = pd.read_csv(TABLE / "E-MTAB-13530_patient_tumor_adjacent_summary.csv")
    stats_table = pd.read_csv(TABLE / "E-MTAB-13530_tumor_vs_adjacent_paired_wilcoxon.csv")

    tumor_sample = "P17_T1"
    adjacent_sample = "P17_B1"
    samples = [adjacent_sample, tumor_sample]
    adatas = {s: load_sample(s) for s in samples}
    selected_metrics = sample_metrics[sample_metrics["sample"].isin(samples)].copy()
    selected_metrics.to_csv(SRC / "figure5_panel_a_c_representative_section_metrics.csv", index=False)
    pd.concat([pd.read_csv(TABLE / f"{s}_spot_scores.csv.gz") for s in samples], ignore_index=True).to_csv(
        SRC / "figure5_panel_a_c_representative_spot_scores.csv.gz", index=False
    )
    sample_metrics.to_csv(SRC / "figure5_panel_d_sample_metrics.csv", index=False)
    patient_summary.to_csv(SRC / "figure5_panel_d_e_patient_paired_metrics.csv", index=False)
    stats_table.to_csv(SRC / "figure5_panel_d_e_paired_wilcoxon_statistics.csv", index=False)

    bach_vmax = np.nanpercentile(np.concatenate([adatas[s].obs["BACH1_log_norm"].to_numpy(float) for s in samples]), 99.3)
    nod_values = np.concatenate([adatas[s].obs["NOD_like_score_scanpy"].to_numpy(float) for s in samples])
    nod_vmin, nod_vmax = np.nanpercentile(nod_values, [1, 99])

    fig = plt.figure(figsize=(13.4, 9.1))
    outer = GridSpec(2, 3, figure=fig, height_ratios=[1.26, 0.92], width_ratios=[1.0, 1.0, 1.02], hspace=0.36, wspace=0.33)
    plate = outer[0, :].subgridspec(2, 5, width_ratios=[1.0, 0.045, 1.0, 0.045, 1.0], hspace=0.17, wspace=0.04)
    quant = outer[1, :].subgridspec(1, 4, width_ratios=[0.92, 0.92, 1.02, 1.20], wspace=0.43)

    rows = [("Adjacent", adjacent_sample), ("Tumor", tumor_sample)]
    spatial_axes = []
    for i, (group, sample) in enumerate(rows):
        adata = adatas[sample]
        row = selected_metrics[selected_metrics["sample"].eq(sample)].iloc[0]
        ax = fig.add_subplot(plate[i, 0])
        sc_b = plot_spatial_metric(ax, adata, "BACH1_log_norm", f"{sample} {group}\nBACH1 expression", "Reds", vmin=0, vmax=bach_vmax)
        add_spatial_annotation(ax, row)
        spatial_axes.append(ax)
        ax = fig.add_subplot(plate[i, 2])
        sc_n = plot_spatial_metric(ax, adata, "NOD_like_score_scanpy", f"{sample} {group}\nNOD-like pathway score", "YlGnBu", vmin=nod_vmin, vmax=nod_vmax)
        spatial_axes.append(ax)
        ax = fig.add_subplot(plate[i, 4])
        plot_cohigh(ax, adata, f"{sample} {group}\nBACH1/NOD high-state map")
        spatial_axes.append(ax)
    add_panel_label(spatial_axes[0], "a", x=-0.10, y=1.04)
    add_panel_label(spatial_axes[1], "b", x=-0.10, y=1.04)
    add_panel_label(spatial_axes[2], "c", x=-0.10, y=1.04)

    cax_b = fig.add_subplot(plate[:, 1])
    cax_n = fig.add_subplot(plate[:, 3])
    cbar_b = fig.colorbar(sc_b, cax=cax_b)
    cbar_b.set_label("BACH1 log-normalized", fontsize=5.6)
    cbar_b.ax.tick_params(labelsize=5, width=0.4)
    cbar_b.outline.set_linewidth(0.4)
    cbar_n = fig.colorbar(sc_n, cax=cax_n)
    cbar_n.set_label("NOD-like score", fontsize=5.6)
    cbar_n.ax.tick_params(labelsize=5, width=0.4)
    cbar_n.outline.set_linewidth(0.4)
    handles = [
        plt.Line2D([0], [0], marker="o", lw=0, ms=4.2, color=COL["nod"], label="NOD high"),
        plt.Line2D([0], [0], marker="o", lw=0, ms=4.2, color=COL["bach1"], label="BACH1 high"),
        plt.Line2D([0], [0], marker="o", lw=0, ms=4.2, color=COL["cohigh"], label="co-high"),
    ]
    spatial_axes[2].legend(
        handles=handles,
        loc="lower left",
        fontsize=5.4,
        handletextpad=0.2,
        labelspacing=0.25,
        borderpad=0.25,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
    )

    ax_d = fig.add_subplot(quant[0, 0])
    paired_metric(ax_d, patient_summary, stats_table, "BACH1_detected_fraction", "BACH1+ spot fraction", "BACH1+ spots expand in tumors")
    add_panel_label(ax_d, "d")

    ax_e = fig.add_subplot(quant[0, 1])
    paired_metric(ax_e, patient_summary, stats_table, "high_high_jaccard", "BACH1/NOD co-high Jaccard", "Co-high overlap increases")
    add_panel_label(ax_e, "e")

    ax_f = fig.add_subplot(quant[0, 2])
    plot_section_correlation(ax_f, sample_metrics)
    add_panel_label(ax_f, "f")

    ax_g = fig.add_subplot(quant[0, 3])
    plot_effect_summary(ax_g, patient_summary, stats_table)
    add_panel_label(ax_g, "g", x=-0.05)

    fig.suptitle(
        "Spatial transcriptomics validates tumor-enriched BACH1 expression and BACH1/NOD-like pathway co-localization",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.92, left=0.04, right=0.985, bottom=0.075)
    save_all(fig, "figure5_spatial_bach1_nod_validation")
    print(f"Wrote formal Figure 5 to {FIG}")
    print(f"Copied final PDFs to {FINAL}")
    print(f"Wrote source data to {SRC}")


if __name__ == "__main__":
    main()
