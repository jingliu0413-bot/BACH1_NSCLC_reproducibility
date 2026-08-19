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
H5AD = BASE / "h5ad_by_sample"
TABLE = BASE / "tables"
OUT = BASE / "spatial_skill_supplement"
FIG = OUT / "figures"
SRC = OUT / "source_data"
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
}
GROUP_COL = {"Tumor": COL["tumor"], "Adjacent": COL["adjacent"], "Healthy": COL["healthy"]}


def add_panel_label(ax, label, x=-0.06, y=1.02):
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


def select_representative_pairs(sample_metrics):
    paired = (
        sample_metrics[sample_metrics["tissue_group"].isin(["Tumor", "Adjacent"])]
        .groupby(["patient_id", "tissue_group"], as_index=False)["BACH1_detected_fraction"]
        .mean()
        .pivot(index="patient_id", columns="tissue_group", values="BACH1_detected_fraction")
        .dropna()
    )
    paired["delta"] = paired["Tumor"] - paired["Adjacent"]
    selected_patients = paired.sort_values("delta", ascending=False).head(4).index.tolist()
    rows = []
    for patient in selected_patients:
        sub = sample_metrics[sample_metrics["patient_id"].eq(patient)]
        adjacent = sub[sub["tissue_group"].eq("Adjacent")].sort_values("BACH1_detected_fraction", ascending=False).iloc[0]
        tumor = sub[sub["tissue_group"].eq("Tumor")].sort_values("BACH1_detected_fraction", ascending=False).iloc[0]
        rows.append(adjacent)
        rows.append(tumor)
    selected = pd.DataFrame(rows)
    selected["display_disease"] = selected["disease"].replace(
        {
            "lung adenocarcinoma": "LUAD",
            "lung squamous cell carcinoma": "LUSC",
            "non-small cell carcinoma": "NSCLC",
        }
    )
    return selected, paired.reset_index()


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


def load_samples(samples):
    return {sample: read_h5ad_compat(H5AD / f"{sample}_bach1_nod_scored.h5ad") for sample in samples}


def plot_bach1_map(ax, adata, row, vmax):
    image, coords = get_image_coords(adata)
    ax.imshow(image, origin="upper")
    vals = adata.obs["BACH1_log_norm"].to_numpy(float)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=6, cmap="Reds", vmin=0, vmax=vmax, linewidth=0, alpha=0.92)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    group_color = GROUP_COL[row.tissue_group]
    ax.text(
        0.02,
        0.98,
        f"{row.sample} | {row.tissue_group}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=COL["dark"],
        bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.2),
    )
    ax.text(
        0.02,
        0.03,
        f"BACH1+ {row.BACH1_detected_fraction:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color=group_color,
        bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.0),
    )
    return sc


def paired_metric(ax, patient_summary, stats_table, metric, ylabel, title):
    wide = patient_summary.pivot(index="patient_id", columns="tissue_group", values=metric).dropna(subset=["Tumor", "Adjacent"])
    x = np.array([0, 1])
    for patient, row in wide.iterrows():
        ax.plot(x, [row["Adjacent"], row["Tumor"]], color="#B8B8B8", lw=0.75, zorder=1)
        ax.scatter(0, row["Adjacent"], s=18, color=COL["adjacent"], edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(1, row["Tumor"], s=18, color=COL["tumor"], edgecolor="white", linewidth=0.45, zorder=3)
    ax.plot(x, [wide["Adjacent"].median(), wide["Tumor"].median()], color=COL["dark"], lw=1.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Adjacent", "Tumor"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=8)
    p = stats_table.loc[stats_table["metric"].eq(metric), "wilcoxon_p"].iloc[0]
    delta = stats_table.loc[stats_table["metric"].eq(metric), "median_delta_tumor_minus_adjacent"].iloc[0]
    ax.text(0.5, 0.98, f"p={p:.3g}; median delta={delta:.3g}", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=COL["grey"])
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    return wide


def plot_patient_delta_bars(ax, paired_table):
    df = paired_table.sort_values("delta", ascending=True).copy()
    y = np.arange(len(df))
    ax.barh(y, df["delta"], color=COL["tumor"], edgecolor="white", lw=0.6)
    ax.axvline(0, color=COL["grey"], lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["patient_id"], fontsize=6)
    ax.set_xlabel("BACH1+ fraction delta\nTumor - Adjacent")
    ax.set_title("Patient-level BACH1 expansion", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    for yi, row in enumerate(df.itertuples(index=False)):
        ax.text(row.delta + 0.003, yi, f"{row.delta:.2f}", va="center", ha="left", fontsize=5.4, color=COL["grey"])


def plot_section_scatter(ax, sample_metrics):
    for group, sub in sample_metrics.groupby("tissue_group"):
        ax.scatter(
            sub["BACH1_detected_fraction"],
            sub["NOD_like_score_mean"],
            s=26,
            color=GROUP_COL[group],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            label=group,
        )
    ax.set_xlabel("BACH1+ spot fraction")
    ax.set_ylabel("mean NOD-like score")
    ax.set_title("Section-level BACH1/NOD relationship", loc="left", fontsize=8)
    ax.grid(color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.6, loc="lower right")


def dataframe_to_markdown(df):
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: f"{x:.4g}")
        else:
            text[col] = text[col].astype(str)
    header = "| " + " | ".join(text.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.to_numpy(dtype=str)]
    return "\n".join([header, divider, *rows])


def write_legend(selected, paired, stats_table):
    out = OUT / "figure3_revised_bach1_spatial_tumor_adjacent_legend.md"
    b_row = stats_table[stats_table["metric"].eq("BACH1_detected_fraction")].iloc[0]
    j_row = stats_table[stats_table["metric"].eq("high_high_jaccard")].iloc[0]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# Revised Figure 3: BACH1 spatial expression across tumor-adjacent sections\n\n")
        fh.write("## Core conclusion\n\n")
        fh.write(
            "Multiple representative patient-matched Visium sections show broader BACH1-positive spatial domains in tumor tissue than in adjacent lung. "
            "Across all eight paired patients, tumor sections have a higher BACH1-positive spot fraction and higher BACH1/NOD co-high spatial overlap.\n\n"
        )
        fh.write("## Representative section selection\n\n")
        fh.write(
            "Patients were ranked by patient-level Tumor - Adjacent BACH1-positive spot fraction. The four largest-delta patients were shown. "
            "For each selected patient, the displayed tumor and adjacent sections are the sections with the highest BACH1-positive spot fraction within that tissue group, which is conservative for adjacent tissue rather than selecting the lowest adjacent section.\n\n"
        )
        fh.write("## Key paired statistics\n\n")
        fh.write(
            f"- BACH1-positive spot fraction: n={int(b_row.n_pairs)} paired patients, Wilcoxon p={b_row.wilcoxon_p:.4g}, median Tumor - Adjacent delta={b_row.median_delta_tumor_minus_adjacent:.4g}\n"
        )
        fh.write(
            f"- BACH1/NOD co-high Jaccard: n={int(j_row.n_pairs)} paired patients, Wilcoxon p={j_row.wilcoxon_p:.4g}, median Tumor - Adjacent delta={j_row.median_delta_tumor_minus_adjacent:.4g}\n\n"
        )
        fh.write("## Displayed sections\n\n")
        fh.write(
            dataframe_to_markdown(
                selected[
                    [
                        "patient_id",
                        "sample",
                        "tissue_group",
                        "display_disease",
                        "BACH1_detected_fraction",
                        "BACH1_mean",
                        "NOD_like_score_mean",
                        "high_high_jaccard",
                    ]
                ]
            )
        )
        fh.write("\n")
    return out


def main():
    sample_metrics = pd.read_csv(TABLE / "E-MTAB-13530_bach1_nod_sample_metrics.csv")
    patient_summary = pd.read_csv(TABLE / "E-MTAB-13530_patient_tumor_adjacent_summary.csv")
    stats_table = pd.read_csv(TABLE / "E-MTAB-13530_tumor_vs_adjacent_paired_wilcoxon.csv")
    selected, paired_table = select_representative_pairs(sample_metrics)
    samples = selected["sample"].tolist()
    adatas = load_samples(samples)
    vmax = np.nanpercentile(np.concatenate([adatas[s].obs["BACH1_log_norm"].to_numpy(float) for s in samples]), 99.2)

    selected.to_csv(SRC / "figure3_revised_selected_bach1_spatial_sections.csv", index=False)
    paired_table.to_csv(SRC / "figure3_revised_patient_bach1_deltas.csv", index=False)
    patient_summary.to_csv(SRC / "figure3_revised_patient_paired_metrics.csv", index=False)
    stats_table.to_csv(SRC / "figure3_revised_paired_statistics.csv", index=False)

    fig = plt.figure(figsize=(13.4, 10.2))
    outer = GridSpec(2, 3, figure=fig, width_ratios=[1.05, 1.05, 0.92], height_ratios=[1, 1], wspace=0.28, hspace=0.30)
    plate = outer[:, :2].subgridspec(4, 2, hspace=0.08, wspace=0.02)
    quant = outer[:, 2].subgridspec(4, 1, hspace=0.48)

    map_axes = []
    for i, patient in enumerate(selected["patient_id"].drop_duplicates()):
        sub = selected[selected["patient_id"].eq(patient)].sort_values("tissue_group")
        # Sort manually so adjacent is left and tumor is right.
        sub = pd.concat([sub[sub["tissue_group"].eq("Adjacent")], sub[sub["tissue_group"].eq("Tumor")]])
        for j, row in enumerate(sub.itertuples(index=False)):
            ax = fig.add_subplot(plate[i, j])
            sc = plot_bach1_map(ax, adatas[row.sample], row, vmax=vmax)
            if i == 0:
                ax.set_title("Adjacent lung" if j == 0 else "Tumor", fontsize=8, pad=2)
            if j == 0:
                ax.text(
                    -0.04,
                    0.50,
                    f"{patient}\n{row.display_disease}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6.4,
                    rotation=90,
                    color=COL["dark"],
                )
            map_axes.append(ax)
    add_panel_label(map_axes[0], "a", x=-0.18, y=1.06)

    ax_b = fig.add_subplot(quant[0, 0])
    paired_metric(ax_b, patient_summary, stats_table, "BACH1_detected_fraction", "BACH1+ spot fraction", "Tumor sections show more BACH1+ spots")
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(quant[1, 0])
    paired_metric(ax_c, patient_summary, stats_table, "high_high_jaccard", "BACH1/NOD co-high Jaccard", "BACH1/NOD overlap is higher in tumors")
    add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(quant[2, 0])
    plot_patient_delta_bars(ax_d, paired_table)
    add_panel_label(ax_d, "d")

    ax_e = fig.add_subplot(quant[3, 0])
    plot_section_scatter(ax_e, sample_metrics)
    add_panel_label(ax_e, "e")

    fig.suptitle(
        "BACH1-positive spatial domains expand in NSCLC tumor sections relative to adjacent lung",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.93, left=0.05, right=0.98, bottom=0.06)
    fig.canvas.draw()
    plate_bbox = mpl.transforms.Bbox.union([ax.get_position() for ax in map_axes])
    quant_bbox = mpl.transforms.Bbox.union([ax.get_position() for ax in [ax_b, ax_c, ax_d, ax_e]])
    cbar_x = min(plate_bbox.x1 + 0.012, quant_bbox.x0 - 0.030)
    cbar_ax = fig.add_axes([cbar_x, plate_bbox.y0, 0.010, plate_bbox.height])
    cb = fig.colorbar(sc, cax=cbar_ax)
    cb.set_label("BACH1 log-normalized", fontsize=6)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)
    save_all(fig, "figure3_revised_bach1_spatial_tumor_adjacent")
    legend = write_legend(selected, paired_table, stats_table)
    print(f"Wrote revised figure to {FIG}")
    print(f"Wrote source data to {SRC}")
    print(f"Wrote legend to {legend}")


if __name__ == "__main__":
    main()
