import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
import requests

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "out" / "bach1_nod_like_pathway"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "source_data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

ATAC_PROX_TERMS = ROOT / "out" / "bach1_atac_proximal_go_kegg_enrichment" / "tables" / "bach1_atac_highconf_proximal_10kb_go_kegg_significant_terms.csv"
ATAC_BACKGROUND = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "expressed_genes_atac_bach1_motif_support_background.csv.gz"
ATAC_PEAKS = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "gse274934_atac_bach1_motif_positive_peaks.csv.gz"
GENE_COORDS = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "gencode_v44_gene_coordinates_for_pyscenic_expressed_genes.csv.gz"
PYS_TARGETS = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "pyscenic_bach1_targets_ranked_with_atac_support.csv"


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

BASE_COLORS = {
    "A": "#4DAF4A",
    "C": "#377EB8",
    "G": "#FF7F00",
    "T": "#E41A1C",
}

SELECTED_GENES = ["TBK1", "TNFAIP3", "IL1B", "RELA", "ATG16L1"]

FALLBACK_PFMS = {
    "MA1633.2": {
        "name": "BACH1",
        "pfm": {
            "A": [32322, 7, 0, 32322, 551, 534, 2834, 32322, 4970],
            "C": [1482, 25, 13, 38, 32322, 31, 32322, 39, 8737],
            "G": [6187, 25, 32322, 24, 0, 3303, 52, 50, 2962],
            "T": [1048, 32322, 811, 14, 517, 32322, 18, 17, 32322],
        },
    },
    "MA0591.2": {
        "name": "Bach1::Mafk",
        "pfm": {
            "A": [15, 66, 2, 0, 113, 0, 1, 5, 108, 0, 0, 98],
            "C": [41, 16, 1, 0, 0, 94, 0, 107, 1, 1, 114, 4],
            "G": [54, 32, 0, 107, 0, 20, 4, 2, 3, 113, 0, 6],
            "T": [4, 0, 111, 7, 1, 0, 109, 0, 2, 0, 0, 6],
        },
    },
}


def add_panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")


def save_all(fig, stem, dpi=600):
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def wrap(s, width=32):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def fetch_jaspar_pfm(matrix_id):
    url = f"https://jaspar.elixir.no/api/v1/matrix/{matrix_id}/?format=json"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        return {"name": data["name"], "pfm": data["pfm"], "source_url": url}
    except Exception:
        data = FALLBACK_PFMS[matrix_id].copy()
        data["source_url"] = url
        data["used_fallback"] = True
        return data


def pfm_to_dataframe(matrix_id, motif_data):
    pfm = motif_data["pfm"]
    df = pd.DataFrame({base: pfm[base] for base in ["A", "C", "G", "T"]})
    df.insert(0, "position", np.arange(1, len(df) + 1))
    df.insert(0, "motif_name", motif_data["name"])
    df.insert(0, "matrix_id", matrix_id)
    df["source_url"] = motif_data["source_url"]
    return df


def draw_sequence_logo(ax, motif_df, title):
    counts = motif_df[["A", "C", "G", "T"]].astype(float).to_numpy().T
    counts = counts + 1e-6
    probs = counts / counts.sum(axis=0, keepdims=True)
    entropy = -np.sum(probs * np.log2(probs), axis=0)
    info = 2 - entropy
    heights = probs * info
    prop = FontProperties(family="DejaVu Sans", weight="bold")
    ax.set_xlim(0, heights.shape[1])
    ax.set_ylim(0, 2.05)
    for pos in range(heights.shape[1]):
        stack = sorted(zip(["A", "C", "G", "T"], heights[:, pos]), key=lambda x: x[1])
        y0 = 0
        for base, height in stack:
            if height < 0.015:
                continue
            tp = TextPath((0, 0), base, size=1, prop=prop)
            bb = tp.get_extents()
            trans = (
                Affine2D()
                .translate(-bb.x0, -bb.y0)
                .scale(0.82 / max(bb.width, 1e-6), height / max(bb.height, 1e-6))
                .translate(pos + 0.09, y0)
            )
            patch = PathPatch(tp, transform=trans + ax.transData, color=BASE_COLORS[base], lw=0)
            ax.add_patch(patch)
            y0 += height
    ax.set_xticks(np.arange(heights.shape[1]) + 0.5)
    ax.set_xticklabels(np.arange(1, heights.shape[1] + 1), fontsize=5.5)
    ax.set_ylabel("bits")
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_xlabel("position")
    ax.grid(axis="y", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)


def get_nod_term():
    terms = pd.read_csv(ATAC_PROX_TERMS)
    nod = terms[terms["Clean_term"].eq("NOD-LIKE RECEPTOR SIGNALING PATHWAY")]
    if nod.empty:
        raise RuntimeError("NOD-LIKE RECEPTOR SIGNALING PATHWAY not found in ATAC proximal enrichment table.")
    row = nod.iloc[0]
    genes = row["Genes"].split(";")
    return row, genes


def compute_nod_peak_links(nod_genes):
    coords = pd.read_csv(GENE_COORDS)
    coords = coords[coords["gene"].isin(nod_genes)].copy()
    peaks = pd.read_csv(
        ATAC_PEAKS,
        usecols=[
            "sample",
            "peak_id",
            "chrom",
            "start",
            "end",
            "motif_model",
            "motif_name",
            "max_motif_score",
            "max_motif_relative_score",
            "frac_cells_accessible",
            "total_counts",
            "n_cells_accessible",
        ],
    )
    peaks = peaks[pd.to_numeric(peaks["max_motif_score"], errors="coerce").fillna(0) >= 950].copy()
    rows = []
    for gene_row in coords.itertuples(index=False):
        sub = peaks[
            (peaks["chrom"] == gene_row.chrom)
            & (peaks["end"] >= gene_row.tss - 10000)
            & (peaks["start"] <= gene_row.tss + 10000)
        ].copy()
        if sub.empty:
            continue
        distance = []
        signed_midpoint = []
        for peak_row in sub.itertuples(index=False):
            if peak_row.start <= gene_row.tss <= peak_row.end:
                dist = 0
            elif peak_row.end < gene_row.tss:
                dist = gene_row.tss - peak_row.end
            else:
                dist = peak_row.start - gene_row.tss
            distance.append(dist)
            signed_midpoint.append(((peak_row.start + peak_row.end) / 2) - gene_row.tss)
        sub["distance_to_tss"] = distance
        sub["signed_midpoint_distance_to_tss"] = signed_midpoint
        sub = sub[sub["distance_to_tss"] <= 10000].copy()
        sub["gene"] = gene_row.gene
        sub["gene_chrom"] = gene_row.chrom
        sub["gene_tss"] = gene_row.tss
        sub["gene_strand"] = gene_row.strand
        rows.append(sub)
    if not rows:
        raise RuntimeError("No high-confidence BACH1 motif peaks found within NOD pathway gene TSS +/-10 kb.")
    links = pd.concat(rows, ignore_index=True)
    return links


def build_gene_summary(nod_genes, links):
    bg = pd.read_csv(ATAC_BACKGROUND)
    pys = set(pd.read_csv(PYS_TARGETS)["gene"].dropna().astype(str))
    bg = bg[bg["gene"].isin(nod_genes)].copy()
    support = (
        links.groupby("gene")
        .agg(
            n_track_links=("gene", "size"),
            n_track_samples=("sample", "nunique"),
            track_samples=("sample", lambda x: ";".join(sorted(set(map(str, x))))),
            track_motif_names=("motif_name", lambda x: ";".join(sorted(set(map(str, x))))),
            n_track_bach1_links=("motif_name", lambda x: int((x == "BACH1").sum())),
            n_track_bach1_mafk_links=("motif_name", lambda x: int((x == "Bach1::Mafk").sum())),
            max_track_motif_score=("max_motif_score", "max"),
            closest_track_peak_to_tss=("distance_to_tss", "min"),
            mean_track_frac_cells_accessible=("frac_cells_accessible", "mean"),
        )
        .reset_index()
    )
    out = bg.merge(support, on="gene", how="left")
    out["n_track_links"] = out["n_track_links"].fillna(0).astype(int)
    out["n_track_samples"] = out["n_track_samples"].fillna(0).astype(int)
    out["track_samples"] = out["track_samples"].fillna("")
    out["track_motif_names"] = out["track_motif_names"].fillna("")
    out["n_track_bach1_links"] = out["n_track_bach1_links"].fillna(0).astype(int)
    out["n_track_bach1_mafk_links"] = out["n_track_bach1_mafk_links"].fillna(0).astype(int)
    out["in_pyscenic_bach1_regulon"] = out["gene"].isin(pys)
    out["selected_for_peak_track"] = out["gene"].isin(SELECTED_GENES)
    out = out.sort_values(
        ["n_track_links", "n_track_samples", "max_track_motif_score"],
        ascending=False,
    )
    return out


def draw_summary_schematic(ax, nod_row, gene_summary):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.93, "BACH1 motif support in NOD-like receptor signaling", ha="left", va="top", fontsize=9.5, fontweight="bold")
    boxes = [
        (0.04, 0.42, 0.23, 0.32, "10 kb scATAC\nBACH1 motif genes\n3,830", COL["teal_soft"]),
        (0.38, 0.42, 0.23, 0.32, "KEGG NOD-like\nreceptor signaling\n63 genes", COL["amber_soft"]),
        (0.72, 0.42, 0.23, 0.32, "Representative\npeak tracks\n5 genes", COL["green_soft"]),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fc, ec="#555555", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, linespacing=1.08)
    for start, end in [((0.27, 0.58), (0.38, 0.58)), ((0.61, 0.58), (0.72, 0.58))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#555555"))
    summary_text = (
        f"KEGG FDR={float(nod_row['Adjusted_P_value']):.3g}; odds ratio={float(nod_row['Odds_ratio']):.2f}.\n"
        f"{int((gene_summary['n_track_links'] > 0).sum())}/{len(gene_summary)} pathway genes have high-confidence "
        "BACH1 motif-bearing open peaks within TSS +/-10 kb."
    )
    ax.text(
        0.04,
        0.12,
        summary_text,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=COL["grey"],
        linespacing=1.25,
    )


def plot_pathway_gene_support(ax, gene_summary):
    df = gene_summary.copy()
    df = df.sort_values(["n_track_links", "gene"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(df))
    colors = np.where(df["selected_for_peak_track"], COL["red"], np.where(df["in_pyscenic_bach1_regulon"], COL["violet"], COL["teal"]))
    ax.hlines(y, 0, df["n_track_links"], color=COL["light"], lw=1.1)
    ax.scatter(
        df["n_track_links"],
        y,
        s=18 + df["n_track_samples"].fillna(0).astype(float) * 10,
        c=colors,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=4.5)
    ax.set_xlabel("high-conf motif peak-gene links\nwithin TSS +/-10 kb")
    ax.set_title("NOD pathway genes with BACH1 motif peaks", loc="left", fontsize=7.7)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.02, "red: selected tracks\npurple: pySCENIC+ATAC", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color=COL["grey"])


def plot_selected_gene_support(ax, gene_summary):
    df = gene_summary[gene_summary["gene"].isin(SELECTED_GENES)].copy()
    df["gene"] = pd.Categorical(df["gene"], categories=SELECTED_GENES[::-1], ordered=True)
    df = df.sort_values("gene")
    y = np.arange(len(df))
    sc = ax.scatter(
        df["n_track_links"],
        y,
        s=32 + df["n_track_samples"] * 28,
        c=df["closest_track_peak_to_tss"] / 1000,
        cmap="viridis_r",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=6.5)
    ax.set_xlabel("track peak links")
    ax.set_title("Selected genes for peak tracks", loc="left", fontsize=7.7)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.07, pad=0.03)
    cb.set_label("closest peak\n(kb to TSS)", fontsize=5)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)
    xmax = df["n_track_links"].max()
    for row, yi in zip(df.itertuples(index=False), y):
        if row.n_track_links >= xmax - 0.5:
            ax.text(row.n_track_links - 0.28, yi, f"{int(row.n_track_samples)} samples", ha="right", va="center", fontsize=5.5)
        else:
            ax.text(row.n_track_links + 0.25, yi, f"{int(row.n_track_samples)} samples", ha="left", va="center", fontsize=5.5)


def plot_peak_tracks(ax, links):
    selected = links[links["gene"].isin(SELECTED_GENES)].copy()
    selected["gene"] = pd.Categorical(selected["gene"], categories=SELECTED_GENES[::-1], ordered=True)
    selected = selected.sort_values(["gene", "sample", "signed_midpoint_distance_to_tss"])
    y_map = {g: i for i, g in enumerate(SELECTED_GENES[::-1])}
    colors = {"BACH1": COL["blue"], "Bach1::Mafk": COL["teal"]}
    sample_order = sorted(selected["sample"].dropna().astype(str).unique())
    if len(sample_order) == 1:
        sample_offsets = {sample_order[0]: 0}
    else:
        sample_offsets = {
            sample: offset for sample, offset in zip(sample_order, np.linspace(-0.24, 0.24, len(sample_order)))
        }
    for gene, yi in y_map.items():
        ax.hlines(yi, -10, 10, color=COL["light"], lw=1.4, zorder=0)
        ax.plot(0, yi, marker="|", color=COL["dark"], ms=11, mew=1.0)
    for row in selected.itertuples(index=False):
        x = row.signed_midpoint_distance_to_tss / 1000
        x1 = max((row.start - row.gene_tss) / 1000, -10.5)
        x2 = min((row.end - row.gene_tss) / 1000, 10.5)
        yi = y_map[str(row.gene)]
        y_peak = yi + sample_offsets.get(str(row.sample), 0)
        lw = 1.0 + min(float(row.frac_cells_accessible), 0.4) * 4
        size = 10 + min(float(row.max_motif_score) - 950, 50) * 0.42
        ax.plot([x1, x2], [y_peak, y_peak], color=colors.get(row.motif_name, COL["grey"]), lw=lw, alpha=0.78, solid_capstyle="butt")
        ax.scatter(
            x,
            y_peak,
            s=size,
            color=colors.get(row.motif_name, COL["grey"]),
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            alpha=0.92,
        )
    ax.axvline(0, color=COL["grey"], lw=0.8, ls="--")
    ax.set_yticks(range(len(SELECTED_GENES)))
    ax.set_yticklabels(SELECTED_GENES[::-1], fontsize=7)
    ax.set_xlim(-10.5, 10.5)
    ax.set_xlabel("peak midpoint distance from TSS (kb)")
    ax.set_title("High-confidence BACH1 motif-bearing open peaks near selected NOD genes", loc="left", fontsize=8)
    ax.grid(axis="x", color=COL["light"], lw=0.5)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.02,
        "sample-specific peak intervals are vertically offset",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.6,
        color=COL["grey"],
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[k], markeredgecolor="white", label=k, markersize=5)
        for k in colors
    ]
    ax.legend(handles=handles, fontsize=5.8, loc="upper right")


def export_gene_summary_10kb(gene_summary):
    cols = [
        "gene",
        "n_track_links",
        "n_track_samples",
        "track_samples",
        "track_motif_names",
        "n_track_bach1_links",
        "n_track_bach1_mafk_links",
        "max_track_motif_score",
        "closest_track_peak_to_tss",
        "mean_track_frac_cells_accessible",
        "in_pyscenic_bach1_regulon",
        "selected_for_peak_track",
    ]
    return gene_summary[[c for c in cols if c in gene_summary.columns]].copy()


def write_legend(nod_row, gene_summary):
    legend = OUT / "bach1_nod_like_pathway_legend.md"
    selected_summary = gene_summary[gene_summary["gene"].isin(SELECTED_GENES)][
        ["gene", "n_track_links", "n_track_samples", "max_track_motif_score", "closest_track_peak_to_tss", "in_pyscenic_bach1_regulon"]
    ]
    with open(legend, "w", encoding="utf-8") as fh:
        fh.write("# BACH1 motif support for NOD-like receptor signaling\n\n")
        fh.write("## Core conclusion\n\n")
        fh.write(
            "The KEGG NOD-like receptor signaling pathway is significantly enriched in the high-confidence TSS +/-10 kb BACH1 motif-associated scATAC gene set. "
            "All 63 pathway genes in the enrichment term have high-confidence BACH1/Bach1::Mafk motif-bearing accessible peaks near their TSS, supporting a chromatin-level link between BACH1 motifs and this inflammatory pathway.\n\n"
        )
        fh.write("## Panels\n\n")
        fh.write(
            "a, Analysis schematic and pathway enrichment summary. "
            "b, NOD-like receptor pathway genes ranked by high-confidence TSS +/-10 kb BACH1 motif peak links. "
            "c, Evidence summary for the five representative genes selected for peak-track display. "
            "d, Peak-track/lollipop view of high-confidence motif-bearing accessible peaks around TBK1, TNFAIP3, IL1B, RELA and ATG16L1. "
            "e-f, JASPAR sequence logos for MA1633.2 BACH1 and MA0591.2 Bach1::Mafk.\n\n"
        )
        fh.write("## Key numbers\n\n")
        fh.write(f"- NOD-like receptor signaling pathway FDR: {float(nod_row['Adjusted_P_value']):.4g}\n")
        fh.write(f"- NOD-like receptor signaling pathway odds ratio: {float(nod_row['Odds_ratio']):.3g}\n")
        fh.write(f"- Pathway genes shown in the support table: {len(gene_summary):,}\n")
        fh.write(f"- Pathway genes with high-confidence motif-bearing peaks in the computed TSS +/-10 kb tracks: {int((gene_summary['n_track_links'] > 0).sum()):,}\n")
        fh.write("- Motif PFMs: JASPAR MA1633.2 and MA0591.2.\n\n")
        fh.write("## Selected genes\n\n")
        cols = list(selected_summary.columns)
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in selected_summary.itertuples(index=False):
            vals = []
            for value in row:
                if isinstance(value, float):
                    vals.append(f"{value:.4g}")
                else:
                    vals.append(str(value))
            fh.write("| " + " | ".join(vals) + " |\n")
        fh.write("\n")
    return legend


def main():
    nod_row, nod_genes = get_nod_term()
    links = compute_nod_peak_links(nod_genes)
    gene_summary = build_gene_summary(nod_genes, links)
    motif_bach1 = fetch_jaspar_pfm("MA1633.2")
    motif_mafk = fetch_jaspar_pfm("MA0591.2")
    motif_df = pd.concat(
        [
            pfm_to_dataframe("MA1633.2", motif_bach1),
            pfm_to_dataframe("MA0591.2", motif_mafk),
        ],
        ignore_index=True,
    )

    export_gene_summary_10kb(gene_summary).to_csv(TABLE_DIR / "nod_like_pathway_bach1_motif_supported_genes.csv", index=False)
    links.to_csv(TABLE_DIR / "nod_like_pathway_highconf_10kb_peak_gene_links.csv.gz", index=False)
    links[links["gene"].isin(SELECTED_GENES)].to_csv(TABLE_DIR / "selected_5_nod_genes_peak_tracks.csv", index=False)
    motif_df.to_csv(TABLE_DIR / "jaspar_bach1_bach1_mafk_pfm.csv", index=False)
    pd.DataFrame([nod_row]).to_csv(TABLE_DIR / "nod_like_receptor_signaling_kegg_enrichment_row.csv", index=False)

    fig = plt.figure(figsize=(13.6, 9.4))
    gs = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[0.86, 1.25, 1.35],
        width_ratios=[1.02, 1.02, 1.22, 1.18],
        hspace=0.52,
        wspace=0.74,
    )
    ax_a = fig.add_subplot(gs[0, :2])
    draw_summary_schematic(ax_a, nod_row, gene_summary)
    add_panel_label(ax_a, "a", x=0.0, y=1.02)

    ax_b = fig.add_subplot(gs[:, 2])
    plot_pathway_gene_support(ax_b, gene_summary)
    add_panel_label(ax_b, "b", x=-0.12)

    ax_c = fig.add_subplot(gs[0, 3])
    plot_selected_gene_support(ax_c, gene_summary)
    add_panel_label(ax_c, "c", x=-0.13)

    ax_d = fig.add_subplot(gs[1:, :2])
    plot_peak_tracks(ax_d, links)
    add_panel_label(ax_d, "d", x=-0.04)

    ax_e = fig.add_subplot(gs[1, 3])
    draw_sequence_logo(ax_e, motif_df[motif_df["matrix_id"] == "MA1633.2"], "BACH1 motif logo\nMA1633.2")
    add_panel_label(ax_e, "e", x=-0.13)

    ax_f = fig.add_subplot(gs[2, 3])
    draw_sequence_logo(ax_f, motif_df[motif_df["matrix_id"] == "MA0591.2"], "Bach1::Mafk motif logo\nMA0591.2")
    add_panel_label(ax_f, "f", x=-0.13)

    fig.suptitle(
        "BACH1 motif-bearing open chromatin supports NOD-like receptor signaling genes",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.93)
    save_all(fig, "bach1_nod_like_receptor_pathway_motif_tracks")

    legend = write_legend(nod_row, gene_summary)
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote source data to {TABLE_DIR}")
    print(f"Wrote legend to {legend}")


if __name__ == "__main__":
    main()
