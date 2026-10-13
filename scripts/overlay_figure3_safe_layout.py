#!/usr/bin/env python3
"""Add non-destructive publication annotations to the locked Figure 3 PDF."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = ROOT / "tmp" / "figure3_render" / "current_300.png"
OUT_DIR = ROOT / "outputs" / "submission_package"
PAGE_W = 901.2
PAGE_H = 627.612


def xy(x: float, y_top: float) -> tuple[float, float]:
    return x / PAGE_W, 1 - y_top / PAGE_H


def text(fig, x: float, y_top: float, value: str, *, size: float, weight: str = "normal",
         color: str = "#252525", ha: str = "left", va: str = "top",
         linespacing: float = 1.05, rotation: float = 0, zorder: int = 20) -> None:
    xf, yf = xy(x, y_top)
    fig.text(
        xf,
        yf,
        value,
        fontsize=size,
        fontweight=weight,
        color=color,
        family="Arial",
        ha=ha,
        va=va,
        linespacing=linespacing,
        rotation=rotation,
        zorder=zorder,
    )


def whiteout(fig, x: float, y_top: float, width: float, height: float, *, zorder: int = 10) -> None:
    fig.patches.append(
        Rectangle(
            (x / PAGE_W, 1 - (y_top + height) / PAGE_H),
            width / PAGE_W,
            height / PAGE_H,
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="none",
            zorder=zorder,
        )
    )


def divider(fig, x: float, y_top: float, y_bottom: float, *, color: str = "#B0B0B0",
            linewidth: float = 0.5) -> None:
    fig.lines.append(
        Line2D(
            [x / PAGE_W, x / PAGE_W],
            [1 - y_bottom / PAGE_H, 1 - y_top / PAGE_H],
            transform=fig.transFigure,
            color=color,
            linewidth=linewidth,
            zorder=11,
        )
        )


def image_panel(fig, panel_image: Image.Image, x: float, y_top: float,
                width: float, height: float, *, zorder: int = 35) -> None:
    """Place a raster crop in page coordinates without changing other panels."""
    panel_ax = fig.add_axes(
        [x / PAGE_W, 1 - (y_top + height) / PAGE_H, width / PAGE_W, height / PAGE_H],
        zorder=zorder,
    )
    panel_ax.imshow(np.asarray(panel_image), extent=(0, 1, 0, 1), interpolation="lanczos")
    panel_ax.set_axis_off()


def main() -> None:
    if not BACKGROUND.exists():
        raise FileNotFoundError(BACKGROUND)
    image = Image.open(BACKGROUND).convert("RGB")
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.asarray(image), extent=(0, 1, 0, 1), interpolation="lanczos")
    ax.set_axis_off()

    # Clear the old floating panel letters/titles before placing a single
    # publication-style header row for each panel.
    whiteout(fig, 0, 0, 450, 31)
    whiteout(fig, 450, 0, 451, 31)
    whiteout(fig, 0, 350, 450, 41)
    whiteout(fig, 450, 350, 451, 41)
    text(fig, 14, 24, "a", size=10.5, weight="bold", va="center")
    text(fig, 40, 25, "Malignant epithelial-state landscape", size=8.5, weight="bold", va="center")
    text(fig, 472, 8, "b", size=10.5, weight="bold", va="top")
    text(fig, 507, 8, "State-level BACH1 and transcriptional programmes", size=7.0, weight="bold", va="top")
    text(fig, 495, 28, "BACH1 detection fraction", size=5.3, weight="bold", ha="center", va="bottom")
    text(fig, 630, 28, "Transcriptional programmes", size=5.8, weight="bold", ha="center", va="bottom")
    text(fig, 25, 175, "UMAP 2", size=6.4, ha="center", va="center", rotation=90)
    whiteout(fig, 748, 135, 30, 105)
    text(fig, 758, 188, "Column z-score", size=6.0, ha="center", va="center", rotation=90)

    # The raster background is letterboxed by imshow's equal aspect, which
    # makes the top-left UMAP appear horizontally compressed. Replace only
    # that panel's source plotting rectangle with a square crop; the title and
    # the lower-row panels remain untouched.
    source_scale = image.width / PAGE_W
    a_src_x, a_src_y, a_src_w, a_src_h = 38, 31, 304, 287
    a_crop = image.crop(
        (
            round(a_src_x * source_scale),
            round(a_src_y * source_scale),
            round((a_src_x + a_src_w) * source_scale),
            round((a_src_y + a_src_h) * source_scale),
        )
    )
    a_square = 287
    a_crop = a_crop.resize(
        (round(a_square * source_scale), round(a_square * source_scale)),
        Image.Resampling.LANCZOS,
    )
    whiteout(fig, 130, a_src_y, 248, a_src_h, zorder=25)
    image_panel(fig, a_crop, a_src_x, a_src_y, a_square, a_square)

    text(fig, 16, 382, "c", size=10.5, weight="bold", va="center")
    text(fig, 44, 383, "Epithelial-state composition accounts for much of\nBACH1–hypoxia covariance", size=7.5, weight="bold", va="center")
    text(fig, 430, 382, "d", size=10.5, weight="bold", va="center")
    text(fig, 458, 383, "Within-state BACH1–hypoxia associations", size=7.5, weight="bold", va="center")

    # Replace the compact source labels/annotations with publication wording
    # while preserving the underlying points and confidence intervals.
    whiteout(fig, 0, 392, 224, 214)
    whiteout(fig, 315, 392, 60, 214)
    whiteout(fig, 375, 392, 70, 214)
    c_labels = [
        "Original scores",
        "Shared genes removed",
        "Dataset + QC adjusted",
        "Dataset + QC + state-PC adjusted",
        "Fully adjusted, ≥20-cell patients",
    ]
    c_y = [407, 455, 503, 551, 599]
    for label, y in zip(c_labels, c_y):
        text(fig, 220, y, label, size=6.7, ha="right", va="center")
    c_annotations = [
        "$\\rho$=0.91 [0.75, 0.97]\nn=16; $P$=1.1$\\times$10$^{-6}$",
        "$\\rho$=0.87 [0.65, 0.95]\nn=16; $P$=1.3$\\times$10$^{-5}$",
        "$\\rho$=0.51 [0.01, 0.80]\nn=16; $P$=0.046",
        "$\\rho$=0.14 [-0.39, 0.59]\nn=16; $P$=0.617",
        "$\\rho$=0.01 [-0.52, 0.54]\nn=14; $P$=0.970",
    ]
    for annotation, y in zip(c_annotations, c_y):
        text(fig, 322, y, annotation, size=7.6, va="center", linespacing=1.0)

    d_labels = ["Stress-associated", "Epithelial core", "AT2-like"]
    for label, y in zip(d_labels, [407, 503, 599]):
        text(fig, 448, y, label, size=6.7, ha="right", va="center")

    whiteout(fig, 630, 392, 95, 214)
    d_annotations = [
        "$\\rho$=0.88 [0.46, 0.98]\nn=8; Holm $P$=0.022",
        "$\\rho$=0.89 [0.26, 0.99]\nn=6; Holm $P$=0.067",
        "$\\rho$=0.10 [-0.65, 0.75]\nn=8; Holm $P$=0.840",
    ]
    for annotation, y in zip(d_annotations, [407, 503, 599]):
        text(fig, 645, y, annotation, size=8.0, va="center", linespacing=1.0)

    # The lower-row title band overlaps the long rotated heatmap labels in the
    # rasterised source. Clear that strip and redraw the labels compactly above
    # the lower row so none of the programme names are clipped.
    whiteout(fig, 470, 318, 380, 73)
    # Make the distinction between the detection column and programme scores
    # visible at a glance, rather than relying on a thin separator alone.
    whiteout(fig, 502, 31, 7, 287, zorder=25)
    divider(fig, 505, 31, 318, color="#B0B0B0", linewidth=0.5)
    programme_labels = [
        "BACH1 detection fraction",
        "DoRothEA BACH1",
        "Hypoxia",
        "Glycolysis",
        "NRF2/oxidative",
        "Heme metabolism",
        "ROS pathway",
        "OXPHOS",
        "EMT",
        "Proliferation",
    ]
    heatmap_centres = [495, 518, 541, 564, 587, 610, 633, 656, 679, 702]
    for x, label in zip(heatmap_centres, programme_labels):
        text(fig, x, 330, label, size=5.7, ha="right", va="top", rotation=42)

    # Rebuild the lower forest plots as a clean vector layer. This gives panel
    # c the larger visual share requested for the primary attenuation result
    # and keeps all statistics in fixed, non-overlapping columns.
    whiteout(fig, 0, 392, PAGE_W, PAGE_H - 392, zorder=30)

    def forest_axes(left: float, top: float, width: float, height: float, color: str):
        axis = fig.add_axes(
            [left / PAGE_W, 1 - (top + height) / PAGE_H, width / PAGE_W, height / PAGE_H],
            facecolor="white",
        )
        axis.set_zorder(40)
        axis.set_xlim(-1, 1)
        axis.set_ylim(-0.5, 4.5 if color == "#333333" else 2.5)
        axis.invert_yaxis()
        axis.axvline(0, color="#B8B8B8", linewidth=0.7, zorder=0)
        axis.tick_params(axis="both", labelsize=6.2, length=2, width=0.6, pad=2)
        axis.set_yticks([])
        axis.set_xticks([-1, -0.5, 0, 0.5, 1])
        axis.set_xlabel("Spearman ρ" if color == "#333333" else "Within-state Spearman ρ", fontsize=6.4, labelpad=5)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#222222")
        return axis

    c_axis = forest_axes(122, 399, 190, 183, "#333333")
    c_rho = [0.91, 0.87, 0.51, 0.14, 0.01]
    c_low = [0.75, 0.65, 0.01, -0.39, -0.52]
    c_high = [0.97, 0.95, 0.80, 0.59, 0.54]
    c_y = np.arange(5)
    c_axis.errorbar(
        c_rho,
        c_y,
        xerr=[np.asarray(c_rho) - np.asarray(c_low), np.asarray(c_high) - np.asarray(c_rho)],
        fmt="o",
        color="#333333",
        ecolor="#333333",
        markersize=5.8,
        linewidth=1.25,
        elinewidth=1.25,
        capsize=3,
        zorder=3,
    )
    c_labels = [
        "Original scores",
        "Shared genes removed",
        "Dataset + QC adjusted",
        "Dataset + QC + state-PC adjusted",
        "Fully adjusted, ≥20-cell patients",
    ]
    c_stat = [
        "$\\rho$=0.91 [0.75, 0.97]\nn=16; $P$=1.1$\\times$10$^{-6}$",
        "$\\rho$=0.87 [0.65, 0.95]\nn=16; $P$=1.3$\\times$10$^{-5}$",
        "$\\rho$=0.51 [0.01, 0.80]\nn=16; $P$=0.046",
        "$\\rho$=0.14 [-0.39, 0.59]\nn=16; $P$=0.617",
        "$\\rho$=0.01 [-0.52, 0.54]\nn=14; $P$=0.970",
    ]
    c_positions = [416, 452, 489, 526, 563]
    for label, y, annotation, y_text in zip(c_labels, c_y, c_stat, c_positions):
        text(fig, 112, y_text, label, size=6.6, ha="right", va="center", zorder=45)
        text(fig, 318, y_text, annotation, size=7.75, va="center", linespacing=1.0, zorder=45)

    d_axis = forest_axes(477, 399, 154, 183, "#4C72B0")
    d_rho = [0.88, 0.89, 0.10]
    d_low = [0.46, 0.26, -0.65]
    d_high = [0.98, 0.99, 0.75]
    d_y = np.arange(3)
    d_axis.errorbar(
        d_rho,
        d_y,
        xerr=[np.asarray(d_rho) - np.asarray(d_low), np.asarray(d_high) - np.asarray(d_rho)],
        fmt="o",
        color="#4C72B0",
        ecolor="#4C72B0",
        markersize=5.8,
        linewidth=1.25,
        elinewidth=1.25,
        capsize=3,
        zorder=3,
    )
    d_labels = ["Stress-associated", "Epithelial core", "AT2-like"]
    d_stat = [
        "$\\rho$=0.88 [0.46, 0.98]\nn=8; Holm $P$=0.022",
        "$\\rho$=0.89 [0.26, 0.99]\nn=6; Holm $P$=0.067",
        "$\\rho$=0.10 [-0.65, 0.75]\nn=8; Holm $P$=0.840",
    ]
    d_positions = [416, 489, 563]
    for label, annotation, y_text in zip(d_labels, d_stat, d_positions):
        text(fig, 469, y_text, label, size=6.6, ha="right", va="center", zorder=45)
        text(fig, 645, y_text, annotation, size=7.95, va="center", linespacing=1.0, zorder=45)

    text(fig, 16, 382, "c", size=10.5, weight="bold", va="center", zorder=45)
    text(fig, 44, 383, "Epithelial-state composition accounts for much of\nBACH1–hypoxia covariance", size=7.5, weight="bold", va="center", zorder=45)
    text(fig, 430, 382, "d", size=10.5, weight="bold", va="center", zorder=45)
    text(fig, 458, 383, "Within-state BACH1–hypoxia associations", size=7.5, weight="bold", va="center", zorder=45)

    output = OUT_DIR / "Figure3.pdf"
    fig.savefig(output, dpi=300, bbox_inches=None, pad_inches=0)
    print(output)


if __name__ == "__main__":
    main()
