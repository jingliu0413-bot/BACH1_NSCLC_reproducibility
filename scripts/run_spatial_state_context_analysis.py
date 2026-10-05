"""Spatial projection and permutation-based context analysis for GSE292299.

This exploratory analysis projects the frozen BACH1/hypoxia scores and the
single-cell-derived epithelial-state programmes onto four lung Visium samples.
Spot-level correlations are descriptive. Spatial concordance is quantified by
bivariate Moran's I on a six-neighbour array-coordinate graph and compared with
within-section label permutations that preserve the observed score distribution.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from scipy import sparse, stats
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "manuscript_submission_20260911"
MATRIX_DIR = ROOT / "data" / "spatial_candidates" / "GSE292299_lung_h5"
SPATIAL_DIR = ROOT / "data" / "spatial_candidates" / "GSE292299_spatial_extracted"
METADATA = ROOT / "data" / "spatial_candidates" / "GSE292299_sample_metadata.csv.gz"
STRESS_SIGNATURE = (
    SOURCE / "stress_ap1_state_signature_genes.csv"
)
OUT = ROOT / "out" / "spatial_state_context_analysis"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20261004
N_PERMUTATIONS = int(os.environ.get("SPATIAL_N_PERMUTATIONS", "1000"))
DEOVERLAP = {"ALDOA", "HMOX1", "IL6"}

STATE_MARKERS = {
    "Stress_AP1": None,
    "Epithelial_core": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "CLDN4", "KRT7"],
    "AT2_like": ["SFTPA1", "SFTPA2", "SFTPB", "SFTPC", "SLC34A2", "ABCA3", "NAPSA", "LAMP3"],
    "Ciliated": ["FOXJ1", "TPPP3", "PIFO", "DNAH5", "DNAI1", "CAPS"],
}


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(values)
    return (values - mean) / sd


def score(adata: sc.AnnData, genes: list[str], weights: list[float]) -> tuple[np.ndarray, list[str]]:
    gene_names = pd.Index(adata.var_names.astype(str)).str.upper()
    parts = []
    present = []
    for gene, weight in zip(genes, weights):
        hits = np.flatnonzero(gene_names.to_numpy() == gene)
        if len(hits) == 0:
            continue
        values = adata.X[:, hits]
        if sparse.issparse(values):
            values = values.toarray()
        values = np.nanmedian(np.asarray(values, dtype=float), axis=1)
        parts.append(float(weight) * zscore(values))
        present.append(gene)
    if not parts:
        return np.full(adata.n_obs, np.nan), []
    weights_present = [float(w) for g, w in zip(genes, weights) if g in present]
    denom = max(np.sum(np.abs(weights_present)), 1e-12)
    return np.sum(parts, axis=0) / denom, present


def load_signature_targets() -> dict[str, tuple[list[str], list[float]]]:
    external = pd.read_csv(SOURCE / "external_bach1_signature_targets.csv")
    phenotype = pd.read_csv(SOURCE / "phenotype_pathway_signature_targets.csv")
    external["target"] = external["target"].astype(str).str.upper()
    phenotype["target"] = phenotype["target"].astype(str).str.upper()
    out = {}
    for signature, source in [
        ("DOROTHEA_BACH1", external[external["signature"].eq("DOROTHEA_BACH1_ABC_TF_ACTIVITY")]),
        ("HALLMARK_HYPOXIA", phenotype[phenotype["signature"].eq("HALLMARK_HYPOXIA")]),
        ("KLENJA2025_BACH1", external[external["signature"].eq("KLENJA2025_BACH1_INVERSE_ACTIVITY")]),
    ]:
        source = source.copy()
        source = source[source["present_in_primary"].astype(str).str.lower().isin(["true", "1", "yes"])]
        source = source[source["target_in_adata"].notna()]
        source["target"] = source["target_in_adata"].astype(str).str.upper()
        source = source.drop_duplicates("target")
        if signature in {"DOROTHEA_BACH1", "HALLMARK_HYPOXIA"}:
            source = source[~source["target"].isin(DEOVERLAP)]
        out[signature] = (source["target"].tolist(), source["weight"].astype(float).tolist())
    if not STRESS_SIGNATURE.exists():
        raise FileNotFoundError(f"Missing regenerated stress signature: {STRESS_SIGNATURE}")
    stress = pd.read_csv(STRESS_SIGNATURE)
    stress_genes = stress["gene"].astype(str).str.upper().tolist()
    out["Stress_AP1"] = (stress_genes, [1.0] * len(stress_genes))
    for state, genes in STATE_MARKERS.items():
        if genes is not None:
            out[state] = ([g.upper() for g in genes], [1.0] * len(genes))
    return out


def find_spatial_files(sample: str) -> dict[str, Path]:
    root = SPATIAL_DIR / sample
    positions = next(root.rglob("tissue_positions.csv"))
    scalefactors = next(root.rglob("scalefactors_json.json"))
    hires = next(root.rglob("tissue_hires_image.png"))
    lowres = next(root.rglob("tissue_lowres_image.png"))
    return {"positions": positions, "scalefactors": scalefactors, "hires": hires, "lowres": lowres}


def six_neighbour_graph(coords: np.ndarray) -> sparse.csr_matrix:
    n = len(coords)
    k = min(7, n)
    _, indices = cKDTree(coords).query(coords, k=k)
    rows = np.repeat(np.arange(n), k - 1)
    cols = indices[:, 1:].reshape(-1)
    data = np.full(len(rows), 1.0 / max(k - 1, 1))
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def bivariate_moran(x: np.ndarray, y: np.ndarray, weights: sparse.csr_matrix) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    if not ok.all():
        raise ValueError("Scores must be finite before spatial graph construction")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
    if denom == 0:
        return np.nan
    n = len(x)
    w_sum = float(weights.sum())
    return float((n / w_sum) * np.dot(xc, weights.dot(yc)) / denom)


def spatial_permutation(x: np.ndarray, y: np.ndarray, weights: sparse.csr_matrix, rng: np.random.Generator) -> tuple[float, float, float, float]:
    observed = bivariate_moran(x, y, weights)
    y_centered = y - y.mean()
    x_centered = x - x.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    n = len(x)
    w_sum = float(weights.sum())
    null = np.empty(N_PERMUTATIONS, dtype=float)
    for start in range(0, N_PERMUTATIONS, 100):
        stop = min(start + 100, N_PERMUTATIONS)
        shuffled = np.column_stack([rng.permutation(y_centered) for _ in range(stop - start)])
        numerators = x_centered @ (weights @ shuffled)
        null[start:stop] = (n / w_sum) * numerators / denom
    p_positive = (1 + np.sum(null >= observed)) / (N_PERMUTATIONS + 1)
    return float(observed), float(p_positive), float(np.nanmean(null)), float(np.nanstd(null))


def descriptive_rho(x: np.ndarray, y: np.ndarray) -> float:
    result = stats.spearmanr(x, y)
    return float(result.statistic)


def overlay_map(ax, image_path: Path, positions: pd.DataFrame, values: np.ndarray, title: str, cmap: str) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    ax.imshow(image, origin="upper")
    finite = values[np.isfinite(values)]
    if finite.size:
        lo, hi = np.quantile(finite, [0.02, 0.98])
        if lo == hi:
            lo, hi = finite.min(), finite.max() + 1e-9
    else:
        lo, hi = 0, 1
    ax.scatter(
        positions["pxl_col_lowres"],
        positions["pxl_row_lowres"],
        c=values,
        s=3,
        alpha=0.75,
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        linewidths=0,
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")


def main() -> None:
    signatures = load_signature_targets()
    metadata = pd.read_csv(METADATA).set_index("ID_Sample")
    rng = np.random.default_rng(RANDOM_SEED)
    score_rows = []
    spot_rows = []
    spatial_rows = []
    map_data = []

    for path in sorted(MATRIX_DIR.glob("*.h5")):
        match = re.search(r"_(NSCLC_P\d+)_filtered", path.name)
        if not match:
            continue
        sample = match.group(1)
        adata = sc.read_10x_h5(path, gex_only=True)
        raw_total = np.asarray(adata.X.sum(axis=1)).ravel() if sparse.issparse(adata.X) else np.asarray(adata.X).sum(axis=1)
        raw_n_genes = np.asarray((adata.X > 0).sum(axis=1)).ravel()
        raw_metrics = pd.DataFrame(
            {
                "raw_total_counts": raw_total,
                "raw_n_genes": raw_n_genes,
            },
            index=adata.obs_names,
        )
        adata.var_names = adata.var_names.astype(str).str.upper()
        adata.var_names_make_unique()
        sc.pp.normalize_total(adata, target_sum=10_000)
        sc.pp.log1p(adata)

        spatial_files = find_spatial_files(sample)
        positions = pd.read_csv(spatial_files["positions"])
        positions["barcode"] = positions["barcode"].astype(str)
        positions = positions[positions["in_tissue"].eq(1)].copy()
        positions = positions.set_index("barcode")
        common = adata.obs_names.intersection(positions.index)
        adata = adata[common].copy()
        positions = positions.loc[common].copy()
        raw_metrics = raw_metrics.loc[common]
        positions["pxl_col_lowres"] = positions["pxl_col_in_fullres"]
        positions["pxl_row_lowres"] = positions["pxl_row_in_fullres"]
        with open(spatial_files["scalefactors"], encoding="utf-8") as handle:
            scalefactors = json.load(handle)
        positions["pxl_col_lowres"] *= float(scalefactors["tissue_lowres_scalef"])
        positions["pxl_row_lowres"] *= float(scalefactors["tissue_lowres_scalef"])

        score_values = {}
        for signature, (genes, weights) in signatures.items():
            values, present = score(adata, genes, weights)
            score_values[signature] = values
            score_rows.append(
                {
                    "sample": sample,
                    "n_spots_in_tissue": len(adata),
                    "signature": signature,
                    "n_targets": len(genes),
                    "n_present": len(present),
                    "coverage": len(present) / len(genes) if genes else np.nan,
                    "mean": float(np.nanmean(values)),
                    "sd": float(np.nanstd(values)),
                    "mean_raw_library_size": float(np.mean(raw_total)),
                    "mean_raw_detected_genes": float(np.mean(raw_n_genes)),
                    }
                )

        spot_table = positions.reset_index().rename(columns={"index": "barcode"})
        spot_table.insert(0, "sample", sample)
        spot_table["histology"] = metadata.loc[sample, "Diagnosis_Subtype"]
        spot_table["treatment_response"] = metadata.loc[sample, "Tx_Response"]
        spot_table["raw_total_counts"] = raw_metrics["raw_total_counts"].to_numpy()
        spot_table["raw_n_genes"] = raw_metrics["raw_n_genes"].to_numpy()
        for signature, values in score_values.items():
            spot_table[signature] = values
        spot_rows.append(spot_table)

        graph_coords = positions[["array_row", "array_col"]].to_numpy(dtype=float)
        graph = six_neighbour_graph(graph_coords)
        comparisons = [
            ("DoRothEA_vs_hypoxia", "DOROTHEA_BACH1", "HALLMARK_HYPOXIA"),
            ("Stress_AP1_vs_hypoxia", "Stress_AP1", "HALLMARK_HYPOXIA"),
            ("Stress_AP1_vs_DoRothEA", "Stress_AP1", "DOROTHEA_BACH1"),
            ("Epithelial_core_vs_hypoxia", "Epithelial_core", "HALLMARK_HYPOXIA"),
            ("AT2_like_vs_hypoxia", "AT2_like", "HALLMARK_HYPOXIA"),
        ]
        for comparison, left, right in comparisons:
            x = score_values[left]
            y = score_values[right]
            rho = descriptive_rho(x, y)
            observed, p_perm, null_mean, null_sd = spatial_permutation(x, y, graph, rng)
            spatial_rows.append(
                {
                    "sample": sample,
                    "biopsy_tissue": metadata.loc[sample, "Bx_Tissue"],
                    "histology": metadata.loc[sample, "Diagnosis_Subtype"],
                    "treatment_response": metadata.loc[sample, "Tx_Response"],
                    "comparison": comparison,
                    "n_spots_in_tissue": len(adata),
                    "spearman_rho_descriptive": rho,
                    "bivariate_moran_I": observed,
                    "permutation_p_positive": p_perm,
                    "permutation_null_mean": null_mean,
                    "permutation_null_sd": null_sd,
                    "n_permutations": N_PERMUTATIONS,
                    "random_seed": RANDOM_SEED,
                    "inference_note": "Section-level exploratory spatial permutation; not a patient-level inferential test.",
                }
            )

        map_data.append((sample, positions.copy(), score_values, spatial_files["lowres"]))

    scores = pd.DataFrame(score_rows)
    spots = pd.concat(spot_rows, ignore_index=True)
    spatial = pd.DataFrame(spatial_rows)
    scores.to_csv(OUT / "GSE292299_spatial_score_projection.csv", index=False)
    spots.to_csv(OUT / "GSE292299_spatial_spot_scores.csv.gz", index=False)
    spatial.to_csv(OUT / "GSE292299_spatial_moran_permutation_summary.csv", index=False)

    fig, axes = plt.subplots(len(map_data), 4, figsize=(12, 3.0 * len(map_data)), squeeze=False)
    for row_index, (sample, positions, values, lowres) in enumerate(map_data):
        overlay_map(axes[row_index, 0], lowres, positions, values["DOROTHEA_BACH1"], f"{sample} DoRothEA", "viridis")
        overlay_map(axes[row_index, 1], lowres, positions, values["HALLMARK_HYPOXIA"], f"{sample} hypoxia", "magma")
        overlay_map(axes[row_index, 2], lowres, positions, values["Stress_AP1"], f"{sample} Stress_AP1", "plasma")
        overlay_map(axes[row_index, 3], lowres, positions, values["Epithelial_core"], f"{sample} epithelial core", "cividis")
    fig.suptitle("GSE292299 spatial score projections (exploratory)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT / "GSE292299_spatial_score_maps.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "GSE292299_spatial_score_maps.pdf", bbox_inches="tight")
    plt.close(fig)

    key = spatial[spatial["comparison"].eq("DoRothEA_vs_hypoxia")].copy()
    summary = {
        "dataset": "GSE292299",
        "n_lung_samples": int(len(map_data)),
        "sample_labels": metadata.loc[[x[0] for x in map_data], ["Bx_Tissue", "Diagnosis_Subtype", "Tx_Response"]].reset_index().to_dict(orient="records"),
        "dorothea_vs_hypoxia": key[["sample", "spearman_rho_descriptive", "bivariate_moran_I", "permutation_p_positive"]].to_dict(orient="records"),
        "direction_consistency": {
            "spearman_positive_n": int((key["spearman_rho_descriptive"] > 0).sum()),
            "moran_positive_n": int((key["bivariate_moran_I"] > 0).sum()),
            "n_samples": int(len(key)),
        },
        "interpretation": "Exploratory section-level spatial concordance; coordinates and tissue images were used, but no patient-level inferential claim is made.",
    }
    (OUT / "GSE292299_spatial_state_context_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(spatial.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
