import json
import math
import tarfile
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from scipy.spatial import cKDTree

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
DATA = ROOT / "data" / "spatial" / "E-MTAB-13530"
RAW = DATA / "raw"
VISIUM = DATA / "visium_dirs"
OUT = ROOT / "out" / "emtab13530_spatial_bach1_nod"
TABLE_DIR = OUT / "tables"
H5AD_DIR = OUT / "h5ad_by_sample"
for path in [VISIUM, TABLE_DIR, H5AD_DIR]:
    path.mkdir(parents=True, exist_ok=True)


NOD_LIKE_GENES = [
    "ANTXR1",
    "ANTXR2",
    "ATG12",
    "ATG16L1",
    "ATL1",
    "BIRC2",
    "BIRC3",
    "BRCC3",
    "CARD16",
    "CARD8",
    "CASP4",
    "CASP8",
    "CTSB",
    "CXCL1",
    "CXCL2",
    "CXCL3",
    "CXCL8",
    "CYBB",
    "ERBIN",
    "FADD",
    "GABARAP",
    "GABARAPL1",
    "GBP5",
    "HSP90AB1",
    "IFI16",
    "IFNAR2",
    "IKBKB",
    "IKBKE",
    "IKBKG",
    "IL18",
    "IL1B",
    "IL6",
    "ITPR1",
    "ITPR2",
    "JUN",
    "MAP3K7",
    "MAPK14",
    "MAVS",
    "MCU",
    "MFN1",
    "MFN2",
    "MYD88",
    "NFKBIA",
    "NLRX1",
    "OAS3",
    "PANX1",
    "PLCB2",
    "PYCARD",
    "RELA",
    "RHOA",
    "RIPK1",
    "STAT1",
    "TAB2",
    "TANK",
    "TBK1",
    "TNFAIP3",
    "TRAF3",
    "TRAF5",
    "TRAF6",
    "TRPM7",
    "TYK2",
    "VDAC1",
    "VDAC2",
]


def prepare_visium_dir(sample: str) -> Path:
    sample_dir = VISIUM / sample
    spatial_dir = sample_dir / "spatial"
    sample_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(exist_ok=True)

    src_h5 = RAW / f"{sample}-filtered_feature_bc_matrix.h5"
    dst_h5 = sample_dir / "filtered_feature_bc_matrix.h5"
    if not dst_h5.exists() or dst_h5.stat().st_size != src_h5.stat().st_size:
        dst_h5.write_bytes(src_h5.read_bytes())

    needed = {
        "aligned_fiducials.jpg",
        "detected_tissue_image.jpg",
        "scalefactors_json.json",
        "tissue_hires_image.png",
        "tissue_lowres_image.png",
        "tissue_positions_list.csv",
    }
    if not needed.issubset({p.name for p in spatial_dir.iterdir()}):
        with tarfile.open(RAW / f"{sample}-spatial.tar") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    member.name = Path(member.name).name
                    tar.extract(member, spatial_dir)
    return sample_dir


def load_sample_metadata() -> pd.DataFrame:
    sdrf = pd.read_csv(RAW / "E-MTAB-13530.sdrf.txt", sep="\t")
    meta = sdrf.drop_duplicates("Source Name").copy()
    meta = meta.rename(
        columns={
            "Source Name": "sample",
            "Characteristics[individual]": "patient",
            "Characteristics[original source name]": "original_source_name",
            "Characteristics[disease]": "disease",
            "Characteristics[disease staging]": "disease_staging",
            "Factor Value[sampling site]": "sampling_site",
            "Characteristics[age]": "age",
            "Characteristics[sex]": "sex",
        }
    )
    meta["patient_id"] = meta["patient"].str.replace("Patient ", "P", regex=False).str.replace("Donor ", "D", regex=False)
    meta["tissue_group"] = np.select(
        [
            meta["sampling_site"].eq("tumor"),
            meta["sampling_site"].eq("normal tissue adjacent to tumor"),
            meta["sampling_site"].eq("healthy tissue"),
        ],
        ["Tumor", "Adjacent", "Healthy"],
        default=meta["sampling_site"],
    )
    meta["sample_prefix"] = meta["sample"].str.extract(r"^([^_]+)_", expand=False)
    return meta[
        [
            "sample",
            "patient",
            "patient_id",
            "sample_prefix",
            "original_source_name",
            "disease",
            "disease_staging",
            "sampling_site",
            "tissue_group",
            "age",
            "sex",
        ]
    ].reset_index(drop=True)


def as_vector(x):
    if sparse.issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 5 or np.nanstd(x[good]) == 0 or np.nanstd(y[good]) == 0:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x[good], y[good])
    return float(rho), float(p)


def weighted_spatial_lag(values, coords, k=6):
    values = np.asarray(values, dtype=float)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(k + 1, len(coords)))
    if idx.ndim == 1:
        return np.full_like(values, np.nan, dtype=float)
    idx = idx[:, 1:]
    return np.nanmean(values[idx], axis=1)


def high_mask(values, quantile=0.75, require_positive=False):
    values = np.asarray(values, dtype=float)
    if require_positive:
        pos = values[values > 0]
        if len(pos) == 0:
            return np.zeros_like(values, dtype=bool)
        cutoff = max(np.nanquantile(values, quantile), np.nanmedian(pos))
        return values >= cutoff
    cutoff = np.nanquantile(values, quantile)
    return values >= cutoff


def fisher_high_high(bach1_high, nod_high):
    bach1_high = np.asarray(bach1_high, dtype=bool)
    nod_high = np.asarray(nod_high, dtype=bool)
    a = int(np.sum(bach1_high & nod_high))
    b = int(np.sum(bach1_high & ~nod_high))
    c = int(np.sum(~bach1_high & nod_high))
    d = int(np.sum(~bach1_high & ~nod_high))
    table = np.array([[a, b], [c, d]])
    odds, p = stats.fisher_exact(table)
    expected = bach1_high.mean() * nod_high.mean()
    observed = a / len(bach1_high) if len(bach1_high) else np.nan
    jaccard = a / max(int(np.sum(bach1_high | nod_high)), 1)
    enrich = observed / expected if expected > 0 else np.nan
    return {
        "high_high_n_spots": a,
        "bach1_high_n_spots": int(bach1_high.sum()),
        "nod_high_n_spots": int(nod_high.sum()),
        "high_high_fraction": observed,
        "high_high_expected_fraction": expected,
        "high_high_enrichment": enrich,
        "high_high_fisher_odds_ratio": odds,
        "high_high_fisher_p": p,
        "high_high_jaccard": jaccard,
    }


def wilcoxon_patient_test(patient_summary, metric):
    wide = patient_summary.pivot(index="patient_id", columns="tissue_group", values=metric)
    wide = wide.dropna(subset=["Tumor", "Adjacent"])
    if len(wide) < 3:
        return {"metric": metric, "n_pairs": len(wide), "wilcoxon_p": np.nan, "median_delta_tumor_minus_adjacent": np.nan}
    try:
        stat, p = stats.wilcoxon(wide["Tumor"], wide["Adjacent"], zero_method="wilcox")
    except ValueError:
        p = np.nan
    return {
        "metric": metric,
        "n_pairs": int(len(wide)),
        "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
        "median_delta_tumor_minus_adjacent": float(np.nanmedian(wide["Tumor"] - wide["Adjacent"])),
    }


def process_sample(row, nod_genes):
    sample = row.sample
    sample_dir = prepare_visium_dir(sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_visium(str(sample_dir))
    adata.var_names_make_unique()
    adata.obs_names = [f"{sample}_{x}" for x in adata.obs_names]
    adata.obs["sample"] = sample
    for col in [
        "patient",
        "patient_id",
        "sample_prefix",
        "original_source_name",
        "disease",
        "disease_staging",
        "sampling_site",
        "tissue_group",
        "age",
        "sex",
    ]:
        adata.obs[col] = getattr(row, col)

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    present_nod = [g for g in nod_genes if g in adata.var_names]
    missing_nod = sorted(set(nod_genes) - set(present_nod))
    sc.tl.score_genes(
        adata,
        gene_list=present_nod,
        score_name="NOD_like_score_scanpy",
        ctrl_size=min(50, len(present_nod)),
        random_state=13,
        use_raw=False,
    )
    if "BACH1" in adata.var_names:
        bach1 = as_vector(adata[:, "BACH1"].X)
    else:
        bach1 = np.full(adata.n_obs, np.nan)
    nod_expr = as_vector(adata[:, present_nod].X.mean(axis=1))

    adata.obs["BACH1_log_norm"] = bach1
    adata.obs["BACH1_detected"] = bach1 > 0
    adata.obs["NOD_like_mean_logexpr"] = nod_expr

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    nod_lag = weighted_spatial_lag(adata.obs["NOD_like_score_scanpy"].to_numpy(), coords, k=6)
    bach1_lag = weighted_spatial_lag(adata.obs["BACH1_log_norm"].to_numpy(), coords, k=6)
    rho, p = safe_spearman(adata.obs["BACH1_log_norm"], adata.obs["NOD_like_score_scanpy"])
    lag_rho, lag_p = safe_spearman(adata.obs["BACH1_log_norm"], nod_lag)
    reverse_lag_rho, reverse_lag_p = safe_spearman(adata.obs["NOD_like_score_scanpy"], bach1_lag)

    bach1_high = high_mask(adata.obs["BACH1_log_norm"], 0.75, require_positive=True)
    nod_high = high_mask(adata.obs["NOD_like_score_scanpy"], 0.75, require_positive=False)
    adata.obs["BACH1_high"] = bach1_high
    adata.obs["NOD_like_high"] = nod_high
    adata.obs["BACH1_NOD_cohigh"] = bach1_high & nod_high
    adata.obs["BACH1_NOD_colocalization_z"] = (
        stats.zscore(adata.obs["BACH1_log_norm"].to_numpy(), nan_policy="omit")
        * stats.zscore(adata.obs["NOD_like_score_scanpy"].to_numpy(), nan_policy="omit")
    )

    high_stats = fisher_high_high(bach1_high, nod_high)
    metrics = {
        "sample": sample,
        "patient_id": row.patient_id,
        "patient": row.patient,
        "disease": row.disease,
        "tissue_group": row.tissue_group,
        "sampling_site": row.sampling_site,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "median_counts": float(np.nanmedian(adata.obs["total_counts"])),
        "median_genes": float(np.nanmedian(adata.obs["n_genes_by_counts"])),
        "median_pct_mt": float(np.nanmedian(adata.obs["pct_counts_mt"])),
        "BACH1_mean": float(np.nanmean(adata.obs["BACH1_log_norm"])),
        "BACH1_median": float(np.nanmedian(adata.obs["BACH1_log_norm"])),
        "BACH1_detected_fraction": float(np.nanmean(adata.obs["BACH1_detected"])),
        "NOD_like_score_mean": float(np.nanmean(adata.obs["NOD_like_score_scanpy"])),
        "NOD_like_score_median": float(np.nanmedian(adata.obs["NOD_like_score_scanpy"])),
        "NOD_like_mean_logexpr_mean": float(np.nanmean(adata.obs["NOD_like_mean_logexpr"])),
        "BACH1_NOD_spearman_rho": rho,
        "BACH1_NOD_spearman_p": p,
        "BACH1_to_neighbor_NOD_spearman_rho": lag_rho,
        "BACH1_to_neighbor_NOD_spearman_p": lag_p,
        "NOD_to_neighbor_BACH1_spearman_rho": reverse_lag_rho,
        "NOD_to_neighbor_BACH1_spearman_p": reverse_lag_p,
        "n_nod_genes_present": int(len(present_nod)),
        "n_nod_genes_missing": int(len(missing_nod)),
        "missing_nod_genes": ";".join(missing_nod),
    }
    metrics.update(high_stats)

    spot_table = adata.obs[
        [
            "sample",
            "patient_id",
            "disease",
            "tissue_group",
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_mt",
            "BACH1_log_norm",
            "BACH1_detected",
            "NOD_like_score_scanpy",
            "NOD_like_mean_logexpr",
            "BACH1_high",
            "NOD_like_high",
            "BACH1_NOD_cohigh",
            "BACH1_NOD_colocalization_z",
            "array_row",
            "array_col",
        ]
    ].copy()
    spot_table["barcode"] = spot_table.index
    spot_table["spatial_x"] = coords[:, 0]
    spot_table["spatial_y"] = coords[:, 1]
    spot_table.to_csv(TABLE_DIR / f"{sample}_spot_scores.csv.gz", index=False)
    adata.write_h5ad(H5AD_DIR / f"{sample}_bach1_nod_scored.h5ad", compression="gzip")
    return metrics, spot_table


def main():
    meta = load_sample_metadata()
    meta.to_csv(TABLE_DIR / "E-MTAB-13530_sample_metadata.csv", index=False)
    (TABLE_DIR / "nod_like_receptor_pathway_gene_set.txt").write_text("\n".join(NOD_LIKE_GENES) + "\n", encoding="utf-8")

    metrics = []
    spot_tables = []
    for row in meta.itertuples(index=False):
        print(f"Processing {row.sample} ({row.tissue_group})", flush=True)
        sample_metrics, spot_table = process_sample(row, NOD_LIKE_GENES)
        metrics.append(sample_metrics)
        spot_tables.append(spot_table)

    sample_metrics = pd.DataFrame(metrics)
    sample_metrics.to_csv(TABLE_DIR / "E-MTAB-13530_bach1_nod_sample_metrics.csv", index=False)
    pd.concat(spot_tables, ignore_index=True).to_csv(TABLE_DIR / "E-MTAB-13530_bach1_nod_all_spot_scores.csv.gz", index=False)

    patient_summary = (
        sample_metrics[sample_metrics["tissue_group"].isin(["Tumor", "Adjacent"])]
        .groupby(["patient_id", "disease", "tissue_group"], as_index=False)
        .agg(
            n_samples=("sample", "nunique"),
            n_spots=("n_spots", "sum"),
            BACH1_mean=("BACH1_mean", "mean"),
            BACH1_detected_fraction=("BACH1_detected_fraction", "mean"),
            NOD_like_score_mean=("NOD_like_score_mean", "mean"),
            NOD_like_mean_logexpr_mean=("NOD_like_mean_logexpr_mean", "mean"),
            BACH1_NOD_spearman_rho=("BACH1_NOD_spearman_rho", "mean"),
            BACH1_to_neighbor_NOD_spearman_rho=("BACH1_to_neighbor_NOD_spearman_rho", "mean"),
            high_high_enrichment=("high_high_enrichment", "mean"),
            high_high_jaccard=("high_high_jaccard", "mean"),
        )
    )
    patient_summary.to_csv(TABLE_DIR / "E-MTAB-13530_patient_tumor_adjacent_summary.csv", index=False)

    test_metrics = [
        "BACH1_mean",
        "BACH1_detected_fraction",
        "NOD_like_score_mean",
        "NOD_like_mean_logexpr_mean",
        "BACH1_NOD_spearman_rho",
        "BACH1_to_neighbor_NOD_spearman_rho",
        "high_high_enrichment",
        "high_high_jaccard",
    ]
    stats_table = pd.DataFrame([wilcoxon_patient_test(patient_summary, m) for m in test_metrics])
    stats_table.to_csv(TABLE_DIR / "E-MTAB-13530_tumor_vs_adjacent_paired_wilcoxon.csv", index=False)

    summary = {
        "n_samples": int(len(meta)),
        "n_tumor_samples": int((meta["tissue_group"] == "Tumor").sum()),
        "n_adjacent_samples": int((meta["tissue_group"] == "Adjacent").sum()),
        "n_healthy_samples": int((meta["tissue_group"] == "Healthy").sum()),
        "n_patients_tumor_adjacent": int(patient_summary["patient_id"].nunique()),
        "n_nod_like_genes_input": int(len(NOD_LIKE_GENES)),
        "output_tables": sorted([p.name for p in TABLE_DIR.glob("*")]),
        "output_h5ad_files": len(list(H5AD_DIR.glob("*.h5ad"))),
    }
    (OUT / "E-MTAB-13530_bach1_nod_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
