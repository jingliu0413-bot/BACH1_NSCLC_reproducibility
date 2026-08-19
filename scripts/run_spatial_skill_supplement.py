import json
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy import sparse, stats
from sklearn.preprocessing import normalize

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
BASE = ROOT / "out" / "emtab13530_spatial_bach1_nod"
H5AD_DIR = BASE / "h5ad_by_sample"
TABLE_DIR = BASE / "tables"
OUT = BASE / "spatial_skill_supplement"
SUP_TABLE = OUT / "tables"
SUP_H5AD = OUT / "h5ad_by_sample"
for path in [SUP_TABLE, SUP_H5AD]:
    path.mkdir(parents=True, exist_ok=True)

SC_MARKERS = (
    ROOT
    / "out"
    / "scanpy_downstream_scrublet"
    / "tables"
    / "nsclc_gse131907_gse274934_major_celltype_auto_markers_top200.csv"
)

SELECTED_MARKER_SAMPLES = ["P10_T1", "P10_B1", "P15_T2"]


def as_array(x):
    if sparse.issparse(x):
        return np.asarray(x.toarray())
    return np.asarray(x)


def vector(x):
    return as_array(x).ravel()


def row_standardized_graph(conn):
    conn = conn.tocsr().astype(float)
    conn.setdiag(0)
    conn.eliminate_zeros()
    rowsum = np.asarray(conn.sum(axis=1)).ravel()
    inv = np.divide(1, rowsum, out=np.zeros_like(rowsum, dtype=float), where=rowsum != 0)
    return sparse.diags(inv) @ conn


def moran_permutation(values, conn, n_perm=199, seed=7):
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 10 or np.nanstd(values[good]) == 0:
        return np.nan, np.nan
    vals = values[good]
    W = conn[good][:, good]
    W = row_standardized_graph(W)
    z = vals - vals.mean()
    denom = np.dot(z, z)
    n = len(z)
    s0 = W.sum()
    observed = float((n / s0) * (z @ (W @ z)) / denom) if s0 > 0 and denom > 0 else np.nan
    if not np.isfinite(observed):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        zp = rng.permutation(z)
        perm[i] = (n / s0) * (zp @ (W @ zp)) / denom
    p = (np.sum(np.abs(perm) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, float(p)


def prepare_marker_sets(spatial_genes):
    markers = pd.read_csv(SC_MARKERS)
    markers = markers[
        (markers["pvals_adj"] <= 0.05)
        & (markers["logfoldchanges"] > 0.5)
        & (markers["pct_nz_group"] > 0.20)
    ].copy()
    markers["names"] = markers["names"].astype(str)
    excluded_prefixes = ("MT-", "RPS", "RPL", "MTRNR")
    markers = markers[~markers["names"].str.upper().str.startswith(excluded_prefixes)]
    markers = markers[markers["names"].isin(spatial_genes)]
    markers = (
        markers.sort_values(["group", "scores"], ascending=[True, False])
        .groupby("group", as_index=False)
        .head(40)
        .copy()
    )
    marker_sets = {
        group: sub["names"].drop_duplicates().tolist()
        for group, sub in markers.groupby("group")
        if sub["names"].nunique() >= 5
    }
    markers.to_csv(SUP_TABLE / "signature_marker_sets_used.csv", index=False)
    return marker_sets, markers


def signature_deconv_proxy(adata, marker_sets, marker_table):
    genes = sorted(set(g for genes in marker_sets.values() for g in genes if g in adata.var_names))
    cell_types = sorted(marker_sets)
    X = as_array(adata[:, genes].X).astype(float)
    S = np.zeros((len(genes), len(cell_types)), dtype=float)
    gene_index = {g: i for i, g in enumerate(genes)}
    ct_index = {ct: i for i, ct in enumerate(cell_types)}
    marker_weights = marker_table[marker_table["names"].isin(genes)].copy()
    for row in marker_weights.itertuples(index=False):
        if row.group in ct_index:
            S[gene_index[row.names], ct_index[row.group]] = max(float(row.logfoldchanges), 0.05)
    colsum = np.linalg.norm(S, axis=0)
    S = S / np.where(colsum == 0, 1, colsum)
    coef = np.linalg.lstsq(S, X.T, rcond=None)[0].T
    coef = np.clip(coef, 0, None)
    row_sum = coef.sum(axis=1, keepdims=True)
    frac = np.divide(coef, row_sum, out=np.zeros_like(coef), where=row_sum > 0)
    for i, ct in enumerate(cell_types):
        adata.obs[f"sig_{ct}"] = frac[:, i]
    if len(cell_types):
        adata.obs["dominant_signature"] = pd.Categorical([cell_types[i] for i in np.argmax(frac, axis=1)])
        adata.obsm["signature_deconv_proxy"] = pd.DataFrame(frac, index=adata.obs_names, columns=cell_types)
    return cell_types


def qc_summary(adata):
    obs = adata.obs
    qc_pass = (
        (obs["total_counts"] >= 500)
        & (obs["n_genes_by_counts"] >= 200)
        & (obs["pct_counts_mt"] <= 20)
    )
    adata.obs["qc_pass_spatial_basic"] = qc_pass.astype(bool)
    return {
        "n_spots": int(adata.n_obs),
        "qc_pass_spots": int(qc_pass.sum()),
        "qc_pass_fraction": float(qc_pass.mean()),
        "median_counts": float(np.median(obs["total_counts"])),
        "median_genes": float(np.median(obs["n_genes_by_counts"])),
        "median_pct_mt": float(np.median(obs["pct_counts_mt"])),
        "p95_counts": float(np.percentile(obs["total_counts"], 95)),
        "p95_genes": float(np.percentile(obs["n_genes_by_counts"], 95)),
    }


def build_spatial_graph(adata):
    try:
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_rings=1)
        graph_method = "squidpy_grid_n_rings_1"
    except Exception:
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
        graph_method = "squidpy_generic_knn_6"
    conn = adata.obsp["spatial_connectivities"].tocsr()
    dist = adata.obsp["spatial_distances"].tocsr()
    nonzero_dist = dist.data[dist.data > 0]
    return {
        "spatial_graph_method": graph_method,
        "spatial_graph_edges_undirected": int(conn.nnz // 2),
        "spatial_graph_mean_degree": float(conn.nnz / adata.n_obs),
        "spatial_graph_mean_neighbor_distance": float(np.mean(nonzero_dist)) if len(nonzero_dist) else np.nan,
        "spatial_graph_components_estimate": np.nan,
    }


def run_domains(adata, sample):
    qc_mask = adata.obs["qc_pass_spatial_basic"].to_numpy(bool)
    ad_qc = adata[qc_mask].copy()
    if ad_qc.n_obs < 100:
        adata.obs["spatial_domain"] = pd.Categorical(["low_QC_or_unassigned"] * adata.n_obs)
        return pd.DataFrame(), pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sc.pp.highly_variable_genes(
                ad_qc,
                n_top_genes=min(1500, ad_qc.n_vars),
                flavor="seurat_v3",
                layer="counts",
            )
        except Exception:
            sc.pp.highly_variable_genes(ad_qc, n_top_genes=min(1500, ad_qc.n_vars), flavor="seurat")
    hvg = ad_qc.var_names[ad_qc.var["highly_variable"]].tolist()
    if len(hvg) < 200:
        hvg = ad_qc.var_names[np.argsort(np.asarray(ad_qc.X.var(axis=0)).ravel())[-1000:]].tolist()
    work = ad_qc[:, hvg].copy()
    sc.pp.scale(work, max_value=10)
    sc.tl.pca(work, n_comps=min(30, work.n_obs - 1, work.n_vars - 1), svd_solver="arpack")
    sc.pp.neighbors(work, n_neighbors=min(15, work.n_obs - 1), n_pcs=min(30, work.obsm["X_pca"].shape[1]))
    try:
        sq.gr.spatial_neighbors(work, coord_type="grid", n_rings=1)
    except Exception:
        sq.gr.spatial_neighbors(work, coord_type="generic", n_neighs=6)
    spatial_norm = normalize(work.obsp["spatial_connectivities"], norm="l1", axis=1)
    expr_norm = normalize(work.obsp["connectivities"], norm="l1", axis=1)
    work.obsp["combined_connectivities"] = sparse.csr_matrix(0.35 * spatial_norm + 0.65 * expr_norm)
    sc.tl.leiden(
        work,
        resolution=0.55,
        key_added="spatial_domain",
        adjacency=work.obsp["combined_connectivities"],
        random_state=13,
    )

    domain_labels = pd.Series("low_QC_or_unassigned", index=adata.obs_names, dtype=object)
    domain_labels.loc[work.obs_names] = [f"D{x}" for x in work.obs["spatial_domain"].astype(str)]
    adata.obs["spatial_domain"] = pd.Categorical(domain_labels)

    domain_rows = []
    for domain, sub in adata.obs.groupby("spatial_domain", observed=True):
        if domain == "low_QC_or_unassigned":
            continue
        domain_rows.append(
            {
                "sample": sample,
                "spatial_domain": domain,
                "n_spots": int(len(sub)),
                "BACH1_detected_fraction": float(sub["BACH1_detected"].mean()),
                "BACH1_mean": float(sub["BACH1_log_norm"].mean()),
                "NOD_like_score_mean": float(sub["NOD_like_score_scanpy"].mean()),
                "cohigh_fraction": float(sub["BACH1_NOD_cohigh"].mean()),
                "dominant_signature": sub["dominant_signature"].mode().iloc[0] if "dominant_signature" in sub and not sub["dominant_signature"].mode().empty else "",
                "Epithelial_signature_mean": float(sub["sig_Epithelial"].mean()) if "sig_Epithelial" in sub else np.nan,
                "Myeloid_signature_mean": float(sub["sig_Myeloid"].mean()) if "sig_Myeloid" in sub else np.nan,
                "T_NK_signature_mean": float(sub["sig_T/NK"].mean()) if "sig_T/NK" in sub else np.nan,
            }
        )
    domain_summary = pd.DataFrame(domain_rows)

    marker_df = pd.DataFrame()
    if sample in SELECTED_MARKER_SAMPLES and work.obs["spatial_domain"].nunique() > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.tl.rank_genes_groups(work, groupby="spatial_domain", method="wilcoxon")
        marker_df = sc.get.rank_genes_groups_df(work, group=None)
        marker_df.insert(0, "sample", sample)
        marker_df = marker_df.groupby(["sample", "group"], as_index=False).head(10)
    return domain_summary, marker_df


def process_sample(path, marker_sets, marker_table):
    sample = path.name.replace("_bach1_nod_scored.h5ad", "")
    print(f"Supplement {sample}", flush=True)
    adata = ad.read_h5ad(path)
    qc = qc_summary(adata)
    graph = build_spatial_graph(adata)
    conn = adata.obsp["spatial_connectivities"].tocsr()
    cell_types = signature_deconv_proxy(adata, marker_sets, marker_table)
    domain_summary, domain_markers = run_domains(adata, sample)

    moran_rows = []
    for key in ["BACH1_log_norm", "NOD_like_score_scanpy", "BACH1_NOD_cohigh"]:
        vals = adata.obs[key].astype(float).to_numpy()
        moran_i, p = moran_permutation(vals, conn, seed=abs(hash((sample, key))) % (2**32))
        moran_rows.append(
            {
                "sample": sample,
                "feature": key,
                "moran_i": moran_i,
                "permutation_p": p,
                "n_permutations": 199,
            }
        )

    sample_row = {
        "sample": sample,
        "patient_id": adata.obs["patient_id"].iloc[0],
        "disease": adata.obs["disease"].iloc[0],
        "tissue_group": adata.obs["tissue_group"].iloc[0],
    }
    sample_row.update(qc)
    sample_row.update(graph)

    sig_cols = [f"sig_{ct}" for ct in cell_types]
    spot_cols = [
        "sample",
        "patient_id",
        "tissue_group",
        "BACH1_log_norm",
        "BACH1_detected",
        "NOD_like_score_scanpy",
        "BACH1_NOD_cohigh",
        "qc_pass_spatial_basic",
        "spatial_domain",
        "dominant_signature",
        "array_row",
        "array_col",
    ] + sig_cols
    spot_export = adata.obs[[c for c in spot_cols if c in adata.obs.columns]].copy()
    spot_export["barcode"] = adata.obs_names
    spot_export.to_csv(SUP_TABLE / f"{sample}_skill_spot_annotations.csv.gz", index=False)
    adata.write_h5ad(SUP_H5AD / f"{sample}_skill_supplement.h5ad", compression="gzip")

    return sample_row, moran_rows, domain_summary, domain_markers, spot_export


def paired_delta_table(sample_table, metric):
    sample_summary = (
        sample_table[sample_table["tissue_group"].isin(["Tumor", "Adjacent"])]
        .groupby(["patient_id", "tissue_group"], as_index=False)[metric]
        .mean()
    )
    wide = sample_summary.pivot(index="patient_id", columns="tissue_group", values=metric).dropna()
    if len(wide) < 3:
        return {"metric": metric, "n_pairs": len(wide), "wilcoxon_p": np.nan, "median_delta": np.nan}
    try:
        p = stats.wilcoxon(wide["Tumor"], wide["Adjacent"]).pvalue
    except ValueError:
        p = np.nan
    return {
        "metric": metric,
        "n_pairs": int(len(wide)),
        "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
        "median_delta": float(np.median(wide["Tumor"] - wide["Adjacent"])),
    }


def main():
    sample_paths = sorted(H5AD_DIR.glob("*_bach1_nod_scored.h5ad"))
    first = ad.read_h5ad(sample_paths[0], backed="r")
    spatial_genes = set(first.var_names.astype(str))
    first.file.close()
    marker_sets, marker_table = prepare_marker_sets(spatial_genes)

    sample_rows = []
    moran_rows = []
    domain_summaries = []
    marker_tables = []
    spot_exports = []
    for path in sample_paths:
        sample_row, sample_moran, domain_summary, domain_markers, spot_export = process_sample(path, marker_sets, marker_table)
        sample_rows.append(sample_row)
        moran_rows.extend(sample_moran)
        if len(domain_summary):
            domain_summaries.append(domain_summary)
        if len(domain_markers):
            marker_tables.append(domain_markers)
        spot_exports.append(spot_export)

    sample_table = pd.DataFrame(sample_rows)
    moran_table = pd.DataFrame(moran_rows)
    domain_table = pd.concat(domain_summaries, ignore_index=True) if domain_summaries else pd.DataFrame()
    marker_table_out = pd.concat(marker_tables, ignore_index=True) if marker_tables else pd.DataFrame()
    spot_table = pd.concat(spot_exports, ignore_index=True)

    sample_table.to_csv(SUP_TABLE / "spatial_skill_sample_qc_graph_summary.csv", index=False)
    moran_table.to_csv(SUP_TABLE / "spatial_skill_moran_statistics.csv", index=False)
    domain_table.to_csv(SUP_TABLE / "spatial_skill_domain_summary.csv", index=False)
    marker_table_out.to_csv(SUP_TABLE / "spatial_skill_domain_markers_selected_samples.csv", index=False)
    spot_table.to_csv(SUP_TABLE / "spatial_skill_all_spot_annotations_deconv_domains.csv.gz", index=False)

    cohigh = spot_table.copy()
    sig_cols = [c for c in cohigh.columns if c.startswith("sig_")]
    enrich_rows = []
    for group, sub in cohigh.groupby("tissue_group", observed=True):
        for col in sig_cols:
            high = sub["BACH1_NOD_cohigh"].astype(bool)
            if high.sum() < 10 or (~high).sum() < 10:
                continue
            enrich_rows.append(
                {
                    "tissue_group": group,
                    "signature": col.replace("sig_", ""),
                    "cohigh_mean": float(sub.loc[high, col].mean()),
                    "non_cohigh_mean": float(sub.loc[~high, col].mean()),
                    "delta": float(sub.loc[high, col].mean() - sub.loc[~high, col].mean()),
                    "mannwhitney_p": float(stats.mannwhitneyu(sub.loc[high, col], sub.loc[~high, col], alternative="two-sided").pvalue),
                    "n_cohigh_spots": int(high.sum()),
                    "n_non_cohigh_spots": int((~high).sum()),
                }
            )
    enrich_table = pd.DataFrame(enrich_rows)
    enrich_table.to_csv(SUP_TABLE / "spatial_skill_cohigh_signature_enrichment.csv", index=False)

    sample_plus = sample_table.merge(
        moran_table.pivot(index="sample", columns="feature", values="moran_i")
        .add_prefix("moran_")
        .reset_index(),
        on="sample",
        how="left",
    )
    paired = pd.DataFrame(
        [
            paired_delta_table(sample_plus, "qc_pass_fraction"),
            paired_delta_table(sample_plus, "moran_BACH1_log_norm"),
            paired_delta_table(sample_plus, "moran_NOD_like_score_scanpy"),
            paired_delta_table(sample_plus, "moran_BACH1_NOD_cohigh"),
        ]
    )
    paired.to_csv(SUP_TABLE / "spatial_skill_paired_tumor_adjacent_supplement_stats.csv", index=False)

    summary = {
        "n_samples_processed": int(len(sample_paths)),
        "n_spots_annotated": int(len(spot_table)),
        "n_cell_type_signatures": int(len(marker_sets)),
        "cell_type_signatures": sorted(marker_sets.keys()),
        "n_domains_total": int(len(domain_table)),
        "selected_samples_with_domain_markers": SELECTED_MARKER_SAMPLES,
        "cell2location_installed": True,
        "deconvolution_note": "Fast scRNA-marker signature regression proxy was used for all sections; full cell2location posterior was not trained in this supplement.",
    }
    (OUT / "spatial_skill_supplement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
