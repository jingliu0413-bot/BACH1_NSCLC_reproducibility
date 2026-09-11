"""Matched-background ATAC permutation for stable BACH1 regulon targets.

The reviewer requested a background matched on expression and locus/ATAC
features rather than an unmatched expressed-gene Fisher test. This script uses
nearest-neighbor matched pools and empirical resampling.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

from project_paths import OUTPUT_DIR, PROJECT_ROOT
from run_bach1_atac_motif_support import (
    ATAC_DIR,
    GENCODE_ALIAS_MAP,
    PROMOTER_BP,
    RESOURCE_DIR,
    STANDARD_CHROMS,
    load_gencode_gene_coords,
)
from run_bach1_atac_motif_support_ucsc_targeted import (
    DEFAULT_OUT as DEFAULT_TARGETED_OUT,
    UCSC_BIGBED,
    _parse_ucsc_bed_lines,
    aggregate_target_support,
    fetch_ucsc_motif_hits,
    link_target_windows,
    load_or_summarize_peaks,
    overlap_hits_with_peaks,
    write_merged_target_windows,
)


DEFAULT_TARGET_TABLE = (
    OUTPUT_DIR
    / "revision_diagnostics"
    / "pyscenic_bach1_consensus_seed_stability_v2"
    / "pyscenic_bach1_stable_ge60pct_targets.csv"
)
DEFAULT_MATRIX = (
    OUTPUT_DIR
    / "bach1_malignant_epithelial_consensus_pyscenic"
    / "malignant_epithelial_counts_filtered_for_pyscenic.csv"
)
DEFAULT_OUT = OUTPUT_DIR / "bach1_atac_motif_support_matched_background"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGET_TABLE)
    parser.add_argument("--pyscenic-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--atac-dir", type=Path, default=ATAC_DIR)
    parser.add_argument("--resource-dir", type=Path, default=RESOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--existing-peak-summary-dir",
        type=Path,
        default=DEFAULT_TARGETED_OUT.with_name("bach1_atac_motif_support_consensus_ucsc_targeted_10kb") / "tables",
    )
    parser.add_argument("--big-bed-to-bed", type=Path, default=PROJECT_ROOT / "work" / "bin" / "bigBedToBed")
    parser.add_argument("--ucsc-bigbed", default=UCSC_BIGBED)
    parser.add_argument("--ucsc-udc-dir", type=Path, default=PROJECT_ROOT / "work" / "ucsc_udc_cache")
    parser.add_argument("--query-mode", choices=["ranges", "bed"], default="ranges")
    parser.add_argument("--query-workers", type=int, default=4)
    parser.add_argument("--query-timeout-sec", type=int, default=12)
    parser.add_argument("--allow-partial-ucsc", action="store_true")
    parser.add_argument("--window-bp", type=int, default=10_000)
    parser.add_argument("--neighbors-per-target", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4101)
    return parser.parse_args()


def run_bigbed_to_bed_timeout(cmd: list[str], timeout_sec: int) -> tuple[list[dict], str, int]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return [], f"Timed out after {timeout_sec} seconds", 124
    return _parse_ucsc_bed_lines(proc.stdout.splitlines(True)), proc.stderr, proc.returncode


def fetch_ucsc_motif_hits_parallel(
    bigbed_to_bed: Path,
    bigbed: str,
    windows_bed: Path,
    out_gz: Path,
    udc_dir: Path,
    query_mode: str,
    allow_partial: bool,
    workers: int,
    timeout_sec: int,
) -> tuple[pd.DataFrame, list[dict]]:
    if query_mode != "ranges" or workers <= 1:
        return fetch_ucsc_motif_hits(
            bigbed_to_bed,
            bigbed,
            windows_bed,
            out_gz,
            udc_dir,
            query_mode,
            allow_partial,
        )
    if not bigbed_to_bed.exists():
        raise FileNotFoundError(f"Missing UCSC bigBedToBed utility: {bigbed_to_bed}")
    udc_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(windows_bed, sep="\t", header=None, names=["chrom", "start", "end", "genes"])
    records = list(windows.itertuples(index=False))
    rows: list[dict] = []
    failures: list[dict] = []

    def query_one(win) -> tuple[list[dict], dict | None]:
        range_arg = f"-range={win.chrom}:{int(win.start) + 1}-{int(win.end)}"
        worker_udc = udc_dir / f"range_worker_{threading.get_ident()}"
        worker_udc.mkdir(parents=True, exist_ok=True)
        cmd = [str(bigbed_to_bed), f"-udcDir={worker_udc}", range_arg, "-tsv", bigbed, "stdout"]
        region_rows = []
        stderr = ""
        code = 1
        for _ in range(1, 4):
            region_rows, stderr, code = run_bigbed_to_bed_timeout(cmd, timeout_sec)
            if code == 0:
                return region_rows, None
        return [], {
            "chrom": win.chrom,
            "start": int(win.start),
            "end": int(win.end),
            "genes": win.genes,
            "stderr": stderr.strip(),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(query_one, win): win for win in records}
        for i, future in enumerate(as_completed(future_map), start=1):
            region_rows, failure = future.result()
            rows.extend(region_rows)
            if failure is not None:
                failures.append(failure)
            if i % 100 == 0 or i == len(records):
                print(f"Queried {i}/{len(records)} UCSC windows", flush=True)

    if failures and not allow_partial:
        preview = "\n".join(f"{x['chrom']}:{x['start']}-{x['end']} {x['stderr']}" for x in failures[:5])
        raise RuntimeError(f"UCSC range extraction failed for {len(failures)} windows:\n{preview}")

    hits = pd.DataFrame(rows)
    if not hits.empty:
        hits = hits.drop_duplicates().sort_values(["chrom", "start", "end", "motif_model", "strand"]).reset_index(drop=True)
    with gzip.open(out_gz, "wt") as handle:
        if hits.empty:
            handle.write("chrom\tstart\tend\tmotif_model\tmotif_name\tucsc_score\tstrand\tucsc_TFName\n")
        else:
            hits.to_csv(handle, sep="\t", index=False)
    return hits, failures


def expression_stats(matrix_csv: Path, out_csv: Path) -> pd.DataFrame:
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return pd.read_csv(out_csv)
    sums = None
    detected = None
    n_rows = 0
    genes = None
    for chunk in pd.read_csv(matrix_csv, index_col=0, chunksize=256):
        if genes is None:
            genes = chunk.columns.to_numpy(str)
            sums = np.zeros(len(genes), dtype=float)
            detected = np.zeros(len(genes), dtype=int)
        arr = chunk.to_numpy(dtype=float, copy=False)
        sums += arr.sum(axis=0)
        detected += (arr > 0).sum(axis=0)
        n_rows += arr.shape[0]
    if genes is None or sums is None or detected is None:
        raise ValueError(f"No expression rows read from {matrix_csv}")
    out = pd.DataFrame(
        {
            "gene": genes,
            "mean_counts": sums / n_rows,
            "detected_fraction": detected / n_rows,
            "detected_cells": detected,
            "n_cells": n_rows,
        }
    )
    out.to_csv(out_csv, index=False)
    return out


def load_peak_summary(args: argparse.Namespace, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    peak_summary = args.existing_peak_summary_dir / "gse274934_atac_peak_accessibility_summary.csv.gz"
    qc_summary = args.existing_peak_summary_dir / "gse274934_atac_sample_qc_summary.csv"
    if peak_summary.exists():
        peaks = pd.read_csv(peak_summary)
        qc = pd.read_csv(qc_summary) if qc_summary.exists() else pd.DataFrame()
        return peaks, qc
    return load_or_summarize_peaks(args.atac_dir, table_dir, force=False)


def overlap_peak_covariates(gene_coords: pd.DataFrame, peaks: pd.DataFrame, window_bp: int) -> pd.DataFrame:
    peaks_by_chrom = {
        chrom: sub.sort_values(["start", "end"]).reset_index(drop=True)
        for chrom, sub in peaks[peaks["chrom"].isin(STANDARD_CHROMS)].groupby("chrom", sort=False)
    }
    rows = []
    for gene in gene_coords.itertuples(index=False):
        row = {
            "gene": gene.gene,
            "tss_peak_count_10kb": 0,
            "promoter_peak_count_2kb": 0,
            "tss_accessibility_mean_10kb": 0.0,
            "promoter_accessibility_mean_2kb": 0.0,
            "tss_accessibility_max_10kb": 0.0,
            "promoter_accessibility_max_2kb": 0.0,
        }
        sub = peaks_by_chrom.get(gene.chrom)
        if sub is None or sub.empty:
            rows.append(row)
            continue
        starts = sub["start"].to_numpy(np.int64)
        win_start = max(0, int(gene.tss) - window_bp)
        win_end = int(gene.tss) + window_bp
        right = np.searchsorted(starts, win_end, side="left")
        cand = sub.iloc[:right]
        cand = cand[cand["end"].to_numpy(np.int64) > win_start]
        if not cand.empty:
            row["tss_peak_count_10kb"] = int(cand.shape[0])
            row["tss_accessibility_mean_10kb"] = float(cand["frac_cells_accessible"].mean())
            row["tss_accessibility_max_10kb"] = float(cand["frac_cells_accessible"].max())
        pro_start = max(0, int(gene.tss) - PROMOTER_BP)
        pro_end = int(gene.tss) + PROMOTER_BP
        pro = cand[(cand["start"].to_numpy(np.int64) < pro_end) & (cand["end"].to_numpy(np.int64) > pro_start)]
        if not pro.empty:
            row["promoter_peak_count_2kb"] = int(pro.shape[0])
            row["promoter_accessibility_mean_2kb"] = float(pro["frac_cells_accessible"].mean())
            row["promoter_accessibility_max_2kb"] = float(pro["frac_cells_accessible"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_matched_pools(features: pd.DataFrame, target_genes: set[str], k: int) -> pd.DataFrame:
    target = features[features["gene"].isin(target_genes)].copy()
    background = features[~features["gene"].isin(target_genes)].copy()
    feature_cols = [
        "mean_counts",
        "detected_fraction",
        "gene_length",
        "tss_peak_count_10kb",
        "promoter_peak_count_2kb",
        "promoter_accessibility_mean_2kb",
    ]
    for col in feature_cols:
        target[col] = pd.to_numeric(target[col], errors="coerce").fillna(0)
        background[col] = pd.to_numeric(background[col], errors="coerce").fillna(0)
    transform = pd.concat([target[feature_cols], background[feature_cols]], ignore_index=True)
    transformed = pd.DataFrame(
        {
            "log_mean_counts": np.log1p(transform["mean_counts"]),
            "detected_fraction": transform["detected_fraction"],
            "log_gene_length": np.log1p(transform["gene_length"]),
            "log_tss_peak_count_10kb": np.log1p(transform["tss_peak_count_10kb"]),
            "log_promoter_peak_count_2kb": np.log1p(transform["promoter_peak_count_2kb"]),
            "promoter_accessibility_mean_2kb": transform["promoter_accessibility_mean_2kb"],
        }
    )
    mu = transformed.mean(axis=0)
    sd = transformed.std(axis=0).replace(0, 1)
    n_target = target.shape[0]
    target_mat = ((transformed.iloc[:n_target] - mu) / sd).to_numpy(float)
    bg_mat = ((transformed.iloc[n_target:] - mu) / sd).to_numpy(float)
    tree = cKDTree(bg_mat)
    kk = min(k, background.shape[0])
    dist, idx = tree.query(target_mat, k=kk)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    rows = []
    bg_genes = background["gene"].to_numpy(str)
    for i, gene in enumerate(target["gene"].to_numpy(str)):
        for rank in range(kk):
            rows.append(
                {
                    "target_gene": gene,
                    "matched_background_gene": bg_genes[int(idx[i, rank])],
                    "neighbor_rank": rank + 1,
                    "scaled_distance": float(dist[i, rank]),
                }
            )
    return pd.DataFrame(rows)


def support_metric_table(support: pd.DataFrame) -> pd.DataFrame:
    out = support.copy()
    out["any_window_support"] = out["has_atac_bach1_motif_support_in_window"].fillna(False).astype(bool)
    out["promoter_support"] = pd.to_numeric(out["n_promoter_2kb_motif_peak_links"], errors="coerce").fillna(0).gt(0)
    out["proximal_support"] = pd.to_numeric(out["n_proximal_10kb_motif_peak_links"], errors="coerce").fillna(0).gt(0)
    out["recurrent_ge2_samples"] = pd.to_numeric(out["n_samples_with_motif_peak_in_window"], errors="coerce").fillna(0).ge(2)
    out["recurrent_ge3_samples"] = pd.to_numeric(out["n_samples_with_motif_peak_in_window"], errors="coerce").fillna(0).ge(3)
    return out


def summarize_metrics(df: pd.DataFrame, genes: list[str]) -> dict:
    sub = df[df["gene"].isin(genes)].copy()
    return {
        "support_proportion": float(sub["any_window_support"].mean()),
        "promoter_support_proportion": float(sub["promoter_support"].mean()),
        "proximal_support_proportion": float(sub["proximal_support"].mean()),
        "recurrent_ge2_samples_proportion": float(sub["recurrent_ge2_samples"].mean()),
        "recurrent_ge3_samples_proportion": float(sub["recurrent_ge3_samples"].mean()),
        "mean_motif_peak_links": float(pd.to_numeric(sub["n_atac_motif_peak_links_in_window"], errors="coerce").fillna(0).mean()),
        "mean_unique_motif_peaks": float(pd.to_numeric(sub["n_unique_motif_peaks_in_window"], errors="coerce").fillna(0).mean()),
        "mean_samples_with_support": float(pd.to_numeric(sub["n_samples_with_motif_peak_in_window"], errors="coerce").fillna(0).mean()),
    }


def empirical_permutation(
    metric_support: pd.DataFrame,
    target_genes: list[str],
    pools: pd.DataFrame,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed = summarize_metrics(metric_support, target_genes)
    pool_map = pools.groupby("target_gene")["matched_background_gene"].apply(list).to_dict()
    target_genes = [g for g in target_genes if g in pool_map and len(pool_map[g]) > 0]
    perm_values = {metric: np.empty(n_perm, dtype=float) for metric in observed}
    for i in range(n_perm):
        sampled = [rng.choice(pool_map[g]) for g in target_genes]
        vals = summarize_metrics(metric_support, sampled)
        for metric, val in vals.items():
            perm_values[metric][i] = val
    rows = []
    for metric, obs in observed.items():
        vals = perm_values[metric]
        vals = vals[np.isfinite(vals)]
        expected = float(vals.mean()) if len(vals) else np.nan
        empirical_p = float((np.sum(vals >= obs) + 1) / (len(vals) + 1)) if len(vals) else np.nan
        rows.append(
            {
                "metric": metric,
                "observed_target_value": obs,
                "matched_background_expected_mean": expected,
                "matched_background_expected_sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "observed_expected_ratio": float(obs / expected) if expected and np.isfinite(expected) else np.nan,
                "empirical_p_greater": empirical_p,
                "permutations": int(n_perm),
                "n_targets_with_matched_pool": int(len(target_genes)),
            }
        )
    out = pd.DataFrame(rows)
    out["empirical_holm_p"] = np.nan
    mask = np.isfinite(out["empirical_p_greater"])
    if mask.any():
        out.loc[mask, "empirical_holm_p"] = multipletests(out.loc[mask, "empirical_p_greater"], method="holm")[1]
    return out


def main() -> None:
    args = parse_args()
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    target_table = pd.read_csv(args.target_table)
    if "gene" not in target_table.columns:
        raise ValueError(f"{args.target_table} does not contain a gene column.")
    target_genes = target_table["gene"].dropna().astype(str).drop_duplicates().tolist()

    expr = expression_stats(args.pyscenic_matrix, table_dir / "pyscenic_background_expression_stats.csv")
    all_genes = expr["gene"].astype(str).tolist()
    gtf = args.resource_dir / "gencode.v44.annotation.gtf.gz"
    gene_coords = load_gencode_gene_coords(gtf, set(all_genes))
    gene_coords.to_csv(table_dir / "gencode_v44_gene_coordinates_for_background_genes.csv.gz", index=False)
    missing_targets = sorted(set(target_genes) - set(gene_coords["gene"]))
    pd.DataFrame({"gene": missing_targets}).to_csv(table_dir / "stable_targets_missing_from_gencode_v44.csv", index=False)
    alias_rows = [
        {"input_gene": old, "gencode_gene_name": new}
        for old, new in GENCODE_ALIAS_MAP.items()
        if old in set(all_genes)
    ]
    pd.DataFrame(alias_rows).to_csv(table_dir / "gencode_v44_symbol_aliases_used.csv", index=False)

    peaks, qc = load_peak_summary(args, table_dir)
    peaks = peaks.copy()
    if "peak_row_id" not in peaks.columns:
        peaks.insert(0, "peak_row_id", np.arange(len(peaks), dtype=np.int64))
    peaks.to_csv(table_dir / "gse274934_atac_peak_accessibility_summary_used.csv.gz", index=False, compression="gzip")
    if not qc.empty:
        qc.to_csv(table_dir / "gse274934_atac_sample_qc_summary_used.csv", index=False)

    peak_cov = overlap_peak_covariates(gene_coords, peaks, args.window_bp)
    features = (
        gene_coords.merge(expr, on="gene", how="left")
        .merge(peak_cov, on="gene", how="left")
        .fillna(
            {
                "mean_counts": 0,
                "detected_fraction": 0,
                "tss_peak_count_10kb": 0,
                "promoter_peak_count_2kb": 0,
                "tss_accessibility_mean_10kb": 0,
                "promoter_accessibility_mean_2kb": 0,
                "tss_accessibility_max_10kb": 0,
                "promoter_accessibility_max_2kb": 0,
            }
        )
    )
    features.to_csv(table_dir / "matched_background_gene_features.csv.gz", index=False, compression="gzip")

    target_genes_with_coords = [g for g in target_genes if g in set(features["gene"])]
    pools = build_matched_pools(features, set(target_genes_with_coords), args.neighbors_per_target)
    pools.to_csv(table_dir / "matched_background_neighbor_pools.csv.gz", index=False, compression="gzip")
    query_genes = sorted(set(target_genes_with_coords) | set(pools["matched_background_gene"]))
    query_coords = gene_coords[gene_coords["gene"].isin(query_genes)].copy()
    windows_bed = table_dir / f"matched_background_query_tss_{args.window_bp // 1000}kb_windows.merged.bed"
    write_merged_target_windows(query_coords, windows_bed, args.window_bp)
    hits_gz = table_dir / "ucsc_jaspar2026_bach1_mafk_hits_in_matched_query_windows.tsv.gz"
    hits, failures = fetch_ucsc_motif_hits_parallel(
        args.big_bed_to_bed,
        args.ucsc_bigbed,
        windows_bed,
        hits_gz,
        args.ucsc_udc_dir,
        args.query_mode,
        args.allow_partial_ucsc,
        args.query_workers,
        args.query_timeout_sec,
    )
    pd.DataFrame(failures).to_csv(table_dir / "ucsc_jaspar2026_failed_matched_query_windows.csv", index=False)
    overlap = overlap_hits_with_peaks(peaks, hits)
    motif_peaks = overlap.merge(peaks, on="peak_row_id", how="left") if not overlap.empty else pd.DataFrame()
    motif_peaks.to_csv(table_dir / "gse274934_atac_bach1_motif_positive_peaks_matched_query_windows.csv.gz", index=False, compression="gzip")
    links = link_target_windows(query_coords, motif_peaks, args.window_bp) if not motif_peaks.empty else pd.DataFrame()
    links.to_csv(table_dir / "matched_query_genes_atac_motif_peak_links.csv.gz", index=False, compression="gzip")

    support = aggregate_target_support(links, query_genes, [300, 400, 500])
    support = support_metric_table(support)
    support.merge(features, on="gene", how="left").to_csv(table_dir / "matched_query_genes_with_atac_support_and_features.csv", index=False)
    target_table.merge(support, on="gene", how="left").to_csv(
        table_dir / "stable_bach1_targets_with_matched_background_atac_support.csv",
        index=False,
    )

    perm = empirical_permutation(support, target_genes_with_coords, pools, args.permutations, args.seed)
    perm.to_csv(table_dir / "matched_background_permutation_summary.csv", index=False)

    summary = {
        "target_table": str(args.target_table),
        "pyscenic_matrix": str(args.pyscenic_matrix),
        "n_input_targets": int(len(target_genes)),
        "n_targets_with_gencode_coordinates": int(len(target_genes_with_coords)),
        "n_background_genes_with_features": int(features.shape[0] - len(target_genes_with_coords)),
        "neighbors_per_target": int(args.neighbors_per_target),
        "n_unique_matched_background_genes_queried": int(pools["matched_background_gene"].nunique()),
        "n_query_genes_for_ucsc": int(len(query_genes)),
        "window_bp": int(args.window_bp),
        "permutations": int(args.permutations),
        "query_mode": args.query_mode,
        "query_workers": int(args.query_workers),
        "query_timeout_sec": int(args.query_timeout_sec),
        "n_ucsc_bach1_mafk_hits_in_query_windows": int(hits.shape[0]),
        "n_ucsc_failed_windows": int(len(failures)),
        "matching_features": [
            "mean expression",
            "detected fraction",
            "gene length",
            "TSS +/- window ATAC peak count",
            "promoter +/-2kb ATAC peak count",
            "promoter accessibility mean",
        ],
        "not_included": "GC content was not included because no local hg38 sequence/GC summary table was available in this deployment.",
        "empirical_tests": perm.to_dict(orient="records"),
        "outputs": sorted(p.name for p in table_dir.glob("*")),
    }
    (args.output_dir / "matched_background_permutation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
