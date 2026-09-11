"""Targeted BACH1 scATAC motif support using UCSC's native JASPAR 2026 track.

This fallback avoids downloading the full genome-wide JASPAR TFBS TSV files.
It queries only TSS-window regions for the supplied BACH1 regulon targets.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import OUTPUT_DIR, PROJECT_ROOT
from run_bach1_atac_motif_support import (
    ATAC_DIR,
    GENCODE_ALIAS_MAP,
    HIGH_CONF_MOTIF_SCORE,
    PROMOTER_BP,
    PROXIMAL_BP,
    RESOURCE_DIR,
    STANDARD_CHROMS,
    WINDOW_BP,
    classify_relation,
    load_gencode_gene_coords,
    peak_distance_to_tss,
    sample_from_path,
    summarize_peak_matrix,
    summarize_singlecell_qc,
)


DEFAULT_TARGET_TABLE = (
    OUTPUT_DIR
    / "bach1_malignant_epithelial_consensus_pyscenic_all_tfs"
    / "tables"
    / "pyscenic_bach1_regulon_targets_integrated.csv"
)
DEFAULT_OUT = OUTPUT_DIR / "bach1_atac_motif_support_consensus_ucsc_targeted"
UCSC_BIGBED = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/jaspar/JASPAR2026.bb"
MOTIFS = {
    "MA1633.2": "BACH1",
    "MA0591.2": "Bach1::Mafk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGET_TABLE)
    parser.add_argument("--atac-dir", type=Path, default=ATAC_DIR)
    parser.add_argument("--resource-dir", type=Path, default=RESOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--big-bed-to-bed", type=Path, default=PROJECT_ROOT / "work" / "bin" / "bigBedToBed")
    parser.add_argument("--ucsc-bigbed", default=UCSC_BIGBED)
    parser.add_argument("--ucsc-udc-dir", type=Path, default=PROJECT_ROOT / "work" / "ucsc_udc_cache")
    parser.add_argument("--query-mode", choices=["ranges", "bed"], default="ranges")
    parser.add_argument("--bed-chunk-size", type=int, default=10)
    parser.add_argument("--bed-chunk-timeout-sec", type=int, default=90)
    parser.add_argument("--allow-partial-ucsc", action="store_true")
    parser.add_argument("--window-bp", type=int, default=WINDOW_BP)
    parser.add_argument("--score-thresholds", type=int, nargs="+", default=[300, 400, 500])
    parser.add_argument("--force-peak-summary", action="store_true")
    return parser.parse_args()


def load_or_summarize_peaks(atac_dir: Path, table_dir: Path, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    peak_summary = table_dir / "gse274934_atac_peak_accessibility_summary.csv.gz"
    qc_summary = table_dir / "gse274934_atac_sample_qc_summary.csv"
    if peak_summary.exists() and not force:
        peaks = pd.read_csv(peak_summary)
        qc = pd.read_csv(qc_summary) if qc_summary.exists() else pd.DataFrame()
        return peaks, qc

    h5_files = sorted(atac_dir.glob("*_ATAC_filtered_peak_bc_matrix.h5"))
    singlecell_files = sorted(atac_dir.glob("*_ATAC_singlecell.csv.gz"))
    if not h5_files:
        raise FileNotFoundError(f"No ATAC h5 files found in {atac_dir}")

    peak_tables = []
    for h5_path in h5_files:
        print(f"Summarizing peaks: {h5_path.name}", flush=True)
        peak_tables.append(summarize_peak_matrix(h5_path))
    peaks = pd.concat(peak_tables, ignore_index=True)
    peaks.insert(0, "peak_row_id", np.arange(len(peaks), dtype=np.int64))
    peaks.to_csv(peak_summary, index=False, compression="gzip")

    qc_rows = [summarize_singlecell_qc(x) for x in singlecell_files]
    qc = pd.DataFrame(qc_rows)
    matrix_cells = peaks.groupby("sample")["n_cells_in_matrix"].max().reset_index(name="n_cells_in_filtered_peak_matrix")
    qc = matrix_cells.merge(qc, on="sample", how="left")
    qc.to_csv(qc_summary, index=False)
    return peaks, qc


def write_merged_target_windows(gene_coords: pd.DataFrame, path: Path, window_bp: int) -> pd.DataFrame:
    windows = gene_coords[["gene", "chrom", "tss"]].copy()
    windows["start"] = (windows["tss"].astype(int) - window_bp).clip(lower=0)
    windows["end"] = windows["tss"].astype(int) + window_bp
    windows = windows[windows["chrom"].isin(STANDARD_CHROMS)].sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    windows[["chrom", "start", "end", "gene"]].to_csv(path.with_suffix(".genes.bed"), sep="\t", header=False, index=False)

    merged_rows = []
    for chrom, sub in windows.groupby("chrom", sort=False):
        current_start = None
        current_end = None
        genes = []
        for row in sub.itertuples(index=False):
            start = int(row.start)
            end = int(row.end)
            if current_start is None or start > current_end:
                if current_start is not None:
                    merged_rows.append({"chrom": chrom, "start": current_start, "end": current_end, "genes": ";".join(genes)})
                current_start = start
                current_end = end
                genes = [row.gene]
            else:
                current_end = max(current_end, end)
                genes.append(row.gene)
        if current_start is not None:
            merged_rows.append({"chrom": chrom, "start": current_start, "end": current_end, "genes": ";".join(genes)})
    merged = pd.DataFrame(merged_rows)
    merged.to_csv(path, sep="\t", header=False, index=False)
    return windows


def _parse_ucsc_bed_lines(lines) -> list[dict]:
    rows = []
    for line in lines:
        line = line.rstrip("\n")
        if not line or line.startswith("chrom\t"):
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        model = fields[3]
        tf_name = fields[6]
        if model not in MOTIFS and tf_name not in set(MOTIFS.values()):
            continue
        rows.append(
            {
                "chrom": fields[0],
                "start": int(fields[1]),
                "end": int(fields[2]),
                "motif_model": model,
                "motif_name": MOTIFS.get(model, tf_name),
                "ucsc_score": float(fields[4]),
                "strand": fields[5],
                "ucsc_TFName": tf_name,
            }
        )
    return rows


def _run_bigbed_to_bed(cmd: list[str], timeout_sec: int | None = None) -> tuple[list[dict], str, int]:
    if timeout_sec is not None and timeout_sec > 0:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
        except subprocess.TimeoutExpired:
            return [], f"Timed out after {timeout_sec} seconds", 124
        return _parse_ucsc_bed_lines(proc.stdout.splitlines(True)), proc.stderr, proc.returncode

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        assert proc.stdout is not None
        rows = _parse_ucsc_bed_lines(proc.stdout)
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        code = proc.wait()
    return rows, stderr, code


def fetch_ucsc_motif_hits(
    bigbed_to_bed: Path,
    bigbed: str,
    windows_bed: Path,
    out_gz: Path,
    udc_dir: Path,
    query_mode: str,
    allow_partial: bool,
    bed_chunk_size: int = 10,
    bed_chunk_timeout_sec: int = 90,
) -> tuple[pd.DataFrame, list[dict]]:
    if not bigbed_to_bed.exists():
        raise FileNotFoundError(f"Missing UCSC bigBedToBed utility: {bigbed_to_bed}")
    udc_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    rows = []

    if query_mode == "bed":
        windows = pd.read_csv(windows_bed, sep="\t", header=None, names=["chrom", "start", "end", "genes"])
        if bed_chunk_size is not None and bed_chunk_size > 0 and len(windows) > bed_chunk_size:
            for chunk_i, start_i in enumerate(range(0, len(windows), bed_chunk_size), start=1):
                chunk = windows.iloc[start_i : start_i + bed_chunk_size]
                chunk_bed = out_gz.parent / f"{windows_bed.stem}.chunk_{chunk_i:04d}.bed"
                chunk.to_csv(chunk_bed, sep="\t", header=False, index=False)
                cmd = [str(bigbed_to_bed), f"-udcDir={udc_dir}", f"-bed={chunk_bed}", "-tsv", bigbed, "stdout"]
                region_rows, stderr, code = _run_bigbed_to_bed(cmd, timeout_sec=bed_chunk_timeout_sec)
                try:
                    chunk_bed.unlink()
                except FileNotFoundError:
                    pass
                if code == 0:
                    rows.extend(region_rows)
                else:
                    failures.append(
                        {
                            "chunk": chunk_i,
                            "n_windows": int(chunk.shape[0]),
                            "chrom": ";".join(chunk["chrom"].astype(str).drop_duplicates().tolist()),
                            "start": int(chunk["start"].min()),
                            "end": int(chunk["end"].max()),
                            "genes": ";".join(chunk["genes"].astype(str).tolist()),
                            "stderr": stderr.strip(),
                        }
                    )
                if chunk_i % 5 == 0 or start_i + bed_chunk_size >= len(windows):
                    print(f"Queried {min(start_i + bed_chunk_size, len(windows))}/{len(windows)} UCSC BED windows", flush=True)
            if failures and not allow_partial:
                preview = "\n".join(
                    f"chunk {x['chunk']} ({x['n_windows']} windows) {x['stderr']}" for x in failures[:5]
                )
                raise RuntimeError(f"UCSC BED extraction failed for {len(failures)} chunks:\n{preview}")
        else:
            cmd = [str(bigbed_to_bed), f"-udcDir={udc_dir}", f"-bed={windows_bed}", "-tsv", bigbed, "stdout"]
            rows, stderr, code = _run_bigbed_to_bed(cmd, timeout_sec=bed_chunk_timeout_sec)
            if code != 0:
                raise RuntimeError(f"bigBedToBed failed with code {code}:\n{stderr}")
    else:
        windows = pd.read_csv(windows_bed, sep="\t", header=None, names=["chrom", "start", "end", "genes"])
        for win in windows.itertuples(index=False):
            range_arg = f"-range={win.chrom}:{int(win.start) + 1}-{int(win.end)}"
            cmd = [str(bigbed_to_bed), f"-udcDir={udc_dir}", range_arg, "-tsv", bigbed, "stdout"]
            region_rows = []
            stderr = ""
            code = 1
            for attempt in range(1, 4):
                region_rows, stderr, code = _run_bigbed_to_bed(cmd)
                if code == 0:
                    break
            if code == 0:
                rows.extend(region_rows)
            else:
                failures.append(
                    {
                        "chrom": win.chrom,
                        "start": int(win.start),
                        "end": int(win.end),
                        "genes": win.genes,
                        "stderr": stderr.strip(),
                    }
                )
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


def overlap_hits_with_peaks(peaks: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    if hits.empty:
        return pd.DataFrame(columns=["peak_row_id", "motif_model", "motif_name", "n_motif_hits", "max_motif_score"])
    peak_groups = {}
    for chrom, sub in peaks.groupby("chrom", sort=False):
        sub = sub.sort_values(["start", "end"]).copy()
        peak_groups[chrom] = {
            "row_id": sub["peak_row_id"].to_numpy(np.int64),
            "start": sub["start"].to_numpy(np.int64),
            "end": sub["end"].to_numpy(np.int64),
        }

    agg = {}
    for chrom, sub_hits in hits.groupby("chrom", sort=False):
        if chrom not in peak_groups:
            continue
        chrom_peaks = peak_groups[chrom]
        starts = chrom_peaks["start"]
        ends = chrom_peaks["end"]
        row_ids = chrom_peaks["row_id"]
        ptr = 0
        active = []
        for hit in sub_hits.sort_values(["start", "end"]).itertuples(index=False):
            while ptr < len(starts) and starts[ptr] < int(hit.end):
                active.append((int(ends[ptr]), int(row_ids[ptr])))
                ptr += 1
            next_active = []
            for peak_end, row_id in active:
                if peak_end <= int(hit.start):
                    continue
                next_active.append((peak_end, row_id))
                key = (row_id, hit.motif_model)
                if key not in agg:
                    agg[key] = [0, -np.inf, hit.motif_name]
                agg[key][0] += 1
                agg[key][1] = max(agg[key][1], float(hit.ucsc_score))
            active = next_active

    rows = [
        {
            "peak_row_id": row_id,
            "motif_model": model,
            "motif_name": values[2],
            "n_motif_hits": values[0],
            "max_motif_score": values[1],
            "max_motif_relative_score": np.nan,
        }
        for (row_id, model), values in agg.items()
    ]
    return pd.DataFrame(rows)


def link_target_windows(gene_coords: pd.DataFrame, motif_peaks: pd.DataFrame, window_bp: int) -> pd.DataFrame:
    peak_groups = {}
    for chrom, sub in motif_peaks.groupby("chrom", sort=False):
        peak_groups[chrom] = sub.sort_values(["start", "end"]).reset_index(drop=True)
    rows = []
    for gene in gene_coords.itertuples(index=False):
        if gene.chrom not in peak_groups:
            continue
        sub = peak_groups[gene.chrom]
        starts = sub["start"].to_numpy(np.int64)
        ends = sub["end"].to_numpy(np.int64)
        win_start = max(0, int(gene.tss) - window_bp)
        win_end = int(gene.tss) + window_bp
        right_idx = np.searchsorted(starts, win_end, side="left")
        candidates = sub.iloc[:right_idx].copy()
        candidates = candidates[candidates["end"] > win_start]
        if candidates.empty:
            continue
        dist = peak_distance_to_tss(
            candidates["start"].to_numpy(np.int64),
            candidates["end"].to_numpy(np.int64),
            int(gene.tss),
        )
        candidates = candidates.assign(
            gene=gene.gene,
            gene_id=gene.gene_id,
            gencode_gene_name=gene.gencode_gene_name,
            gene_type=gene.gene_type,
            gene_chrom=gene.chrom,
            gene_tss=int(gene.tss),
            distance_to_tss=dist.astype(int),
        )
        candidates["regulatory_distance_class"] = candidates["distance_to_tss"].map(classify_relation)
        rows.append(candidates)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_target_support(links: pd.DataFrame, genes: list[str], thresholds: list[int]) -> pd.DataFrame:
    rows = []
    if not links.empty:
        for gene, sub in links.groupby("gene", sort=False):
            promoter = sub["regulatory_distance_class"].eq("promoter_2kb")
            proximal = sub["regulatory_distance_class"].isin(["promoter_2kb", "proximal_10kb"])
            bach1 = sub["motif_model"].eq("MA1633.2")
            mafk = sub["motif_model"].eq("MA0591.2")
            row = {
                "gene": gene,
                "has_atac_bach1_motif_support_in_window": True,
                "n_atac_motif_peak_links_in_window": int(sub.shape[0]),
                "n_unique_motif_peaks_in_window": int(sub[["sample", "peak_id"]].drop_duplicates().shape[0]),
                "n_samples_with_motif_peak_in_window": int(sub["sample"].nunique()),
                "samples_with_motif_peak_in_window": ";".join(sorted(sub["sample"].unique())),
                "motif_models_supported": ";".join(sorted(sub["motif_model"].unique())),
                "n_bach1_motif_peak_links_in_window": int(bach1.sum()),
                "n_bach1_mafk_motif_peak_links_in_window": int(mafk.sum()),
                "n_promoter_2kb_motif_peak_links": int(promoter.sum()),
                "n_proximal_10kb_motif_peak_links": int(proximal.sum()),
                "n_nonproximal_window_motif_peak_links": int((~proximal).sum()),
                "closest_motif_peak_distance_to_tss": int(sub["distance_to_tss"].min()),
                "max_ucsc_motif_score": float(sub["max_motif_score"].max()),
                "max_frac_cells_accessible": float(sub["frac_cells_accessible"].max()),
                "max_total_counts": int(sub["total_counts"].max()),
            }
            for threshold in thresholds:
                high = sub["max_motif_score"] >= threshold
                row[f"n_ucsc_score_ge_{threshold}_motif_peak_links_in_window"] = int(high.sum())
                row[f"n_ucsc_score_ge_{threshold}_proximal_10kb_motif_peak_links"] = int((high & proximal).sum())
            rows.append(row)
    support = pd.DataFrame(rows)
    out = pd.DataFrame({"gene": genes}).merge(support, on="gene", how="left")
    base_columns = [
        "has_atac_bach1_motif_support_in_window",
        "n_atac_motif_peak_links_in_window",
        "n_unique_motif_peaks_in_window",
        "n_samples_with_motif_peak_in_window",
        "samples_with_motif_peak_in_window",
        "motif_models_supported",
        "n_bach1_motif_peak_links_in_window",
        "n_bach1_mafk_motif_peak_links_in_window",
        "n_promoter_2kb_motif_peak_links",
        "n_proximal_10kb_motif_peak_links",
        "n_nonproximal_window_motif_peak_links",
        "closest_motif_peak_distance_to_tss",
        "max_ucsc_motif_score",
        "max_frac_cells_accessible",
        "max_total_counts",
    ]
    for threshold in thresholds:
        base_columns.extend(
            [
                f"n_ucsc_score_ge_{threshold}_motif_peak_links_in_window",
                f"n_ucsc_score_ge_{threshold}_proximal_10kb_motif_peak_links",
            ]
        )
    for col in base_columns:
        if col not in out:
            out[col] = False if col.startswith("has_") else ""
            if col.startswith("n_") or col.startswith("max_") or col.startswith("closest_"):
                out[col] = 0
    for col in out.columns:
        if col.startswith("n_") or col in ["closest_motif_peak_distance_to_tss", "max_ucsc_motif_score", "max_frac_cells_accessible", "max_total_counts"]:
            out[col] = out[col].fillna(0)
    out["has_atac_bach1_motif_support_in_window"] = (
        out["has_atac_bach1_motif_support_in_window"].astype("boolean").fillna(False).astype(bool)
    )
    for col in ["samples_with_motif_peak_in_window", "motif_models_supported"]:
        if col in out:
            out[col] = out[col].fillna("")
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    target_table = pd.read_csv(args.target_table)
    if "gene" not in target_table:
        raise ValueError(f"{args.target_table} does not contain a gene column.")
    target_genes = target_table["gene"].dropna().astype(str).unique().tolist()
    if not target_genes:
        summary = {
            "date": str(date.today()),
            "target_table": str(args.target_table),
            "status": "skipped",
            "reason": "No target genes were supplied.",
            "n_bach1_targets": 0,
            "n_targets_with_gencode_coordinates": 0,
            "n_atac_samples": 0,
            "atac_samples": [],
            "n_atac_sample_specific_peaks": 0,
            "n_ucsc_bach1_mafk_tfbs_hits_in_target_windows": 0,
            "n_ucsc_failed_windows": 0,
            "n_motif_positive_sample_specific_peaks_in_target_windows": 0,
            "window_bp": int(args.window_bp),
            "n_bach1_targets_with_atac_motif_support_in_window": 0,
            "n_bach1_targets_with_proximal_10kb_motif_peak": 0,
            "n_bach1_targets_with_promoter_2kb_motif_peak": 0,
            "outputs": {},
        }
        with (out_dir / "bach1_atac_motif_support_ucsc_targeted_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        (table_dir / "pyscenic_bach1_targets_with_ucsc_jaspar2026_atac_support.csv").write_text(
            "gene,has_atac_bach1_motif_support_in_window\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return

    peaks, qc = load_or_summarize_peaks(args.atac_dir, table_dir, args.force_peak_summary)
    gtf = args.resource_dir / "gencode.v44.annotation.gtf.gz"
    gene_coords = load_gencode_gene_coords(gtf, set(target_genes))
    gene_coords.to_csv(table_dir / "gencode_v44_gene_coordinates_for_bach1_targets.csv", index=False)

    missing = sorted(set(target_genes) - set(gene_coords["gene"]))
    pd.DataFrame({"gene": missing}).to_csv(table_dir / "bach1_targets_missing_from_gencode_v44.csv", index=False)
    alias_rows = [
        {"input_gene": old, "gencode_gene_name": new}
        for old, new in GENCODE_ALIAS_MAP.items()
        if old in set(target_genes)
    ]
    pd.DataFrame(alias_rows).to_csv(table_dir / "gencode_v44_symbol_aliases_used.csv", index=False)

    window_label = f"{args.window_bp // 1000}kb" if args.window_bp % 1000 == 0 else f"{args.window_bp}bp"
    windows_bed = table_dir / f"bach1_target_tss_{window_label}_windows.merged.bed"
    write_merged_target_windows(gene_coords, windows_bed, args.window_bp)

    hits_gz = table_dir / "ucsc_jaspar2026_bach1_mafk_hits_in_target_windows.tsv.gz"
    hits, ucsc_failures = fetch_ucsc_motif_hits(
        args.big_bed_to_bed,
        args.ucsc_bigbed,
        windows_bed,
        hits_gz,
        args.ucsc_udc_dir,
        args.query_mode,
        args.allow_partial_ucsc,
        args.bed_chunk_size,
        args.bed_chunk_timeout_sec,
    )
    pd.DataFrame(ucsc_failures).to_csv(table_dir / "ucsc_jaspar2026_failed_windows.csv", index=False)
    overlap = overlap_hits_with_peaks(peaks, hits)
    motif_peaks = overlap.merge(peaks, on="peak_row_id", how="left") if not overlap.empty else pd.DataFrame()
    motif_peaks.to_csv(table_dir / "gse274934_atac_bach1_motif_positive_peaks_target_windows.csv.gz", index=False, compression="gzip")

    target_coords = gene_coords[gene_coords["gene"].isin(target_genes)].copy()
    links = link_target_windows(target_coords, motif_peaks, args.window_bp) if not motif_peaks.empty else pd.DataFrame()
    links.to_csv(table_dir / "bach1_regulon_targets_atac_motif_peak_links_target_windows.csv.gz", index=False, compression="gzip")

    support = aggregate_target_support(links, target_genes, args.score_thresholds)
    integrated = target_table.merge(support, on="gene", how="left")
    integrated.to_csv(table_dir / "pyscenic_bach1_targets_with_ucsc_jaspar2026_atac_support.csv", index=False)

    threshold_rows = []
    for threshold in args.score_thresholds:
        if links.empty:
            sub = links
        else:
            sub = links[links["max_motif_score"] >= threshold]
        threshold_rows.append(
            {
                "ucsc_motif_score_threshold": threshold,
                "n_target_genes_supported_in_window": int(sub["gene"].nunique()) if not sub.empty else 0,
                "n_motif_peak_links_in_window": int(sub.shape[0]) if not sub.empty else 0,
                "n_target_genes_supported_proximal_10kb": int(sub.loc[sub["distance_to_tss"].abs() <= PROXIMAL_BP, "gene"].nunique()) if not sub.empty else 0,
                "n_target_genes_supported_promoter_2kb": int(sub.loc[sub["distance_to_tss"].abs() <= PROMOTER_BP, "gene"].nunique()) if not sub.empty else 0,
            }
        )
    pd.DataFrame(threshold_rows).to_csv(table_dir / "ucsc_jaspar2026_score_threshold_sensitivity.csv", index=False)

    summary = {
        "date": str(date.today()),
        "resource_note": (
            "UCSC native hg38 jaspar2026 bigBed was used as a fallback because the "
            "JASPAR Mencius per-motif TFBS TSV server was unreachable locally."
        ),
        "ucsc_bigbed": args.ucsc_bigbed,
        "motif_models": MOTIFS,
        "ucsc_score_note": (
            "UCSC bigBed score scale differs from the repository's original JASPAR TSV "
            f"high-confidence threshold of {HIGH_CONF_MOTIF_SCORE}; score-threshold rows are "
            "reported as UCSC-specific sensitivity checks."
        ),
        "n_bach1_targets": int(len(target_genes)),
        "n_targets_with_gencode_coordinates": int(gene_coords["gene"].nunique()),
        "n_atac_samples": int(qc["sample"].nunique()) if "sample" in qc else int(peaks["sample"].nunique()),
        "atac_samples": sorted(peaks["sample"].astype(str).unique().tolist()),
        "n_atac_sample_specific_peaks": int(peaks.shape[0]),
        "n_ucsc_bach1_mafk_tfbs_hits_in_target_windows": int(hits.shape[0]),
        "n_ucsc_failed_windows": int(len(ucsc_failures)),
        "n_motif_positive_sample_specific_peaks_in_target_windows": int(motif_peaks[["sample", "peak_id"]].drop_duplicates().shape[0]) if not motif_peaks.empty else 0,
        "window_bp": int(args.window_bp),
        "n_bach1_targets_with_atac_motif_support_in_window": int(integrated["has_atac_bach1_motif_support_in_window"].fillna(False).sum()),
        "n_bach1_targets_with_proximal_10kb_motif_peak": int((integrated["n_proximal_10kb_motif_peak_links"].fillna(0) > 0).sum()),
        "n_bach1_targets_with_promoter_2kb_motif_peak": int((integrated["n_promoter_2kb_motif_peak_links"].fillna(0) > 0).sum()),
        "outputs": {
            "integrated_target_support": str(table_dir / "pyscenic_bach1_targets_with_ucsc_jaspar2026_atac_support.csv"),
            "target_links": str(table_dir / "bach1_regulon_targets_atac_motif_peak_links_target_windows.csv.gz"),
            "motif_peaks": str(table_dir / "gse274934_atac_bach1_motif_positive_peaks_target_windows.csv.gz"),
            "ucsc_hits": str(hits_gz),
        },
    }
    with (out_dir / "bach1_atac_motif_support_ucsc_targeted_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report = out_dir / "bach1_atac_motif_support_ucsc_targeted_report.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Targeted BACH1 ATAC motif support via UCSC JASPAR 2026\n\n")
        handle.write(f"Date: {summary['date']}\n\n")
        handle.write("## Scope\n\n")
        handle.write(
            "This analysis queries UCSC's hg38 `jaspar2026` track only within TSS +/- "
            f"{args.window_bp:,} bp windows for the supplied BACH1 regulon targets, then intersects "
            "MA1633.2/BACH1 and MA0591.2/Bach1::Mafk sites with GSE274934 scATAC peaks.\n\n"
        )
        handle.write("## Important limitation\n\n")
        handle.write(summary["ucsc_score_note"] + "\n\n")
        handle.write("## Summary\n\n")
        for key, value in summary.items():
            if key != "outputs":
                handle.write(f"- {key}: {value}\n")
        handle.write("\n## Outputs\n\n")
        for key, value in summary["outputs"].items():
            handle.write(f"- {key}: {value}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
