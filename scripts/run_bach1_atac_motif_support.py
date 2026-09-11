import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
ATAC_DIR = ROOT / "data" / "GSE274934" / "ATAC"
RESOURCE_DIR = ROOT / "resources" / "atac_motif_support"
PYSCENIC_DIR = ROOT / "out" / "bach1_malignant_epithelial_pyscenic"
OUT_DIR = ROOT / "out" / "bach1_atac_motif_support"
TABLE_DIR = OUT_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TABLE = PYSCENIC_DIR / "tables" / "pyscenic_bach1_regulon_targets_integrated.csv"
PYSCENIC_MATRIX = PYSCENIC_DIR / "malignant_epithelial_counts_filtered_for_pyscenic.csv"
GTF = RESOURCE_DIR / "gencode.v44.annotation.gtf.gz"

MOTIF_FILES = [
    {
        "model": "MA1633.2",
        "motif_name": "BACH1",
        "path": RESOURCE_DIR / "MA1633.2.tsv.gz",
    },
    {
        "model": "MA0591.2",
        "motif_name": "Bach1::Mafk",
        "path": RESOURCE_DIR / "MA0591.2.tsv.gz",
    },
]

WINDOW_BP = 100_000
PROMOTER_BP = 2_000
PROXIMAL_BP = 10_000
HIGH_CONF_MOTIF_SCORE = 950
STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
GENCODE_ALIAS_MAP = {
    "ARNTL": "BMAL1",
    "WDR66": "CFAP251",
    "TMEM161B-AS1": "TMEM161B-DT",
}


def dataframe_to_simple_markdown(df: pd.DataFrame) -> str:
    """Small markdown writer to avoid depending on pandas' optional tabulate package."""
    if df.empty:
        return ""
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.to_numpy()]
    return "\n".join([header, sep] + rows)


def sample_from_path(path: Path) -> str:
    match = re.search(r"_(AL\d+)_ATAC_", path.name)
    if not match:
        raise ValueError(f"Cannot parse sample from {path.name}")
    return match.group(1)


def decode_array(x):
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in x])


def parse_peak_name(name: str):
    chrom, rest = name.split(":")
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def summarize_peak_matrix(h5_path: Path) -> pd.DataFrame:
    sample = sample_from_path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        names = decode_array(h5["matrix/features/name"][:])
        n_features, n_cells = [int(x) for x in h5["matrix/shape"][:]]
        indices_ds = h5["matrix/indices"]
        data_ds = h5["matrix/data"]

        total_counts = np.zeros(n_features, dtype=np.int64)
        n_cells_accessible = np.zeros(n_features, dtype=np.int32)
        chunk = 5_000_000
        n_nnz = len(data_ds)
        for start in range(0, n_nnz, chunk):
            end = min(start + chunk, n_nnz)
            idx = indices_ds[start:end].astype(np.int64)
            val = data_ds[start:end].astype(np.int64)
            total_counts += np.bincount(idx, weights=val, minlength=n_features).astype(np.int64)
            n_cells_accessible += np.bincount(idx, minlength=n_features).astype(np.int32)

    parsed = [parse_peak_name(x) for x in names]
    out = pd.DataFrame(parsed, columns=["chrom", "start", "end"])
    out.insert(0, "sample", sample)
    out.insert(1, "peak_id", names)
    out["n_cells_in_matrix"] = n_cells
    out["total_counts"] = total_counts
    out["n_cells_accessible"] = n_cells_accessible
    out["frac_cells_accessible"] = out["n_cells_accessible"] / float(n_cells)
    out["mean_counts_per_cell"] = out["total_counts"] / float(n_cells)
    out = out[out["chrom"].isin(STANDARD_CHROMS)].reset_index(drop=True)
    return out


def summarize_singlecell_qc(singlecell_path: Path) -> dict:
    sample = sample_from_path(singlecell_path)
    usecols = [
        "barcode",
        "passed_filters",
        "is__cell_barcode",
        "TSS_fragments",
        "peak_region_fragments",
        "blacklist_region_fragments",
    ]
    df = pd.read_csv(singlecell_path, usecols=usecols)
    cells = df[df["is__cell_barcode"] == 1].copy()
    return {
        "sample": sample,
        "n_called_cells_singlecell_csv": int(cells.shape[0]),
        "median_passed_filters": float(cells["passed_filters"].median()) if len(cells) else np.nan,
        "median_TSS_fragments": float(cells["TSS_fragments"].median()) if len(cells) else np.nan,
        "median_peak_region_fragments": float(cells["peak_region_fragments"].median()) if len(cells) else np.nan,
        "median_blacklist_region_fragments": float(cells["blacklist_region_fragments"].median()) if len(cells) else np.nan,
    }


def parse_gene_name(attrs: str) -> str | None:
    match = re.search(r'gene_name "([^"]+)"', attrs)
    return match.group(1) if match else None


def parse_gene_id(attrs: str) -> str | None:
    match = re.search(r'gene_id "([^"]+)"', attrs)
    return match.group(1) if match else None


def parse_gene_type(attrs: str) -> str | None:
    match = re.search(r'gene_type "([^"]+)"', attrs)
    return match.group(1) if match else None


def load_gencode_gene_coords(gtf_path: Path, gene_names: set[str]) -> pd.DataFrame:
    rows = []
    alias_to_original = {v: k for k, v in GENCODE_ALIAS_MAP.items() if k in gene_names}
    wanted_gene_names = set(gene_names) | set(alias_to_original)
    with gzip.open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom = fields[0]
            if chrom not in STANDARD_CHROMS:
                continue
            attrs = fields[8]
            gencode_gene_name = parse_gene_name(attrs)
            if gencode_gene_name not in wanted_gene_names:
                continue
            gene = alias_to_original.get(gencode_gene_name, gencode_gene_name)
            start1 = int(fields[3])
            end1 = int(fields[4])
            strand = fields[6]
            start0 = start1 - 1
            tss = start0 if strand == "+" else end1
            rows.append(
                {
                    "gene": gene,
                    "gencode_gene_name": gencode_gene_name,
                    "gene_id": parse_gene_id(attrs),
                    "gene_type": parse_gene_type(attrs),
                    "chrom": chrom,
                    "start": start0,
                    "end": end1,
                    "strand": strand,
                    "tss": int(tss),
                    "gene_length": int(end1 - start0),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["gene", "gencode_gene_name", "gene_id", "gene_type", "chrom", "start", "end", "strand", "tss", "gene_length"])
    df = pd.DataFrame(rows)
    df["is_protein_coding"] = df["gene_type"].eq("protein_coding").astype(int)
    df = (
        df.sort_values(["gene", "is_protein_coding", "gene_length"], ascending=[True, False, False])
        .drop_duplicates("gene", keep="first")
        .drop(columns=["is_protein_coding"])
        .reset_index(drop=True)
    )
    return df


def build_peaks_by_chrom(peaks: pd.DataFrame):
    out = {}
    for chrom, sub in peaks.groupby("chrom", sort=False):
        sub = sub.sort_values(["start", "end"]).copy()
        out[chrom] = {
            "row_id": sub["peak_row_id"].to_numpy(np.int64),
            "start": sub["start"].to_numpy(np.int64),
            "end": sub["end"].to_numpy(np.int64),
        }
    return out


def update_peak_motif_overlaps(peaks_by_chrom, motif_file: Path, model: str, motif_name: str):
    if not motif_file.exists():
        raise FileNotFoundError(
            f"Missing motif track {motif_file}. "
            "Run scripts/download_reference_resources.py, or provide --resource-dir with "
            "MA1633.2.tsv.gz and MA0591.2.tsv.gz."
        )
    agg = {}
    active = []
    current_chrom = None
    ptr = 0
    starts = ends = row_ids = None

    with gzip.open(motif_file, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                continue
            chrom = fields[0]
            if chrom not in peaks_by_chrom:
                continue
            motif_start = int(fields[1])
            motif_end = int(fields[2])
            score = float(fields[4])
            # JASPAR TSV tracks use score/relative-score/strand; UCSC exports use
            # score/strand/TFName. Keep both readable so provenance checks can use
            # either official representation.
            rel_score = np.nan
            if len(fields) > 5 and fields[5] not in {"+", "-"}:
                rel_score = float(fields[5])

            if chrom != current_chrom:
                current_chrom = chrom
                chrom_peaks = peaks_by_chrom[chrom]
                starts = chrom_peaks["start"]
                ends = chrom_peaks["end"]
                row_ids = chrom_peaks["row_id"]
                ptr = 0
                active = []

            n_peaks = len(starts)
            while ptr < n_peaks and starts[ptr] < motif_end:
                active.append((int(ends[ptr]), int(row_ids[ptr])))
                ptr += 1

            if active:
                next_active = []
                for peak_end, row_id in active:
                    if peak_end <= motif_start:
                        continue
                    next_active.append((peak_end, row_id))
                    key = (row_id, model)
                    if key in agg:
                        agg[key][0] += 1
                        agg[key][1] = max(agg[key][1], score)
                        if not np.isnan(rel_score):
                            agg[key][2] = max(agg[key][2], rel_score)
                    else:
                        agg[key] = [1, score, rel_score, motif_name]
                active = next_active

    rows = [
        {
            "peak_row_id": row_id,
            "motif_model": model,
            "motif_name": values[3],
            "n_motif_hits": values[0],
            "max_motif_score": values[1],
            "max_motif_relative_score": values[2],
        }
        for (row_id, _), values in agg.items()
    ]
    return pd.DataFrame(rows)


def peak_distance_to_tss(peak_start: np.ndarray, peak_end: np.ndarray, tss: int) -> np.ndarray:
    left = np.maximum(0, peak_start - tss)
    right = np.maximum(0, tss - peak_end)
    return left + right


def classify_relation(distance: int) -> str:
    if distance <= PROMOTER_BP:
        return "promoter_2kb"
    if distance <= PROXIMAL_BP:
        return "proximal_10kb"
    return "distal_100kb"


def link_motif_peaks_to_genes(gene_coords: pd.DataFrame, motif_peaks: pd.DataFrame) -> pd.DataFrame:
    peak_groups = {}
    for chrom, sub in motif_peaks.groupby("chrom", sort=False):
        sub = sub.sort_values(["start", "end"]).reset_index(drop=True)
        peak_groups[chrom] = sub

    link_rows = []
    for _, gene in gene_coords.iterrows():
        chrom = gene["chrom"]
        if chrom not in peak_groups:
            continue
        sub = peak_groups[chrom]
        starts = sub["start"].to_numpy(np.int64)
        ends = sub["end"].to_numpy(np.int64)
        win_start = max(0, int(gene["tss"]) - WINDOW_BP)
        win_end = int(gene["tss"]) + WINDOW_BP
        right_idx = np.searchsorted(starts, win_end, side="left")
        if right_idx == 0:
            continue
        candidate = sub.iloc[:right_idx].copy()
        candidate = candidate[candidate["end"] > win_start]
        if candidate.empty:
            continue
        dist = peak_distance_to_tss(
            candidate["start"].to_numpy(np.int64),
            candidate["end"].to_numpy(np.int64),
            int(gene["tss"]),
        )
        candidate = candidate.assign(
            gene=gene["gene"],
            gene_id=gene["gene_id"],
            gencode_gene_name=gene["gencode_gene_name"],
            gene_type=gene["gene_type"],
            gene_chrom=chrom,
            gene_tss=int(gene["tss"]),
            distance_to_tss=dist.astype(int),
        )
        candidate["regulatory_distance_class"] = candidate["distance_to_tss"].map(classify_relation)
        link_rows.append(candidate)

    if not link_rows:
        return pd.DataFrame()

    columns = [
        "gene",
        "gene_id",
        "gencode_gene_name",
        "gene_type",
        "gene_chrom",
        "gene_tss",
        "sample",
        "peak_id",
        "chrom",
        "start",
        "end",
        "distance_to_tss",
        "regulatory_distance_class",
        "motif_model",
        "motif_name",
        "n_motif_hits",
        "max_motif_score",
        "max_motif_relative_score",
        "n_cells_in_matrix",
        "total_counts",
        "n_cells_accessible",
        "frac_cells_accessible",
        "mean_counts_per_cell",
    ]
    return pd.concat(link_rows, ignore_index=True)[columns]


def aggregate_gene_support(links: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    rows = []
    if links.empty:
        return pd.DataFrame({"gene": genes})
    for gene, sub in links.groupby("gene", sort=False):
        promoter = sub["regulatory_distance_class"].eq("promoter_2kb")
        proximal = sub["regulatory_distance_class"].isin(["promoter_2kb", "proximal_10kb"])
        bach1 = sub["motif_model"].eq("MA1633.2")
        mafk = sub["motif_model"].eq("MA0591.2")
        high_conf = sub["max_motif_score"] >= HIGH_CONF_MOTIF_SCORE
        high_conf_promoter = high_conf & promoter
        high_conf_proximal = high_conf & proximal
        rows.append(
            {
                "gene": gene,
                "has_atac_bach1_motif_support_100kb": True,
                "has_high_conf_atac_bach1_motif_support_100kb": bool(high_conf.any()),
                "n_atac_motif_peak_links_100kb": int(sub.shape[0]),
                "n_unique_motif_peaks_100kb": int(sub[["sample", "peak_id"]].drop_duplicates().shape[0]),
                "n_samples_with_motif_peak_100kb": int(sub["sample"].nunique()),
                "samples_with_motif_peak_100kb": ";".join(sorted(sub["sample"].unique())),
                "motif_models_supported": ";".join(sorted(sub["motif_model"].unique())),
                "n_bach1_motif_peak_links_100kb": int(bach1.sum()),
                "n_bach1_mafk_motif_peak_links_100kb": int(mafk.sum()),
                "n_promoter_2kb_motif_peak_links": int(promoter.sum()),
                "n_proximal_10kb_motif_peak_links": int(proximal.sum()),
                "n_distal_100kb_motif_peak_links": int((~proximal).sum()),
                "n_high_conf_motif_peak_links_100kb": int(high_conf.sum()),
                "n_high_conf_unique_motif_peaks_100kb": int(sub.loc[high_conf, ["sample", "peak_id"]].drop_duplicates().shape[0]),
                "n_samples_with_high_conf_motif_peak_100kb": int(sub.loc[high_conf, "sample"].nunique()),
                "samples_with_high_conf_motif_peak_100kb": ";".join(sorted(sub.loc[high_conf, "sample"].unique())),
                "n_high_conf_bach1_motif_peak_links_100kb": int((high_conf & bach1).sum()),
                "n_high_conf_bach1_mafk_motif_peak_links_100kb": int((high_conf & mafk).sum()),
                "n_high_conf_promoter_2kb_motif_peak_links": int(high_conf_promoter.sum()),
                "n_high_conf_proximal_10kb_motif_peak_links": int(high_conf_proximal.sum()),
                "closest_high_conf_motif_peak_distance_to_tss": int(sub.loc[high_conf, "distance_to_tss"].min()) if high_conf.any() else np.nan,
                "closest_motif_peak_distance_to_tss": int(sub["distance_to_tss"].min()),
                "max_motif_score": float(sub["max_motif_score"].max()),
                "max_motif_relative_score": float(sub["max_motif_relative_score"].max()),
                "max_frac_cells_accessible": float(sub["frac_cells_accessible"].max()),
                "max_total_counts": int(sub["total_counts"].max()),
            }
        )
    support = pd.DataFrame(rows)
    out = pd.DataFrame({"gene": genes}).merge(support, on="gene", how="left")
    fill_zero = [
        "n_atac_motif_peak_links_100kb",
        "n_unique_motif_peaks_100kb",
        "n_samples_with_motif_peak_100kb",
        "n_bach1_motif_peak_links_100kb",
        "n_bach1_mafk_motif_peak_links_100kb",
        "n_promoter_2kb_motif_peak_links",
        "n_proximal_10kb_motif_peak_links",
        "n_distal_100kb_motif_peak_links",
    ]
    for col in fill_zero:
        out[col] = out[col].fillna(0).astype(int)
    high_conf_zero = [
        "n_high_conf_motif_peak_links_100kb",
        "n_high_conf_unique_motif_peaks_100kb",
        "n_samples_with_high_conf_motif_peak_100kb",
        "n_high_conf_bach1_motif_peak_links_100kb",
        "n_high_conf_bach1_mafk_motif_peak_links_100kb",
        "n_high_conf_promoter_2kb_motif_peak_links",
        "n_high_conf_proximal_10kb_motif_peak_links",
    ]
    for col in high_conf_zero:
        out[col] = out[col].fillna(0).astype(int)
    out["has_atac_bach1_motif_support_100kb"] = out["has_atac_bach1_motif_support_100kb"].fillna(False).astype(bool)
    out["has_high_conf_atac_bach1_motif_support_100kb"] = out["has_high_conf_atac_bach1_motif_support_100kb"].fillna(False).astype(bool)
    for col in ["samples_with_motif_peak_100kb", "motif_models_supported"]:
        out[col] = out[col].fillna("")
    out["samples_with_high_conf_motif_peak_100kb"] = out["samples_with_high_conf_motif_peak_100kb"].fillna("")
    return out


def read_expressed_genes(matrix_csv: Path) -> list[str]:
    cols = pd.read_csv(matrix_csv, nrows=0).columns.tolist()
    if cols and cols[0].lower().startswith("unnamed"):
        cols = cols[1:]
    return [x for x in cols if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atac-dir", type=Path, default=ATAC_DIR)
    parser.add_argument("--resource-dir", type=Path, default=RESOURCE_DIR)
    parser.add_argument("--pyscenic-dir", type=Path, default=PYSCENIC_DIR)
    parser.add_argument("--target-table", type=Path, default=None)
    parser.add_argument("--pyscenic-matrix", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def run():
    global ATAC_DIR, RESOURCE_DIR, PYSCENIC_DIR, OUT_DIR, TABLE_DIR, TARGET_TABLE, PYSCENIC_MATRIX, GTF, MOTIF_FILES

    args = parse_args()
    ATAC_DIR = args.atac_dir
    RESOURCE_DIR = args.resource_dir
    PYSCENIC_DIR = args.pyscenic_dir
    OUT_DIR = args.output_dir
    TABLE_DIR = OUT_DIR / "tables"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_TABLE = args.target_table or (PYSCENIC_DIR / "tables" / "pyscenic_bach1_regulon_targets_integrated.csv")
    PYSCENIC_MATRIX = args.pyscenic_matrix or (PYSCENIC_DIR / "malignant_epithelial_counts_filtered_for_pyscenic.csv")
    GTF = RESOURCE_DIR / "gencode.v44.annotation.gtf.gz"
    MOTIF_FILES = [
        {
            "model": "MA1633.2",
            "motif_name": "BACH1",
            "path": RESOURCE_DIR / "MA1633.2.tsv.gz",
        },
        {
            "model": "MA0591.2",
            "motif_name": "Bach1::Mafk",
            "path": RESOURCE_DIR / "MA0591.2.tsv.gz",
        },
    ]

    h5_files = sorted(ATAC_DIR.glob("*_ATAC_filtered_peak_bc_matrix.h5"))
    singlecell_files = sorted(ATAC_DIR.glob("*_ATAC_singlecell.csv.gz"))
    if not h5_files:
        raise FileNotFoundError(f"No ATAC h5 files found in {ATAC_DIR}")

    peak_tables = []
    for h5_path in h5_files:
        print(f"Summarizing peaks: {h5_path.name}", flush=True)
        peak_tables.append(summarize_peak_matrix(h5_path))
    peaks = pd.concat(peak_tables, ignore_index=True)
    peaks.insert(0, "peak_row_id", np.arange(len(peaks), dtype=np.int64))
    peaks.to_csv(TABLE_DIR / "gse274934_atac_peak_accessibility_summary.csv.gz", index=False)

    qc_rows = [summarize_singlecell_qc(x) for x in singlecell_files]
    qc = pd.DataFrame(qc_rows)
    matrix_cells = peaks.groupby("sample")["n_cells_in_matrix"].max().reset_index(name="n_cells_in_filtered_peak_matrix")
    qc = matrix_cells.merge(qc, on="sample", how="left")
    qc.to_csv(TABLE_DIR / "gse274934_atac_sample_qc_summary.csv", index=False)

    print("Intersecting BACH1 motif tracks with ATAC peaks", flush=True)
    peaks_by_chrom = build_peaks_by_chrom(peaks)
    motif_overlaps = []
    for motif in MOTIF_FILES:
        print(f"  motif {motif['model']} {motif['motif_name']}", flush=True)
        motif_overlaps.append(update_peak_motif_overlaps(peaks_by_chrom, motif["path"], motif["model"], motif["motif_name"]))
    motif_overlap = pd.concat(motif_overlaps, ignore_index=True)
    motif_peaks = motif_overlap.merge(peaks, on="peak_row_id", how="left")
    motif_peaks = motif_peaks[
        [
            "peak_row_id",
            "sample",
            "peak_id",
            "chrom",
            "start",
            "end",
            "motif_model",
            "motif_name",
            "n_motif_hits",
            "max_motif_score",
            "max_motif_relative_score",
            "n_cells_in_matrix",
            "total_counts",
            "n_cells_accessible",
            "frac_cells_accessible",
            "mean_counts_per_cell",
        ]
    ]
    motif_peaks.to_csv(TABLE_DIR / "gse274934_atac_bach1_motif_positive_peaks.csv.gz", index=False)

    motif_threshold_rows = []
    for threshold in [850, 900, 925, 950, 975]:
        sub = motif_peaks[motif_peaks["max_motif_score"] >= threshold]
        motif_threshold_rows.append(
            {
                "max_motif_score_threshold": threshold,
                "n_motif_peak_rows": int(sub.shape[0]),
                "n_unique_sample_specific_peaks": int(sub[["sample", "peak_id"]].drop_duplicates().shape[0]),
            }
        )
    motif_threshold_summary = pd.DataFrame(motif_threshold_rows)
    motif_threshold_summary.to_csv(TABLE_DIR / "gse274934_atac_bach1_motif_score_threshold_summary.csv", index=False)

    target_table = pd.read_csv(TARGET_TABLE)
    target_genes = target_table["gene"].dropna().astype(str).unique().tolist()
    expressed_genes = read_expressed_genes(PYSCENIC_MATRIX)
    all_genes = sorted(set(target_genes) | set(expressed_genes))

    print("Parsing GENCODE gene coordinates", flush=True)
    gene_coords = load_gencode_gene_coords(GTF, set(all_genes))
    gene_coords.to_csv(TABLE_DIR / "gencode_v44_gene_coordinates_for_pyscenic_expressed_genes.csv.gz", index=False)
    alias_rows = [
        {"input_gene": old, "gencode_gene_name": new}
        for old, new in GENCODE_ALIAS_MAP.items()
        if old in set(all_genes)
    ]
    pd.DataFrame(alias_rows).to_csv(TABLE_DIR / "gencode_v44_symbol_aliases_used.csv", index=False)
    missing_target_genes = sorted(set(target_genes) - set(gene_coords["gene"]))
    pd.DataFrame({"gene": missing_target_genes}).to_csv(TABLE_DIR / "bach1_targets_missing_from_gencode_v44.csv", index=False)

    target_coords = gene_coords[gene_coords["gene"].isin(target_genes)].copy()
    print("Linking motif-positive ATAC peaks to BACH1 regulon targets", flush=True)
    target_links = link_motif_peaks_to_genes(target_coords, motif_peaks)
    target_links.to_csv(TABLE_DIR / "bach1_regulon_targets_atac_motif_peak_links.csv.gz", index=False)

    target_support = aggregate_gene_support(target_links, target_genes)
    integrated = target_table.merge(target_support, on="gene", how="left")
    integrated["has_atac_bach1_motif_support_100kb"] = integrated["has_atac_bach1_motif_support_100kb"].fillna(False).astype(bool)
    integrated["has_high_conf_atac_bach1_motif_support_100kb"] = integrated["has_high_conf_atac_bach1_motif_support_100kb"].fillna(False).astype(bool)
    for col in [c for c in integrated.columns if c.startswith("n_") and c.endswith(("100kb", "links", "peaks"))]:
        integrated[col] = integrated[col].fillna(0)
    integrated.to_csv(TABLE_DIR / "pyscenic_bach1_regulon_targets_with_atac_motif_support.csv", index=False)

    print("Computing expressed-gene background enrichment", flush=True)
    expressed_coords = gene_coords[gene_coords["gene"].isin(expressed_genes)].copy()
    expressed_links = link_motif_peaks_to_genes(expressed_coords, motif_peaks)
    expressed_support = aggregate_gene_support(expressed_links, expressed_genes)
    expressed_support.to_csv(TABLE_DIR / "expressed_genes_atac_bach1_motif_support_background.csv.gz", index=False)

    target_set = set(target_genes)
    expressed_support["is_pyscenic_bach1_regulon_target"] = expressed_support["gene"].isin(target_set)
    enrichment_rows = []
    for label, mask in [
        ("any_BACH1_or_Bach1_Mafk_motif_peak_100kb", expressed_support["has_atac_bach1_motif_support_100kb"]),
        ("BACH1_MA1633_motif_peak_100kb", expressed_support["n_bach1_motif_peak_links_100kb"] > 0),
        ("Bach1_Mafk_MA0591_motif_peak_100kb", expressed_support["n_bach1_mafk_motif_peak_links_100kb"] > 0),
        ("promoter_2kb_motif_peak", expressed_support["n_promoter_2kb_motif_peak_links"] > 0),
        ("proximal_10kb_motif_peak", expressed_support["n_proximal_10kb_motif_peak_links"] > 0),
        (f"high_conf_score_ge_{HIGH_CONF_MOTIF_SCORE}_motif_peak_100kb", expressed_support["has_high_conf_atac_bach1_motif_support_100kb"]),
        (f"high_conf_score_ge_{HIGH_CONF_MOTIF_SCORE}_BACH1_MA1633_motif_peak_100kb", expressed_support["n_high_conf_bach1_motif_peak_links_100kb"] > 0),
        (f"high_conf_score_ge_{HIGH_CONF_MOTIF_SCORE}_promoter_2kb_motif_peak", expressed_support["n_high_conf_promoter_2kb_motif_peak_links"] > 0),
        (f"high_conf_score_ge_{HIGH_CONF_MOTIF_SCORE}_proximal_10kb_motif_peak", expressed_support["n_high_conf_proximal_10kb_motif_peak_links"] > 0),
    ]:
        a = int((expressed_support["is_pyscenic_bach1_regulon_target"] & mask).sum())
        b = int((expressed_support["is_pyscenic_bach1_regulon_target"] & ~mask).sum())
        c = int((~expressed_support["is_pyscenic_bach1_regulon_target"] & mask).sum())
        d = int((~expressed_support["is_pyscenic_bach1_regulon_target"] & ~mask).sum())
        oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        enrichment_rows.append(
            {
                "test": label,
                "target_supported": a,
                "target_not_supported": b,
                "background_supported": c,
                "background_not_supported": d,
                "oddsratio": oddsratio,
                "fisher_pvalue_greater": pvalue,
            }
        )
    enrichment = pd.DataFrame(enrichment_rows)
    enrichment.to_csv(TABLE_DIR / "bach1_regulon_atac_motif_support_enrichment_summary.csv", index=False)

    top_for_review = integrated.sort_values(
        [
            "candidate_score",
            "has_high_conf_atac_bach1_motif_support_100kb",
            "n_samples_with_high_conf_motif_peak_100kb",
            "n_high_conf_proximal_10kb_motif_peak_links",
            "max_motif_score",
        ],
        ascending=[False, False, False, False, False],
    )
    top_for_review.to_csv(TABLE_DIR / "pyscenic_bach1_targets_ranked_with_atac_support.csv", index=False)

    summary = {
        "n_atac_samples": int(len(h5_files)),
        "atac_samples": sorted([sample_from_path(x) for x in h5_files]),
        "n_atac_sample_specific_peaks": int(peaks.shape[0]),
        "n_motif_positive_sample_specific_peaks": int(motif_peaks[["sample", "peak_id"]].drop_duplicates().shape[0]),
        "n_pyscenic_bach1_targets": int(len(target_genes)),
        "n_pyscenic_bach1_targets_with_gencode_coordinates": int(target_coords["gene"].nunique()),
        "n_pyscenic_bach1_targets_with_atac_motif_support_100kb": int(integrated["has_atac_bach1_motif_support_100kb"].sum()),
        "n_pyscenic_bach1_targets_with_promoter_2kb_motif_peak": int((integrated["n_promoter_2kb_motif_peak_links"].fillna(0) > 0).sum()),
        "n_pyscenic_bach1_targets_with_proximal_10kb_motif_peak": int((integrated["n_proximal_10kb_motif_peak_links"].fillna(0) > 0).sum()),
        "high_conf_motif_score_threshold": int(HIGH_CONF_MOTIF_SCORE),
        "n_pyscenic_bach1_targets_with_high_conf_atac_motif_support_100kb": int(integrated["has_high_conf_atac_bach1_motif_support_100kb"].sum()),
        "n_pyscenic_bach1_targets_with_high_conf_promoter_2kb_motif_peak": int((integrated["n_high_conf_promoter_2kb_motif_peak_links"].fillna(0) > 0).sum()),
        "n_pyscenic_bach1_targets_with_high_conf_proximal_10kb_motif_peak": int((integrated["n_high_conf_proximal_10kb_motif_peak_links"].fillna(0) > 0).sum()),
        "top_targets_with_integrated_and_atac_support": top_for_review.head(30)["gene"].tolist(),
    }
    with open(OUT_DIR / "bach1_atac_motif_support_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    report = OUT_DIR / "bach1_atac_motif_support_report.md"
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("# BACH1 ATAC motif support report\n\n")
        fh.write("Date: 2026-06-18\n\n")
        fh.write("## Definition\n\n")
        fh.write(
            "ATAC motif support is defined as an accessible scATAC peak from GSE274934 "
            "that overlaps a precomputed hg38 JASPAR BACH1 motif site and lies within "
            f"{WINDOW_BP:,} bp of the target gene TSS. Distances are classified as "
            f"promoter <= {PROMOTER_BP:,} bp, proximal <= {PROXIMAL_BP:,} bp, and distal <= {WINDOW_BP:,} bp.\n\n"
        )
        fh.write("## Resources\n\n")
        fh.write("- JASPAR 2026 hg38 TFBS: MA1633.2 BACH1 and MA0591.2 Bach1::Mafk\n")
        fh.write("- GENCODE v44 hg38 gene annotation\n\n")
        fh.write(f"High-confidence motif support uses max motif score >= {HIGH_CONF_MOTIF_SCORE}.\n\n")
        fh.write("## Summary\n\n")
        for key, value in summary.items():
            fh.write(f"- {key}: {value}\n")
        fh.write("\n## Enrichment against expressed-gene background\n\n")
        fh.write(dataframe_to_simple_markdown(enrichment))
        fh.write("\n\n## Main tables\n\n")
        fh.write("- tables/pyscenic_bach1_regulon_targets_with_atac_motif_support.csv\n")
        fh.write("- tables/bach1_regulon_targets_atac_motif_peak_links.csv.gz\n")
        fh.write("- tables/gse274934_atac_bach1_motif_positive_peaks.csv.gz\n")
        fh.write("- tables/bach1_regulon_atac_motif_support_enrichment_summary.csv\n")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
