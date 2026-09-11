"""Run a full-TF pySCENIC workflow and keep outputs separate from BACH1-only results."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from project_paths import OUTPUT_DIR, PROJECT_RESOURCES_DIR


INPUT = OUTPUT_DIR / "bach1_malignant_epithelial_pyscenic"
OUT = OUTPUT_DIR / "bach1_malignant_epithelial_pyscenic_all_tfs"
RES = PROJECT_RESOURCES_DIR / "pyscenic"

EXPR = INPUT / "malignant_epithelial_counts_filtered_for_pyscenic.csv"
TFS = INPUT / "hs_hgnc_tfs_expressed_in_malignant_epithelial.txt"
ADJ = OUT / "pyscenic_all_tfs_grnboost2_adjacencies.csv"
REG = OUT / "pyscenic_all_tfs_regulons.csv"
AUC = OUT / "pyscenic_all_tfs_aucell.csv"

DB_500 = RES / "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
DB_10K = RES / "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
ANNOT = RES / "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"


def execute(command: list[str], stem: str, dry_run: bool = False, skip_existing: Path | None = None) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    if skip_existing is not None and skip_existing.exists() and skip_existing.stat().st_size > 0:
        print(f"Exists, skipping: {skip_existing}", flush=True)
        return
    with (OUT / f"{stem}.log").open("w", encoding="utf-8") as stdout, (
        OUT / f"{stem}.err.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, check=True, stdout=stdout, stderr=stderr)


def regulons_are_empty(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    except pd.errors.EmptyDataError:
        return True
    return df.empty


def main() -> None:
    global INPUT, OUT, EXPR, TFS, ADJ, REG, AUC

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--pyscenic", default=shutil.which("pyscenic"))
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true", help="Skip completed output files.")
    parser.add_argument(
        "--allow-empty-regulons",
        action="store_true",
        help="Treat an empty ctx regulon table as a completed seed and skip AUCell.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    INPUT = args.input_dir
    OUT = args.output_dir
    EXPR = INPUT / "malignant_epithelial_counts_filtered_for_pyscenic.csv"
    TFS = INPUT / "hs_hgnc_tfs_expressed_in_malignant_epithelial.txt"
    ADJ = OUT / "pyscenic_all_tfs_grnboost2_adjacencies.csv"
    REG = OUT / "pyscenic_all_tfs_regulons.csv"
    AUC = OUT / "pyscenic_all_tfs_aucell.csv"

    if not args.pyscenic:
        raise RuntimeError("pyscenic executable was not found; activate the pySCENIC environment first")

    required = [EXPR, TFS, DB_500, DB_10K, ANNOT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing full-TF pySCENIC inputs:\n" + "\n".join(missing))

    OUT.mkdir(parents=True, exist_ok=True)
    workers = str(args.workers)
    seed = str(args.seed)

    execute(
        [
            args.pyscenic,
            "grn",
            "--method",
            "grnboost2",
            "--seed",
            seed,
            "--num_workers",
            workers,
            "-o",
            str(ADJ),
            str(EXPR),
            str(TFS),
        ],
        "pyscenic_all_tfs_grn",
        args.dry_run,
        ADJ if args.resume else None,
    )
    execute(
        [
            args.pyscenic,
            "ctx",
            str(ADJ),
            str(DB_500),
            str(DB_10K),
            "--annotations_fname",
            str(ANNOT),
            "--expression_mtx_fname",
            str(EXPR),
            "--mask_dropouts",
            "--mode",
            "dask_multiprocessing",
            "--num_workers",
            workers,
            "--min_genes",
            "10",
            "-o",
            str(REG),
        ],
        "pyscenic_all_tfs_ctx",
        args.dry_run,
        REG if args.resume else None,
    )
    if args.allow_empty_regulons and regulons_are_empty(REG):
        print(f"Regulon table is empty for seed {seed}; skipping AUCell because --allow-empty-regulons was set.", flush=True)
        return
    execute(
        [
            args.pyscenic,
            "aucell",
            str(EXPR),
            str(REG),
            "--num_workers",
            workers,
            "--seed",
            seed,
            "-o",
            str(AUC),
        ],
        "pyscenic_all_tfs_aucell",
        args.dry_run,
        AUC if args.resume else None,
    )


if __name__ == "__main__":
    main()
