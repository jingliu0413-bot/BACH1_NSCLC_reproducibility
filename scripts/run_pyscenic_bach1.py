"""Run the exact BACH1-focused pySCENIC command sequence used in the study."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from project_paths import OUTPUT_DIR, PROJECT_RESOURCES_DIR


OUT = OUTPUT_DIR / "bach1_malignant_epithelial_pyscenic"
RES = PROJECT_RESOURCES_DIR / "pyscenic"
EXPR = OUT / "malignant_epithelial_counts_filtered_for_pyscenic.csv"
TFS = OUT / "bach1_only_tf.txt"
ADJ = OUT / "pyscenic_bach1_grnboost2_adjacencies.csv"
REG = OUT / "pyscenic_bach1_regulons.csv"
AUC = OUT / "pyscenic_bach1_aucell.csv"
DB_500 = RES / "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
DB_10K = RES / "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
ANNOT = RES / "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"


def execute(command: list[str], stem: str, dry_run: bool = False) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    with (OUT / f"{stem}.log").open("w", encoding="utf-8") as stdout, (
        OUT / f"{stem}.err.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, check=True, stdout=stdout, stderr=stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pyscenic", default=shutil.which("pyscenic"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.pyscenic:
        raise RuntimeError("pyscenic executable was not found; activate the pySCENIC environment first")

    required = [EXPR, DB_500, DB_10K, ANNOT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing pySCENIC inputs:\n" + "\n".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        TFS.write_text("BACH1\n", encoding="ascii")
    workers = str(args.workers)

    execute(
        [args.pyscenic, "grn", "--method", "grnboost2", "--seed", "777", "--num_workers", workers, "-o", str(ADJ), str(EXPR), str(TFS)],
        "pyscenic_bach1_grn",
        args.dry_run,
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
        "pyscenic_bach1_ctx",
        args.dry_run,
    )
    execute(
        [args.pyscenic, "aucell", str(EXPR), str(REG), "--num_workers", workers, "--seed", "777", "-o", str(AUC)],
        "pyscenic_bach1_aucell",
        args.dry_run,
    )


if __name__ == "__main__":
    main()
