"""Download external reference files without adding them to the code repository."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

from project_paths import PROJECT_RESOURCES_DIR


FILES = {
    "pyscenic/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather": (
        "https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
        "9c4026a3a8e25fe07cf96749644e2ca028b787410829b30b9932574dc6e78bdb",
    ),
    "pyscenic/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather": (
        "https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
        "50bd3a2e0da17bf74ad31b1a2f284ebbacfad700a07a540a7a1d3db5e185eaac",
    ),
    "pyscenic/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl": (
        "https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
        "81eb754118e27e854974301b1400fcf519489f8be5249239671fb288cb501c31",
    ),
    "pyscenic/hs_hgnc_tfs.txt": (
        "https://raw.githubusercontent.com/aertslab/pySCENIC/0.12.1/resources/hs_hgnc_tfs.txt",
        "b3d59559bc1c04df61954b4051f1a4e0c17c849f532cd6271f749d832a18509a",
    ),
    "atac_motif_support/MA1633.2.tsv.gz": (
        "https://mencius.uio.no/JASPAR/JASPAR_TFBSs/2026/hg38/MA1633.2.tsv.gz",
        "5c3882c49f193a6849f0e71039942783ba68b03b2ba01e9b97d61c847bc8598c",
    ),
    "atac_motif_support/MA0591.2.tsv.gz": (
        "https://mencius.uio.no/JASPAR/JASPAR_TFBSs/2026/hg38/MA0591.2.tsv.gz",
        "eb6dbffc0de5f0abbc1d367f19d953c413ab94f470800715a2dbba92ffd3a9c6",
    ),
    "atac_motif_support/gencode.v44.annotation.gtf.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz",
        "01f817afed65feee863361b4baf30a95e722ee5c5d508ff77b04106ef7ba20d3",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for relative, (url, expected) in FILES.items():
        path = PROJECT_RESOURCES_DIR / relative
        if args.force or not path.exists() or sha256(path) != expected:
            print(f"Downloading {url}", flush=True)
            download(url, path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {actual}")
        print(f"Verified {relative}", flush=True)


if __name__ == "__main__":
    main()
