"""Download and selectively extract the GEO inputs used by this study."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import requests

from project_paths import DATA_DIR


GSE131907_FILES = {
    "GSE131907_Lung_Cancer_cell_annotation.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_cell_annotation.txt.gz",
    "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz",
    "GSE131907_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/matrix/GSE131907_series_matrix.txt.gz",
}
GSE274934_FILES = {
    "GSE274934_RAW.tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE274nnn/GSE274934/suppl/GSE274934_RAW.tar",
    "GSE274934_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE274nnn/GSE274934/matrix/GSE274934_series_matrix.txt.gz",
}


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Exists: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def selected_destination(name: str, root: Path) -> Path | None:
    if name.endswith("_GEX_filtered_feature_bc_matrix.h5"):
        return root / "GEX_filtered_h5" / name
    if "_ATAC_" in name and (
        name.endswith("_filtered_peak_bc_matrix.h5")
        or name.endswith("_singlecell.csv.gz")
        or name.endswith("_peaks.bed.gz")
    ):
        return root / "ATAC" / name
    return None


def extract_gse274934(tar_path: Path, root: Path) -> None:
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            destination = selected_destination(name, root)
            if destination is None or not member.isfile():
                continue
            if destination.exists() and destination.stat().st_size == member.size:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read {member.name} from {tar_path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            print(f"Extracted {destination.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gse274934-raw-download", action="store_true", help="Use an existing GSE274934_RAW.tar.")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    gse131907 = DATA_DIR / "GSE131907"
    for name, url in GSE131907_FILES.items():
        download(url, gse131907 / name)

    gse274934 = DATA_DIR / "GSE274934"
    for name, url in GSE274934_FILES.items():
        if name == "GSE274934_RAW.tar" and args.skip_gse274934_raw_download:
            continue
        download(url, gse274934 / name)
    tar_path = gse274934 / "GSE274934_RAW.tar"
    if not args.no_extract:
        if not tar_path.exists():
            raise FileNotFoundError(f"Missing {tar_path}")
        extract_gse274934(tar_path, gse274934)


if __name__ == "__main__":
    main()
