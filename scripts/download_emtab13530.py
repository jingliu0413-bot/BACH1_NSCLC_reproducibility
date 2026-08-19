"""Download the E-MTAB-13530 Visium matrices, spatial archives and metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from project_paths import DATA_DIR


BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/530/E-MTAB-13530/Files"
SAMPLES = [
    "D1_1", "D1_2", "D2_1", "D2_2",
    "P10_B1", "P10_B2", "P10_T1", "P10_T2", "P10_T3", "P10_T4",
    "P11_B1", "P11_B2", "P11_T1", "P11_T2", "P11_T3", "P11_T4",
    "P15_B1", "P15_B2", "P15_T1", "P15_T2",
    "P16_B1", "P16_B2", "P16_T1", "P16_T2",
    "P17_B1", "P17_B2", "P17_T1", "P17_T2",
    "P19_B1", "P19_B2", "P19_T1", "P19_T2",
    "P24_B1", "P24_B2", "P24_T1", "P24_T2",
    "P25_B1", "P25_B2", "P25_T1", "P25_T2",
]


def download(name: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Exists: {destination.name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{BASE_URL}/{name}"
    print(f"Downloading {url}", flush=True)
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", default=SAMPLES)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.samples) - set(SAMPLES))
    if unknown:
        raise ValueError(f"Unknown sample names: {unknown}")

    raw = DATA_DIR / "spatial" / "E-MTAB-13530" / "raw"
    for name in ["E-MTAB-13530.idf.txt", "E-MTAB-13530.sdrf.txt"]:
        download(name, raw / name)
    if args.metadata_only:
        return
    for sample in args.samples:
        for suffix in ["filtered_feature_bc_matrix.h5", "spatial.tar"]:
            name = f"{sample}-{suffix}"
            download(name, raw / name)


if __name__ == "__main__":
    main()
