from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import infercnvpy as cnv
import scanpy as sc

from project_paths import OUTPUT_DIR

BASE_DIR = OUTPUT_DIR
SOURCE_SCRIPT = Path(__file__).with_name("run_epithelial_cnv_infercnvpy.py")
INPUT_H5AD = (
    BASE_DIR
    / "epithelial_cnv_infercnvpy"
    / "nsclc_gse131907_gse274934_epithelial_cnv_input_epithelial_tnk_reference_autosomes.h5ad"
)
OUT_DIR = BASE_DIR / "epithelial_cnv_infercnvpy_sensitivity_no_dynamic_threshold"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PREFIX = "nsclc_gse131907_gse274934_epithelial_cnv_no_dynamic_threshold"
RANDOM_STATE = 7


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_helpers():
    spec = importlib.util.spec_from_file_location("epithelial_cnv_helpers", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT_DIR = OUT_DIR
    module.FIG_DIR = FIG_DIR
    module.TABLE_DIR = TABLE_DIR
    module.PREFIX = PREFIX
    module.INPUT_H5AD = INPUT_H5AD
    return module


def main() -> None:
    helper = load_helpers()
    helper.ensure_dirs()
    helper.apply_publication_style()

    params = {
        "input_h5ad": str(INPUT_H5AD),
        "source": "sensitivity rerun using the same epithelial + 5% T/NK reference input",
        "infercnv": {
            "window_size": 100,
            "step": 10,
            "dynamic_threshold": None,
            "lfc_clip": 3,
            "reference_key": "cnv_input_group",
            "reference_cat": "T/NK_reference",
            "n_jobs": 1,
            "chunksize": 4000,
        },
    }
    with open(OUT_DIR / f"{PREFIX}_params.json", "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)

    log(f"Reading saved CNV input subset: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    log("Running infercnvpy.tl.infercnv without dynamic threshold")
    cnv.tl.infercnv(
        adata,
        reference_key="cnv_input_group",
        reference_cat="T/NK_reference",
        lfc_clip=3,
        window_size=100,
        step=10,
        dynamic_threshold=None,
        exclude_chromosomes=None,
        chunksize=4000,
        n_jobs=1,
        inplace=True,
        key_added="cnv",
        calculate_gene_values=False,
    )

    log("Running CNV PCA, neighbors, UMAP and Leiden")
    cnv.tl.pca(adata, n_comps=30, random_state=RANDOM_STATE)
    cnv.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    cnv.tl.leiden(adata, resolution=0.5, random_state=RANDOM_STATE)
    cnv.tl.umap(adata, random_state=RANDOM_STATE)
    cnv.tl.cnv_score(adata, groupby="cnv_leiden", use_rep="cnv")

    log("Classifying candidate malignant epithelial cells")
    thresholds = helper.classify_cnv(adata)

    log("Writing tables and figures")
    helper.write_result_tables(adata)
    helper.plot_embedding(
        adata,
        "X_umap",
        "cnv_malignancy_call",
        FIG_DIR / f"{PREFIX}_integrated_umap_cnv_call",
        "Integrated UMAP: epithelial CNV calls (no dynamic threshold)",
        categorical=True,
    )
    helper.plot_embedding(
        adata,
        "X_umap",
        "cnv_burden",
        FIG_DIR / f"{PREFIX}_integrated_umap_cnv_burden",
        "Integrated UMAP: CNV burden (no dynamic threshold)",
        categorical=False,
    )
    helper.plot_embedding(
        adata,
        "X_cnv_umap",
        "cnv_malignancy_call",
        FIG_DIR / f"{PREFIX}_cnv_umap_cnv_call",
        "CNV UMAP: inferred epithelial status (no dynamic threshold)",
        categorical=True,
    )
    helper.plot_embedding(
        adata,
        "X_cnv_umap",
        "cnv_burden",
        FIG_DIR / f"{PREFIX}_cnv_umap_cnv_burden",
        "CNV UMAP: CNV burden (no dynamic threshold)",
        categorical=False,
    )
    helper.plot_burden_violin(adata)
    helper.plot_chromosome_summary(adata)

    output_h5ad = OUT_DIR / f"{PREFIX}_analysis.h5ad"
    log(f"Writing final sensitivity CNV object: {output_h5ad}")
    adata.write_h5ad(output_h5ad)
    helper.write_report(adata, thresholds)
    log("Done")


if __name__ == "__main__":
    main()
