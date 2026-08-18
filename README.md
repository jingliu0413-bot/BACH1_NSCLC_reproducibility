# BACH1 regulatory analysis in NSCLC

This repository contains the analysis code and figure source data for the integrated scRNA-seq, expression-derived CNV, BACH1-focused pySCENIC, scATAC-seq motif, pathway-enrichment and spatial-transcriptomic analyses in the accompanying NSCLC study.

## Analysis scope

- GSE131907 scRNA-seq: tumour lung and adjacent lung cells only.
- GSE274934 scRNA-seq: nine tumour samples.
- GSE274934 scATAC-seq: five author-filtered samples.
- E-MTAB-13530 spatial transcriptomics: 40 Visium sections, including eight tumour-adjacent patient pairs.
- TCR sequencing and non-lung GSE131907 tissues are excluded.

The principal chromatin definition is a high-confidence BACH1 or Bach1::Mafk motif-bearing accessible peak within TSS +/-10 kb. The pySCENIC/scATAC overlap contains dual-evidence candidates; it is not presented as proof of direct BACH1 binding. Spatial co-localisation is likewise interpreted as association rather than causality.

## Repository contents

- `scripts/`: analysis, download and plotting scripts.
- `environment/`: pinned analysis and pySCENIC environments.
- `source_data/`: lightweight source tables for the reported figures.
- `resources/`: checksums and instructions for external databases.
- `docs/`: workflow, data manifest and transfer notes.

Raw data, intermediate AnnData objects, virtual environments and large third-party databases are intentionally excluded.

## Setup

Python 3.11.5 was used. The main analysis and pySCENIC were run in separate environments.

```bash
conda create -n bach1-nsclc-analysis python=3.11.5 pip -y
conda activate bach1-nsclc-analysis
pip install -r environment/requirements-analysis.txt
```

```bash
conda create -n bach1-pyscenic python=3.11.5 pip -y
conda activate bach1-pyscenic
pip install -r environment/requirements-pyscenic.txt
```

Scripts default to treating the repository as the project root, with `data/`, `out/` and `resources/` beneath it. Existing data and outputs can be stored elsewhere by defining:

```powershell
$env:NSCLC_PROJECT_ROOT = "D:\BACH1_NSCLC_project"
$env:NSCLC_DATA_DIR = "D:\BACH1_NSCLC_project\data"
$env:NSCLC_OUTPUT_DIR = "D:\BACH1_NSCLC_project\out"
$env:NSCLC_RESOURCES_DIR = "D:\BACH1_NSCLC_project\resources"
```

On Linux or macOS, use `export` with the same variable names.

## Workflow

Run the following commands from the repository root. Computationally intensive stages are deliberately not wrapped in an unconditional `run_all` command.

```bash
# Public inputs and reference databases
python scripts/download_geo_data.py
python scripts/prepare_gse274934_gex.py
python scripts/download_reference_resources.py

# Integrated scRNA-seq atlas and Scanpy QC
python scripts/integrate_nsclc_scrna.py
python scripts/run_scanpy_standard_qc.py
python scripts/run_scanpy_downstream.py

# CNV support and epithelial reclustering
python scripts/run_epithelial_cnv_infercnvpy.py
python scripts/run_epithelial_cnv_infercnvpy_sensitivity_no_dynamic_threshold.py
python scripts/run_epithelial_cnv_adjacent_epithelial_reference.py
python scripts/combine_epithelial_cnv_calls.py
python scripts/run_epithelial_reclustering.py

# BACH1 expression and co-expression
python scripts/run_bach1_malignant_epithelial_analysis.py
python scripts/prepare_pyscenic_input.py
```

Activate the isolated pySCENIC environment for the next command:

```bash
python scripts/run_pyscenic_bach1.py --workers 4
```

Return to the main analysis environment:

```bash
python scripts/postprocess_bach1_pyscenic.py
python scripts/run_bach1_atac_motif_support.py
python scripts/run_bach1_intersection_enrichment.py
python scripts/run_bach1_atac_proximal_go_kegg_enrichment.py
python scripts/plot_bach1_nod_like_pathway.py

# Spatial validation
python scripts/download_emtab13530.py
python scripts/analyze_emtab13530_bach1_nod.py
python scripts/run_spatial_skill_supplement.py

# Final figures
python scripts/plot_bach1_story_4figures_10kb.py
python scripts/plot_cnv_supplement_figure_s1.py
python scripts/plot_figure5_formal_spatial_bach1_nod_validation.py
python scripts/plot_figure3_revised_bach1_multislices.py
```

Detailed inputs, outputs and fixed thresholds are listed in `docs/WORKFLOW.md`.

## Reproducibility notes

- QC uses `n_genes_by_counts >= 200` and `< 5000`, `total_counts >= 500`, mitochondrial fraction `< 15%` and haemoglobin fraction `< 1%`. Ribosomal fraction is summarised but not used as a hard filter.
- Random seed 7 is used for the principal scRNA-seq and CNV workflows; pySCENIC uses seed 777.
- BACH1 pySCENIC is a targeted one-regulator analysis, not an all-transcription-factor network reconstruction.
- Enrichr libraries are retrieved through GSEApy. The reported source tables are included because online libraries can be updated after publication.
- Full reanalysis requires substantial memory and storage; the original project occupied approximately 51 GB before packaging.

## License

Code is released under the MIT License. Public data and external reference databases remain subject to their original providers' terms.
