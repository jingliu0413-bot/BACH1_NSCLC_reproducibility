# BACH1 regulatory-context analysis in NSCLC

This repository contains the reproducible analysis code, environment files, documentation and lightweight source tables for the NSCLC BACH1 manuscript. The current code release matches the spatially extended manuscript submission package prepared on 2026-10-05.

## Analysis Scope

- GSE131907 scRNA-seq: lung tumour and adjacent lung cells.
- GSE274934 scRNA-seq: nine tumour GEX samples.
- GSE274934 scATAC-seq: five author-filtered tumour samples.
- GSE292299 spatial transcriptomics: four NSCLC tissue sections used for independent spatial replication.
- TCGA-LUAD and TCGA-LUSC: bulk primary tumour score-association replication.

The primary single-cell analysis uses a 13,694-cell malignant epithelial set from 16 patients. The pySCENIC sensitivity analysis uses a 4,906-cell restricted CNV-consensus set and summarizes BACH1 target recurrence across seeds 777-781. scATAC-seq is used as tumour-tissue proximal accessible-motif context, not as proof of direct BACH1 binding.

## Repository Contents

- `scripts/`: download, preprocessing, analysis, robustness, plotting and supplementary-table preparation scripts.
- `environment/`: pinned analysis and pySCENIC environments.
- `source_data/`: lightweight source tables; `source_data/manuscript_submission_20260911/` contains the manuscript source tables and spatial state-de-overlap audit tables.
- `resources/`: instructions and checksums for external resources; large third-party databases are downloaded locally and not committed.
- `docs/`: workflow and data/resource manifests.

Raw data, intermediate AnnData objects, virtual environments, complete pySCENIC databases, large reference resources, local manuscript builds and reviewer-comment working files are intentionally excluded from Git.

## Setup

Python 3.11.5 was used for the main analysis environment. pySCENIC was run in a separate environment.

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

Scripts default to the repository root, with `data/`, `out/` and `resources/` underneath it. Existing data and outputs can be stored elsewhere by defining:

```bash
export NSCLC_PROJECT_ROOT=/path/to/BACH1_NSCLC_project
export NSCLC_DATA_DIR=/path/to/BACH1_NSCLC_project/data
export NSCLC_OUTPUT_DIR=/path/to/BACH1_NSCLC_project/out
export NSCLC_RESOURCES_DIR=/path/to/BACH1_NSCLC_project/resources
```

## Current Workflow

Run commands from the repository root. Computationally intensive stages are not wrapped in an unconditional `run_all` command.

```bash
# Public inputs and reference databases
python scripts/download_geo_data.py
python scripts/prepare_gse274934_gex.py
python scripts/download_reference_resources.py

# Integrated scRNA-seq atlas and epithelial/CNV analysis
python scripts/integrate_nsclc_scrna.py
python scripts/run_scanpy_standard_qc.py
python scripts/run_scanpy_downstream.py
python scripts/run_epithelial_cnv_infercnvpy.py
python scripts/run_epithelial_cnv_infercnvpy_sensitivity_no_dynamic_threshold.py
python scripts/run_epithelial_cnv_adjacent_epithelial_reference.py
python scripts/combine_epithelial_cnv_calls.py
python scripts/run_epithelial_reclustering.py

# Primary malignant epithelial analysis and external BACH1 activity robustness
python scripts/run_bach1_malignant_epithelial_analysis.py --malignant-set primary
python scripts/run_external_bach1_activity_analysis.py
python scripts/run_bach1_hypoxia_robustness.py
python scripts/run_final_inference_sensitivity.py
python scripts/run_tcga_bach1_hypoxia_validation.py
```

Activate the isolated pySCENIC environment for the restricted-consensus full-TF runs:

```bash
python scripts/run_bach1_malignant_epithelial_analysis.py --malignant-set consensus
python scripts/prepare_pyscenic_input.py \
  --input-h5ad out/bach1_malignant_epithelial_consensus/nsclc_malignant_epithelial_consensus_bach1_analysis_object.h5ad \
  --output-dir out/bach1_malignant_epithelial_consensus_pyscenic

python scripts/run_pyscenic_all_tfs.py --seed 777 --input-dir out/bach1_malignant_epithelial_consensus_pyscenic --output-dir out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs --resume
python scripts/run_pyscenic_all_tfs.py --seed 778 --input-dir out/bach1_malignant_epithelial_consensus_pyscenic --output-dir out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed778 --resume
python scripts/run_pyscenic_all_tfs.py --seed 779 --input-dir out/bach1_malignant_epithelial_consensus_pyscenic --output-dir out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed779 --resume
python scripts/run_pyscenic_all_tfs.py --seed 780 --input-dir out/bach1_malignant_epithelial_consensus_pyscenic --output-dir out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed780 --resume
python scripts/run_pyscenic_all_tfs.py --seed 781 --input-dir out/bach1_malignant_epithelial_consensus_pyscenic --output-dir out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed781 --resume
```

Return to the main analysis environment:

```bash
python scripts/postprocess_pyscenic_all_tfs.py \
  --input-h5ad out/bach1_malignant_epithelial_consensus/nsclc_malignant_epithelial_consensus_bach1_analysis_object.h5ad \
  --pyscenic-dir out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs

python scripts/compare_pyscenic_seed_stability.py \
  --output-dir out/revision_diagnostics/pyscenic_bach1_consensus_full_tf_seed_stability_v1 \
  --run seed777 out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs \
  --run seed778 out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed778 \
  --run seed779 out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed779 \
  --run seed780 out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed780 \
  --run seed781 out/bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed781

python scripts/run_bach1_atac_motif_support_ucsc_targeted.py --query-mode bed
python scripts/plot_supplementary_figure5_scatac_target_window_context.py
python scripts/run_spatial_external_score_feasibility.py
python scripts/run_spatial_state_context_analysis.py
python scripts/run_spatial_state_deoverlap_sensitivity.py
python scripts/run_spatial_state_pc_block_null.py
python scripts/plot_spatial_extension_draft.py
python scripts/replot_figure4_pyscenic_from_source_data.py
python scripts/plot_bach1_story_4figures_10kb.py
python scripts/plot_cnv_supplement_figure_s1.py
python scripts/plot_spatial_skill_supplement_figure3.py
python scripts/prepare_supplementary_tables.py
```

Optional post-submission ICI feasibility work is kept separate from the
manuscript analysis. The public GSE243013 metadata audit can be run with:

```bash
python scripts/run_gse243013_state_mpr_feasibility.py
```

OAK/POPLAR interaction analysis is gated on approved EGA-derived expression
and clinical files. The access request and frozen model plan are documented in
`docs/OAK_POPLAR_EGA_ACCESS_REQUEST.md` and
`docs/OAK_POPLAR_ICI_INTERACTION_ANALYSIS_PLAN.md`.

Detailed inputs, outputs and fixed analysis definitions are listed in `docs/WORKFLOW.md`.

## Current Manuscript Positioning

- The central result is a patient-level DoRothEA-derived BACH1 score associated with a hypoxia-related transcriptional context in primary NSCLC malignant epithelium.
- The de-overlapped DoRothEA-hypoxia matched-null analysis uses an independent 81-gene null distribution after excluding ALDOA, HMOX1 and IL6.
- TCGA-LUAD/LUSC provides bulk tumour replication of the score association.
- pySCENIC provides seed-sensitive recurrent candidate prioritisation, with five-seed recurrence summarized independently from the seed-777 visualization.
- scATAC-seq provides tumour-tissue accessible-motif support for candidate prioritisation.
- Independent spatial transcriptomics reproduces positive BACH1-score/hypoxia covariance across four tissue sections, with attenuation after epithelial-state adjustment.
- Spatial block-null and state-de-overlap analyses are retained as reproducibility checks for the spatial extension; the earlier exploratory NOD analysis is not part of the submission package.

## License

Code is released under the MIT License. Public datasets and external reference databases remain subject to their original providers' terms.
