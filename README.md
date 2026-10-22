# BACH1 regulatory-context analysis in NSCLC

This branch (`manuscript-v1-hypoxia-states`, release tag `v1.0-hypoxia-states`) is the code and source-data release for the manuscript

> Liu J, Tang H, Golpon H, Dzik R, Janciauskiene S. *Patient-resolved single-cell analysis links a BACH1-associated transcriptional score to hypoxia-related epithelial states in non-small cell lung cancer.* (submitted, October 2026)

The submitted manuscript, figures and supplementary workbook are in `submission/`. Later analyses on `main` (spatial transcriptomics extension, immunotherapy feasibility work) are not part of this manuscript; see "Scope of this release" below.

## Analysis Scope

- GSE131907 scRNA-seq: lung tumour and adjacent lung cells.
- GSE274934 scRNA-seq: nine tumour GEX samples.
- GSE274934 scATAC-seq: five author-filtered tumour samples (28,500 cells; 640,447 sample-specific peaks).
- TCGA-LUAD and TCGA-LUSC: bulk primary tumour score-association replication.

The primary single-cell analysis uses a 13,694-cell malignant epithelial set from 16 patients. The pySCENIC analysis uses a 4,906-cell restricted CNV-consensus set (a subset of the primary set) and summarizes BACH1 target recurrence across seeds 777-781. scATAC-seq is used as tumour-tissue proximal accessible-motif context, not as proof of direct BACH1 binding.

## Repository Contents

- `scripts/`: download, preprocessing, analysis, robustness, plotting and supplementary-table preparation scripts.
- `environment/`: pinned analysis and pySCENIC environments.
- `source_data/`: lightweight source tables; `source_data/manuscript_submission_20260911/` contains the manuscript source tables (spatial audit tables in the same folder belong to the later extension).
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
python scripts/run_state_specificity_clinical_revision.py   # epithelial states, within-state tests, pan-TF benchmark, cohort split
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
python scripts/replot_figure4_pyscenic_from_source_data.py
python scripts/plot_bach1_story_4figures_10kb.py
python scripts/plot_cnv_supplement_figure_s1.py
python scripts/plot_pan_tf_tcga_figure.py                 # Figure 4
python scripts/prepare_supplementary_tables.py
```

Clinical-feasibility explorations are intentionally kept outside this public
manuscript release. They are not required to reproduce the current Figures
1–5 or Supplementary Figures 1–5 and are not part of the submission package.

Detailed inputs, outputs and fixed analysis definitions are listed in `docs/WORKFLOW.md`.

## Figure and table map

| Manuscript item | Script(s) | Source table(s) |
|---|---|---|
| Fig. 1 atlas, malignant epithelium, BACH1 detection | integrate_nsclc_scrna.py, run_scanpy_*, run_epithelial_*, combine_epithelial_cnv_calls.py, run_bach1_malignant_epithelial_analysis.py, rebuild_final_figure_architecture.py | source_data/figure1_* |
| Fig. 2 BACH1 scores, DoRothEA-hypoxia, LOPO, matched null | run_external_bach1_activity_analysis.py, run_bach1_hypoxia_robustness.py, plot_streamlined_figure2.py | figure2c_within_patient_score_comparison_holm_family.csv, dorothea_hypoxia_deoverlap_*, matched_random_gene_set_* |
| Fig. 3 epithelial states, attenuation, within-state tests | run_final_inference_sensitivity.py, run_state_specificity_clinical_revision.py | patient_level_residualized_*, per_patient_qc_adjusted_*, Supplementary Table 4E-4M |
| Fig. 4 pan-TF benchmark, cohort split, TCGA | run_state_specificity_clinical_revision.py, run_tcga_bach1_hypoxia_validation.py, plot_pan_tf_tcga_figure.py | dorothea_all_tf_hypoxia_contextual_benchmark.csv, scrna_cohort_split_dorothea_hypoxia_correlations.csv, tcga_* |
| Fig. 5, Suppl. Fig. 2 pySCENIC and scATAC candidate context | run_pyscenic_all_tfs.py, compare_pyscenic_seed_stability.py, run_bach1_atac_motif_support_ucsc_targeted.py, replot_figure4_pyscenic_from_source_data.py, plot_supplementary_figure5_scatac_target_window_context.py | pyscenic_consensus_full_tf_seed_stability/, gse274934_atac_*, figure4*_current_* |
| Suppl. Fig. 1 CNV selection | plot_cnv_supplement_figure_s1.py | source_data/figure1_*cnv* |
| Supplementary Table workbook | prepare_supplementary_tables.py | submission/Supplementary_Table.xlsx |

Source tables for this manuscript are in `source_data/manuscript_submission_20260911/` (the folder name reflects the date the table layout was frozen). Changes made when this release was frozen are listed in `docs/RELEASE_NOTES_v1.0.md`.

## Scope of this release

Scripts present in the repository but not used for this manuscript: `*spatial*`, `prepare_spatial_rctd_inputs.py`, `run_spatial_rctd_spacexr.R`, `score_spatial_stable_bach1_regulon.py`, `summarize_spatial_rctd_results.py`, `analyze_emtab13530_bach1_nod.py`, `download_emtab13530.py`, `plot_bach1_nod_like_pathway.py`, `plot_figure5_formal_spatial_bach1_nod_validation.py`, `run_oak_poplar_ici_interaction.py`, `run_gse243013_state_mpr_feasibility.py`, `update_manuscript_spatial_methods.py`, `update_supplementary_*`, and the GSE292299 / spatial / E-MTAB-13530 tables in `source_data/`. They belong to later or exploratory analyses.

## Resources

DoRothEA (confidence A-C) and CollecTRI networks used for scoring were retrieved through decoupler/OmniPath on 2026-09-10 and are included in `resources/bach1_external_signatures/` so that scores can be reproduced without a live download. Large reference databases (cisTarget rankings, GENCODE annotation, JASPAR bigBed) are not redistributed; see `resources/README.md` and `scripts/download_reference_resources.py`.

## License

Code is released under the MIT License. Public datasets and external reference databases remain subject to their original providers' terms.
