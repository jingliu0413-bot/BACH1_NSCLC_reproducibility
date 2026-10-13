# Reproducible Workflow

This document describes the current manuscript submission analysis workflow. Large raw data, intermediate AnnData objects and external databases are not tracked in Git; scripts write those products under `data/`, `out/` or a user-defined output directory.

## Ordered Analysis Stages

| Order | Script | Principal input | Principal output |
|---:|---|---|---|
| 0.1 | `download_geo_data.py` | GEO accessions | Selected GSE131907 and GSE274934 inputs |
| 0.2 | `prepare_gse274934_gex.py` | Nine filtered 10x HDF5 matrices | Merged GSE274934 AnnData |
| 0.3 | `download_reference_resources.py` | Provider URLs and embedded checksums | pySCENIC, JASPAR and GENCODE resources |
| 1 | `integrate_nsclc_scrna.py` | GSE131907 and GSE274934 counts | Integrated atlas and initial PCA object |
| 2 | `run_scanpy_standard_qc.py` | Integrated raw-count AnnData | QC-filtered raw-count AnnData and QC figures |
| 3 | `run_scanpy_downstream.py` | QC-filtered counts | Scrublet-filtered atlas, Harmony, UMAP, Leiden clusters and marker tables |
| 4.1 | `run_epithelial_cnv_infercnvpy.py` | Integrated atlas | Strict T/NK-referenced CNV result |
| 4.2 | `run_epithelial_cnv_infercnvpy_sensitivity_no_dynamic_threshold.py` | CNV input subset | Sensitive T/NK-referenced CNV result |
| 4.3 | `run_epithelial_cnv_adjacent_epithelial_reference.py` | CNV input subset | Adjacent-epithelial-referenced CNV result |
| 4.4 | `combine_epithelial_cnv_calls.py` | Three CNV call tables | Primary, working and restricted CNV-consensus malignant-cell definitions |
| 5 | `run_epithelial_reclustering.py` | Integrated atlas and CNV calls | Epithelial reclustering object and marker tables |
| 6.1 | `run_bach1_malignant_epithelial_analysis.py --malignant-set primary` | Epithelial reclustering object | 13,694-cell primary malignant epithelial BACH1 object |
| 6.2 | `run_external_bach1_activity_analysis.py` | Primary malignant epithelial object | External BACH1 activity scores and patient-level summaries |
| 6.3 | `run_bach1_hypoxia_robustness.py` | Primary activity outputs | DoRothEA-hypoxia overlap, 84-gene and 81-gene matched-null tests |
| 6.4 | `run_final_inference_sensitivity.py` | Primary activity outputs | QC-adjusted and residualized patient-level sensitivity analyses |
| 6.5 | `run_tcga_bach1_hypoxia_validation.py` | UCSC Xena TCGA-LUAD/LUSC resources | TCGA score-association replication and covariate-adjusted models |
| 7.1 | `run_bach1_malignant_epithelial_analysis.py --malignant-set consensus` | Epithelial reclustering object | 4,906-cell restricted CNV-consensus BACH1 object |
| 7.2 | `prepare_pyscenic_input.py` | Restricted consensus BACH1 object | Filtered pySCENIC matrix and expressed TF list |
| 7.3 | `run_pyscenic_all_tfs.py` | Expression matrix, TF list and cisTarget resources | Full-TF pySCENIC seed-specific outputs |
| 7.4 | `postprocess_pyscenic_all_tfs.py` | Seed-777 full-TF pySCENIC output | Seed-777 AUCell and motif-pruning summaries for visualization |
| 7.5 | `compare_pyscenic_seed_stability.py` | Seeds 777-781 full-TF pySCENIC outputs | Five-seed BACH1 target recurrence, Jaccard and weighted-importance tables |
| 8.1 | `run_bach1_atac_motif_support_ucsc_targeted.py` | Five ATAC matrices, GENCODE and UCSC JASPAR2026 bigBed | Target-window BACH1-family motif support and score-threshold summaries |
| 8.2 | `plot_supplementary_figure5_scatac_target_window_context.py` | ATAC target-window outputs | Supplementary scATAC figure and source tables |
| 9.1 | `run_spatial_state_context_analysis.py` | GSE292299 expression and coordinate inputs | Projected spatial scores, section-level correlations and state-PC residualization |
| 9.2 | `run_spatial_state_pc_block_null.py` | GSE292299 projected score table | Raw 20-by-20 block null and state-adjusted block-size sensitivity summaries |
| 9.3 | `run_spatial_state_deoverlap_sensitivity.py` | Locked GSE292299 score table and epithelial-state gene sets | State-score de-overlap and residual-correlation audit |
| 10.1 | `rebuild_final_figure_architecture.py` | Final analysis outputs | Main Figures 1-4 and Supplementary Figures 1-3 |
| 10.2 | `plot_spatial_extension_draft.py` | GSE292299 spatial summaries | Main Figure 5 |
| 10.3 | `update_supplementary_figures45_legends.py` | Locked manuscript legends and sensitivity outputs | Supplementary Figures 4-5 legend synchronization |
| 10.4 | `prepare_supplementary_tables.py` | Final result tables | Combined supplementary workbook source sheets |

## Fixed Analysis Definitions

- Scanpy QC: `n_genes_by_counts >= 200`, `n_genes_by_counts < 5000`, `total_counts >= 500`, mitochondrial fraction `< 15%` and haemoglobin fraction `< 1%`.
- Ribosomal percentage: displayed and summarised, not filtered.
- Broad atlas: 3,000 highly variable genes, 40 PCs, 15 neighbours, Leiden resolutions 0.3 and 0.8.
- Epithelial reclustering: epithelial-specific HVGs, 40 PCs, 15 neighbours and principal Leiden resolution 0.6.
- Primary malignant epithelial set: GSE131907 tumour epithelial cells from author tS1/tS2/tS3 annotations plus GSE274934 tumour epithelial cells supported by at least one expression-derived CNV strategy; final primary set, 13,694 cells from 16 patients.
- Restricted CNV-consensus set: malignant by at least two CNV strategies; final pySCENIC sensitivity set, 4,906 cells, including 784 BACH1-detected cells.
- BACH1 transcript-detection grouping: BACH1-detected cells have non-zero log-normalised BACH1 expression; BACH1-undetected cells have no detected transcript.
- External BACH1 score implementation: directional mean-z scoring for DoRothEA A-C, CollecTRI and lung BACH1 effector resources, with Klenja-Skudrinja depletion-induced genes inverted for BACH1-like activity.
- De-overlapped DoRothEA-hypoxia analysis: ALDOA, HMOX1 and IL6 are removed from both scores, leaving 81 DoRothEA genes and 191 Hallmark hypoxia genes.
- Matched random gene-set tests: original 84-gene DoRothEA null uses seed 1701; de-overlapped 81-gene null uses seed 1702; each permutation preserves tested gene count and uses the corresponding weight vector.
- TCGA replication: LUAD/LUSC primary tumour sample-type 01 matrices from UCSC Xena; models use complete-case covariates without imputation.
- pySCENIC visualization: seed 777 full-TF run in the restricted CNV-consensus set is used for AUCell UMAP/distribution and motif-pruning display.
- pySCENIC stability: seeds 777-781 are summarized independently; recurrence in at least 3/5 seeds defines the recurrent restricted-consensus candidate set.
- scATAC target-window context: tumour ATAC peaks are intersected with UCSC JASPAR2026 BACH1/Bach1::Mafk motif hits in candidate TSS +/-10-kb windows; UCSC field 5 score thresholds 300, 400 and 500 are reported as sensitivity checks.
- GSE292299 spatial projection: de-overlapped DoRothEA BACH1, Hallmark hypoxia and epithelial-state programme scores are projected to P5, P8, P12 and P15 using the frozen score definitions.
- Spatial state adjustment: DoRothEA-hypoxia residualization uses three PCs from four projected epithelial programmes; Stress_AP1-hypoxia uses leave-one-programme-out PCs to avoid self-adjustment.
- Spatial neighbourhood: six nearest array-coordinate neighbours with row-standardised edge weights.
- Spatial nulls: Figure 5d uses the raw DoRothEA-hypoxia pair with a 20-by-20 locally constrained block null; Supplementary Figure 5 uses the state-PC-residualized pair at block sizes 10, 15, 20 and 30.

## Current Figure Mapping

- Figure 1: cellular context and study design.
- Figure 2: external BACH1 activity estimates, patient-level DoRothEA-hypoxia association and matched-null robustness.
- Figure 3: malignant epithelial-state architecture and attenuation of patient-level BACH1-hypoxia covariance.
- Figure 4: pan-TF contextualisation and TCGA-LUAD/LUSC bulk score-association replication.
- Figure 5: independent GSE292299 spatial replication and state-PC attenuation.
- Supplementary Figure 1: CNV malignant epithelial selection.
- Supplementary Figure 2: tumour scATAC proximal accessible-motif context.
- Supplementary Figure 3: multi-omic prioritisation of BACH1-associated candidate regulatory features.
- Supplementary Figure 4: scRNA cohort-stratified BACH1-hypoxia covariance.
- Supplementary Figure 5: state-adjusted spatial block-size sensitivity.

## Interpretation Scope

The analysis identifies a patient-level DoRothEA-derived BACH1 score linked to a hypoxia-associated transcriptional context in NSCLC tumour epithelium. Much of this covariance is embedded within malignant epithelial-state architecture. TCGA provides independent bulk-cohort reproducibility, and GSE292299 provides independent section-level spatial replication of the same state-embedded pattern. pySCENIC and ATAC analyses prioritize candidates for follow-up; they do not by themselves establish BACH1 protein activity, direct occupancy, enhancer-promoter regulation or causality.
