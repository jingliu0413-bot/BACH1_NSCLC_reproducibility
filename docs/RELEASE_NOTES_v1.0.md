# Release notes: v1.0-hypoxia-states (manuscript version of record)

Branch `manuscript-v1-hypoxia-states` was created from `main` commit ad68022 to freeze the code and source data for the manuscript in `submission/`.

## Corrected source tables
- `matched_random_gene_set_hypoxia_correlation_summary.csv`, `matched_random_gene_set_hypoxia_correlations.csv` and `matched_random_gene_set_first50_matches.csv` were replaced with the output of the current `scripts/run_bach1_hypoxia_robustness.py` (run 2026-09-11). The previously committed tables came from an earlier version of the script that reused one 84-gene null for both comparisons (null mean 0.476, de-overlapped P = 0.003). The corrected tables use independent nulls (seed 1701, 84 genes; seed 1702, 81 genes); the de-overlapped null mean is 0.504 and the one-sided empirical P is 0.002, as reported in the manuscript and Supplementary Table 2.
- Added `matched_random_gene_set_hypoxia_correlations_long.csv` and `matched_random_gene_set_input_gene_counts.csv` from the same run.

## Added source tables
- `pyscenic_consensus_full_tf_seed_stability/`: five-seed (777-781) BACH1 target stability in the 4,906-cell restricted set (331 union targets; 46 in >=3 seeds; 14 in >=4 seeds; mean pairwise Jaccard 0.143).
- `gse274934_atac_sample_qc_summary.csv` (28,500 cells across AL05, AL08, AL10, AL11, AL12) and `gse274934_atac_target_window_summary.json` (640,447 sample-specific peaks).
- `resources/bach1_external_signatures/dorothea_human_ABC_decoupler.csv` and `collectri_human_decoupler.csv`: the regulon networks used for scoring.
- `submission/`: submitted manuscript, figures and supplementary workbook; `manuscript.docx` at the repository root was replaced with the submitted version.

## Code changes
- `scripts/compare_pyscenic_seed_stability.py`: default runs now list all five seeds (777-781), matching the analysis; previously the five-seed run required explicit `--run` arguments.

## Notes for readers (implementation details not spelled out in the Methods)
- TCGA: mean-z scores are standardised within LUAD and LUSC before pooling; the combined Spearman correlation therefore uses cohort-centred scores. Holm correction is applied within each cohort across the four score comparisons (DoRothEA original and de-overlapped, CollecTRI, Klenja-Skudrinja). The cohort-interaction model is fitted in `run_final_inference_sensitivity.py`.
- scRNA cohort split (Fig. 4b): P values are scipy asymptotic Spearman P values; for GSE274934 (n = 5, rho = 0.5) the exact permutation P is 0.45.
- CNV calling is made per CNV cluster (>=20 epithelial cells, median burden above threshold; adjacent-epithelium reference strategy also requires <=10% reference cells).
- Per-patient QC models: epithelial states with <10 cells are pooled into "Other" and the state term is dropped when residual df < 5; covariates are z-scored within patient.
- Within-patient BACH1-detected versus undetected comparisons require >=1 detected and >=1 undetected cell (15 patients); QC models use >=20 cells with >=3 detected and >=3 undetected (14 patients).
- The Holm values reported for the five external BACH1 scores come from `run_bach1_hypoxia_robustness.py` (`figure2c_within_patient_score_comparison_holm_family.csv`); the broader Holm family written by `run_external_bach1_activity_analysis.py` is not used in the manuscript.
