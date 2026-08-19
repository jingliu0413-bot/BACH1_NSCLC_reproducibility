# Reproducible workflow

## Ordered analysis stages

| Order | Script | Principal input | Principal output |
|---:|---|---|---|
| 0.1 | `download_geo_data.py` | GEO accessions | Selected GSE131907 and GSE274934 inputs |
| 0.2 | `prepare_gse274934_gex.py` | Nine filtered 10x HDF5 matrices | Merged GSE274934 AnnData |
| 0.3 | `download_reference_resources.py` | Provider URLs and embedded checksums | cisTarget, JASPAR and GENCODE resources |
| 1 | `integrate_nsclc_scrna.py` | GSE131907 and GSE274934 counts | Raw integrated atlas and initial PCA object |
| 2 | `run_scanpy_standard_qc.py` | Integrated raw-count AnnData | QC-filtered raw-count AnnData and QC figures |
| 3 | `run_scanpy_downstream.py` | QC-filtered counts | Scrublet-filtered atlas, Harmony, UMAP, Leiden clusters and marker tables |
| 4.1 | `run_epithelial_cnv_infercnvpy.py` | Integrated atlas | Strict T/NK-referenced CNV result |
| 4.2 | `run_epithelial_cnv_infercnvpy_sensitivity_no_dynamic_threshold.py` | CNV input subset | Sensitive T/NK-referenced CNV result |
| 4.3 | `run_epithelial_cnv_adjacent_epithelial_reference.py` | CNV input subset | Adjacent-epithelial-referenced CNV result |
| 4.4 | `combine_epithelial_cnv_calls.py` | Three CNV call tables | Working and consensus malignant-cell sets |
| 5 | `run_epithelial_reclustering.py` | Integrated atlas and CNV calls | Epithelial reclustering object and marker tables |
| 6 | `run_bach1_malignant_epithelial_analysis.py` | Working malignant epithelial cells | BACH1 expression, correlation and candidate tables |
| 7.1 | `prepare_pyscenic_input.py` | Malignant epithelial count layer | Filtered pySCENIC matrix |
| 7.2 | `run_pyscenic_bach1.py` | Expression matrix and cisTarget resources | GRNBoost2, cisTarget and AUCell outputs |
| 7.3 | `postprocess_bach1_pyscenic.py` | pySCENIC outputs | Integrated regulon tables and figures |
| 8 | `run_bach1_atac_motif_support.py` | Five filtered ATAC matrices, JASPAR and GENCODE | Motif-positive peaks and TSS-linked genes |
| 9.1 | `run_bach1_intersection_enrichment.py` | pySCENIC and TSS +/-10-kb ATAC sets | Dual-evidence genes and GO/KEGG enrichment |
| 9.2 | `run_bach1_atac_proximal_go_kegg_enrichment.py` | Broader TSS +/-10-kb ATAC set | Contextual GO/KEGG enrichment |
| 10 | `plot_bach1_nod_like_pathway.py` | NOD-like pathway enrichment and peak links | Five-locus peak and motif-logo display |
| 11.1 | `download_emtab13530.py` | E-MTAB-13530 | Forty Visium input sections |
| 11.2 | `analyze_emtab13530_bach1_nod.py` | Visium matrices and SDRF metadata | BACH1/NOD scores and paired statistics |
| 11.3 | `run_spatial_skill_supplement.py` | Scored Visium objects | Spatial domains, autocorrelation and signatures |
| 12 | `plot_bach1_story_4figures_10kb.py` | Principal analysis outputs | Figures 1-4 |
| 13 | `plot_figure5_formal_spatial_bach1_nod_validation.py` | Spatial outputs | Figure 5 |
| 14 | `plot_cnv_supplement_figure_s1.py` | CNV outputs | Three-panel Figure S1 |
| 15 | `plot_figure3_revised_bach1_multislices.py` | Spatial supplement outputs | Multi-section supplementary spatial figure |

## Fixed analysis definitions

- Scanpy QC: `n_genes_by_counts >= 200`, `n_genes_by_counts < 5000`, `total_counts >= 500`, `pct_counts_mt < 15` and `pct_counts_hb < 1`.
- Ribosomal percentage: displayed and summarised, not filtered.
- Broad atlas: 3,000 highly variable genes, 40 PCs, 15 neighbours, Leiden resolutions 0.3 and 0.8.
- Epithelial reclustering: epithelial-specific HVGs, 40 PCs, 15 neighbours and principal Leiden resolution 0.6.
- CNV reference: 5% T/NK cells sampled within each sample; infercnvpy window size 100 genes and step 10 genes.
- Working malignant set: no-dynamic-threshold T/NK-reference call.
- Higher-confidence CNV set: malignant according to at least two of three infercnvpy analyses.
- pySCENIC: BACH1-only GRNBoost2, seed 777, four workers, dropout masking, minimum module size 10 and AUCell seed 777.
- High-confidence chromatin motif: JASPAR TFBS score at least 950.
- Principal peak-gene link: accessible BACH1 or Bach1::Mafk motif peak within TSS +/-10 kb.
- Enrichment background: genes represented in the malignant-epithelial pySCENIC matrix.
- Spatial high states: sample-specific 75th-percentile thresholds, with BACH1 additionally required to have positive expression.
- Spatial neighbourhood: six nearest Visium spots.
- Paired spatial statistics: patient-level two-sided Wilcoxon signed-rank test across eight tumour-adjacent pairs.

## Interpretation boundaries

The 65-gene overlap is a dual-evidence candidate set. Motif presence does not demonstrate occupancy, accessibility does not establish regulation, pySCENIC does not demonstrate physical binding and spatial co-localisation does not establish causality.
