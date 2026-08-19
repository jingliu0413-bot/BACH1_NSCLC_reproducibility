# Data and resource manifest

## Public datasets

| Accession | Modality | Included material | Local location |
|---|---|---|---|
| GSE131907 | scRNA-seq | Lung tumour (`tLung`) and adjacent lung (`nLung`) cells only; other organs excluded | `data/GSE131907/` |
| GSE274934 | scRNA-seq | Nine tumour GEX samples: AL02, AL03, AL04, AL05, AL06, AL08, AL10, AL11 and AL12 | `data/GSE274934/GEX_filtered_h5/` |
| GSE274934 | scATAC-seq | Five author-filtered samples: AL05, AL08, AL10, AL11 and AL12 | `data/GSE274934/ATAC/` |
| E-MTAB-13530 | Visium | Forty tumour, adjacent and healthy lung sections; tumour-adjacent inference uses eight paired patients | `data/spatial/E-MTAB-13530/raw/` |

TCR data and GSE274934 ATAC fragment files are not used. The GEO download script selectively extracts filtered GEX matrices, filtered ATAC peak matrices, ATAC single-cell QC tables and peak BED files from `GSE274934_RAW.tar`.

## Expected primary files

GSE131907 requires:

- `GSE131907_Lung_Cancer_cell_annotation.txt.gz`
- `GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz`
- `GSE131907_series_matrix.txt.gz`

GSE274934 requires nine `*_GEX_filtered_feature_bc_matrix.h5` files, five `*_ATAC_filtered_peak_bc_matrix.h5` files, five `*_ATAC_singlecell.csv.gz` files and `GSE274934_series_matrix.txt.gz`.

E-MTAB-13530 requires `E-MTAB-13530.sdrf.txt`, one filtered feature-barcode HDF5 file per section and one spatial TAR archive per section.

## Files excluded from Git

- Raw data and intermediate matrices.
- AnnData objects and virtual environments.
- cisTarget, JASPAR TFBS and GENCODE database files.
- Logs, manuscript drafts, reviewer comments and figure PDFs.

The public repository should contain code, environment files, checksums, documentation and lightweight source-data tables only.
