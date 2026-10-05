# OAK/POPLAR EGA access request

The OAK/POPLAR feasibility analysis requires the expression and clinical
datasets governed by the same Genentech DAC. Do not combine EGA expression
with Vivli clinical data from the same trials.

## Requested datasets

- `EGAD00001008390`: POPLAR log2(TPM + 1) expression matrix
- `EGAD00001008391`: OAK log2(TPM + 1) expression matrix
- `EGAD00001008548`: POPLAR clinical data
- `EGAD00001008549`: OAK clinical data
- Optional: `EGAD00001008550`: OAK PD-L1, TMB, STK11, KEAP1 and EGFR biomarkers

DAC: `EGAC00001002120` (Genentech). The request should cover both expression
and clinical data under one approved project and should state that the analysis
will use gene-expression data, clinical outcomes, and prespecified biomarker
variables where available.

## Proposed request text

> To evaluate whether a predefined, externally derived DoRothEA BACH1 transcriptional score modifies clinical benefit from atezolizumab versus chemotherapy in randomized OAK and POPLAR NSCLC cohorts. The score definition will be frozen before outcome analysis and will not be optimized using clinical outcomes. Primary analyses will test treatment-by-score interaction for PFS and OS, with response as a secondary endpoint.

The score is the de-overlapped 81-target DoRothEA BACH1 score already used in
the NSCLC manuscript, with ALDOA, HMOX1 and IL6 excluded. Treatment coding is
fixed as chemotherapy=0 (reference) and atezolizumab=1. Within each trial,
the score will be standardized to one standard deviation before model fitting.
The planned analysis is exploratory.
