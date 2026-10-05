# Final ICI clinical feasibility audit

## GSE243013 metadata audit

The public GEO metadata were downloaded and audited locally. The series contains
1,254,749 immune cells from 243 patients:

- 233 patients had documented anti-PD-1 exposure;
- 212 met the strict anti-PD-1 plus chemotherapy definition;
- among the strict cohort, 113 were `MPR_or_pCR`, 98 were `non-MPR`, and 1 had an unknown response;
- the major compartments are B cell, myeloid cell and T/NK cell;
- no malignant epithelial compartment is present in the metadata.

The frozen 81-gene de-overlapped DoRothEA BACH1 score has complete gene-index
coverage in GSE243013. The expression matrix is still required before any
cell- or patient-level score/MPR analysis can be run. Until then, this dataset
supports an immune-state/MPR feasibility analysis only; it is not a direct
replication of the malignant-epithelial result in the manuscript.

Outputs are in `out/clinical_ici_feasibility/GSE243013/`.

## OAK/POPLAR access status

The OAK/POPLAR expression and clinical data are controlled-access EGA data.
The analysis therefore has not been run locally and no outcome-driven result
has been added to the manuscript. The request draft and frozen model plan are:

- `docs/OAK_POPLAR_EGA_ACCESS_REQUEST.md`
- `docs/OAK_POPLAR_ICI_INTERACTION_ANALYSIS_PLAN.md`

Once the EGA files are approved and downloaded, run
`scripts/run_oak_poplar_ici_interaction.py` with explicit column mappings.
The primary estimand is the score-by-atezolizumab interaction for PFS and OS,
with response as a secondary endpoint. Treatment coding is fixed as
chemotherapy=0 and atezolizumab=1, and the score enters models after
within-trial 1-SD standardization. Primary Cox fits use complete cases without
outcome-informed imputation, report Schoenfeld-residual PH diagnostics, and
only add a time-varying interaction as a sensitivity analysis if the PH check
indicates clear non-proportionality.
