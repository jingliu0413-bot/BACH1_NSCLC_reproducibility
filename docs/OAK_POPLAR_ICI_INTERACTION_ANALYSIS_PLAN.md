# OAK/POPLAR ICI interaction analysis plan

This plan is frozen before access to clinical outcomes is used for score
selection.

## Score

Use the existing DoRothEA A-C BACH1 target list and weights. Remove the three
genes shared with Hallmark hypoxia (`ALDOA`, `HMOX1`, `IL6`) and retain the
resulting 81-gene vector without outcome-based filtering. Within each trial,
compute gene-wise z scores across the provided log2(TPM + 1) samples and take
the signed weighted mean normalized by the sum of absolute weights. The final
score is then standardized within trial to one standard deviation before it
enters any model, so interaction effects are interpreted per within-trial 1-SD
higher score. Record target coverage and missing genes before fitting models.

Treatment coding is fixed in advance: chemotherapy is the reference arm
(`treatment=0`) and atezolizumab is the exposed arm (`treatment=1`).

## Primary model

Fit one pooled model containing trial, histology and the treatment-by-score
interaction. The primary estimand is the interaction term, not separate
within-arm significance tests:

`endpoint ~ score + treatment + score:treatment + trial + histology`

Primary endpoints are PFS and OS. Response is secondary and uses a logistic
interaction model with the same score and treatment terms. Any additional
biomarker adjustment using PD-L1, TMB, STK11, KEAP1 or EGFR is sensitivity
analysis only and is limited to samples with the relevant measurements.

Primary Cox models use complete cases for the endpoint and prespecified
covariates; no outcome-informed imputation will be performed. Proportional
hazards will be assessed using Schoenfeld-residual diagnostics. If the
score-by-treatment interaction shows clear evidence of non-proportionality,
a time-varying interaction model will be reported as a sensitivity analysis,
not as a replacement for the primary model.

No cutoff optimization, outcome-driven gene selection, or arm-specific score
threshold will be performed. OAK and POPLAR will also be reported separately
before the pooled estimate.
