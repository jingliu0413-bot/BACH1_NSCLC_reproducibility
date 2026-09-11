# Source Data

This directory contains lightweight generated tables that support inspection of the reported analyses. Raw sequencing matrices, intermediate AnnData objects and external databases are intentionally excluded.

## Current Submission Tables

`if5_submission_20260911/` contains the current IF5-oriented submission source tables, including:

- external BACH1 signature and patient-level activity summaries;
- DoRothEA-hypoxia overlap, de-overlapped scoring, matched-null and QC-adjusted sensitivity outputs;
- TCGA-LUAD/LUSC score-association and adjusted-model outputs;
- current restricted-consensus Figure 4 pySCENIC source data;
- spatial threshold and leave-one-patient-out sensitivity tables.

These tables are generated from scripts in `scripts/` and correspond to the manuscript version prepared on 2026-09-11.

## Legacy Root-Level Tables

Root-level `figure*` source tables are retained from the first code-release package to preserve earlier figure reproducibility. The current manuscript figure mapping is documented in `docs/WORKFLOW.md`; use `if5_submission_20260911/` for the latest submission-level tabular outputs.
