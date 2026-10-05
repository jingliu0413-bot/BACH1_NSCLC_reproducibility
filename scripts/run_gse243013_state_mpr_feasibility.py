#!/usr/bin/env python3
"""Audit GSE243013 for an exploratory BACH1/state-to-MPR analysis.

This stage uses the public patient/cell metadata only. It deliberately does
not infer a score without the expression matrix, and it records that the
series contains immune compartments rather than malignant epithelial cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "manuscript_submission_20260911"
INPUT = ROOT / "data" / "clinical_extension" / "GSE243013" / "GSE243013_NSCLC_immune_scRNA_metadata.csv.gz"
GENES = ROOT / "data" / "clinical_extension" / "GSE243013" / "GSE243013_genes.csv.gz"
OUT = ROOT / "out" / "clinical_ici_feasibility" / "GSE243013"


def first_value(series: pd.Series) -> object:
    values = series.dropna().astype(str)
    values = values[~values.str.lower().isin({"", "nan", "none"})]
    return values.mode().iloc[0] if not values.empty else pd.NA


def patient_manifest(meta: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sampleID",
        "cancer_type",
        "gender",
        "age",
        "smoking_history",
        "pre_treatment_staging",
        "anti-PD1_therapy",
        "chemotherapy",
        "targeted_therapy",
        "cycles",
        "pathological_response",
        "pathological_response_rate",
        "radiological_response",
    ]
    rows = []
    for sample, group in meta.groupby("sampleID", observed=True, sort=True):
        row = {"sampleID": sample}
        for column in columns[1:]:
            row[column] = first_value(group[column])
        rows.append(row)
    patients = pd.DataFrame(rows)

    def known(value: object, negative: set[str]) -> bool:
        return str(value).strip().lower() not in negative

    patients["anti_pd1_exposed"] = patients["anti-PD1_therapy"].map(
        lambda x: known(x, {"", "nan", "none", "no", "unknowm", "unknown"})
    )
    patients["chemotherapy_exposed"] = patients["chemotherapy"].map(
        lambda x: known(x, {"", "nan", "none", "no", "unknowm", "unknown"})
    )
    patients["strict_chemoimmunotherapy"] = patients["anti_pd1_exposed"] & patients["chemotherapy_exposed"]
    patients["response_group"] = patients["pathological_response"].map(
        {"pCR": "MPR_or_pCR", "MPR": "MPR_or_pCR", "non-MPR": "non-MPR"}
    ).fillna("unknown")
    return patients


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Download the GEO metadata first: {INPUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(INPUT, compression="gzip", low_memory=False)
    patients = patient_manifest(meta)
    patients.to_csv(OUT / "GSE243013_patient_manifest.csv", index=False)

    cell_type = (
        meta.groupby(["sampleID", "major_cell_type"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cell_type.to_csv(OUT / "GSE243013_patient_celltype_counts.csv", index=False)

    response_by_histology = pd.crosstab(
        patients["cancer_type"], patients["response_group"], dropna=False
    ).reset_index()
    response_by_histology.to_csv(OUT / "GSE243013_response_by_histology.csv", index=False)

    target_table = pd.read_csv(SOURCE / "external_bach1_signature_targets.csv")
    targets = target_table.loc[
        target_table["signature"].eq("DOROTHEA_BACH1_ABC_TF_ACTIVITY")
        & target_table["present_in_primary"].eq(True),
        "target",
    ].astype(str).str.upper()
    targets = sorted(set(targets) - {"ALDOA", "HMOX1", "IL6"})
    available_genes = set(pd.read_csv(GENES, compression="gzip")["geneSymbol"].astype(str).str.upper()) if GENES.exists() else set()
    score_coverage = pd.DataFrame(
        {
            "requested_deoverlapped_targets": [len(targets)],
            "targets_present_in_GSE243013_gene_index": [len(set(targets) & available_genes)],
            "missing_targets": [";".join(sorted(set(targets) - available_genes))],
            "expression_matrix_downloaded": [
                (ROOT / "data" / "clinical_extension" / "GSE243013" / "GSE243013_NSCLC_immune_scRNA_counts.mtx.gz").exists()
            ],
        }
    )
    score_coverage.to_csv(OUT / "GSE243013_BACH1_score_coverage_audit.csv", index=False)

    summary = {
        "dataset": "GSE243013",
        "n_cells": int(len(meta)),
        "n_patients": int(patients["sampleID"].nunique()),
        "n_anti_pd1_exposed": int(patients["anti_pd1_exposed"].sum()),
        "n_strict_chemoimmunotherapy": int(patients["strict_chemoimmunotherapy"].sum()),
        "anti_pd1_response_counts": patients.loc[patients["anti_pd1_exposed"], "response_group"].value_counts().to_dict(),
        "strict_chemoimmunotherapy_response_counts": patients.loc[patients["strict_chemoimmunotherapy"], "response_group"].value_counts().to_dict(),
        "major_cell_types": sorted(meta["major_cell_type"].dropna().astype(str).unique().tolist()),
        "malignant_epithelial_compartment_present": bool(
            meta["major_cell_type"].astype(str).str.contains("epithelial|malignant", case=False, regex=True).any()
        ),
        "primary_endpoint_status": "metadata feasibility only; expression matrix required for BACH1/state scores",
        "interpretation_boundary": "This immune-only series can test exploratory immune-state/MPR associations, not direct replication of the malignant-epithelial BACH1 score result.",
    }
    with (OUT / "GSE243013_feasibility_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
