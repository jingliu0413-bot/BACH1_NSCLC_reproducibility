#!/usr/bin/env python3
"""Prepare IF5-oriented supplementary table CSVs from final source tables."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "outputs" / "submission_ready_20260911"
READY_TABLES = READY / "tables"
ROBUST_TABLES = ROOT / "out" / "external_bach1_activity_robustness" / "tables"
OUT = ROOT / "outputs" / "submission_if5_20260911" / "supplementary_tables"
CSV_DIR = OUT / "csv"
JSON_DIR = OUT / "json"

ATAC_TARGETED = ROOT / "out" / "bach1_atac_motif_support_consensus_full_tf_5seed_stable_ge60_ucsc_bedmode_10kb"
PYSCENIC_STABILITY = ROOT / "out" / "revision_diagnostics" / "pyscenic_bach1_consensus_full_tf_seed_stability_v1"

SHARED_DOROTHEA_HYPOXIA_GENES = {"ALDOA", "HMOX1", "IL6"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def direction_from_weight(value: str) -> str:
    if value == "":
        return ""
    weight = float(value)
    if weight > 0:
        return "positive/BACH1-like"
    if weight < 0:
        return "negative/inverse"
    return "zero"


def copy_csv(source: Path, target_name: str) -> None:
    rows = read_csv(source)
    write_csv(CSV_DIR / target_name, rows)


def unique_present_count(rows: list[dict[str, str]]) -> int:
    return len({r.get("target_in_adata") or r.get("target") or r.get("gene") for r in rows if truthy(r.get("present_in_primary"))})


def build_signature_tables() -> None:
    external = read_csv(READY_TABLES / "external_bach1_signature_targets.csv")
    phenotype = read_csv(READY_TABLES / "phenotype_pathway_signature_targets.csv")
    hypoxia_present = {
        r["target_in_adata"]
        for r in phenotype
        if r["signature"] == "HALLMARK_HYPOXIA" and truthy(r["present_in_primary"])
    }

    rows: list[dict[str, object]] = []
    for r in external:
        matrix_gene = r.get("target_in_adata") or r.get("target")
        is_dor = r["signature"] == "DOROTHEA_BACH1_ABC_TF_ACTIVITY"
        removed = bool(is_dor and matrix_gene in SHARED_DOROTHEA_HYPOXIA_GENES)
        rows.append(
            {
                "supplementary_table": "Supplementary Table 1",
                "score_family": "external_BACH1_score",
                "signature": r["signature"],
                "gene_symbol": r["target"],
                "matrix_gene_symbol": matrix_gene,
                "present_in_primary": truthy(r.get("present_in_primary")),
                "used_in_primary_score": truthy(r.get("present_in_primary")),
                "used_after_shared_gene_removal": truthy(r.get("present_in_primary")) and not removed,
                "removed_for_deoverlapped_analysis": removed,
                "present_in_hallmark_hypoxia": matrix_gene in hypoxia_present,
                "regulatory_direction": direction_from_weight(r.get("weight", "")),
                "weight": r.get("weight", ""),
                "source_type": r.get("source_type", ""),
                "source_id": r.get("source_id", ""),
                "confidence": r.get("confidence", ""),
                "resources": r.get("resources", ""),
                "reference": r.get("source_id", ""),
                "pmid": r.get("pmid", ""),
                "pmcid": r.get("pmcid", ""),
                "doi": r.get("doi", ""),
                "geo_chipseq": r.get("geo_chipseq", ""),
                "geo_rnaseq": r.get("geo_rnaseq", ""),
                "notes": r.get("note", ""),
            }
        )

    for r in phenotype:
        if r["signature"] == "LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY":
            continue
        matrix_gene = r.get("target_in_adata") or r.get("target")
        removed = bool(r["signature"] == "HALLMARK_HYPOXIA" and matrix_gene in SHARED_DOROTHEA_HYPOXIA_GENES)
        source = r.get("source_type", "")
        if r["signature"].startswith("HALLMARK_"):
            resources = "MSigDB Hallmark via decoupler"
            confidence = "curated_gene_set"
        elif r["signature"].startswith("CURATED_NRF2"):
            resources = "curated NRF2 antioxidant-response genes"
            confidence = "curated_gene_set"
        else:
            resources = source
            confidence = ""
        rows.append(
            {
                "supplementary_table": "Supplementary Table 1",
                "score_family": "phenotype_context_score",
                "signature": r["signature"],
                "gene_symbol": r["target"],
                "matrix_gene_symbol": matrix_gene,
                "present_in_primary": truthy(r.get("present_in_primary")),
                "used_in_primary_score": truthy(r.get("present_in_primary")),
                "used_after_shared_gene_removal": truthy(r.get("present_in_primary")) and not removed,
                "removed_for_deoverlapped_analysis": removed,
                "present_in_hallmark_hypoxia": matrix_gene in hypoxia_present,
                "regulatory_direction": direction_from_weight(r.get("weight", "")),
                "weight": r.get("weight", ""),
                "source_type": source,
                "source_id": r["signature"],
                "confidence": confidence,
                "resources": resources,
                "reference": "",
                "pmid": "",
                "pmcid": "",
                "doi": "",
                "geo_chipseq": "",
                "geo_rnaseq": "",
                "notes": "Phenotype/context score used for association or overlap analyses.",
            }
        )

    sig_fields = [
        "supplementary_table",
        "score_family",
        "signature",
        "gene_symbol",
        "matrix_gene_symbol",
        "present_in_primary",
        "used_in_primary_score",
        "used_after_shared_gene_removal",
        "removed_for_deoverlapped_analysis",
        "present_in_hallmark_hypoxia",
        "regulatory_direction",
        "weight",
        "source_type",
        "source_id",
        "confidence",
        "resources",
        "reference",
        "pmid",
        "pmcid",
        "doi",
        "geo_chipseq",
        "geo_rnaseq",
        "notes",
    ]
    write_csv(CSV_DIR / "Supplementary_Table_1_signature_genes.csv", rows, sig_fields)

    summary_rows: list[dict[str, object]] = []
    external_sigs = sorted({r["signature"] for r in external})
    phenotype_sigs = sorted({r["signature"] for r in phenotype if r["signature"] != "LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY"})
    for signature in external_sigs + phenotype_sigs:
        source_rows = [r for r in external if r["signature"] == signature]
        score_family = "external_BACH1_score"
        source = ""
        direction = "weighted directional mean-z"
        manuscript_role = "Figure 2 external BACH1 score"
        if not source_rows:
            source_rows = [r for r in phenotype if r["signature"] == signature]
            score_family = "phenotype_context_score"
            source = source_rows[0].get("source_type", "") if source_rows else ""
            direction = "unweighted positive mean-z"
            manuscript_role = "phenotype/context score"
        else:
            source = source_rows[0].get("source_type", "")
        present_genes = {
            r.get("target_in_adata") or r.get("target")
            for r in source_rows
            if truthy(r.get("present_in_primary"))
        }
        after_shared = present_genes
        if signature == "DOROTHEA_BACH1_ABC_TF_ACTIVITY":
            after_shared = present_genes - SHARED_DOROTHEA_HYPOXIA_GENES
            manuscript_role = "primary external BACH1 score in Figure 2 and TCGA Figure 3"
        elif signature == "HALLMARK_HYPOXIA":
            after_shared = present_genes - SHARED_DOROTHEA_HYPOXIA_GENES
            manuscript_role = "primary phenotype outcome in Figure 2 and TCGA Figure 3"
        elif signature == "COLLECTRI_BACH1_TF_ACTIVITY":
            manuscript_role = "external BACH1 sensitivity score"
        elif signature == "KLENJA2025_BACH1_INVERSE_ACTIVITY":
            manuscript_role = "lung cancer perturbation/ChIP-seq sensitivity score"
        elif signature == "LITERATURE_LUNG_BACH1_EFFECTOR_ACTIVITY":
            manuscript_role = "descriptive lung-cancer BACH1 effector score"
        summary_rows.append(
            {
                "supplementary_table": "Supplementary Table 1",
                "score_family": score_family,
                "signature": signature,
                "source": source,
                "input_rows_or_edges": len(source_rows),
                "unique_genes_requested": len({r.get("target") or r.get("gene") for r in source_rows}),
                "unique_genes_present_primary": len(present_genes),
                "genes_used_after_shared_gene_removal": len(after_shared),
                "direction_handling": direction,
                "manuscript_role": manuscript_role,
                "notes": "Shared-gene removal affects only DoRothEA and Hallmark hypoxia by excluding ALDOA, HMOX1 and IL6.",
            }
        )
    write_csv(CSV_DIR / "Supplementary_Table_1_score_set_summary.csv", summary_rows)
    copy_csv(READY_TABLES / "bach1_phenotype_gene_set_overlap_summary.csv", "Supplementary_Table_1_overlap_summary.csv")
    copy_csv(READY_TABLES / "bach1_phenotype_gene_set_overlap_genes.csv", "Supplementary_Table_1_overlap_genes.csv")


def script_constant(name: str) -> str:
    script = (ROOT / "scripts" / "run_bach1_hypoxia_robustness.py").read_text(encoding="utf-8")
    match = re.search(rf"^{name}\s*=\s*([0-9]+)", script, flags=re.MULTILINE)
    return match.group(1) if match else ""


def build_matched_null_tables() -> None:
    summary = read_csv(ROBUST_TABLES / "matched_random_gene_set_hypoxia_correlation_summary.csv")
    params = [
        (
            "score_comparisons",
            "original 84-gene DoRothEA versus Hallmark hypoxia; de-overlapped 81-gene DoRothEA versus de-overlapped Hallmark hypoxia",
            "The two observed correlations were tested against independently sampled null distributions.",
        ),
        (
            "random_seeds",
            "original_hypoxia=1701; deoverlapped_hypoxia=1702",
            "Fixed NumPy default_rng seeds in scripts/run_bach1_hypoxia_robustness.py.",
        ),
        ("permutations", script_constant("N_PERMUTATIONS"), "Number of matched random gene sets."),
        ("matching_bins", "20 mean-expression quantile bins x 20 detection-fraction quantile bins", "Bins were computed in the primary malignant epithelial set."),
        ("gene_count_rule", "null set gene count equals the tested DoRothEA score gene count", "Original comparison used 84 genes; de-overlapped comparison used 81 genes."),
        ("candidate_exclusions", "all original DoRothEA genes; all Hallmark hypoxia genes; BACH1", "The null pool excluded score genes and BACH1 itself; ALDOA, HMOX1 and IL6 could not re-enter the de-overlapped null."),
        ("sampling_rule", "without replacement within each null set", "Each expressed DoRothEA gene was replaced once per permutation."),
        ("empty_bin_rule", "expand both bin dimensions stepwise until candidates are available", "Maximum search radius was 20 bins."),
        ("weights", "original DoRothEA weights reused", "Random score preserved the target-level weight vector."),
        ("empirical_p_formula", "(b+1)/(B+1)", "b is the number of null correlations >= the observed correlation; B is the number of permutations."),
    ]
    param_rows = [
        {
            "supplementary_table": "Supplementary Table 2",
            "parameter": key,
            "value": value,
            "details": details,
            "source_file": "scripts/run_bach1_hypoxia_robustness.py",
        }
        for key, value, details in params
    ]
    for row in summary:
        if row["comparison"] == "deoverlapped_hypoxia":
            param_rows.append(
                {
                    "supplementary_table": "Supplementary Table 2",
                    "parameter": "observed_deoverlapped_rho",
                    "value": row["observed_rho"],
                    "details": "Observed patient-level Spearman correlation used for the empirical-tail test.",
                    "source_file": "out/external_bach1_activity_robustness/tables/matched_random_gene_set_hypoxia_correlation_summary.csv",
                }
            )
            param_rows.append(
                {
                    "supplementary_table": "Supplementary Table 2",
                    "parameter": "deoverlapped_empirical_one_sided_p",
                    "value": row["empirical_p_greater_equal_observed"],
                    "details": "One-sided empirical P value for null correlations >= observed correlation.",
                    "source_file": "out/external_bach1_activity_robustness/tables/matched_random_gene_set_hypoxia_correlation_summary.csv",
                }
            )
    write_csv(CSV_DIR / "Supplementary_Table_2_matched_null_parameters.csv", param_rows)
    copy_csv(ROBUST_TABLES / "matched_random_gene_set_hypoxia_correlation_summary.csv", "Supplementary_Table_2_matched_null_summary.csv")
    copy_csv(ROBUST_TABLES / "matched_random_gene_set_input_gene_counts.csv", "Supplementary_Table_2_matched_null_input_gene_counts.csv")
    copy_csv(ROBUST_TABLES / "matched_random_gene_set_first50_matches.csv", "Supplementary_Table_2_first50_null_matches.csv")


def build_atac_tables() -> None:
    targeted_json = json.loads((ATAC_TARGETED / "bach1_atac_motif_support_ucsc_targeted_summary.json").read_text(encoding="utf-8"))
    resource_rows = [
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "input_recurrent_targets",
            "value": targeted_json["n_bach1_targets"],
            "details": "BACH1 candidate targets present in >=3/5 restricted-consensus full-TF pySCENIC seeds.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "targets_with_gencode_coordinates",
            "value": targeted_json["n_targets_with_gencode_coordinates"],
            "details": "PALM2-AKAP2 was not mapped to GENCODE v44 and was excluded from window-based ATAC matching.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "TSS_window_bp",
            "value": targeted_json["window_bp"],
            "details": "TSS +/- 10-kb windows were queried.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "ATAC_samples",
            "value": "; ".join(targeted_json["atac_samples"]),
            "details": f"{targeted_json['n_atac_samples']} tumour scATAC samples with author-filtered peaks.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "motif_models",
            "value": "; ".join(f"{key}={value}" for key, value in targeted_json["motif_models"].items()),
            "details": "BACH1-family motif occurrence from UCSC JASPAR2026 bigBed.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "ucsc_bigbed",
            "value": targeted_json["ucsc_bigbed"],
            "details": targeted_json.get("resource_note", ""),
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "ucsc_score_field",
            "value": "field 5 / score",
            "details": "Recorded as a UCSC-track sensitivity value; the submitted analysis reports score-threshold sensitivity rather than a single binding-confidence cutoff.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "target_window_query_mode",
            "value": "bigBedToBed -bed on merged recurrent-target TSS windows",
            "details": "Target-window querying used the UCSC bigBed server directly and completed without failed target windows.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "target_window_failed_queries",
            "value": targeted_json["n_ucsc_failed_windows"],
            "details": "Failed UCSC target-window queries in the canonical descriptive target-window analysis.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "motif_positive_sample_specific_peaks_in_target_windows",
            "value": targeted_json["n_motif_positive_sample_specific_peaks_in_target_windows"],
            "details": "Accessible tumour scATAC peaks intersecting BACH1-family UCSC JASPAR2026 motif occurrences within recurrent-candidate TSS windows.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "targets_with_window_support",
            "value": targeted_json["n_bach1_targets_with_atac_motif_support_in_window"],
            "details": "Coordinate-mapped recurrent candidates with at least one motif-overlapping accessible peak in the TSS +/- 10-kb window.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "targets_with_promoter_2kb_support",
            "value": targeted_json["n_bach1_targets_with_promoter_2kb_motif_peak"],
            "details": "Coordinate-mapped recurrent candidates with at least one motif-overlapping accessible peak within promoter +/- 2 kb.",
        },
        {
            "supplementary_table": "Supplementary Table 3",
            "parameter": "matched_background_status",
            "value": "not retained as an inferential analysis",
            "details": "The full UCSC bigBed is approximately 196 GB, per-motif JASPAR TSV downloads were unavailable locally, and remote extraction of matched background windows did not complete reliably enough for submission.",
        },
    ]
    write_csv(CSV_DIR / "Supplementary_Table_3_atac_resource_and_matching_summary.csv", resource_rows)
    copy_csv(ATAC_TARGETED / "tables" / "pyscenic_bach1_targets_with_ucsc_jaspar2026_atac_support.csv", "Supplementary_Table_3_targets_with_ucsc_target_window_support.csv")
    copy_csv(ATAC_TARGETED / "tables" / "ucsc_jaspar2026_score_threshold_sensitivity.csv", "Supplementary_Table_3_ucsc_score_threshold_sensitivity.csv")
    copy_csv(ATAC_TARGETED / "tables" / "bach1_targets_missing_from_gencode_v44.csv", "Supplementary_Table_3_targets_missing_from_gencode.csv")


def build_pyscenic_and_resource_tables() -> None:
    copy_csv(PYSCENIC_STABILITY / "pyscenic_bach1_seed_stability_run_summary.csv", "Supplementary_Table_4_pyscenic_seed_run_summary.csv")
    copy_csv(PYSCENIC_STABILITY / "pyscenic_bach1_seed_stability_pairwise_jaccard.csv", "Supplementary_Table_4_pyscenic_pairwise_jaccard.csv")
    copy_csv(READY_TABLES / "figure4e_current_bach1_recurrent_targets.csv", "Supplementary_Table_4_recurrent_targets_figure4e.csv")

    rows = [
        ("Main analysis environment", "Python", "3.11.5", "environment/environment-analysis.yml", "Used for the main scRNA-seq, CNV, scATAC and spatial workflows."),
        ("Robustness/TCGA environment", "Python", "3.12.5", "environment/revision_robustness_venv", "Used for final robustness and TCGA scripts."),
        ("Robustness/TCGA environment", "Scanpy", "1.11.4", "environment/revision_robustness_venv", "Version recorded in Methods."),
        ("Robustness/TCGA environment", "AnnData", "0.12.2", "environment/revision_robustness_venv", "Version recorded in Methods."),
        ("Robustness/TCGA environment", "NumPy", "1.26.4", "environment/revision_robustness_venv", "Version recorded in Methods."),
        ("Robustness/TCGA environment", "pandas", "2.3.3", "environment/revision_robustness_venv", "Version recorded in Methods."),
        ("Robustness/TCGA environment", "SciPy", "1.16.2", "environment/revision_robustness_venv", "Version recorded in Methods."),
        ("Robustness/TCGA environment", "matplotlib", "3.10.6", "environment/revision_robustness_venv", "Version recorded in Methods."),
        ("Robustness/TCGA environment", "seaborn", "0.13.2", "environment/revision_robustness_venv", "Version recorded in Methods."),
        ("Robustness/TCGA environment", "statsmodels", "0.14.5", "environment/revision_robustness_venv", "Version recorded in Methods."),
        ("pySCENIC environment", "Python", "3.11.5", "environment/environment-pyscenic.yml", "Used for pySCENIC v0.12.1 runs."),
        ("pySCENIC environment", "pySCENIC", "0.12.1", "environment/environment-pyscenic.yml", "BACH1-only and full-TF sensitivity runs."),
        ("pySCENIC environment", "arboreto", "0.1.6", "environment/environment-pyscenic.yml", "GRNBoost2 backend."),
        ("pySCENIC environment", "ctxcore", "0.2.0", "environment/environment-pyscenic.yml", "cisTarget motif-pruning backend."),
        ("External TF resources", "DoRothEA human A-C", "resource table generated through decoupler/OmniPath", "resources/bach1_external_signatures/dorothea_human_ABC_decoupler.csv", "Target membership used for primary DoRothEA mean-z score."),
        ("External TF resources", "CollecTRI human", "resource table generated through decoupler/OmniPath", "resources/bach1_external_signatures/collectri_human_decoupler.csv", "Alternative BACH1 activity resource."),
        ("GO/KEGG enrichment", "Enrichr via GSEApy", "accessed 19 June 2026", "GO_Biological_Process_2026; GO_Molecular_Function_2026; GO_Cellular_Component_2026; KEGG_2026", "Libraries used for over-representation analysis; interpreted as candidate context, not pathway activation."),
        ("Genome annotation", "GENCODE", "v44", "resources/atac_motif_support/gencode.v44.annotation.gtf.gz", "Gene coordinates and aliases for TSS-window matching."),
        ("Motif annotation", "UCSC JASPAR2026 bigBed", "accessed 10-11 September 2026", "http://hgdownload.soe.ucsc.edu/gbdb/hg38/jaspar/JASPAR2026.bb", "Used for descriptive BACH1-family motif occurrence in hg38 target windows; complete matched-background enrichment was not retained for inference."),
        ("TCGA resources", "UCSC Xena TCGA sampleMap", "downloaded for final TCGA replication scripts", "TCGA-LUAD and TCGA-LUSC expression and clinicalMatrix resources", "Primary tumour sample type code 01 only."),
    ]
    write_csv(
        CSV_DIR / "Supplementary_Table_4_software_and_resource_versions.csv",
        [
            {
                "supplementary_table": "Supplementary Table 4",
                "resource_family": family,
                "resource_or_package": package,
                "version_or_access_date": version,
                "local_or_remote_source": source,
                "notes": notes,
            }
            for family, package, version, source, notes in rows
        ],
    )


def build_readme() -> None:
    rows = [
        ("Supplementary_Table_1_signature_genes.csv", "Gene-level membership, direction, source, confidence and inclusion flags for external BACH1 and phenotype/context score sets."),
        ("Supplementary_Table_1_score_set_summary.csv", "Score-set-level counts including requested genes, present genes and shared-gene removal counts."),
        ("Supplementary_Table_1_overlap_summary.csv", "Gene-set overlap metrics between BACH1 resources and phenotype/context signatures."),
        ("Supplementary_Table_2_matched_null_parameters.csv", "Matched random gene-set algorithm parameters and empirical-tail-test definitions."),
        ("Supplementary_Table_2_matched_null_summary.csv", "Observed versus null DoRothEA-hypoxia patient-level correlations."),
        ("Supplementary_Table_2_matched_null_input_gene_counts.csv", "Input gene counts and exclusions for original and de-overlapped matched-null analyses."),
        ("Supplementary_Table_2_first50_null_matches.csv", "First 50 sampled matched-random gene sets for diagnostic reproducibility."),
        ("Supplementary_Table_3_atac_resource_and_matching_summary.csv", "ATAC motif resource, target mapping, UCSC target-window query status and matched-background analysis status."),
        ("Supplementary_Table_3_targets_with_ucsc_target_window_support.csv", "Recurrent BACH1-associated candidates with descriptive UCSC JASPAR2026 target-window accessible-motif support."),
        ("Supplementary_Table_3_ucsc_score_threshold_sensitivity.csv", "Descriptive UCSC score-threshold sensitivity for target-window accessible-motif support."),
        ("Supplementary_Table_3_targets_missing_from_gencode.csv", "Recurrent targets lacking GENCODE v44 coordinates."),
        ("Supplementary_Table_4_pyscenic_seed_run_summary.csv", "BACH1 target counts by restricted-consensus full-TF pySCENIC seed."),
        ("Supplementary_Table_4_pyscenic_pairwise_jaccard.csv", "Pairwise Jaccard overlap between restricted-consensus full-TF pySCENIC seeds."),
        ("Supplementary_Table_4_recurrent_targets_figure4e.csv", "Five-seed recurrence-weighted BACH1 candidate targets used in Figure 4e."),
        ("Supplementary_Table_4_software_and_resource_versions.csv", "Software versions and external resource access dates cited in Methods."),
    ]
    write_csv(
        CSV_DIR / "README_supplementary_tables.csv",
        [
            {
                "file": file_name,
                "description": desc,
                "generated_by": "scripts/prepare_if5_supplementary_tables.py",
            }
            for file_name, desc in rows
        ],
    )


def write_json_manifest() -> None:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_by": "scripts/prepare_if5_supplementary_tables.py",
        "output_directory": str(OUT.relative_to(ROOT)),
        "source_submission_table_directory": str(READY_TABLES.relative_to(ROOT)),
        "source_robustness_table_directory": str(ROBUST_TABLES.relative_to(ROOT)),
        "source_atac_target_window_directory": str(ATAC_TARGETED.relative_to(ROOT)),
        "supplementary_table_groups": {
            "Supplementary Table 1": "Signature genes, score-set counts and overlap diagnostics.",
            "Supplementary Table 2": "Original and de-overlapped DoRothEA-hypoxia matched random gene-set robustness analyses.",
            "Supplementary Table 3": "Descriptive scATAC target-window BACH1-family motif occurrence and resource diagnostics.",
            "Supplementary Table 4": "pySCENIC seed stability plus software and external resources.",
        },
    }
    (JSON_DIR / "if5_supplementary_tables_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    build_signature_tables()
    build_matched_null_tables()
    build_atac_tables()
    build_pyscenic_and_resource_tables()
    build_readme()
    write_json_manifest()
    print(OUT)


if __name__ == "__main__":
    main()
