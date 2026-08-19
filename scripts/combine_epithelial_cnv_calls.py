"""Combine the three infercnvpy call sets used to define malignant epithelial cells."""

from __future__ import annotations

from itertools import combinations_with_replacement

import pandas as pd

from project_paths import DATA_DIR, OUTPUT_DIR


OUT = OUTPUT_DIR / "epithelial_cnv_infercnvpy"
STRICT = OUT / "tables" / "nsclc_gse131907_gse274934_epithelial_cnv_epithelial_cell_cnv_calls.csv.gz"
SENSITIVE = (
    OUTPUT_DIR
    / "epithelial_cnv_infercnvpy_sensitivity_no_dynamic_threshold"
    / "tables"
    / "nsclc_gse131907_gse274934_epithelial_cnv_no_dynamic_threshold_epithelial_cell_cnv_calls.csv.gz"
)
ADJACENT = (
    OUTPUT_DIR
    / "epithelial_cnv_adjacent_epithelial_reference"
    / "tables"
    / "nsclc_gse131907_gse274934_epithelial_cnv_adjacent_epithelial_ref_cell_cnv_calls.csv.gz"
)
AUTHOR_ANNOTATION = DATA_DIR / "GSE131907" / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"


def main() -> None:
    strict = pd.read_csv(STRICT)
    sensitive = pd.read_csv(SENSITIVE)
    adjacent = pd.read_csv(ADJACENT)
    identity = ["cell_id", "dataset", "sample", "sample_id", "tissue_status", "leiden_major", "leiden_sub"]
    merged = strict[
        identity + ["cnv_leiden", "cnv_burden", "cnv_cellwise_high", "cnv_malignancy_call"]
    ].rename(
        columns={
            "cnv_leiden": "strict_cnv_leiden",
            "cnv_burden": "strict_cnv_burden",
            "cnv_cellwise_high": "strict_cellwise_high",
            "cnv_malignancy_call": "strict_cnv_malignancy_call",
        }
    )
    merged = merged.merge(
        sensitive[["cell_id", "cnv_leiden", "cnv_burden", "cnv_cellwise_high", "cnv_malignancy_call"]].rename(
            columns={
                "cnv_leiden": "sensitive_cnv_leiden",
                "cnv_burden": "sensitive_cnv_burden",
                "cnv_cellwise_high": "sensitive_cellwise_high",
                "cnv_malignancy_call": "sensitive_cnv_malignancy_call",
            }
        ),
        on="cell_id",
        how="inner",
    )
    merged["recommended_cnv_malignancy_call"] = merged["sensitive_cnv_malignancy_call"]
    merged = merged.merge(
        adjacent[
            ["cell_id", "cnv_leiden", "cnv_burden_adjacent_ref", "cnv_adjacent_ref_cellwise_high", "cnv_adjacent_ref_malignancy_call"]
        ].rename(columns={"cnv_leiden": "adjacent_ref_cnv_leiden"}),
        on="cell_id",
        how="left",
    )

    annotation = pd.read_csv(AUTHOR_ANNOTATION, sep="\t")
    merged["annotation_index"] = pd.NA
    is_gse131907 = merged["dataset"].eq("GSE131907")
    merged.loc[is_gse131907, "annotation_index"] = merged.loc[is_gse131907, "cell_id"].str.replace(
        r"^GSE131907_", "", regex=True
    )
    annotation = annotation[["Index", "Cell_type", "Cell_type.refined", "Cell_subtype", "Sample_Origin"]]
    merged = merged.merge(annotation, left_on="annotation_index", right_on="Index", how="left")
    author_state = merged["Cell_subtype"].isin(["tS1", "tS2", "tS3"]).astype("boolean")
    author_state.loc[~is_gse131907] = pd.NA
    merged["author_gse131907_tumor_epithelial_state"] = author_state

    merged["tnk_strict_malignant"] = merged["strict_cnv_malignancy_call"].eq("malignant_epithelial")
    merged["tnk_no_dynamic_malignant"] = merged["sensitive_cnv_malignancy_call"].eq("malignant_epithelial")
    merged["adjacent_ref_malignant"] = merged["cnv_adjacent_ref_malignancy_call"].eq("malignant_epithelial")
    support_columns = ["tnk_strict_malignant", "tnk_no_dynamic_malignant", "adjacent_ref_malignant"]
    merged["n_cnv_methods_malignant"] = merged[support_columns].sum(axis=1).astype(int)
    merged["cnv_consensus_call"] = "not_malignant_by_cnv_methods"
    merged.loc[merged["n_cnv_methods_malignant"].eq(1), "cnv_consensus_call"] = "single_cnv_method_malignant_candidate"
    merged.loc[merged["n_cnv_methods_malignant"].ge(2), "cnv_consensus_call"] = "consensus_malignant_epithelial"
    merged["recommended_working_malignant"] = merged["tnk_no_dynamic_malignant"]

    comparison = OUT / "nsclc_gse131907_gse274934_epithelial_cnv_strict_vs_no_dynamic_cell_call_comparison.csv.gz"
    merged.to_csv(OUT / "nsclc_gse131907_gse274934_epithelial_multimethod_malignancy_calls.csv.gz", index=False)
    merged[identity + [c for c in merged.columns if c.startswith("strict_") or c.startswith("sensitive_")]].to_csv(
        comparison, index=False
    )
    merged.loc[merged["cnv_consensus_call"].eq("consensus_malignant_epithelial"), "cell_id"].to_csv(
        OUT / "nsclc_gse131907_gse274934_epithelial_cnv_consensus_malignant_cell_ids.txt", index=False, header=False
    )
    merged.loc[merged["recommended_working_malignant"], "cell_id"].to_csv(
        OUT / "nsclc_gse131907_gse274934_epithelial_cnv_working_malignant_cell_ids.txt", index=False, header=False
    )

    definitions = [
        ("tnk_strict_dynamic_threshold", merged["tnk_strict_malignant"]),
        ("tnk_no_dynamic_threshold", merged["tnk_no_dynamic_malignant"]),
        ("adjacent_epithelial_reference", merged["adjacent_ref_malignant"]),
        ("cnv_consensus_at_least_two_methods", merged["cnv_consensus_call"].eq("consensus_malignant_epithelial")),
        ("single_cnv_method_only", merged["cnv_consensus_call"].eq("single_cnv_method_malignant_candidate")),
        ("recommended_working_set", merged["recommended_working_malignant"]),
    ]
    summary_rows = []
    for label, mask in definitions:
        subset = merged[mask]
        summary_rows.append(
            {
                "method_or_set": label,
                "n_cells": len(subset),
                "n_adjacent_normal": int(subset["tissue_status"].eq("adjacent_normal").sum()),
                "n_tumor": int(subset["tissue_status"].eq("tumor").sum()),
                "n_GSE131907": int(subset["dataset"].eq("GSE131907").sum()),
                "n_GSE274934": int(subset["dataset"].eq("GSE274934").sum()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        OUT / "nsclc_gse131907_gse274934_epithelial_multimethod_malignancy_summary.csv", index=False
    )
    merged.groupby(["dataset", "sample_id", "tissue_status", "cnv_consensus_call"], observed=True).size().reset_index(
        name="n_cells"
    ).to_csv(OUT / "nsclc_gse131907_gse274934_epithelial_multimethod_consensus_by_sample.csv", index=False)

    sets = {
        "tnk_strict": set(merged.loc[merged["tnk_strict_malignant"], "cell_id"]),
        "tnk_no_dynamic": set(merged.loc[merged["tnk_no_dynamic_malignant"], "cell_id"]),
        "adjacent_ref": set(merged.loc[merged["adjacent_ref_malignant"], "cell_id"]),
    }
    overlaps = [
        {"set_a": a, "set_b": b, "overlap_n": len(sets[a] & sets[b])}
        for a, b in combinations_with_replacement(sets, 2)
    ]
    pd.DataFrame(overlaps).to_csv(
        OUT / "nsclc_gse131907_gse274934_epithelial_cnv_method_pairwise_overlaps.csv", index=False
    )
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
