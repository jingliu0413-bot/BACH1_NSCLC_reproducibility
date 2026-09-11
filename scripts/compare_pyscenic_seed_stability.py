"""Compare BACH1 regulon target stability across pySCENIC seed runs."""

from __future__ import annotations

import argparse
import ast
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import OUTPUT_DIR


DEFAULT_RUNS = {
    "seed777": OUTPUT_DIR / "bach1_malignant_epithelial_consensus_pyscenic_all_tfs",
    "seed778": OUTPUT_DIR / "bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed778",
    "seed779": OUTPUT_DIR / "bach1_malignant_epithelial_consensus_pyscenic_all_tfs_seed779",
}
OUT = OUTPUT_DIR / "revision_diagnostics"


def bach1_targets_from_regulons(regulon_csv: Path) -> tuple[set[str], pd.DataFrame]:
    reg = pd.read_csv(regulon_csv, header=[0, 1], index_col=[0, 1])
    reg.index.names = ["TF", "MotifID"]
    rows = []
    genes: set[str] = set()
    bach1_rows = reg.loc[reg.index.get_level_values("TF") == "BACH1"]
    if bach1_rows.empty:
        return genes, pd.DataFrame()
    for (_, motif), row in bach1_rows.iterrows():
        targets = ast.literal_eval(row[("Enrichment", "TargetGenes")])
        nes = float(row[("Enrichment", "NES")])
        auc = float(row[("Enrichment", "AUC")])
        annotation = row[("Enrichment", "Annotation")]
        context = row[("Enrichment", "Context")]
        for gene, importance in targets:
            gene = str(gene)
            if gene == "BACH1":
                continue
            genes.add(gene)
            rows.append(
                {
                    "MotifID": str(motif),
                    "gene": gene,
                    "importance": float(importance),
                    "NES": nes,
                    "AUC": auc,
                    "Annotation": annotation,
                    "Context": context,
                }
            )
    return genes, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        help="Add a pySCENIC run directory. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    runs = {label: Path(path) for label, path in args.run} if args.run else DEFAULT_RUNS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_sets: dict[str, set[str]] = {}
    target_rows = []
    missing = []
    for label, run_dir in runs.items():
        regulons = run_dir / "pyscenic_all_tfs_regulons.csv"
        if not regulons.exists() or regulons.stat().st_size == 0:
            missing.append({"label": label, "path": str(regulons)})
            continue
        genes, rows = bach1_targets_from_regulons(regulons)
        target_sets[label] = genes
        if not rows.empty:
            rows.insert(0, "run", label)
            target_rows.append(rows)

    if len(target_sets) < 2:
        raise RuntimeError(f"Need at least two completed pySCENIC runs; missing/incomplete: {missing}")

    all_targets = sorted(set().union(*target_sets.values()))
    freq = pd.DataFrame({"gene": all_targets})
    for label, genes in target_sets.items():
        freq[f"in_{label}"] = freq["gene"].isin(genes)
    freq["n_runs_detected"] = freq[[c for c in freq.columns if c.startswith("in_")]].sum(axis=1).astype(int)
    freq["frequency"] = freq["n_runs_detected"] / len(target_sets)
    freq = freq.sort_values(["n_runs_detected", "gene"], ascending=[False, True])

    pairwise_rows = []
    for a, b in combinations(target_sets, 2):
        inter = target_sets[a] & target_sets[b]
        union = target_sets[a] | target_sets[b]
        pairwise_rows.append(
            {
                "run_a": a,
                "run_b": b,
                "n_a": len(target_sets[a]),
                "n_b": len(target_sets[b]),
                "overlap_n": len(inter),
                "jaccard": len(inter) / len(union) if union else np.nan,
                "overlap_genes": ";".join(sorted(inter)),
            }
        )
    pairwise = pd.DataFrame(pairwise_rows)

    run_summary = pd.DataFrame(
        [{"run": label, "n_bach1_unique_targets": len(genes)} for label, genes in target_sets.items()]
    )
    all_target_rows = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()
    total_runs = len(target_sets)
    stable60_n = int(np.ceil(total_runs * 0.60))
    stable70_n = int(np.ceil(total_runs * 0.70))
    if not all_target_rows.empty:
        evidence = (
            all_target_rows.groupby("gene", as_index=False)
            .agg(
                motifs=("MotifID", lambda s: ";".join(sorted(set(map(str, s))))),
                runs_with_evidence=("run", lambda s: ";".join(sorted(set(map(str, s))))),
                n_motif_rows=("MotifID", "size"),
                mean_importance=("importance", "mean"),
                max_importance=("importance", "max"),
                mean_NES=("NES", "mean"),
                max_NES=("NES", "max"),
                mean_AUC=("AUC", "mean"),
                max_AUC=("AUC", "max"),
            )
        )
        freq = freq.merge(evidence, on="gene", how="left")
    else:
        for col in [
            "motifs",
            "runs_with_evidence",
            "n_motif_rows",
            "mean_importance",
            "max_importance",
            "mean_NES",
            "max_NES",
            "mean_AUC",
            "max_AUC",
        ]:
            freq[col] = np.nan
    freq["n_runs_total"] = total_runs
    freq["stable60_min_runs"] = stable60_n
    freq["stable70_min_runs"] = stable70_n
    freq["weighted_importance"] = freq["frequency"] * freq["mean_importance"].fillna(0)
    freq["stability_class"] = np.select(
        [
            freq["n_runs_detected"].eq(total_runs),
            freq["n_runs_detected"].ge(stable70_n),
            freq["n_runs_detected"].ge(stable60_n),
        ],
        ["core_100pct", "stable_ge70pct", "stable_ge60pct"],
        default="extended_lt60pct",
    )
    freq = freq.sort_values(
        ["n_runs_detected", "weighted_importance", "gene"],
        ascending=[False, False, True],
    )

    run_summary.to_csv(args.output_dir / "pyscenic_bach1_seed_stability_run_summary.csv", index=False)
    pairwise.to_csv(args.output_dir / "pyscenic_bach1_seed_stability_pairwise_jaccard.csv", index=False)
    freq.to_csv(args.output_dir / "pyscenic_bach1_seed_stability_target_frequency.csv", index=False)
    freq[freq["n_runs_detected"].eq(total_runs)].to_csv(
        args.output_dir / "pyscenic_bach1_core_100pct_targets.csv",
        index=False,
    )
    freq[freq["n_runs_detected"].ge(stable70_n)].to_csv(
        args.output_dir / "pyscenic_bach1_stable_ge70pct_targets.csv",
        index=False,
    )
    freq[freq["n_runs_detected"].ge(stable60_n)].to_csv(
        args.output_dir / "pyscenic_bach1_stable_ge60pct_targets.csv",
        index=False,
    )
    freq.to_csv(args.output_dir / "pyscenic_bach1_extended_union_targets.csv", index=False)
    if not all_target_rows.empty:
        all_target_rows.to_csv(
            args.output_dir / "pyscenic_bach1_seed_stability_target_rows.csv.gz",
            index=False,
            compression="gzip",
        )

    summary = {
        "n_completed_runs": len(target_sets),
        "runs": list(target_sets),
        "missing_or_incomplete": missing,
        "n_union_targets": int(len(all_targets)),
        "n_targets_detected_in_all_runs": int(freq["n_runs_detected"].eq(len(target_sets)).sum()),
        "stable60_min_runs": stable60_n,
        "n_targets_detected_in_at_least_60pct_runs": int(freq["frequency"].ge(0.6).sum()),
        "stable70_min_runs": stable70_n,
        "n_targets_detected_in_at_least_70pct_runs": int(freq["n_runs_detected"].ge(stable70_n).sum()),
        "mean_pairwise_jaccard": float(pairwise["jaccard"].mean()),
        "outputs": {
            "run_summary": str(args.output_dir / "pyscenic_bach1_seed_stability_run_summary.csv"),
            "pairwise_jaccard": str(args.output_dir / "pyscenic_bach1_seed_stability_pairwise_jaccard.csv"),
            "target_frequency": str(args.output_dir / "pyscenic_bach1_seed_stability_target_frequency.csv"),
            "core_100pct": str(args.output_dir / "pyscenic_bach1_core_100pct_targets.csv"),
            "stable_ge70pct": str(args.output_dir / "pyscenic_bach1_stable_ge70pct_targets.csv"),
            "stable_ge60pct": str(args.output_dir / "pyscenic_bach1_stable_ge60pct_targets.csv"),
            "extended_union": str(args.output_dir / "pyscenic_bach1_extended_union_targets.csv"),
        },
    }
    (args.output_dir / "pyscenic_bach1_seed_stability_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
