import json
import math
import re
from pathlib import Path

import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
PYSCENIC_DIR = ROOT / "out" / "bach1_malignant_epithelial_pyscenic"
ATAC_DIR = ROOT / "out" / "bach1_atac_motif_support"
OUT_DIR = ROOT / "out" / "bach1_intersection_go_kegg_enrichment"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PYSCENIC_TARGETS = PYSCENIC_DIR / "tables" / "pyscenic_bach1_regulon_targets_integrated.csv"
ATAC_TARGETS = ATAC_DIR / "tables" / "pyscenic_bach1_targets_ranked_with_atac_support.csv"
BACKGROUND_MATRIX = PYSCENIC_DIR / "malignant_epithelial_counts_filtered_for_pyscenic.csv"
SYMBOL_ALIAS_FILE = ATAC_DIR / "tables" / "gencode_v44_symbol_aliases_used.csv"

LIBRARIES = [
    "GO_Biological_Process_2026",
    "GO_Molecular_Function_2026",
    "GO_Cellular_Component_2026",
    "KEGG_2026",
]

PRIMARY_SET = "pyscenic_intersect_atac_highconf_proximal_10kb"
FDR_CUTOFF = 0.05


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def read_background_genes(path: Path) -> list[str]:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    if cols and cols[0].lower().startswith("unnamed"):
        cols = cols[1:]
    return [str(x) for x in cols if str(x)]


def read_alias_map(path: Path) -> dict[str, str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return dict(zip(df["input_gene"].astype(str), df["gencode_gene_name"].astype(str)))


def normalize_symbols(genes: list[str], alias_map: dict[str, str]) -> list[str]:
    out = []
    seen = set()
    for gene in genes:
        mapped = alias_map.get(gene, gene)
        if mapped not in seen:
            out.append(mapped)
            seen.add(mapped)
    return out


def bh_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out.tolist()


def clean_term(term: str) -> str:
    term = re.sub(r"\s*\([^)]*(GO|R-HSA|WP|hsa|KEGG)[^)]*\)\s*$", "", str(term))
    term = re.sub(r"\s+", " ", term).strip()
    return term


def run_ora(query_genes: list[str], background_genes: list[str], libraries: list[str]) -> pd.DataFrame:
    query = set(query_genes)
    background = set(background_genes)
    rows = []
    for library in libraries:
        print(f"Downloading/loading library: {library}", flush=True)
        gene_sets = gp.get_library(name=library, organism="Human")
        for term, genes in gene_sets.items():
            term_genes = set(genes) & background
            if len(term_genes) < 5:
                continue
            overlap_genes = sorted(query & term_genes)
            a = len(overlap_genes)
            if a == 0:
                continue
            b = len(query - term_genes)
            c = len(term_genes - query)
            d = len(background - query - term_genes)
            oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "Gene_set": library,
                    "Term": term,
                    "Clean_term": clean_term(term),
                    "Overlap": a,
                    "Query_size": len(query),
                    "Term_size_in_background": len(term_genes),
                    "Background_size": len(background),
                    "Odds_ratio": oddsratio,
                    "P_value": pvalue,
                    "Genes": ";".join(overlap_genes),
                }
            )
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    adjusted = []
    for library, sub in res.groupby("Gene_set", sort=False):
        qvals = bh_adjust(sub["P_value"].tolist())
        adjusted.extend(zip(sub.index, qvals))
    q = pd.Series(index=res.index, dtype=float)
    for idx, value in adjusted:
        q.loc[idx] = value
    res["Adjusted_P_value"] = q
    res["Combined_score"] = -np.log10(res["P_value"].clip(lower=np.nextafter(0, 1))) * np.log2(res["Odds_ratio"].replace(np.inf, np.nan))
    res["Combined_score"] = res["Combined_score"].replace([np.inf, -np.inf], np.nan)
    res["Neg_log10_FDR"] = -np.log10(res["Adjusted_P_value"].clip(lower=np.nextafter(0, 1)))
    res["Gene_ratio"] = res["Overlap"] / res["Query_size"]
    return res.sort_values(["Adjusted_P_value", "P_value", "Odds_ratio"], ascending=[True, True, False]).reset_index(drop=True)


def select_top_terms(res: pd.DataFrame, top_n_per_library: int = 10) -> pd.DataFrame:
    if res.empty:
        return res
    parts = []
    for library, sub in res.groupby("Gene_set", sort=False):
        sig = sub[sub["Adjusted_P_value"] <= FDR_CUTOFF]
        use = sig if not sig.empty else sub
        parts.append(use.sort_values(["Adjusted_P_value", "P_value"]).head(top_n_per_library))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def wrap_label(label: str, width: int = 46) -> str:
    words = str(label).split()
    lines = []
    current = []
    n = 0
    for word in words:
        if current and n + len(word) + 1 > width:
            lines.append(" ".join(current))
            current = [word]
            n = len(word)
        else:
            current.append(word)
            n += len(word) + (1 if current[:-1] else 0)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def save_pub(fig: plt.Figure, stem: Path):
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_enrichment_dotplot(res: pd.DataFrame, set_name: str, max_terms_per_library: int = 8):
    if res.empty:
        return
    plot_df = []
    for library, sub in res.groupby("Gene_set", sort=False):
        sig = sub[sub["Adjusted_P_value"] <= FDR_CUTOFF]
        use = sig if not sig.empty else sub
        plot_df.append(use.sort_values(["Adjusted_P_value", "P_value"]).head(max_terms_per_library))
    df = pd.concat(plot_df, ignore_index=True)
    df = df.sort_values(["Gene_set", "Neg_log10_FDR", "Gene_ratio"], ascending=[True, True, True])
    df["Display_term"] = df["Clean_term"].map(lambda x: wrap_label(x, 44))
    df["Library_short"] = df["Gene_set"].str.replace("_2026", "", regex=False).str.replace("_", " ")

    n_terms = len(df)
    fig_h = max(4.8, 0.24 * n_terms + 1.0)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    palette = {
        "GO Biological Process": "#4E79A7",
        "GO Molecular Function": "#59A14F",
        "GO Cellular Component": "#9C755F",
        "KEGG": "#E15759",
    }
    color_values = df["Library_short"].map(palette).fillna("#6B7280")
    sizes = 18 + (df["Overlap"] / df["Overlap"].max()) * 120
    ax.scatter(df["Neg_log10_FDR"], np.arange(n_terms), s=sizes, c=color_values, alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.set_yticks(np.arange(n_terms))
    ax.set_yticklabels(df["Display_term"])
    ax.set_xlabel("-log10(FDR)")
    ax.set_ylabel("")
    ax.axvline(-math.log10(FDR_CUTOFF), color="#9CA3AF", lw=0.8, ls="--")
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"BACH1 intersected targets: GO/KEGG enrichment ({set_name})", loc="left", fontsize=8, pad=8)

    handles = []
    for label, color in palette.items():
        if label in set(df["Library_short"]):
            handles.append(plt.Line2D([0], [0], marker="o", color="none", label=label, markerfacecolor=color, markersize=5))
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), title="Library", fontsize=6, title_fontsize=6)
    fig.tight_layout()
    save_pub(fig, FIG_DIR / f"{set_name}_go_kegg_dotplot")


def plot_library_summary(res: pd.DataFrame, set_name: str):
    if res.empty:
        return
    summary = (
        res.assign(significant=res["Adjusted_P_value"] <= FDR_CUTOFF)
        .groupby("Gene_set", as_index=False)
        .agg(n_terms_tested=("Term", "count"), n_terms_fdr_0_05=("significant", "sum"), min_fdr=("Adjusted_P_value", "min"))
    )
    summary["Library_short"] = summary["Gene_set"].str.replace("_2026", "", regex=False).str.replace("_", " ")
    summary = summary.sort_values("n_terms_fdr_0_05", ascending=False)

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    colors = ["#4E79A7", "#59A14F", "#9C755F", "#E15759"]
    ax.barh(summary["Library_short"], summary["n_terms_fdr_0_05"], color=colors[: len(summary)], height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel("Significant terms (FDR < 0.05)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)
    for i, row in summary.reset_index(drop=True).iterrows():
        ax.text(row["n_terms_fdr_0_05"] + 0.15, i, f"min FDR={row['min_fdr']:.2g}", va="center", fontsize=6)
    ax.set_title(f"Enrichment breadth ({set_name})", loc="left", fontsize=8)
    fig.tight_layout()
    save_pub(fig, FIG_DIR / f"{set_name}_library_summary")


def run():
    pyscenic = pd.read_csv(PYSCENIC_TARGETS)
    atac = pd.read_csv(ATAC_TARGETS)
    alias_map = read_alias_map(SYMBOL_ALIAS_FILE)

    pyscenic_genes = set(pyscenic["gene"].dropna().astype(str))
    atac_proximal = set(atac.loc[atac["n_high_conf_proximal_10kb_motif_peak_links"].fillna(0).astype(float) > 0, "gene"].astype(str))

    gene_sets = {
        PRIMARY_SET: sorted(pyscenic_genes & atac_proximal),
    }

    background_raw = read_background_genes(BACKGROUND_MATRIX)
    background = normalize_symbols(background_raw, alias_map)

    manifest_rows = []
    for set_name, genes_raw in gene_sets.items():
        normalized = normalize_symbols(genes_raw, alias_map)
        raw_df = pd.DataFrame({"gene": genes_raw})
        raw_df["enrichment_gene_symbol"] = raw_df["gene"].map(lambda x: alias_map.get(x, x))
        raw_df.to_csv(TABLE_DIR / f"{set_name}_genes.csv", index=False)
        manifest_rows.append(
            {
                "set_name": set_name,
                "n_genes_original": len(genes_raw),
                "n_genes_after_symbol_normalization": len(normalized),
                "genes_file": f"tables/{set_name}_genes.csv",
            }
        )
        print(f"Running ORA for {set_name}: {len(normalized)} genes", flush=True)
        res = run_ora(normalized, background, LIBRARIES)
        res.insert(0, "query_set", set_name)
        res.to_csv(TABLE_DIR / f"{set_name}_go_kegg_ora_all_terms.csv", index=False)
        top = select_top_terms(res, top_n_per_library=15)
        top.to_csv(TABLE_DIR / f"{set_name}_go_kegg_ora_top_terms.csv", index=False)
        plot_enrichment_dotplot(res, set_name)
        plot_library_summary(res, set_name)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(TABLE_DIR / "bach1_intersection_gene_sets_manifest.csv", index=False)
    all_top = []
    for set_name in gene_sets:
        p = TABLE_DIR / f"{set_name}_go_kegg_ora_top_terms.csv"
        if p.exists() and p.stat().st_size > 0:
            all_top.append(pd.read_csv(p))
    if all_top:
        pd.concat(all_top, ignore_index=True).to_csv(TABLE_DIR / "all_gene_sets_go_kegg_ora_top_terms.csv", index=False)

    summary = {
        "libraries": LIBRARIES,
        "background_n_genes": len(background),
        "sets": manifest_rows,
        "primary_set_definition": "pySCENIC BACH1 regulon candidates intersected with scATAC high-confidence BACH1/Bach1::Mafk motif-positive accessible peaks within TSS +/-10 kb.",
        "symbol_aliases_used": alias_map,
    }
    with open(OUT_DIR / "bach1_intersection_go_kegg_enrichment_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    report = OUT_DIR / "bach1_intersection_go_kegg_enrichment_report.md"
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("# BACH1 intersection GO/KEGG enrichment report\n\n")
        fh.write("Date: 2026-06-19\n\n")
        fh.write("## Input definition\n\n")
        fh.write("- pySCENIC BACH1 targets: `pyscenic_bach1_regulon_targets_integrated.csv`\n")
        fh.write("- ATAC-supported BACH1 targets: `pyscenic_bach1_targets_ranked_with_atac_support.csv`\n")
        fh.write("- Primary intersection: pySCENIC candidates with high-confidence ATAC motif support within TSS +/-10 kb.\n")
        fh.write("- Background: genes expressed in the malignant epithelial pySCENIC input matrix.\n\n")
        fh.write("## Gene set sizes\n\n")
        for row in manifest_rows:
            fh.write(f"- {row['set_name']}: {row['n_genes_original']} genes; normalized symbols: {row['n_genes_after_symbol_normalization']}\n")
        fh.write("\n## Libraries\n\n")
        for lib in LIBRARIES:
            fh.write(f"- {lib}\n")
        fh.write("\n## Key result tables\n\n")
        fh.write("- `tables/pyscenic_intersect_atac_highconf_proximal_10kb_go_kegg_ora_all_terms.csv`\n")
        fh.write("- `tables/all_gene_sets_go_kegg_ora_top_terms.csv`\n")
        fh.write("\n## Notes\n\n")
        fh.write("ORA was computed locally with Fisher's exact test using the malignant epithelial expressed genes as background. FDR was adjusted per library by Benjamini-Hochberg.\n")
        fh.write("High-confidence ATAC motif support uses motif score >= 950 from JASPAR hg38 TFBS tracks.\n")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
