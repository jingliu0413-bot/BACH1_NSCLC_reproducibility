import json
import math
import re
from pathlib import Path

import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
ATAC_TABLE = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "expressed_genes_atac_bach1_motif_support_background.csv.gz"
SYMBOL_ALIAS_FILE = ROOT / "out" / "bach1_atac_motif_support" / "tables" / "gencode_v44_symbol_aliases_used.csv"
OUT_DIR = ROOT / "out" / "bach1_atac_proximal_go_kegg_enrichment"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LIBRARIES = [
    "GO_Biological_Process_2026",
    "GO_Molecular_Function_2026",
    "GO_Cellular_Component_2026",
    "KEGG_2026",
]
FDR_CUTOFF = 0.05
SET_NAME = "bach1_atac_highconf_proximal_10kb"

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
        mapped = alias_map.get(str(gene), str(gene))
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


def run_ora(query_genes: list[str], background_genes: list[str]) -> pd.DataFrame:
    query = set(query_genes)
    background = set(background_genes)
    rows = []
    for library in LIBRARIES:
        print(f"Loading {library}", flush=True)
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
    q = pd.Series(index=res.index, dtype=float)
    for _, sub in res.groupby("Gene_set", sort=False):
        for idx, value in zip(sub.index, bh_adjust(sub["P_value"].tolist())):
            q.loc[idx] = value
    res["Adjusted_P_value"] = q
    res["Neg_log10_FDR"] = -np.log10(res["Adjusted_P_value"].clip(lower=np.nextafter(0, 1)))
    res["Gene_ratio"] = res["Overlap"] / res["Query_size"]
    return res.sort_values(["Adjusted_P_value", "P_value", "Odds_ratio"], ascending=[True, True, False]).reset_index(drop=True)


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


def plot_dotplot(res: pd.DataFrame, max_terms_per_library: int = 10):
    parts = []
    for library, sub in res.groupby("Gene_set", sort=False):
        sig = sub[sub["Adjusted_P_value"] <= FDR_CUTOFF]
        use = sig if not sig.empty else sub
        parts.append(use.sort_values(["Adjusted_P_value", "P_value"]).head(max_terms_per_library))
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["Gene_set", "Neg_log10_FDR", "Gene_ratio"], ascending=[True, True, True])
    df["Display_term"] = df["Clean_term"].map(lambda x: wrap_label(x, 44))
    df["Library_short"] = df["Gene_set"].str.replace("_2026", "", regex=False).str.replace("_", " ")

    palette = {
        "GO Biological Process": "#4E79A7",
        "GO Molecular Function": "#59A14F",
        "GO Cellular Component": "#9C755F",
        "KEGG": "#E15759",
    }
    n_terms = len(df)
    fig_h = max(5.2, 0.24 * n_terms + 1.0)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    sizes = 18 + (df["Overlap"] / df["Overlap"].max()) * 130
    ax.scatter(
        df["Neg_log10_FDR"],
        np.arange(n_terms),
        s=sizes,
        c=df["Library_short"].map(palette).fillna("#6B7280"),
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(np.arange(n_terms))
    ax.set_yticklabels(df["Display_term"])
    ax.set_xlabel("-log10(FDR)")
    ax.set_ylabel("")
    ax.axvline(-math.log10(FDR_CUTOFF), color="#9CA3AF", lw=0.8, ls="--")
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("BACH1 ATAC proximal targets: GO/KEGG enrichment", loc="left", fontsize=8, pad=8)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", label=label, markerfacecolor=color, markersize=5)
        for label, color in palette.items()
        if label in set(df["Library_short"])
    ]
    ax.legend(handles=handles, loc="lower right", title="Library", fontsize=6, title_fontsize=6)
    fig.tight_layout()
    save_pub(fig, FIG_DIR / f"{SET_NAME}_go_kegg_dotplot")


def plot_significant_bar(res: pd.DataFrame):
    sig = res[res["Adjusted_P_value"] <= FDR_CUTOFF].copy()
    sig.to_csv(TABLE_DIR / f"{SET_NAME}_go_kegg_significant_terms.csv", index=False)
    if sig.empty:
        return
    sig = sig.sort_values(["Adjusted_P_value", "P_value"]).head(30).iloc[::-1].reset_index(drop=True)
    sig["Display"] = sig["Clean_term"].map(lambda x: wrap_label(x, 50))
    sig["Library_short"] = sig["Gene_set"].str.replace("_2026", "", regex=False).str.replace("_", " ")
    palette = {
        "GO Biological Process": "#4E79A7",
        "GO Molecular Function": "#59A14F",
        "GO Cellular Component": "#9C755F",
        "KEGG": "#E15759",
    }
    fig_h = max(3.8, 0.34 * len(sig) + 1.0)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    y = np.arange(len(sig))
    ax.barh(y, sig["Neg_log10_FDR"], color=sig["Library_short"].map(palette), height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(sig["Display"])
    ax.set_xlabel("-log10(FDR)")
    ax.set_ylabel("")
    ax.axvline(-math.log10(FDR_CUTOFF), color="#9CA3AF", lw=0.8, ls="--")
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)
    for i, row in sig.iterrows():
        ax.text(row["Neg_log10_FDR"] + 0.04, i, f"{int(row['Overlap'])} genes", va="center", fontsize=6)
    ax.set_title("Significant GO/KEGG terms for BACH1 ATAC proximal targets", loc="left", fontsize=8)
    fig.tight_layout()
    save_pub(fig, FIG_DIR / f"{SET_NAME}_go_kegg_significant_terms")


def main():
    alias_map = read_alias_map(SYMBOL_ALIAS_FILE)
    support = pd.read_csv(ATAC_TABLE)
    query_raw = support.loc[support["n_high_conf_proximal_10kb_motif_peak_links"].fillna(0).astype(float) > 0, "gene"].astype(str).tolist()
    background_raw = support["gene"].astype(str).tolist()
    query = normalize_symbols(query_raw, alias_map)
    background = normalize_symbols(background_raw, alias_map)

    pd.DataFrame({"gene": query_raw, "enrichment_gene_symbol": [alias_map.get(x, x) for x in query_raw]}).to_csv(
        TABLE_DIR / f"{SET_NAME}_genes.csv", index=False
    )

    res = run_ora(query, background)
    res.to_csv(TABLE_DIR / f"{SET_NAME}_go_kegg_ora_all_terms.csv", index=False)
    top = pd.concat(
        [
            sub.sort_values(["Adjusted_P_value", "P_value"]).head(20)
            for _, sub in res.groupby("Gene_set", sort=False)
        ],
        ignore_index=True,
    )
    top.to_csv(TABLE_DIR / f"{SET_NAME}_go_kegg_ora_top_terms.csv", index=False)

    plot_dotplot(res)
    plot_significant_bar(res)

    sig_counts = (
        res.assign(significant=res["Adjusted_P_value"] <= FDR_CUTOFF)
        .groupby("Gene_set", as_index=False)
        .agg(n_terms_tested=("Term", "count"), n_significant_terms=("significant", "sum"), min_fdr=("Adjusted_P_value", "min"))
    )
    sig_counts.to_csv(TABLE_DIR / f"{SET_NAME}_library_summary.csv", index=False)

    summary = {
        "set_name": SET_NAME,
        "definition": "Genes with high-confidence BACH1/Bach1::Mafk motif-positive open ATAC peaks within TSS +/-10 kb.",
        "n_query_genes": len(query),
        "n_background_genes": len(background),
        "libraries": LIBRARIES,
        "fdr_cutoff": FDR_CUTOFF,
        "symbol_aliases_used": alias_map,
        "n_significant_terms_total": int((res["Adjusted_P_value"] <= FDR_CUTOFF).sum()),
    }
    with open(OUT_DIR / f"{SET_NAME}_go_kegg_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(OUT_DIR / f"{SET_NAME}_go_kegg_report.md", "w", encoding="utf-8") as fh:
        fh.write("# BACH1 ATAC proximal GO/KEGG enrichment\n\n")
        fh.write("Date: 2026-06-19\n\n")
        fh.write("## Definition\n\n")
        fh.write(summary["definition"] + "\n\n")
        fh.write(f"- Query genes: {len(query)}\n")
        fh.write(f"- Background genes: {len(background)}\n")
        fh.write("- Background: all malignant epithelial expressed genes with ATAC motif support annotation table.\n")
        fh.write("- ORA: Fisher's exact test, Benjamini-Hochberg FDR adjusted per library.\n\n")
        fh.write("## Main outputs\n\n")
        fh.write(f"- tables/{SET_NAME}_genes.csv\n")
        fh.write(f"- tables/{SET_NAME}_go_kegg_ora_all_terms.csv\n")
        fh.write(f"- tables/{SET_NAME}_go_kegg_ora_top_terms.csv\n")
        fh.write(f"- tables/{SET_NAME}_go_kegg_significant_terms.csv\n")
        fh.write(f"- figures/{SET_NAME}_go_kegg_dotplot.png/svg/pdf\n")
        fh.write(f"- figures/{SET_NAME}_go_kegg_significant_terms.png/svg/pdf\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
