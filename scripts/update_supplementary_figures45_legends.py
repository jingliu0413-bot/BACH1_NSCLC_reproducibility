from html import escape
from pathlib import Path
from re import DOTALL, subn
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


LEGEND_4 = (
    "Supplementary Figure 4. Cohort-stratified BACH1–hypoxia covariance. "
    "Patient-level correlations for the de-overlapped DoRothEA BACH1 and Hallmark hypoxia scores in all 16 patients and separately in GSE131907 and GSE274934. "
    "Points indicate Spearman's ρ and horizontal bars indicate approximate Fisher-z 95% confidence intervals; P values are unadjusted. "
    "The GSE274934 estimate is descriptive because it is based on five patients and has limited precision."
)

LEGEND_5 = (
    "Supplementary Figure 5. Spatial block-size sensitivity of state-adjusted BACH1–hypoxia concordance. "
    "Observed minus block-null mean bivariate Moran's I for the state-PC-residualized DoRothEA BACH1–Hallmark hypoxia score pair across block sizes of 10, 15, 20 and 30 array coordinates. "
    "Positive values indicate that observed spatial concordance exceeds the corresponding locally constrained null mean; they do not by themselves establish statistical significance across block sizes. "
    "P15 is the only LUSC section and is interpreted descriptively."
)

LEGEND_MAIN_5 = (
    "Figure 5. Independent spatial replication of state-embedded BACH1–hypoxia covariance. "
    "a, Four lung sections from the independent GSE292299 NSCLC cohort, shown as P5, P8, P12 and P15 with histology and in-tissue spot counts; P15 is the single LUSC section. "
    "b, Representative P12 (LUAD) projections of the de-overlapped DoRothEA BACH1 score, de-overlapped Hallmark hypoxia score and Stress-associated programme score; each map has an independently scaled score colour bar, so colour intensity is not directly comparable across programmes. "
    "c, Raw section-level Spearman correlations for BACH1–hypoxia and Stress programme–hypoxia pairs. "
    "d, Raw-score bivariate Moran's I compared with the 20-by-20 locally constrained block-null mean; the raw DoRothEA score was held fixed while raw hypoxia labels were permuted within blocks, and error bars indicate the 2.5th–97.5th percentiles of 1,000 permutations. "
    "e, DoRothEA–hypoxia correlation before adjustment, after QC adjustment and after adjustment for QC features plus three projected epithelial-state PCs. Spatial summaries are section-level and exploratory; the block null preserves local block structure but not the full variogram. The single LUSC section was interpreted descriptively. Block-size sensitivity for the state-PC-residualized score pair is shown in Supplementary Figure 5."
)


def replace_legend(xml: str, number: int, replacement: str) -> str:
    pattern = rf'(<w:t[^>]*>)Supplementary Figure {number}\..*?(</w:t>)'
    updated, count = subn(
        pattern,
        lambda match: match.group(1) + escape(replacement, quote=False) + match.group(2),
        xml,
        count=1,
        flags=DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Expected one Supplementary Figure {number} legend, found {count}")
    return updated


def replace_main_figure5(xml: str) -> str:
    pattern = r'(<w:t[^>]*>)Figure 5\..*?(</w:t>)'
    updated, count = subn(
        pattern,
        lambda match: match.group(1) + escape(LEGEND_MAIN_5, quote=False) + match.group(2),
        xml,
        count=1,
        flags=DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Expected one main Figure 5 legend, found {count}")
    return updated


def update_docx(path: Path) -> None:
    with ZipFile(path, "r") as source:
        document_xml = source.read("word/document.xml").decode("utf-8")
        document_xml = replace_main_figure5(document_xml)
        document_xml = replace_legend(document_xml, 4, LEGEND_4)
        document_xml = replace_legend(document_xml, 5, LEGEND_5)
        old = "partial BACH1 correlation"
        if document_xml.count(old) != 2:
            raise RuntimeError("Expected two candidate-matrix correlation labels")
        document_xml = document_xml.replace(old, "partial BACH1 transcript correlation")
        old_definition = "the corresponding raw values are annotated in the cells."
        if document_xml.count(old_definition) != 1:
            raise RuntimeError("Expected one candidate-matrix definition")
        document_xml = document_xml.replace(
            old_definition,
            old_definition
            + " The partial correlation compares candidate-gene expression with BACH1 mRNA after covariate residualisation; it does not use the DoRothEA score.",
        )
        with NamedTemporaryFile(dir=path.parent, suffix=".docx", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as target:
                for item in source.infolist():
                    payload = document_xml.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                    target.writestr(item, payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for docx in (root / "manuscript.docx", root / "outputs/submission_package/manuscript.docx"):
        update_docx(docx)
