from html import escape
from pathlib import Path
from re import DOTALL, subn
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


LEGEND = (
    "Supplementary Figure 3. Multi-omic prioritisation of BACH1-associated candidate regulatory features. "
    "a, Candidate-selection funnel across five independent full-TF pySCENIC runs with different random seeds. "
    "The 331 seed-union targets were reduced to 46 targets detected in at least three runs and 14 targets detected in at least four runs; the funnel is schematic and not to scale. "
    "Tier 1 comprised candidates detected in all five runs, with motif/accessibility context in all five GSE274934 tumour scATAC-seq samples and at least one promoter-window proximity link. "
    "Tier 2 comprised candidates detected in exactly four runs that met the same chromatin-context criteria but did not meet Tier 1. "
    "A sample was counted as ATAC-supportive when an accessible peak overlapped a BACH1-family motif record (BACH1, JASPAR MA1633.2; BACH1::MafK, JASPAR MA0591.2) within the TSS ±10-kb window; promoter-window support required at least one such record within ±2 kb of the TSS. "
    "Context-limited candidates were recurrent candidates that failed one or more chromatin-context criteria: FSTL1 had ATAC context in 3/5 samples and no promoter-window support; FAT1 had ATAC context in 4/5 samples and therefore did not meet the 5/5 ATAC requirement; and PALM2-AKAP2 had no motif-overlapping accessible peak and is shown as 0/5 ATAC support in panel b. "
    "b, Candidate-level chromatin-context summaries for the same 14 recurrent genes, ordered first by evidence class, then by seed recurrence and GRNBoost2 importance. "
    "The seed-recurrence column reports the number of independent pySCENIC runs in which each candidate was recovered. "
    "Green bars show descriptive raw counts of UCSC score ≥300 motif–peak–candidate proximity links in the broader TSS ±10-kb window; orange bars show descriptive raw counts of retained promoter-window proximity links within TSS ±2 kb. "
    "The ATAC column shows the number of samples with motif-overlapping accessible-peak context out of five. "
    "The fixed scATAC sample order was AL05, AL08, AL10, AL11 and AL12. Colour bands denote tiers as in panel a. "
    "These link counts did not enter tier assignment, are not normalized for gene length or accessible-peak burden, and do not imply that a larger count represents a higher evidence tier. "
    "This panel shows the 14-gene recurrent subset and should not be compared directly with counts for the broader candidate set in Supplementary Figure 2. "
    "c, State/network evidence matrix for the same 14 candidates. Columns show GRNBoost2 importance, partial BACH1 correlation and stress-state expression enrichment; the corresponding raw values are annotated in the cells. "
    "Seed recurrence is shown in panel b and is not re-encoded as a heatmap colour in panel c. GRNBoost2 importance uses a sequential scale from 0.30 to 1.50; partial BACH1 correlation uses a fixed ±0.05 scale (observed range, −0.022 to 0.028); and stress-state enrichment uses a fixed ±0.20 scale (observed range, −0.077 to 0.196). Values outside these fixed display ranges were clipped to the displayed limits. "
    "These columns are contextual and did not enter tier assignment. Tier colours are retained in the gene labels to align this matrix with panel b. "
    "Evidence categories are not equivalent, and motif occurrence, chromatin accessibility and transcription-start-site proximity do not establish direct BACH1 occupancy or transcriptional regulation."
)


def update_docx(path: Path) -> None:
    with ZipFile(path, "r") as source:
        document_xml = source.read("word/document.xml").decode("utf-8")
        updated, count = subn(
            r'(<w:t[^>]*>)(Supplementary Figure 3\. Multi-omic prioritisation.*?)(</w:t>)',
            lambda match: match.group(1) + escape(LEGEND, quote=False) + match.group(3),
            document_xml,
            count=1,
            flags=DOTALL,
        )
        if count != 1:
            raise RuntimeError(f"Expected one Supplementary Figure 3 legend in {path}, found {count}")

        with NamedTemporaryFile(dir=path.parent, suffix=".docx", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as target:
                for item in source.infolist():
                    payload = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                    target.writestr(item, payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for docx in (root / "manuscript.docx", root / "outputs/submission_package/manuscript.docx"):
        update_docx(docx)
