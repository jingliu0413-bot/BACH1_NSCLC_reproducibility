# External Resources

Large reference files are not redistributed. Run:

```bash
python scripts/download_reference_resources.py
```

The script downloads and verifies the exact resources used in the study:

- cisTarget hg38 10-kb and 500-bp/100-bp gene-based ranking databases.
- motifs-v10nr clustered human motif annotation.
- pySCENIC human HGNC transcription-factor list.
- JASPAR 2026 hg38 MA1633.2 BACH1 and MA0591.2 Bach1::Mafk TFBS tracks.
- GENCODE human release 44 GRCh38.p14 annotation.

The two provider-supplied cisTarget SHA1 files are retained in this directory. SHA256 checksums for every external resource are embedded in the download script.

The current scATAC target-window analysis can also query the UCSC hg38 JASPAR2026 bigBed track directly with `scripts/run_bach1_atac_motif_support_ucsc_targeted.py`, avoiding redistribution of genome-wide TFBS files.
