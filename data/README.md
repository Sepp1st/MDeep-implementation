# Data layout

This directory separates current study datasets from raw reads and archived
experiments.

- `raw_data/` contains the China, Croatia, and Hungary source reads.
- `3Countries/` and `China-Hungary/` contain the current study-specific QIIME
  and model-input artifacts.
- `final_otu_97_preprocessed/` contains the final OTU-97 clustering outputs.
- `intermediate/` contains non-final QIIME artifacts retained to reproduce
  preprocessing.
- `archive/` contains superseded experiments retained for provenance; it is
  not an input to the current model workflow.

The raw Mai source of record is outside this directory at
`Test-data/Mai/SRR8647671.fastq`.  See `archive/mai_2026-03/README.md` for why
the earlier Mai-derived files are archived.
