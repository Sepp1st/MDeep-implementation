# Archived Mai preprocessing (March 2026)

Source raw read: `Test-data/Mai/SRR8647671.fastq` (outside `data/`).

This archive preserves the prior Mai-derived artifacts, but they must not be
used as inputs to the current models:

| Artifact group | Shape | Finding |
| --- | --- | --- |
| `assembled_52_otus/Mai/mai_X.npy` | 24 x 52 | Non-zero assembled abundance matrix. Its companion correlation matrix is 52 x 52 and labels contain 24 disease samples. |
| Former 3-country evaluation matrix | 24 x 272 | Entirely zero; deleted during cleanup. Current 3-country model arrays use 271 features. |
| Former China-Hungary evaluation matrix | 24 x 241 | Entirely zero; deleted during cleanup. Current China-Hungary model arrays use 240 features. |
| Former Croatia-Hungary evaluation matrix | 24 x 157 | Entirely zero; deleted during cleanup. Current Croatia-Hungary model arrays use 156 features. |

The matching identical all-disease one-hot labels were deleted with the zero
matrices. The helper scripts are retained in `helper_scripts/` because they
generated these legacy evaluation artifacts. Reprocess from the raw FASTQ with
the current feature schema before using Mai for inference.
