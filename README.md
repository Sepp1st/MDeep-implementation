# MDeep-implementation
Implementation of MDeep and evaluation based on Bukavina et al [1]

## Data Management

This repository contains the implementation code for MDeep. The genomic sequencing data files (`.gz`, `.fq.gz`, `.fastq.gz`) are not included in this repository due to their large size (4.2GB total).

### Data Structure
The data directory contains genomic sequencing data from three countries:
- `data/China/` - Chinese cohort samples (BC1-BC62, CT1-CT19)
- `data/Croatia/` - Croatian cohort samples (SAMEA4924324-SAMEA4924348)
- `data/Hungary/` - Hungarian cohort samples (UI01-UI04, UN01-UN06)

### Getting the Data
If you need access to the genomic data files, please:
1. Contact the repository maintainer
2. Use external data hosting solutions like:
   - Git LFS (Large File Storage)
   - Cloud storage (AWS S3, Google Cloud, etc.)
   - Research data repositories (SRA, ENA, etc.)

### Alternative Solutions
For working with large genomic datasets, consider:
1. **Git LFS**: For version-controlled large files
2. **DVC (Data Version Control)**: For data pipeline management
3. **Symlinks**: Link to data stored elsewhere
4. **Download scripts**: Automated data fetching from public repositories
