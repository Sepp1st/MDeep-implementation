#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
# 1) Classify an already filtered representative-seqs artifact
#    bash scripts/run_qiime_taxonomy_conda.sh \
#      --env qiime2-2021.4 \
#      --seqs data/final_Croa_Hung_seqs.qza \
#      --classifier data/silva-138-99-nb-classifier.qza \
#      --outdir data/Croatia_Hung/taxonomy_export
#
# 2) Filter first, then classify
#    bash scripts/run_qiime_taxonomy_conda.sh \
#      --env qiime2-2021.4 \
#      --seqs data/Croa_Hung_seqs.qza \
#      --feature-ids data/Croa_Hung_feature_ids.txt \
#      --filtered-seqs data/final_Croa_Hung_seqs.qza \
#      --classifier data/silva-138-99-nb-classifier.qza \
#      --outdir data/Croatia_Hung/taxonomy_export

ENV_NAME="qiime2-2021.4"
SEQS_QZA=""
CLASSIFIER_QZA=""
OUTDIR=""
FEATURE_IDS=""
FILTERED_SEQS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --seqs) SEQS_QZA="$2"; shift 2 ;;
    --classifier) CLASSIFIER_QZA="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --feature-ids) FEATURE_IDS="$2"; shift 2 ;;
    --filtered-seqs) FILTERED_SEQS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$SEQS_QZA" || -z "$CLASSIFIER_QZA" || -z "$OUTDIR" ]]; then
  echo "Required: --seqs, --classifier, --outdir"
  exit 1
fi

if [[ ! -f "$SEQS_QZA" ]]; then
  echo "Missing sequences artifact: $SEQS_QZA"
  exit 1
fi
if [[ ! -f "$CLASSIFIER_QZA" ]]; then
  echo "Missing classifier artifact: $CLASSIFIER_QZA"
  exit 1
fi

mkdir -p "$OUTDIR"

# Ensure conda commands are available in non-interactive shells
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Initialize conda in this shell first."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Using QIIME from env: $ENV_NAME"
qiime info | sed -n '1,20p'

READS_FOR_CLASSIFY="$SEQS_QZA"

if [[ -n "$FEATURE_IDS" || -n "$FILTERED_SEQS" ]]; then
  if [[ -z "$FEATURE_IDS" || -z "$FILTERED_SEQS" ]]; then
    echo "When filtering, both --feature-ids and --filtered-seqs are required."
    exit 1
  fi
  if [[ ! -f "$FEATURE_IDS" ]]; then
    echo "Missing feature ID metadata file: $FEATURE_IDS"
    exit 1
  fi

  echo "Filtering sequences to final feature IDs..."
  qiime feature-table filter-seqs \
    --i-data "$SEQS_QZA" \
    --m-metadata-file "$FEATURE_IDS" \
    --o-filtered-data "$FILTERED_SEQS"

  READS_FOR_CLASSIFY="$FILTERED_SEQS"
fi

TAXONOMY_QZA="$OUTDIR/taxonomy.qza"

echo "Running classify-sklearn..."
qiime feature-classifier classify-sklearn \
  --i-classifier "$CLASSIFIER_QZA" \
  --i-reads "$READS_FOR_CLASSIFY" \
  --o-classification "$TAXONOMY_QZA"

echo "Exporting taxonomy.tsv..."
qiime tools export \
  --input-path "$TAXONOMY_QZA" \
  --output-path "$OUTDIR"

echo "Done. Taxonomy TSV: $OUTDIR/taxonomy.tsv"
