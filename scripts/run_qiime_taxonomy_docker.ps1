param(
    [string]$WorkspacePath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$SeqsQza = "data/Croa_Hung_seqs.qza",
    [string]$ClassifierQza = "data/silva-99-nb-classifier.qza",
    [string]$OutputDir = "data/Croatia_Hung/taxonomy_export",
  [string]$FeatureIdsTxt = "",
  [switch]$FilterBeforeClassify,
  [string]$FilteredSeqsQza = "data/final_filtered_seqs_for_taxonomy.qza",
    [string]$QiimeImage = "quay.io/qiime2/core:2024.10"
)

$ErrorActionPreference = "Stop"

$seqsHost = Join-Path $WorkspacePath $SeqsQza
$classifierHost = Join-Path $WorkspacePath $ClassifierQza
$outHost = Join-Path $WorkspacePath $OutputDir
$featureIdsHost = $null

if (-not (Test-Path $seqsHost)) {
    throw "Sequences artifact not found: $seqsHost"
}
if (-not (Test-Path $classifierHost)) {
    throw "Classifier artifact not found: $classifierHost"
}

if ($FilterBeforeClassify) {
  if ([string]::IsNullOrWhiteSpace($FeatureIdsTxt)) {
    throw "-FeatureIdsTxt is required when -FilterBeforeClassify is used."
  }

  $featureIdsHost = Join-Path $WorkspacePath $FeatureIdsTxt
  if (-not (Test-Path $featureIdsHost)) {
    throw "Feature IDs metadata file not found: $featureIdsHost"
  }
}

New-Item -ItemType Directory -Force -Path $outHost | Out-Null

$seqsContainer = "/work/" + ($SeqsQza -replace "\\", "/")
$classifierContainer = "/work/" + ($ClassifierQza -replace "\\", "/")
$outputDirContainer = "/work/" + ($OutputDir -replace "\\", "/")
$taxonomyQzaContainer = "$outputDirContainer/taxonomy.qza"
$featureIdsContainer = ""
$filteredSeqsContainer = "/work/" + ($FilteredSeqsQza -replace "\\", "/")
$readsForClassification = $seqsContainer

if ($FilterBeforeClassify) {
  $featureIdsContainer = "/work/" + ($FeatureIdsTxt -replace "\\", "/")
  $readsForClassification = $filteredSeqsContainer
}

Write-Host "Pulling image: $QiimeImage"
docker pull $QiimeImage

$containerCmd = @"
set -e
"@

if ($FilterBeforeClassify) {
    $containerCmd += @"
qiime feature-table filter-seqs \
  --i-data $seqsContainer \
  --m-metadata-file $featureIdsContainer \
  --o-filtered-data $filteredSeqsContainer

"@
}

$containerCmd += @"
qiime feature-classifier classify-sklearn \
  --i-classifier $classifierContainer \
  --i-reads $readsForClassification \
  --o-classification $taxonomyQzaContainer

qiime tools export \
  --input-path $taxonomyQzaContainer \
  --output-path $outputDirContainer

echo "Done. taxonomy.tsv is in $outputDirContainer"
"@

Write-Host "Running QIIME2 taxonomy in Docker..."
docker run --rm `
  -v "${WorkspacePath}:/work" `
  $QiimeImage `
  /bin/bash -lc "$containerCmd"

Write-Host "Taxonomy export completed at: $outHost"
