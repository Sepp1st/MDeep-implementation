"""Guarded QIIME route runner for OTU-99 and DADA2 ASV representations.

The runner creates QIIME manifests from the repository's raw-read layout and
keeps all mutable artifacts outside the legacy OTU-97 directories. It supports
two explicit modes: ``quality-report`` creates the QIIME visualizations needed
to select DADA2 parameters, while ``all`` runs a configured route end to end.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


class RouteError(RuntimeError):
    """Raised when route inputs or artifact reuse are unsafe."""


@dataclass(frozen=True)
class SampleReadPair:
    source: str
    sample_id: str
    label: int
    forward: Path
    reverse: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_qza_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RouteError(f"Artifact does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("metadata.yaml")]
        if len(names) != 1:
            raise RouteError(f"{path} has no unique QIIME metadata.yaml")
        text = archive.read(names[0]).decode("utf-8", errors="strict")
    return dict(re.findall(r"^(uuid|type|format):\s*(.+)$", text, flags=re.MULTILINE))


def qza_framework_versions(path: Path) -> set[str]:
    """Read framework versions from provenance without extracting the artifact."""
    versions: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith("action/action.yaml"):
                continue
            text = archive.read(name).decode("utf-8", errors="replace")
            match = re.search(r"framework:\s*\n\s+version:\s*([^\n]+)", text)
            if match:
                versions.add(match.group(1).strip().strip("'\""))
    return versions


def qza_table_ids(path: Path) -> set[str]:
    """Read sample and feature IDs from a feature-table QZA when needed.

    The expensive biom dependency is imported only after all lightweight reuse
    checks have passed.
    """
    try:
        import biom
    except ImportError as exc:  # pragma: no cover - only runs in QIIME environment
        raise RouteError("biom-format is required to validate a reusable table") from exc
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as directory:
        names = [name for name in archive.namelist() if name.endswith("/data/feature-table.biom")]
        if len(names) != 1:
            raise RouteError(f"{path} does not contain one feature-table.biom")
        extracted = Path(directory) / "feature-table.biom"
        extracted.write_bytes(archive.read(names[0]))
        table = biom.load_table(str(extracted))
        return set(table.ids(axis="sample")) | {f"feature:{value}" for value in table.ids(axis="observation")}


def qza_sequence_ids(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data/dna-sequences.fasta")]
        if len(names) != 1:
            raise RouteError(f"{path} does not contain one dna-sequences.fasta")
        text = archive.read(names[0]).decode("utf-8", errors="strict")
    return {line[1:].split()[0] for line in text.splitlines() if line.startswith(">")}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RouteError(f"Cannot load config {path}: {exc}") from exc
    if config.get("schema_version") != 1:
        raise RouteError("Unsupported feature route configuration schema")
    return config


class FeatureRouteRunner:
    def __init__(self, config: dict[str, Any], representation: str, route: str, dry_run: bool, resume: bool):
        if representation not in {"otu_99", "asv"}:
            raise RouteError(f"Unsupported representation: {representation}")
        if route not in config["routes"]:
            raise RouteError(f"Unknown route: {route}")
        self.config = config
        self.representation = representation
        self.route = route
        self.dry_run = dry_run
        self.resume = resume
        self.sources = config["routes"][route]
        self.raw_root = resolve_path(config["paths"]["raw_data"])
        self.intermediate_root = resolve_path(config["paths"]["intermediate"])
        self.work_dir = self.intermediate_root / representation / route
        self.final_dir = REPO_ROOT / (
            "data/final_otu_99_preprocessed" if representation == "otu_99" else "data/final_asv_preprocessed"
        ) / route
        self.reuse_report: dict[str, Any] = {"representation": representation, "route": route, "candidates": []}

    def source_pairs(self, source: str) -> list[SampleReadPair]:
        source_config = self.config["sources"][source]
        directory = self.raw_root / source_config["directory"]
        if not directory.is_dir():
            raise RouteError(f"Raw source directory does not exist: {directory}")
        pairs: list[SampleReadPair] = []
        for sample_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
            files = sorted(path for path in sample_dir.iterdir() if path.is_file())
            # China has legacy ``*_f1.gz``/``*_r2.gz`` names, whereas the
            # Croatia and Hungary files use ``*_1.fastq.gz``/``*_2.fastq.gz``.
            forward = [path for path in files if re.search(r"(?:_f1|_1)(?:\.(?:fastq|fq))?\.gz$", path.name, re.I)]
            reverse = [path for path in files if re.search(r"(?:_r2|_2)(?:\.(?:fastq|fq))?\.gz$", path.name, re.I)]
            if len(forward) != 1 or len(reverse) != 1:
                raise RouteError(
                    f"{source}/{sample_dir.name} needs exactly one forward and reverse read; "
                    f"found {len(forward)} and {len(reverse)}"
                )
            matching_prefixes = [prefix for prefix in source_config["label_prefixes"] if sample_dir.name.startswith(prefix)]
            if len(matching_prefixes) != 1:
                raise RouteError(f"Cannot derive a label for {source}/{sample_dir.name}")
            prefix = matching_prefixes[0]
            pairs.append(
                SampleReadPair(
                    source=source,
                    sample_id=f"{source_config['sample_prefix']}{sample_dir.name}",
                    label=int(source_config["label_prefixes"][prefix]),
                    forward=forward[0].resolve(),
                    reverse=reverse[0].resolve(),
                )
            )
        if not pairs:
            raise RouteError(f"No sample directories found in {directory}")
        return pairs

    def all_pairs(self) -> list[SampleReadPair]:
        pairs = [pair for source in self.sources for pair in self.source_pairs(source)]
        identifiers = [pair.sample_id for pair in pairs]
        if len(identifiers) != len(set(identifiers)):
            raise RouteError("Route contains duplicate sample IDs")
        labels = {pair.label for pair in pairs}
        if labels != {0, 1}:
            raise RouteError(f"Route must contain both labels; found {labels}")
        return pairs

    def write_manifests(self, pairs: Iterable[SampleReadPair]) -> tuple[dict[str, Path], Path, Path]:
        manifests_dir = self.work_dir / "manifests"
        labels_path = manifests_dir / "labels.tsv"
        checksums_path = manifests_dir / "raw_checksums.json"
        by_source: dict[str, list[SampleReadPair]] = {source: [] for source in self.sources}
        for pair in pairs:
            by_source[pair.source].append(pair)
        if self.dry_run:
            return ({source: manifests_dir / f"{source}.tsv" for source in self.sources}, labels_path, checksums_path)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_paths: dict[str, Path] = {}
        for source, source_pairs in by_source.items():
            manifest = manifests_dir / f"{source}.tsv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["sample-id", "absolute-filepath", "direction"])
                for pair in source_pairs:
                    writer.writerow([pair.sample_id, pair.forward, "forward"])
                    writer.writerow([pair.sample_id, pair.reverse, "reverse"])
            manifest_paths[source] = manifest
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["sample-id", "label"])
            for pair in sorted(pairs, key=lambda value: value.sample_id):
                writer.writerow([pair.sample_id, pair.label])
        checksums = {
            pair.sample_id: {"forward": sha256_file(pair.forward), "reverse": sha256_file(pair.reverse)}
            for pair in pairs
        }
        checksums_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_paths, labels_path, checksums_path

    def run_command(self, args: list[str], outputs: Iterable[Path] = ()) -> None:
        outputs = tuple(outputs)
        if self.resume and outputs and all(path.exists() for path in outputs):
            print("[resume]", " ".join(args))
            return
        print("[qiime]", " ".join(args))
        if not self.dry_run:
            subprocess.run(args, cwd=REPO_ROOT, check=True)

    def qiime_command(self, *args: str) -> list[str]:
        return [self.config["qiime"]["executable"], *args]

    def validate_reuse(self, pairs: list[SampleReadPair]) -> tuple[Path, Path] | None:
        candidate = self.config.get("reuse_candidates", {}).get(self.representation, {}).get(self.route)
        if not candidate:
            return None
        result: dict[str, Any] = {"candidate": candidate, "accepted": False, "reasons": []}
        self.reuse_report["candidates"].append(result)
        table_path = resolve_path(candidate["table"])
        sequence_path = resolve_path(candidate["sequences"])
        try:
            table_metadata = read_qza_metadata(table_path)
            sequence_metadata = read_qza_metadata(sequence_path)
            if table_metadata.get("type") != "FeatureTable[Frequency]":
                result["reasons"].append("table artifact is not FeatureTable[Frequency]")
            if sequence_metadata.get("type") != "FeatureData[Sequence]":
                result["reasons"].append("sequence artifact is not FeatureData[Sequence]")
            expected_version = candidate.get("expected_qiime_version")
            versions = qza_framework_versions(table_path) | qza_framework_versions(sequence_path)
            result["qiime_versions"] = sorted(versions)
            if expected_version not in versions:
                result["reasons"].append(f"expected QIIME {expected_version}, found {sorted(versions)}")
            checksums_file = candidate.get("raw_checksums")
            if not checksums_file:
                result["reasons"].append("candidate has no recorded raw-file checksum manifest")
            else:
                recorded = json.loads(resolve_path(checksums_file).read_text(encoding="utf-8"))
                current = {
                    pair.sample_id: {"forward": sha256_file(pair.forward), "reverse": sha256_file(pair.reverse)}
                    for pair in pairs
                }
                if recorded != current:
                    result["reasons"].append("raw-file checksum manifest does not match current route inputs")
            if not result["reasons"]:
                identifiers = qza_table_ids(table_path)
                sample_ids = {value for value in identifiers if not value.startswith("feature:")}
                feature_ids = {value.removeprefix("feature:") for value in identifiers if value.startswith("feature:")}
                if sample_ids != {pair.sample_id for pair in pairs}:
                    result["reasons"].append("table sample IDs do not exactly match the target route")
                if feature_ids != qza_sequence_ids(sequence_path):
                    result["reasons"].append("table feature IDs do not match representative sequence IDs")
        except (OSError, ValueError, zipfile.BadZipFile, RouteError) as exc:
            result["reasons"].append(str(exc))
        if result["reasons"]:
            return None
        result["accepted"] = True
        result["table_uuid"] = table_metadata.get("uuid")
        result["sequence_uuid"] = sequence_metadata.get("uuid")
        result["table_sha256"] = sha256_file(table_path)
        result["sequence_sha256"] = sha256_file(sequence_path)
        return table_path, sequence_path

    def source_demux(self, source: str, manifest: Path) -> Path:
        source_dir = self.work_dir / "sources" / source
        demux = source_dir / "demux.qza"
        self.run_command(
            self.qiime_command(
                "tools", "import", "--type", "SampleData[PairedEndSequencesWithQuality]",
                "--input-path", str(manifest), "--input-format", "PairedEndFastqManifestPhred33V2",
                "--output-path", str(demux),
            ),
            (demux,),
        )
        return demux

    def quality_reports(self, manifests: dict[str, Path]) -> None:
        for source, manifest in manifests.items():
            demux = self.source_demux(source, manifest)
            visualization = self.work_dir / "sources" / source / "demux_summary.qzv"
            self.run_command(
                self.qiime_command("demux", "summarize", "--i-data", str(demux), "--o-visualization", str(visualization)),
                (visualization,),
            )

    def ensure_dada2_settings(self) -> None:
        missing: list[str] = []
        for source in self.sources:
            values = self.config["sources"][source]["dada2"]
            for key in ("trim_left_f", "trim_left_r", "trunc_len_f", "trunc_len_r"):
                if not isinstance(values.get(key), int) or values[key] < 0:
                    missing.append(f"{source}.{key}")
        if missing:
            raise RouteError(
                "DADA2 parameters are unset. Run --stage quality-report, review each .qzv, "
                "then set non-negative trim/truncation values in config/feature_routes.json: "
                + ", ".join(missing)
            )

    def build_otu99_precluster(self, manifests: dict[str, Path]) -> tuple[Path, Path]:
        tables: list[Path] = []
        sequences: list[Path] = []
        for source, manifest in manifests.items():
            source_dir = self.work_dir / "sources" / source
            demux = self.source_demux(source, manifest)
            merged = source_dir / "joined.qza"
            filtered = source_dir / "quality_filtered.qza"
            table = source_dir / "derep_table.qza"
            seqs = source_dir / "derep_sequences.qza"
            self.run_command(self.qiime_command("vsearch", "merge-pairs", "--i-demultiplexed-seqs", str(demux), "--o-merged-sequences", str(merged)), (merged,))
            self.run_command(self.qiime_command("quality-filter", "q-score", "--i-demux", str(merged), "--o-filtered-sequences", str(filtered), "--o-filter-stats", str(source_dir / "quality_filter_stats.qza")), (filtered,))
            self.run_command(self.qiime_command("vsearch", "dereplicate-sequences", "--i-sequences", str(filtered), "--o-dereplicated-table", str(table), "--o-dereplicated-sequences", str(seqs)), (table, seqs))
            tables.append(table)
            sequences.append(seqs)
        route_dir = self.work_dir / "route_features"
        merged_table, merged_sequences = route_dir / "precluster_table.qza", route_dir / "precluster_sequences.qza"
        table_args = ["feature-table", "merge"] + [part for table in tables for part in ("--i-tables", str(table))] + ["--o-merged-table", str(merged_table)]
        seq_args = ["feature-table", "merge-seqs"] + [part for seq in sequences for part in ("--i-data", str(seq))] + ["--o-merged-data", str(merged_sequences)]
        self.run_command(self.qiime_command(*table_args), (merged_table,))
        self.run_command(self.qiime_command(*seq_args), (merged_sequences,))
        return merged_table, merged_sequences

    def build_asv_features(self, manifests: dict[str, Path]) -> tuple[Path, Path]:
        self.ensure_dada2_settings()
        tables: list[Path] = []
        sequences: list[Path] = []
        for source, manifest in manifests.items():
            source_dir = self.work_dir / "sources" / source
            demux = self.source_demux(source, manifest)
            settings = self.config["sources"][source]["dada2"]
            table, seqs = source_dir / "asv_table.qza", source_dir / "asv_sequences.qza"
            self.run_command(
                self.qiime_command(
                    "dada2", "denoise-paired", "--i-demultiplexed-seqs", str(demux),
                    "--p-trim-left-f", str(settings["trim_left_f"]), "--p-trim-left-r", str(settings["trim_left_r"]),
                    "--p-trunc-len-f", str(settings["trunc_len_f"]), "--p-trunc-len-r", str(settings["trunc_len_r"]),
                    "--p-n-threads", str(self.config["qiime"]["threads"]),
                    "--o-table", str(table), "--o-representative-sequences", str(seqs),
                    "--o-denoising-stats", str(source_dir / "dada2_stats.qza"),
                ),
                (table, seqs),
            )
            tables.append(table)
            sequences.append(seqs)
        route_dir = self.work_dir / "route_features"
        merged_table, merged_sequences = route_dir / "asv_table.qza", route_dir / "asv_sequences.qza"
        table_args = ["feature-table", "merge"] + [part for table in tables for part in ("--i-tables", str(table))] + ["--o-merged-table", str(merged_table)]
        seq_args = ["feature-table", "merge-seqs"] + [part for seq in sequences for part in ("--i-data", str(seq))] + ["--o-merged-data", str(merged_sequences)]
        self.run_command(self.qiime_command(*table_args), (merged_table,))
        self.run_command(self.qiime_command(*seq_args), (merged_sequences,))
        return merged_table, merged_sequences

    def otu99_features(self, table: Path, sequences: Path) -> tuple[Path, Path]:
        route_dir = self.work_dir / "route_features"
        clustered_table, clustered_sequences = route_dir / "clustered_table.qza", route_dir / "clustered_sequences.qza"
        self.run_command(self.qiime_command("vsearch", "cluster-features-de-novo", "--i-table", str(table), "--i-sequences", str(sequences), "--p-perc-identity", "0.99", "--p-threads", str(self.config["qiime"]["threads"]), "--o-clustered-table", str(clustered_table), "--o-clustered-sequences", str(clustered_sequences)), (clustered_table, clustered_sequences))
        return self.chimera_filter(clustered_table, clustered_sequences)

    def chimera_filter(self, table: Path, sequences: Path) -> tuple[Path, Path]:
        route_dir = self.work_dir / "route_features"
        nonchimeras = route_dir / "nonchimeras.qza"
        self.run_command(self.qiime_command("vsearch", "uchime-denovo", "--i-table", str(table), "--i-sequences", str(sequences), "--o-chimeras", str(route_dir / "chimeras.qza"), "--o-nonchimeras", str(nonchimeras), "--o-stats", str(route_dir / "uchime_stats.qza")), (nonchimeras,))
        filtered_table, filtered_sequences = route_dir / "feature_table.qza", route_dir / "representative_sequences.qza"
        self.run_command(self.qiime_command("feature-table", "filter-features", "--i-table", str(table), "--m-metadata-file", str(nonchimeras), "--o-filtered-table", str(filtered_table)), (filtered_table,))
        self.run_command(self.qiime_command("feature-table", "filter-seqs", "--i-data", str(sequences), "--m-metadata-file", str(nonchimeras), "--o-filtered-data", str(filtered_sequences)), (filtered_sequences,))
        return filtered_table, filtered_sequences

    def downstream(self, table: Path, sequences: Path, labels_path: Path) -> None:
        exported = self.work_dir / "exported_table"
        biom_path = exported / "feature-table.biom"
        self.run_command(self.qiime_command("tools", "export", "--input-path", str(table), "--output-path", str(exported)), (biom_path,))
        processing = self.config["processing"]
        self.run_command([sys.executable, str(REPO_ROOT / "src/route_matrices.py"), "prepare", "--biom", str(biom_path), "--labels", str(labels_path), "--output", str(self.final_dir), "--prevalence-threshold", str(processing["prevalence_threshold"]), "--median-nonzero-count", str(processing["median_nonzero_count"]), "--winsor-quantile", str(processing["winsor_quantile"])], (self.final_dir / "feature_ids.npy",))
        final_sequences = self.work_dir / "final_representative_sequences.qza"
        self.run_command(self.qiime_command("feature-table", "filter-seqs", "--i-data", str(sequences), "--m-metadata-file", str(self.final_dir / "feature_ids.txt"), "--o-filtered-data", str(final_sequences)), (final_sequences,))
        classifier = resolve_path(self.config["qiime"]["classifier"])
        taxonomy = self.final_dir / "taxonomy.qza"
        self.run_command(self.qiime_command("feature-classifier", "classify-sklearn", "--i-classifier", str(classifier), "--i-reads", str(final_sequences), "--o-classification", str(taxonomy)), (taxonomy,))
        self.run_command(self.qiime_command("tools", "export", "--input-path", str(taxonomy), "--output-path", str(self.final_dir / "taxonomy_export")), (self.final_dir / "taxonomy_export" / "taxonomy.tsv",))
        rooted_tree = self.work_dir / "rooted_tree.qza"
        self.run_command(self.qiime_command("alignment", "mafft-fasttree", "--i-sequences", str(final_sequences), "--o-alignment", str(self.work_dir / "aligned_rep_seqs.qza"), "--o-masked-alignment", str(self.work_dir / "masked_aligned_rep_seqs.qza"), "--o-tree", str(self.work_dir / "unrooted_tree.qza"), "--o-rooted-tree", str(rooted_tree)), (rooted_tree,))
        tree_export = self.work_dir / "exported_tree"
        tree_path = tree_export / "tree.nwk"
        self.run_command(self.qiime_command("tools", "export", "--input-path", str(rooted_tree), "--output-path", str(tree_export)), (tree_path,))
        correlation = self.final_dir / "c.npy"
        self.run_command([sys.executable, str(REPO_ROOT / "src/compute_correlation_matrix.py"), "--tree", str(tree_path), "--otu-ids", str(self.final_dir / "feature_ids.npy"), "--output", str(correlation)], (correlation,))
        self.run_command([sys.executable, str(REPO_ROOT / "src/route_matrices.py"), "finalize", "--output", str(self.final_dir), "--correlation", str(correlation), "--test-size", str(processing["test_size"]), "--random-seed", str(processing["random_seed"])], (self.final_dir / "X_eval.npy", self.final_dir / "c_clustered.npy"))

    def assert_runtime_dependencies(self) -> None:
        if shutil.which(self.config["qiime"]["executable"]) is None:
            raise RouteError(f"QIIME executable not found: {self.config['qiime']['executable']}")
        classifier = resolve_path(self.config["qiime"]["classifier"])
        if not classifier.is_file():
            raise RouteError(f"QIIME taxonomy classifier not found: {classifier}")

    def run(self, stage: str) -> None:
        pairs = self.all_pairs()
        manifests, labels_path, _ = self.write_manifests(pairs)
        if stage == "quality-report":
            self.quality_reports(manifests)
            return
        if not self.dry_run:
            self.assert_runtime_dependencies()
        if self.representation == "asv":
            self.ensure_dada2_settings()
            table, sequences = self.build_asv_features(manifests)
        else:
            reusable = self.validate_reuse(pairs)
            table, sequences = reusable if reusable else self.build_otu99_precluster(manifests)
            table, sequences = self.otu99_features(table, sequences)
        self.downstream(table, sequences, labels_path)
        if not self.dry_run:
            self.final_dir.mkdir(parents=True, exist_ok=True)
            (self.final_dir / "reuse_provenance.json").write_text(json.dumps(self.reuse_report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build guarded OTU-99 or DADA2-ASV feature routes")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/feature_routes.json")
    parser.add_argument("--representation", choices=("otu_99", "asv"), required=True)
    parser.add_argument("--route", choices=("3_countries", "China_Hung", "Croatia_Hung"), required=True)
    parser.add_argument("--stage", choices=("quality-report", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print QIIME commands without executing or writing manifests")
    parser.add_argument("--resume", action="store_true", help="Skip commands whose declared outputs already exist")
    args = parser.parse_args()
    try:
        runner = FeatureRouteRunner(load_config(args.config), args.representation, args.route, args.dry_run, args.resume)
        runner.run(args.stage)
    except (RouteError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"feature route failed: {exc}\n")


if __name__ == "__main__":
    main()
