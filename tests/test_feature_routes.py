import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_routes import FeatureRouteRunner, RouteError  # noqa: E402
from route_matrices import finalize  # noqa: E402


class FeatureRouteRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        raw = self.root / "raw"
        sources = {
            "China": ("China", {"BC": 1, "CT": 0}),
            "Croatia": ("Croatia", {"BC": 1, "CT": 0}),
            "Hungary": ("Hungary", {"UI": 1, "UN": 1}),
        }
        for source, (directory, prefixes) in sources.items():
            for prefix in prefixes:
                sample_dir = raw / directory / f"{prefix}1"
                sample_dir.mkdir(parents=True)
                (sample_dir / "reads_1.fastq.gz").write_bytes(b"forward")
                (sample_dir / "reads_2.fastq.gz").write_bytes(b"reverse")
        self.config = {
            "schema_version": 1,
            "qiime": {"executable": "qiime", "expected_version": "2024.10.1", "classifier": "missing.qza", "threads": 1},
            "paths": {"raw_data": str(raw), "intermediate": str(self.root / "intermediate")},
            "processing": {"prevalence_threshold": 0.1, "median_nonzero_count": 10, "winsor_quantile": 0.97, "test_size": 0.2, "random_seed": 42},
            "sources": {
                source: {
                    "directory": directory,
                    "sample_prefix": source + "_",
                    "label_prefixes": prefixes,
                    "dada2": {"trim_left_f": None, "trim_left_r": None, "trunc_len_f": None, "trunc_len_r": None},
                }
                for source, (directory, prefixes) in sources.items()
            },
            "routes": {"3_countries": ["China", "Croatia", "Hungary"], "China_Hung": ["China", "Hungary"], "Croatia_Hung": ["Croatia", "Hungary"]},
            "reuse_candidates": {"otu_99": {"3_countries": {"table": str(self.root / "missing-table.qza"), "sequences": str(self.root / "missing-sequences.qza"), "raw_checksums": None, "expected_qiime_version": "2024.10.1"}}},
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_manifest_and_labels_are_created_from_raw_layout(self):
        runner = FeatureRouteRunner(self.config, "otu_99", "3_countries", dry_run=False, resume=False)
        pairs = runner.all_pairs()
        manifests, labels, checksums = runner.write_manifests(pairs)
        self.assertEqual(set(manifests), {"China", "Croatia", "Hungary"})
        self.assertIn("China_BC1", labels.read_text(encoding="utf-8"))
        self.assertIn("Croatia_BC1", labels.read_text(encoding="utf-8"))
        self.assertEqual(len(json.loads(checksums.read_text(encoding="utf-8"))), 6)

    def test_candidate_without_checksum_provenance_is_rejected(self):
        runner = FeatureRouteRunner(self.config, "otu_99", "3_countries", dry_run=True, resume=False)
        self.assertIsNone(runner.validate_reuse(runner.all_pairs()))
        reasons = runner.reuse_report["candidates"][0]["reasons"]
        self.assertTrue(reasons)

    def test_asv_requires_reviewed_dada2_parameters(self):
        runner = FeatureRouteRunner(self.config, "asv", "China_Hung", dry_run=True, resume=False)
        with self.assertRaisesRegex(RouteError, "DADA2 parameters are unset"):
            runner.ensure_dada2_settings()


class RouteMatrixTests(unittest.TestCase):
    def test_finalize_creates_aligned_clustered_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            np.save(output / "X.npy", np.arange(24, dtype=float).reshape(8, 3))
            np.save(output / "Y.npy", np.array([0, 0, 0, 0, 1, 1, 1, 1]))
            np.save(output / "feature_ids.npy", np.array(["f1", "f2", "f3"]))
            correlation = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]])
            correlation_path = output / "input_c.npy"
            np.save(correlation_path, correlation)

            finalize(output, correlation_path, test_size=0.5, random_seed=42)

            order = np.load(output / "feature_order.npy", allow_pickle=False)
            clustered = np.load(output / "c_clustered.npy", allow_pickle=False)
            self.assertEqual(sorted(order.tolist()), [0, 1, 2])
            self.assertTrue(np.allclose(clustered, clustered.T))
            self.assertEqual(np.load(output / "X_eval.npy", allow_pickle=False).shape, (4, 3))
            self.assertEqual(np.load(output / "X_clustered.npy", allow_pickle=False).shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
