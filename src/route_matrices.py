"""Prepare and finalize MDeep matrices for an OTU-99 or ASV route.

This module intentionally has two commands. ``prepare`` runs only after a
route feature table has been exported from QIIME; ``finalize`` runs only after
the route tree has been converted into ``c.npy``. Keeping these operations
separate prevents a feature/table/tree mismatch from being hidden.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split


class MatrixError(RuntimeError):
    """Raised when a route cannot safely produce model input matrices."""


def load_labels(path: Path) -> pd.Series:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample-id", "label"}
        if set(reader.fieldnames or ()) != required:
            raise MatrixError(f"{path} must contain exactly: sample-id, label")
        labels: dict[str, int] = {}
        for row in reader:
            sample_id = row["sample-id"]
            label = int(row["label"])
            if label not in (0, 1):
                raise MatrixError(f"Invalid binary label for {sample_id}: {label}")
            if sample_id in labels:
                raise MatrixError(f"Duplicate label for sample {sample_id}")
            labels[sample_id] = label
    return pd.Series(labels, dtype=np.int64, name="label")


def gmpr_normalize(table: pd.DataFrame) -> pd.DataFrame:
    """Run the same GMPR normalization used by the existing merge script.

    Falling back to TSS would silently produce a different scientific result,
    so this command fails when R/rpy2/GUniFrac are unavailable.
    """
    try:
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri
    except ImportError as exc:  # pragma: no cover - depends on local R setup
        raise MatrixError(
            "GMPR requires rpy2 and the R GUniFrac package; refusing to "
            "substitute a different normalization method."
        ) from exc

    rpy2.robjects.numpy2ri.activate()
    factors = np.asarray(importr("GUniFrac").GMPR(table.T.values))
    if factors.shape != (len(table),) or not np.isfinite(factors).all() or (factors <= 0).any():
        raise MatrixError("GMPR returned invalid sample size factors")
    return table.div(factors, axis=0)


def prepare(
    biom_path: Path,
    labels_path: Path,
    output_dir: Path,
    prevalence_threshold: float,
    median_nonzero_count: int,
    winsor_quantile: float,
) -> None:
    try:
        import biom
        from skbio.diversity import beta_diversity
    except ImportError as exc:  # pragma: no cover - dependency is environment-owned
        raise MatrixError("prepare requires biom-format and scikit-bio") from exc

    table = biom.load_table(str(biom_path)).to_dataframe(dense=True).T
    labels = load_labels(labels_path)
    missing = set(table.index) - set(labels.index)
    unknown = set(labels.index) - set(table.index)
    if missing or unknown:
        raise MatrixError(
            f"Label/table mismatch: missing labels={sorted(missing)[:5]}, "
            f"unknown labels={sorted(unknown)[:5]}"
        )
    labels = labels.loc[table.index]
    if labels.nunique() != 2:
        raise MatrixError("A binary route must retain both cancer and control samples")

    # Existing MDeep pipeline: remove Bray-Curtis outlier samples (Oj > 2).
    nonempty = table.loc[table.sum(axis=1) > 0]
    if len(nonempty) >= 2:
        distances = beta_diversity("braycurtis", nonempty.values, nonempty.index).to_data_frame()
        medians = distances.median(axis=1)
        overall_median = medians.median()
        if overall_median > 0:
            keep_samples = medians.index[(medians / overall_median) <= 2]
            table = table.loc[keep_samples]
            labels = labels.loc[keep_samples]

    if len(table) < 3 or labels.nunique() != 2:
        raise MatrixError("Outlier filtering left too few samples/classes for a stratified split")

    prevalence = (table > 0).sum(axis=0)
    median_nonzero = table.mask(table == 0).median(axis=0).fillna(0)
    keep_features = (prevalence >= len(table) * prevalence_threshold) & (
        median_nonzero >= median_nonzero_count
    )
    table = table.loc[:, keep_features]
    if table.shape[1] == 0:
        raise MatrixError("Feature filtering removed every feature")

    normalized = gmpr_normalize(table)
    caps = normalized.quantile(winsor_quantile, axis=0)
    transformed = np.sqrt(normalized.clip(upper=caps, axis=1))
    if not np.isfinite(transformed.values).all() or np.allclose(transformed.values, 0):
        raise MatrixError("Preprocessed abundance matrix is non-finite or all zero")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", transformed.values.astype(np.float64))
    np.save(output_dir / "Y.npy", labels.values.astype(np.int64))
    np.save(output_dir / "sample_ids.npy", transformed.index.values.astype(str))
    np.save(output_dir / "feature_ids.npy", transformed.columns.values.astype(str))
    with (output_dir / "feature_ids.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("#OTUID\n")
        handle.writelines(f"{feature_id}\n" for feature_id in transformed.columns)

    report = {
        "samples": int(transformed.shape[0]),
        "features": int(transformed.shape[1]),
        "label_counts": {str(label): int(count) for label, count in labels.value_counts().items()},
        "prevalence_threshold": prevalence_threshold,
        "median_nonzero_count": median_nonzero_count,
        "winsor_quantile": winsor_quantile,
        "normalization": "GMPR",
    }
    (output_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def hac_order(correlation: np.ndarray) -> np.ndarray:
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise MatrixError("Correlation matrix must be square")
    if not np.isfinite(correlation).all() or not np.allclose(correlation, correlation.T):
        raise MatrixError("Correlation matrix must be finite and symmetric")
    if not np.allclose(np.diag(correlation), 1.0):
        raise MatrixError("Correlation diagonal must be one")

    indexes = np.arange(correlation.shape[0]).reshape(-1, 1)

    def distance(left: np.ndarray, right: np.ndarray) -> float:
        return 1.0 - correlation[int(left[0]), int(right[0])]

    linked = linkage(indexes, metric=distance, method="single")
    return np.asarray([int(value) for value in dendrogram(linked, no_plot=True)["ivl"]], dtype=np.int64)


def finalize(output_dir: Path, correlation_path: Path, test_size: float, random_seed: int) -> None:
    x = np.load(output_dir / "X.npy", allow_pickle=False)
    y = np.load(output_dir / "Y.npy", allow_pickle=False)
    features = np.load(output_dir / "feature_ids.npy", allow_pickle=False)
    c = np.load(correlation_path, allow_pickle=False)
    if x.ndim != 2 or y.shape != (x.shape[0],) or features.shape != (x.shape[1],):
        raise MatrixError("X, Y, and feature IDs have incompatible shapes")
    if c.shape != (x.shape[1], x.shape[1]):
        raise MatrixError(f"c shape {c.shape} does not match {x.shape[1]} features")
    if len(np.unique(y)) != 2 or min(np.bincount(y)) < 2:
        raise MatrixError("Both labels need at least two samples for a stratified split")

    indexes = np.arange(x.shape[0])
    train_idx, eval_idx = train_test_split(
        indexes, test_size=test_size, random_state=random_seed, stratify=y
    )
    order = hac_order(c)
    c_clustered = c[np.ix_(order, order)]
    x_train, x_eval = x[train_idx], x[eval_idx]
    x_train_clustered, x_eval_clustered = x_train[:, order], x_eval[:, order]

    np.save(output_dir / "c.npy", c.astype(np.float64))
    np.save(output_dir / "c_clustered.npy", c_clustered.astype(np.float64))
    np.save(output_dir / "feature_order.npy", order)
    np.save(output_dir / "X_train.npy", x_train.astype(np.float64))
    np.save(output_dir / "X_eval.npy", x_eval.astype(np.float64))
    np.save(output_dir / "Y_train.npy", y[train_idx].astype(np.int64))
    np.save(output_dir / "Y_eval.npy", y[eval_idx].astype(np.int64))
    np.save(output_dir / "X_train_clustered.npy", x_train_clustered.astype(np.float64))
    np.save(output_dir / "X_eval_clustered.npy", x_eval_clustered.astype(np.float64))
    # Legacy consumers expect this name to contain the evaluation matrix.
    np.save(output_dir / "X_clustered.npy", x_eval_clustered.astype(np.float64))
    with (output_dir / "feature_order_mapping.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("clustered_position\toriginal_index\tfeature_name\n")
        for position, original_index in enumerate(order):
            handle.write(f"{position}\t{original_index}\t{features[original_index]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/finalize MDeep route matrices")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--biom", required=True, type=Path)
    prepare_parser.add_argument("--labels", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--prevalence-threshold", type=float, default=0.1)
    prepare_parser.add_argument("--median-nonzero-count", type=int, default=10)
    prepare_parser.add_argument("--winsor-quantile", type=float, default=0.97)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output", required=True, type=Path)
    finalize_parser.add_argument("--correlation", required=True, type=Path)
    finalize_parser.add_argument("--test-size", type=float, default=0.2)
    finalize_parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.biom, args.labels, args.output, args.prevalence_threshold,
            args.median_nonzero_count, args.winsor_quantile,
        )
    else:
        finalize(args.output, args.correlation, args.test_size, args.random_seed)


if __name__ == "__main__":
    main()
