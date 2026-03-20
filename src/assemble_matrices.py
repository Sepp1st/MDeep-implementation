"""
Step F: Final Output Assembly for MDeep

F1: Create Y labels from metadata
F2: Save matrices (X, c, Y) to .npy files
F3: (Optional) Train/eval split for multi-sample datasets

Usage:
    python assemble_matrices.py --X X_preprocessed.npy --c c.npy --output data/Mai/
"""

import numpy as np
import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MatrixAssembler:
    """
    Assemble final matrices for MDeep training/evaluation.
    """
    
    def __init__(self, X_path, c_path, output_dir, sample_ids_path=None):
        """
        Parameters
        ----------
        X_path : str
            Path to X_preprocessed.npy (samples × OTUs)
        c_path : str
            Path to c.npy (OTU × OTU correlation matrix)
        output_dir : str
            Output directory for final matrices
        sample_ids_path : str, optional
            Path to sample_ids.npy (for metadata extraction)
        """
        self.X_path = Path(X_path)
        self.c_path = Path(c_path)
        self.output_dir = Path(output_dir)
        self.sample_ids_path = Path(sample_ids_path) if sample_ids_path else None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.X = None
        self.c = None
        self.Y = None
        self.sample_ids = None
        
    def load_matrices(self):
        """
        Load preprocessed matrices.
        """
        logger.info(f"Loading X from {self.X_path}")
        self.X = np.load(self.X_path)
        logger.info(f"X shape: {self.X.shape} (samples × OTUs)")
        
        logger.info(f"Loading c from {self.c_path}")
        self.c = np.load(self.c_path)
        logger.info(f"c shape: {self.c.shape} (OTU × OTU)")
        
        # Verify dimensions match
        assert self.X.shape[1] == self.c.shape[0], \
            f"X has {self.X.shape[1]} OTUs but c is {self.c.shape[0]}×{self.c.shape[1]}"
        
        # Load sample IDs if available
        if self.sample_ids_path and self.sample_ids_path.exists():
            logger.info(f"Loading sample IDs from {self.sample_ids_path}")
            self.sample_ids = np.load(self.sample_ids_path, allow_pickle=True)
            logger.info(f"Loaded {len(self.sample_ids)} sample IDs")
        else:
            self.sample_ids = np.array([f"Sample_{i}" for i in range(self.X.shape[0])])
        
        return self
    
    def create_labels_mai(self):
        """
        F1 for Mai: All samples are cancer (Y=1).
        """
        logger.info("Creating labels for Mai dataset (all cancer)")
        n_samples = self.X.shape[0]
        self.Y = np.ones(n_samples, dtype=np.int32)  # All cancer = 1
        logger.info(f"Created Y with shape {self.Y.shape}: all {n_samples} samples marked as cancer (1)")
        return self
    
    def create_labels_from_filenames(self, cancer_keyword='Cancer', control_keyword='Control'):
        """
        F1 for Wu: Extract labels from sample IDs/filenames.
        
        Maps:
        - Samples with cancer_keyword → label 1 (cancer)
        - Samples with control_keyword → label 0 (control)
        
        Parameters
        ----------
        cancer_keyword : str, default 'Cancer'
            Substring indicating cancer sample
        control_keyword : str, default 'Control'
            Substring indicating control sample
        """
        logger.info(f"Creating labels from sample IDs (Cancer→1, Control→0)")
        
        n_samples = self.X.shape[0]
        self.Y = np.zeros(n_samples, dtype=np.int32)
        
        cancer_count = 0
        control_count = 0
        unknown_count = 0
        
        for i, sample_id in enumerate(self.sample_ids):
            sample_id_str = str(sample_id)
            if cancer_keyword in sample_id_str:
                self.Y[i] = 1
                cancer_count += 1
            elif control_keyword in sample_id_str:
                self.Y[i] = 0
                control_count += 1
            else:
                logger.warning(f"Sample {i} ({sample_id_str}) doesn't contain '{cancer_keyword}' or '{control_keyword}'")
                unknown_count += 1
        
        logger.info(f"Label distribution:")
        logger.info(f"  - Cancer (1): {cancer_count}")
        logger.info(f"  - Control (0): {control_count}")
        if unknown_count > 0:
            logger.warning(f"  - Unknown: {unknown_count}")
        
        return self
    
    def verify_matrices(self):
        """
        Verify matrix consistency.
        """
        logger.info("Verifying matrix consistency...")
        
        # Check shapes
        n_samples, n_otus = self.X.shape
        assert self.c.shape == (n_otus, n_otus), f"c shape mismatch: expected ({n_otus}, {n_otus}), got {self.c.shape}"
        assert self.Y.shape == (n_samples,), f"Y shape mismatch: expected ({n_samples},), got {self.Y.shape}"
        assert len(self.sample_ids) == n_samples, f"sample_ids length mismatch"
        
        # Check for NaN/inf
        assert not np.isnan(self.X).any(), "X contains NaN values"
        assert not np.isnan(self.c).any(), "c contains NaN values"
        assert not np.isinf(self.X).any(), "X contains inf values"
        assert not np.isinf(self.c).any(), "c contains inf values"
        
        # Check c is symmetric with diagonal = 1
        assert np.allclose(self.c, self.c.T), "c is not symmetric"
        assert np.allclose(np.diag(self.c), 1.0), "c diagonal is not all 1.0"
        
        # Check c values in [0, 1] (for correlation matrices)
        assert (self.c >= 0).all() and (self.c <= 1).all(), f"c values not in [0,1]: [{self.c.min()}, {self.c.max()}]"
        
        logger.info("✓ All matrix checks passed")
        return self
    
    def save_matrices(self, prefix=''):
        """
        F2: Save final matrices to .npy files.
        
        Parameters
        ----------
        prefix : str, optional
            Prefix for output filenames
        """
        logger.info(f"Saving matrices to {self.output_dir}")
        
        # Save X
        X_path = self.output_dir / f'{prefix}X.npy'
        np.save(X_path, self.X.astype(np.float32))
        logger.info(f"Saved X to {X_path}")
        
        # Save c
        c_path = self.output_dir / f'{prefix}c.npy'
        np.save(c_path, self.c.astype(np.float32))
        logger.info(f"Saved c to {c_path}")
        
        # Save Y
        Y_path = self.output_dir / f'{prefix}Y.npy'
        np.save(Y_path, self.Y.astype(np.int32))
        logger.info(f"Saved Y to {Y_path}")
        
        # Save sample IDs
        sample_ids_path = self.output_dir / f'{prefix}sample_ids.npy'
        np.save(sample_ids_path, self.sample_ids)
        logger.info(f"Saved sample_ids to {sample_ids_path}")
        
        # Save a summary report
        report_path = self.output_dir / f'{prefix}assembly_report.txt'
        with open(report_path, 'w') as f:
            f.write("MDeep Final Assembly Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Input files:\n")
            f.write(f"  X: {self.X_path}\n")
            f.write(f"  c: {self.c_path}\n\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            f.write(f"Matrix dimensions:\n")
            f.write(f"  X (abundance): {self.X.shape[0]} samples × {self.X.shape[1]} OTUs\n")
            f.write(f"  c (correlation): {self.c.shape[0]} × {self.c.shape[1]} OTUs\n")
            f.write(f"  Y (labels): {self.Y.shape[0]} samples\n\n")
            f.write(f"Label distribution:\n")
            unique, counts = np.unique(self.Y, return_counts=True)
            for label, count in zip(unique, counts):
                f.write(f"  Label {label}: {count} samples\n")
            f.write(f"\nX statistics:\n")
            f.write(f"  dtype: {self.X.dtype}\n")
            f.write(f"  min: {self.X.min():.6f}\n")
            f.write(f"  max: {self.X.max():.6f}\n")
            f.write(f"  mean: {self.X.mean():.6f}\n")
            f.write(f"  sparsity: {(self.X == 0).sum() / self.X.size * 100:.2f}%\n")
        logger.info(f"Saved report to {report_path}")
        
        return {
            'X': X_path,
            'c': c_path,
            'Y': Y_path,
            'sample_ids': sample_ids_path,
            'report': report_path
        }
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """
        F3: Split data into train/eval sets (optional, for multi-sample datasets).
        
        Parameters
        ----------
        test_size : float, default 0.2
            Fraction of data for evaluation
        random_state : int, default 42
            Random seed for reproducibility
        """
        logger.info(f"Performing train/eval split with test_size={test_size}")
        
        indices = np.arange(self.X.shape[0])
        
        # Stratified split to maintain label balance
        X_train_idx, X_eval_idx = train_test_split(
            indices, 
            test_size=test_size, 
            stratify=self.Y,
            random_state=random_state
        )
        
        X_train = self.X[X_train_idx]
        X_eval = self.X[X_eval_idx]
        Y_train = self.Y[X_train_idx]
        Y_eval = self.Y[X_eval_idx]
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Eval set: {X_eval.shape[0]} samples")
        
        # Save split matrices
        train_path = self.output_dir / 'X_train.npy'
        eval_path = self.output_dir / 'X_eval.npy'
        y_train_path = self.output_dir / 'Y_train.npy'
        y_eval_path = self.output_dir / 'Y_eval.npy'
        
        np.save(train_path, X_train.astype(np.float32))
        np.save(eval_path, X_eval.astype(np.float32))
        np.save(y_train_path, Y_train.astype(np.int32))
        np.save(y_eval_path, Y_eval.astype(np.int32))
        
        logger.info(f"Saved train/eval splits to {self.output_dir}")
        
        return {
            'X_train': train_path,
            'X_eval': eval_path,
            'Y_train': y_train_path,
            'Y_eval': y_eval_path
        }
    
    def run_pipeline(self, dataset_type='mai', prefix='', split=False):
        """
        Execute full assembly pipeline: F1 → F2 → (F3).
        
        Parameters
        ----------
        dataset_type : str, default 'mai'
            'mai' (all cancer) or 'wu' (cancer/control from filenames)
        prefix : str, optional
            Prefix for output filenames
        split : bool, default False
            Whether to perform train/eval split
        """
        logger.info("="*60)
        logger.info("Starting final matrix assembly pipeline (Steps F1-F2)")
        logger.info("="*60)
        
        self.load_matrices()
        
        # F1: Create labels
        if dataset_type.lower() == 'mai':
            self.create_labels_mai()
        elif dataset_type.lower() == 'wu':
            self.create_labels_from_filenames()
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        # Verify matrices
        self.verify_matrices()
        
        # F2: Save matrices
        output_files = self.save_matrices(prefix=prefix)
        
        # F3: (Optional) Train/eval split
        split_files = None
        if split and self.X.shape[0] > 10:  # Only split if enough samples
            split_files = self.train_test_split()
        
        logger.info("="*60)
        logger.info("Final assembly pipeline completed!")
        logger.info("="*60)
        
        return output_files, split_files


def main():
    """
    CLI interface for matrix assembly.
    
    Example:
        # For Mai (all cancer)
        python assemble_matrices.py \\
            --X Test-data/Mai/after_D_processed/mai_X_preprocessed.npy \\
            --c Test-data/Mai/after_D_processed/mai_c.npy \\
            --sample-ids Test-data/Mai/after_D_processed/mai_sample_ids.npy \\
            --output data/Mai \\
            --dataset-type mai \\
            --prefix mai_
        
        # For Wu (cancer/control)
        python assemble_matrices.py \\
            --X path/to/wu_X_preprocessed.npy \\
            --c path/to/wu_c.npy \\
            --sample-ids path/to/wu_sample_ids.npy \\
            --output data/Wu \\
            --dataset-type wu \\
            --split
    """
    parser = argparse.ArgumentParser(
        description="Assemble final matrices for MDeep (Steps F1-F3)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--X', type=str, required=True,
                       help='Path to X_preprocessed.npy')
    parser.add_argument('--c', type=str, required=True,
                       help='Path to c.npy (correlation matrix)')
    parser.add_argument('--sample-ids', type=str, default=None,
                       help='Path to sample_ids.npy (for label extraction)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--dataset-type', type=str, default='mai',
                       choices=['mai', 'wu'],
                       help='Dataset type for label creation (mai=all cancer, wu=from filenames)')
    parser.add_argument('--prefix', type=str, default='',
                       help='Prefix for output filenames')
    parser.add_argument('--split', action='store_true',
                       help='Perform train/eval split (F3)')
    
    args = parser.parse_args()
    
    assembler = MatrixAssembler(
        X_path=args.X,
        c_path=args.c,
        output_dir=args.output,
        sample_ids_path=args.sample_ids
    )
    
    output_files, split_files = assembler.run_pipeline(
        dataset_type=args.dataset_type,
        prefix=args.prefix,
        split=args.split
    )
    
    print("\n✓ Output files created:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")
    
    if split_files:
        print("\n✓ Train/eval split files:")
        for key, path in split_files.items():
            print(f"  {key}: {path}")


if __name__ == '__main__':
    main()
