"""
Data preprocessing for MDeep following the paper's pipeline:
"A novel deep learning method for predictive modeling of microbiome data"

Steps D1-D6:
  D1: Load BIOM → pandas DataFrame
  D2: Remove outlier samples (IQR-based)
  D3: Remove noisy/less-informative OTUs (prevalence < 10%, median non-zero < 10)
  D4: GMPR normalization (handles zero-inflation)
  D5: Winsorization at 97% quantile
  D6: Square-root transformation
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.mstats import winsorize
import argparse
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MicrobiomePreprocessor:
    """
    Implements the 5-step preprocessing pipeline from MDeep paper.
    """
    
    def __init__(self, biom_path, output_dir=None, sample_labels=None):
        """
        Parameters
        ----------
        biom_path : str
            Path to the BIOM file (JSON format from QIIME2 export)
        output_dir : str, optional
            Directory to save preprocessed matrices. 
            If None, uses same directory as biom_path.
        sample_labels : array-like, optional
            Binary labels (0/1) for each sample in BIOM.
            If provided, saved to output_dir/Y.npy
        """
        self.biom_path = Path(biom_path)
        self.output_dir = Path(output_dir) if output_dir else self.biom_path.parent
        self.sample_labels = sample_labels
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.otu_table = None  # samples × OTUs (after transpose)
        self.otu_ids = None
        self.sample_ids = None
        
    def load_biom(self):
        """
        Step D1: Load BIOM → pandas DataFrame
        
        Handles both JSON and HDF5 BIOM formats.
        Transposes so rows=samples, columns=OTUs.
        """
        logger.info(f"Loading BIOM from {self.biom_path}")
        
        try:
            import biom
        except ImportError:
            raise ImportError(
                "biom package not found. Install with: pip install biom-format"
            )
        
        table = biom.load_table(str(self.biom_path))
        
        # Convert to pandas DataFrame (rows=OTUs, cols=samples by default)
        df = table.to_dataframe().T  # Transpose: rows=samples, cols=OTUs
        
        # Convert sparse to dense if loaded as sparse
        try:
            if hasattr(df, 'sparse'):
                logger.info("Converting sparse DataFrame to dense format")
                df = df.to_numpy()  # Convert to dense numpy, then back to DataFrame
                df = pd.DataFrame(df, index=table.ids(axis='sample'), columns=table.ids(axis='observation'))
            elif any(pd.api.types.is_sparse(df[col]) for col in df.columns):
                logger.info("Converting sparse columns to dense")
                df = df.astype('float64')
        except:
            pass  # If not sparse, continue
        
        self.otu_table = df
        self.sample_ids = df.index.values
        self.otu_ids = df.columns.values
        
        logger.info(f"Loaded OTU table: {df.shape[0]} samples × {df.shape[1]} OTUs")
        return self
    
    def remove_outlier_samples(self):
        """
        Step D2: Remove outlier samples using IQR method
        
        Removes samples with extreme library sizes:
        - Below Q1 - 1.5*IQR
        - Above Q3 + 1.5*IQR
        """
        logger.info("Step D2: Removing outlier samples by library size")
        
        lib_sizes = self.otu_table.sum(axis=1)
        q1 = lib_sizes.quantile(0.25)
        q3 = lib_sizes.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers
        outliers = (lib_sizes < lower_bound) | (lib_sizes > upper_bound)
        n_outliers = outliers.sum()
        
        if n_outliers > 0:
            logger.info(f"Removing {n_outliers} outlier samples (lib size bounds: [{lower_bound:.0f}, {upper_bound:.0f}])")
            self.otu_table = self.otu_table[~outliers]
            if self.sample_labels is not None:
                self.sample_labels = self.sample_labels[~outliers.values]
        else:
            logger.info("No outlier samples detected")
        
        logger.info(f"After outlier removal: {self.otu_table.shape[0]} samples × {self.otu_table.shape[1]} OTUs")
        return self
    
    def filter_otu_features(self, prevalence_threshold=0.10, median_count_threshold=10):    
        """
        Step D3: Remove less informative and noisy OTUs
        
        Removes OTUs where:
        - Prevalence < 10% (present in < 10% of samples)
        - Median non-zero count < 10 (among samples where OTU is present)
        
        Parameters
        ----------
        prevalence_threshold : float, default 0.10
            Minimum prevalence (fraction of samples with count > 0)
        median_count_threshold : int, default 10
            Minimum median non-zero count
        """
        logger.info("Step D3: Filtering OTUs by prevalence and median non-zero count")
        
        n_samples = self.otu_table.shape[0]
        min_prevalence = prevalence_threshold * n_samples
        
        # Calculate prevalence (number of samples with count > 0)
        prevalence = (self.otu_table > 0).sum(axis=0)
        
        # Calculate median non-zero count (handle sparse arrays)
        def median_nonzero(col):
            # Convert sparse to dense if needed
            if hasattr(col, 'sparse'):
                col_values = col.sparse.to_dense().values
            else:
                col_values = col.values
            
            nonzero = col_values[col_values > 0]
            return np.median(nonzero) if len(nonzero) > 0 else 0
        
        median_counts = self.otu_table.apply(median_nonzero, axis=0)
        
        # Filter OTUs
        keep_mask = (prevalence >= min_prevalence) & (median_counts >= median_count_threshold)
        n_removed = (~keep_mask).sum()
        
        logger.info(f"Removing {n_removed} OTUs with prevalence < {prevalence_threshold*100:.1f}% or median count < {median_count_threshold}")
        logger.info(f"  - OTUs failing prevalence: {(prevalence < min_prevalence).sum()}")
        logger.info(f"  - OTUs failing median count: {(median_counts < median_count_threshold).sum()}")
        
        self.otu_table = self.otu_table.loc[:, keep_mask]
        self.otu_ids = self.otu_table.columns.values
        
        logger.info(f"After OTU filtering: {self.otu_table.shape[0]} samples × {self.otu_table.shape[1]} OTUs")
        return self  
    
    def gmpr_normalization(self, verbose=True):
        """
        Step D4: GMPR normalization (Geometric Mean of Pairwise Ratios)
        
        A robust normalization method for handling zero-inflated OTU counts.
        Requires R and the GUniFrac package (install: conda install -c bioconda gunifrac).
        
        Falls back to TSS (Total Sum Scaling) if R/rpy2 not available.
        """
        logger.info("Step D4: GMPR normalization")
        
        try:
            import rpy2
            from rpy2.robjects.packages import importr
            from rpy2.robjects import numpy2ri
            numpy2ri.activate()
            
            # Load GUniFrac package
            try:
                gunifrac = importr('GUniFrac')
            except Exception as e:
                logger.warning(f"GUniFrac R package not found: {e}")
                logger.warning("Falling back to TSS normalization")
                self._tss_normalization()
                return self
            
            # Convert to R-compatible format: OTUs × samples
            counts = self.otu_table.values.astype(int)  # Current: samples × OTUs (24 × 52)
            counts_t = counts.T  # Transpose to OTUs × samples (52 × 24)
            
            logger.info(f"Transposing counts for GMPR: {counts.shape} -> {counts_t.shape}")
            
            # Call GMPR from R (expects OTUs in rows, samples in columns)
            from rpy2.robjects import r
            r.assign('otu_matrix', counts_t)
            gmpr_factors = np.array(r('GUniFrac::GMPR(otu_matrix)'))  # Returns factors for each sample
            
            logger.info(f"GMPR factors shape: {gmpr_factors.shape}, counts shape: {counts.shape}")
            
            # Normalize: gmpr_factors has one value per sample, divide element-wise
            # For each sample i: normalized[i, :] = counts[i, :] / gmpr_factors[i]
            normalized = counts.astype(float) / gmpr_factors[:, np.newaxis]
            
            # Handle division by zero
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.otu_table = pd.DataFrame(normalized, 
                                         index=self.otu_table.index,
                                         columns=self.otu_table.columns)
            
            logger.info("GMPR normalization completed successfully")
            
        except ImportError:
            logger.warning("rpy2 not found. Falling back to TSS (Total Sum Scaling) normalization")
            logger.warning("For production, install: pip install rpy2")
            logger.warning("and R package: conda install -c bioconda gunifrac")
            self._tss_normalization()
        
        return self
    
    def _tss_normalization(self):
        """
        Fallback: Total Sum Scaling normalization (simple rarefaction alternative).
        """
        logger.info("Applying TSS normalization (fallback)")
        lib_sizes = self.otu_table.sum(axis=1)
        self.otu_table = self.otu_table.div(lib_sizes, axis=0)  # Normalize by row sum
        logger.info("TSS normalization completed")
    
    def winsorize_outlier_counts(self, quantile=0.97):
        """
        Step D5: Winsorization at 97% quantile
        
        Caps outlier counts to reduce the influence of extreme values.
        Replaces values above the 97th percentile with the 97th percentile value.
        
        Parameters
        ----------
        quantile : float, default 0.97
            Quantile threshold for capping (0.97 = 97%)
        """
        logger.info(f"Step D5: Winsorization at {quantile*100:.0f}% quantile")
        
        # Apply winsorization per OTU (column)
        # limits=(0, 1-quantile) means keep lower limita, cap upper (1-0.97=0.03)
        winsorized = np.zeros_like(self.otu_table.values, dtype=float)
        
        for i, col in enumerate(self.otu_table.columns):
            col_data = self.otu_table[col].values
            # winsorize returns 1D array
            winsorized[:, i] = winsorize(col_data, limits=[0, 1 - quantile])
        
        self.otu_table = pd.DataFrame(winsorized,
                                     index=self.otu_table.index,
                                     columns=self.otu_table.columns)
        
        logger.info("Winsorization completed")
        return self
    
    def sqrt_transform(self):
        """
        Step D6: Square-root transformation
        
        Reduces the influence of highly abundant taxa by applying sqrt to all counts.
        """
        logger.info("Step D6: Square-root transformation")
        
        self.otu_table = np.sqrt(self.otu_table)
        
        logger.info("Square-root transformation completed")
        return self
    
    def save_matrices(self, prefix=''):
        """
        Save preprocessed matrices to .npy files.
        
        Parameters
        ----------
        prefix : str, optional
            Prefix for output filenames (e.g., 'X_' → 'X_preprocessed.npy')
        
        Returns
        -------
        dict
            Paths to saved files
        """
        logger.info(f"Saving matrices to {self.output_dir}")
        
        # Save abundance matrix
        X_path = self.output_dir / f'{prefix}X_preprocessed.npy'
        np.save(X_path, self.otu_table.values.astype(np.float32))
        logger.info(f"Saved X (abundance matrix) to {X_path}")
        
        # Save sample IDs
        sample_ids_path = self.output_dir / f'{prefix}sample_ids.npy'
        np.save(sample_ids_path, self.sample_ids)
        
        # Save OTU IDs
        otu_ids_path = self.output_dir / f'{prefix}otu_ids.npy'
        np.save(otu_ids_path, self.otu_ids)
        logger.info(f"Saved OTU IDs ({len(self.otu_ids)} OTUs) to {otu_ids_path}")
        
        # Save labels if provided
        if self.sample_labels is not None:
            Y_path = self.output_dir / f'{prefix}Y.npy'
            np.save(Y_path, self.sample_labels.astype(np.int32))
            logger.info(f"Saved Y (labels) to {Y_path}")
        
        # Save preprocessing report
        report_path = self.output_dir / f'{prefix}preprocessing_report.txt'
        with open(report_path, 'w') as f:
            f.write("MDeep Preprocessing Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Input BIOM: {self.biom_path}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Final shape: {self.otu_table.shape[0]} samples × {self.otu_table.shape[1]} OTUs\n")
            f.write(f"Data type: {self.otu_table.values.dtype}\n")
            f.write(f"Value range: [{self.otu_table.values.min():.6f}, {self.otu_table.values.max():.6f}]\n")
            f.write(f"Sparsity: {(self.otu_table == 0).sum().sum() / self.otu_table.size * 100:.2f}%\n")
        logger.info(f"Saved preprocessing report to {report_path}")
        
        return {
            'X': X_path,
            'sample_ids': sample_ids_path,
            'otu_ids': otu_ids_path,
            'Y': self.output_dir / f'{prefix}Y.npy' if self.sample_labels is not None else None,
            'report': report_path
        }
    
    def run_pipeline(self, output_prefix=''):
        """
        Execute the full preprocessing pipeline: D1–D6.
        
        Parameters
        ----------
        output_prefix : str, optional
            Prefix for output files
        
        Returns
        -------
        dict
            Paths to saved outputs
        """
        logger.info("="*60)
        logger.info("Starting MDeep preprocessing pipeline (Steps D1-D6)")
        logger.info("="*60)
        
        self.load_biom()
        self.remove_outlier_samples()
        self.filter_otu_features(prevalence_threshold=0.10, median_count_threshold=10)
        self.gmpr_normalization()
        self.winsorize_outlier_counts(quantile=0.97)
        self.sqrt_transform()
        
        output_paths = self.save_matrices(prefix=output_prefix)
        
        logger.info("="*60)
        logger.info("Preprocessing pipeline completed!")
        logger.info("="*60)
        
        return output_paths


def main():
    """
    CLI interface for preprocessing BIOM files.
    
    Example:
        python preprocess.py --biom Test-data/Mai/mai_exported/feature-table.biom \\
                            --output data/Mai \\
                            --labels 24 1 0 0 1 1 ... (24 labels for Mai samples)
    """
    parser = argparse.ArgumentParser(
        description="MDeep Preprocessing Pipeline (Steps D1-D6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess Mai BIOM
  python preprocess.py --biom feature-table.biom --output data/Mai
  
  # With sample labels
  python preprocess.py --biom feature-table.biom --output data/Mai --labels-file labels.txt
        """
    )
    
    parser.add_argument('--biom', type=str, required=True,
                       help='Path to BIOM file (JSON format from QIIME2)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: same as BIOM parent dir)')
    parser.add_argument('--labels-file', type=str, default=None,
                       help='Path to file with sample labels (one per line, 0/1)')
    parser.add_argument('--prefix', type=str, default='',
                       help='Prefix for output filenames (default: no prefix)')
    
    args = parser.parse_args()
    
    # Load labels if provided
    labels = None
    if args.labels_file:
        logger.info(f"Loading labels from {args.labels_file}")
        labels = np.loadtxt(args.labels_file, dtype=int)
    
    # Run preprocessing
    preprocessor = MicrobiomePreprocessor(
        biom_path=args.biom,
        output_dir=args.output,
        sample_labels=labels
    )
    
    output_paths = preprocessor.run_pipeline(output_prefix=args.prefix)
    
    print("\nOutput files saved:")
    for key, path in output_paths.items():
        if path is not None:
            print(f"  {key}: {path}")


if __name__ == '__main__':
    main()