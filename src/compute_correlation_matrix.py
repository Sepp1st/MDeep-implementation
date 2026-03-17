"""
Step E: Compute phylogenetic correlation matrix C from Newick tree.

E1: Patristic distance matrix D
E2: D → C transformation (distance to correlation)
E3: Align indices with preprocessed OTU matrix

Usage:
    python compute_correlation_matrix.py --tree tree.nwk --otu-ids otu_ids.npy --output c.npy
"""

import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PhylogeneticCorrelationMatrix:
    """
    Compute phylogenetic correlation matrix from Newick tree.
    """
    
    def __init__(self, tree_path, otu_ids=None):
        """
        Parameters
        ----------
        tree_path : str
            Path to Newick format tree file (from QIIME2 export)
        otu_ids : array-like, optional
            Filtered OTU IDs from preprocessing (order matters!)
            If provided, subsets tree to only include these OTUs
        """
        self.tree_path = Path(tree_path)
        self.otu_ids = np.array(otu_ids) if otu_ids is not None else None
        self.tree = None
        self.distance_matrix = None
        self.correlation_matrix = None
        self.ordered_tips = None
        
    def load_tree(self):
        """
        Step E1a: Load Newick tree using skbio
        """
        logger.info(f"Loading Newick tree from {self.tree_path}")
        
        try:
            import skbio
        except ImportError:
            raise ImportError("skbio not found. Install with: pip install scikit-bio")
        
        self.tree = skbio.TreeNode.read(str(self.tree_path), format='newick')
        tip_list = list(self.tree.tips())
        logger.info(f"Loaded tree with {len(tip_list)} tips")
        
        return self
    
    def get_tree_tip_names(self):
        """
        Extract tip (OTU) names from tree in order.
        """
        tip_names = [tip.name for tip in self.tree.tips()]
        logger.info(f"Tree tip names (first 10): {tip_names[:10]}")
        return tip_names
    
    def subset_tree_to_otus(self):
        """
        Filter tree to keep only OTUs in self.otu_ids.
        
        This ensures the tree matches the preprocessed abundance matrix.
        Note: This approach rebuilds a minimal tree connecting only the target OTUs.
        """
        if self.otu_ids is None:
            logger.info("No OTU filtering: using all tree tips")
            return self
        
        logger.info(f"Subsetting tree to {len(self.otu_ids)} filtered OTUs")
        
        current_tips = list(self.tree.tips())
        current_tip_names = set([tip.name for tip in current_tips])
        otu_ids_set = set(self.otu_ids)
        
        # Find common OTUs
        common_otus = current_tip_names & otu_ids_set
        logger.info(f"Common OTUs between tree and filtered set: {len(common_otus)}")
        
        if len(common_otus) < len(self.otu_ids):
            missing = otu_ids_set - current_tip_names
            logger.warning(f"OTUs in filtered set but not in tree ({len(missing)}): {list(missing)[:10]}")
        
        # For simplicity, don't actually prune the tree - just work with common OTUs
        # Use tree_prune_to_tips from skbio
        try:
            from skbio.tree import TreeNode
            # Get list of tips to keep (those in common_otus)
            tips_to_keep = [tip for tip in current_tips if tip.name in common_otus]
            tip_names_to_keep = [tip.name for tip in tips_to_keep]
            
            logger.info(f"Keeping {len(tip_names_to_keep)} tips, pruning internal nodes...")
            
            # Create a new tree pruned to these tips
            # Method: copy tree and remove others
            self.tree = self.tree.copy()
            tips_to_remove = [tip for tip in list(self.tree.tips()) if tip.name not in tip_names_to_keep]
            
            # Remove tips and clean up empty internal nodes
            for tip in tips_to_remove:
                if tip.parent is not None:
                    tip.parent.remove(tip)
            
            # Remove internal nodes with only one child (simplify)
            self._simplify_tree()
            
        except Exception as e:
            logger.warning(f"Tree subsetting encountered issue: {e}. Working with all tips.")
        
        remaining_tips = list(self.tree.tips())
        logger.info(f"After subsetting: tree has {len(remaining_tips)} tips")
        return self
    
    def _simplify_tree(self):
        """Remove internal nodes with only one child."""
        def _has_single_child(node):
            return len(node.children) == 1
        
        def _remove_single_children(node):
            # Recursively process children
            for child in list(node.children):
                _remove_single_children(child)
            
            # If this node has only one child, remove it
            if len(node.children) == 1 and node.parent is not None:
                child = node.children[0]
                # Transfer child's branch length to parent
                if child.length is not None and node.length is not None:
                    child.length = child.length + node.length
                elif child.length is None and node.length is not None:
                    child.length = node.length
                
                # Remove this node from parent
                node.parent.remove(node)
                node.parent.add_child(child)
        
        _remove_single_children(self.tree)
    
    def compute_patristic_distances(self):
        """
        Step E1b: Compute patristic distance matrix D from tree.
        
        Patristic distance = sum of branch lengths between two tips.
        """
        logger.info("Computing patristic distance matrix")
        
        try:
            import skbio
        except ImportError:
            raise ImportError("skbio not found")
        
        # Get tips in consistent order
        tips = list(self.tree.tips())
        tip_names = [tip.name for tip in tips]
        n_tips = len(tips)
        
        logger.info(f"Computing distances for {n_tips} tips")
        
        # Initialize distance matrix
        D = np.zeros((n_tips, n_tips))
        
        # Compute pairwise distances
        for i, tip_i in enumerate(tips):
            for j, tip_j in enumerate(tips):
                if i == j:
                    D[i, j] = 0.0
                elif i < j:
                    # Compute distance using tip_i.distance(tip_j)
                    d = tip_i.distance(tip_j)
                    D[i, j] = d
                    D[j, i] = d  # Symmetric
        
        self.distance_matrix = D
        self.ordered_tips = np.array(tip_names)
        
        logger.info(f"Distance matrix shape: {D.shape}")
        logger.info(f"Distance range: [{D.min():.6f}, {D.max():.6f}]")
        
        return self
    
    def distance_to_correlation(self, method='exponential', alpha=0.5):
        """
        Step E2: Convert distance matrix D to correlation matrix C.
        
        Methods:
        - 'exponential': C[i,j] = exp(-alpha * D[i,j])  (default, smooth decay)
        - 'linear': C[i,j] = 1 - D[i,j] / max(D)        (linear scaling to [0,1])
        - 'inverse': C[i,j] = 1 / (1 + alpha * D[i,j])  (inverse relationship)
        
        Parameters
        ----------
        method : str, default 'exponential'
            Transformation method
        alpha : float, default 0.5
            Parameter for exponential/inverse methods.
            For exponential: larger alpha → stronger decay with distance
            For inverse: controls scaling
        """
        logger.info(f"Converting distance to correlation using '{method}' method")
        
        D = self.distance_matrix
        
        if method == 'exponential':
            # C[i,j] = exp(-alpha * D[i,j])
            # When D=0: C=1, When D→∞: C→0
            C = np.exp(-alpha * D)
            logger.info(f"Using exponential: C = exp(-{alpha} * D)")
            
        elif method == 'linear':
            # C[i,j] = 1 - D[i,j] / max(D)
            # Normalize to [0, 1] range
            max_d = D.max()
            if max_d > 0:
                C = 1.0 - (D / max_d)
            else:
                C = np.ones_like(D)
            logger.info(f"Using linear: C = 1 - D / {max_d:.6f}")
            
        elif method == 'inverse':
            # C[i,j] = 1 / (1 + alpha * D[i,j])
            # Smooth inverse relationship
            C = 1.0 / (1.0 + alpha * D)
            logger.info(f"Using inverse: C = 1 / (1 + {alpha} * D)")
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'exponential', 'linear', or 'inverse'")
        
        # Ensure symmetric and diagonal = 1
        C = (C + C.T) / 2  # Ensure symmetry
        np.fill_diagonal(C, 1.0)  # Self-correlation = 1
        
        # Clip to [0, 1] to handle numerical errors
        C = np.clip(C, 0.0, 1.0)
        
        self.correlation_matrix = C
        
        logger.info(f"Correlation matrix shape: {C.shape}")
        logger.info(f"Correlation range: [{C.min():.6f}, {C.max():.6f}]")
        logger.info(f"Diagonal (self-correlation): {np.diag(C)[:5]}... (all should be 1.0)")
        
        return self
    
    def align_with_otu_ids(self, otu_ids):
        """
        Step E3: Align correlation matrix with preprocessed OTU IDs.
        
        Reorders rows/columns of C to match the order of otu_ids from preprocessing.
        
        Parameters
        ----------
        otu_ids : array-like
            OTU IDs from preprocessing (X_preprocessed.npy columns)
        
        Returns
        -------
        np.ndarray
            Reordered correlation matrix
        """
        logger.info("Step E3: Aligning correlation matrix with preprocessed OTU IDs")
        
        otu_ids = np.array(otu_ids)
        logger.info(f"Aligning {len(otu_ids)} OTU IDs from preprocessing")
        logger.info(f"First 10 OTU IDs: {otu_ids[:10]}")
        logger.info(f"First 10 tree tips: {self.ordered_tips[:10]}")
        
        # Create mapping from tree tip name to index in C
        tip_to_idx = {name: i for i, name in enumerate(self.ordered_tips)}
        
        # Create reordering indices
        reorder_idx = []
        missing_otus = []
        for otu_id in otu_ids:
            if otu_id in tip_to_idx:
                reorder_idx.append(tip_to_idx[otu_id])
            else:
                missing_otus.append(otu_id)
        
        if missing_otus:
            logger.warning(f"OTUs not found in tree ({len(missing_otus)}): {missing_otus[:10]}")
            if len(missing_otus) > len(otu_ids) * 0.5:
                raise ValueError(f"Too many OTUs missing from tree: {len(missing_otus)}/{len(otu_ids)}")
        
        reorder_idx = np.array(reorder_idx)
        
        # Reorder correlation matrix
        C_aligned = self.correlation_matrix[np.ix_(reorder_idx, reorder_idx)]
        
        logger.info(f"Aligned correlation matrix shape: {C_aligned.shape}")
        assert C_aligned.shape == (len(otu_ids), len(otu_ids)), "Shape mismatch after alignment"
        
        return C_aligned
    
    def save_correlation_matrix(self, output_path, otu_ids=None):
        """
        Save correlation matrix to .npy file.
        
        If otu_ids provided, aligns matrix first.
        
        Parameters
        ----------
        output_path : str
            Path to save c.npy
        otu_ids : array-like, optional
            OTU IDs to align with
        """
        if otu_ids is not None:
            C = self.align_with_otu_ids(otu_ids)
        else:
            C = self.correlation_matrix
        
        np.save(output_path, C.astype(np.float32))
        logger.info(f"Saved correlation matrix to {output_path}")
        logger.info(f"C shape: {C.shape}, dtype: {C.dtype}")
    
    def run_pipeline(self, otu_ids=None, output_path=None, method='exponential', alpha=0.5):
        """
        Execute full pipeline: E1 → E2 → E3.
        
        Parameters
        ----------
        otu_ids : array-like, optional
            Filtered OTU IDs from preprocessing
        output_path : str, optional
            Path to save c.npy
        method : str, default 'exponential'
            Distance-to-correlation method
        alpha : float, default 0.5
            Transformation parameter
        """
        logger.info("="*60)
        logger.info("Starting phylogenetic correlation matrix pipeline (Steps E1-E3)")
        logger.info("="*60)
        
        self.load_tree()
        if otu_ids is not None:
            self.otu_ids = otu_ids
            self.subset_tree_to_otus()
        self.compute_patristic_distances()
        self.distance_to_correlation(method=method, alpha=alpha)
        
        if output_path:
            self.save_correlation_matrix(output_path, otu_ids=otu_ids)
        
        logger.info("="*60)
        logger.info("Correlation matrix pipeline completed!")
        logger.info("="*60)
        
        return self.correlation_matrix


def main():
    """
    CLI interface for computing correlation matrix.
    
    Example:
        python compute_correlation_matrix.py \\
            --tree tree.nwk \\
            --otu-ids otu_ids.npy \\
            --output c.npy \\
            --method exponential \\
            --alpha 0.5
    """
    parser = argparse.ArgumentParser(
        description="Compute phylogenetic correlation matrix (Steps E1-E3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python compute_correlation_matrix.py --tree tree.nwk --otu-ids otu_ids.npy --output c.npy
  
  # With custom transformation (linear normalization)
  python compute_correlation_matrix.py \\
    --tree tree.nwk --otu-ids otu_ids.npy --output c.npy \\
    --method linear
    
  # Exponential with custom alpha
  python compute_correlation_matrix.py \\
    --tree tree.nwk --otu-ids otu_ids.npy --output c.npy \\
    --method exponential --alpha 1.0
        """
    )
    
    parser.add_argument('--tree', type=str, required=True,
                       help='Path to Newick format tree file')
    parser.add_argument('--otu-ids', type=str, default=None,
                       help='Path to otu_ids.npy from preprocessing (for subsetting/reordering)')
    parser.add_argument('--output', type=str, default='c.npy',
                       help='Output path for correlation matrix (default: c.npy)')
    parser.add_argument('--method', type=str, default='exponential', 
                       choices=['exponential', 'linear', 'inverse'],
                       help='Distance-to-correlation transformation method')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Parameter for exponential/inverse transformation (default: 0.5)')
    
    args = parser.parse_args()
    
    # Load OTU IDs if provided
    otu_ids = None
    if args.otu_ids:
        otu_ids = np.load(args.otu_ids, allow_pickle=True)
        logger.info(f"Loaded {len(otu_ids)} OTU IDs from {args.otu_ids}")
    
    # Compute correlation matrix
    pcm = PhylogeneticCorrelationMatrix(tree_path=args.tree, otu_ids=otu_ids)
    C = pcm.run_pipeline(otu_ids=otu_ids, output_path=args.output, 
                         method=args.method, alpha=args.alpha)
    
    print(f"\nCorrelation matrix saved to: {args.output}")
    print(f"Shape: {C.shape}")
    print(f"Value range: [{C.min():.4f}, {C.max():.4f}]")


if __name__ == '__main__':
    main()
