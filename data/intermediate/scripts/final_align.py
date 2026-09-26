import argparse
import numpy as np
import pandas as pd

def align_matrices(abundance_npz_path, c_matrix_npy_path, c_matrix_ids_npy_path, 
                   output_X_path, output_C_path):
    """
    Loads the final abundance matrix (X) and correlation matrix (C),
    aligns the order of features (OTUs) in both, and saves the final,
    aligned matrices ready for the MDeep modeling script.
    """
    
    # --- Step 1: Load all data components ---
    print("--- Step 1: Loading final data components ---")
    
    # Load the processed abundance data
    abundance_data = np.load(abundance_npz_path, allow_pickle=True)
    # The MDeep code expects Samples x Features, so we transpose here.
    X_original = abundance_data['matrix'].T  # Shape: (Samples, OTUs)
    otu_ids_from_X = abundance_data['feature_ids']
    sample_ids = abundance_data['sample_ids']
    print(f"Loaded X matrix with shape (Samples, OTUs): {X_original.shape}")

    # Load the phylogenetic correlation data
    C_original = np.load(c_matrix_npy_path)
    otu_ids_from_C = np.load(c_matrix_ids_npy_path)
    print(f"Loaded C matrix with shape: {C_original.shape}")
    
    # --- Step 2: Align X to match the order of C ---
    print("\n--- Step 2: Aligning abundance matrix (X) to match the order of C ---")
    
    # Create pandas DataFrames for easy, index-based reordering
    # Features (OTUs) are the columns in df_X
    df_X = pd.DataFrame(X_original, index=sample_ids, columns=otu_ids_from_X)
    
    # The "master" or "target" order will be the order from the C matrix
    target_otu_order = otu_ids_from_C
    
    # Reorder the COLUMNS of X to match the target OTU order
    # This will automatically handle adding/dropping OTUs if the sets aren't identical
    # by using the intersection of the columns.
    df_X_aligned = df_X.reindex(columns=target_otu_order, fill_value=0)
    
    # --- Step 3: Save the final, aligned matrices ---
    # Convert back to NumPy arrays for the model
    X_final = df_X_aligned.to_numpy()
    # C is already in the target order, so we can use the original one
    C_final = C_original
    
    print("\n--- Step 3: Saving final, aligned matrices ---")
    print(f"Final aligned X shape (Samples, OTUs): {X_final.shape}")
    print(f"Final aligned C shape (OTUs, OTUs): {C_final.shape}")
    
    # Save the final matrices with the names the MDeep script expects
    np.save(output_X_path, X_final)
    print(f"Saved final aligned X matrix to {output_X_path}")
    
    np.save(output_C_path, C_final)
    print(f"Saved final aligned C matrix to {output_C_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Align final data for MDeep model.")
    parser.add_argument("--abundance_npz", required=True, help="Path to the processed abundance .npz file (contains X).")
    parser.add_argument("--c_matrix", required=True, help="Path to the C matrix .npy file.")
    parser.add_argument("--c_matrix_ids", required=True, help="Path to the C matrix IDs .npy file.")
    parser.add_argument("--out_X", required=True, help="Output path for the final aligned X matrix (.npy).")
    parser.add_argument("--out_C", required=True, help="Output path for the final aligned C matrix (.npy).")
    args = parser.parse_args()
    
    align_matrices(args.abundance_npz, args.c_matrix, args.c_matrix_ids, args.out_X, args.out_C)