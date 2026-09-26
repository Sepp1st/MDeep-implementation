import argparse
import biom
import pandas as pd
import numpy as np
from skbio.diversity import beta_diversity
from pygmp import gmpr

def merge_and_preprocess(biom_paths, output_npz_path, output_ids_path):
    """
    Loads multiple BIOM tables, merges them, and applies the 5-step MDeep
    preprocessing pipeline (as per Chen et al., 2018).
    
    Saves the final processed abundance matrix (X) and the list of
    surviving OTU IDs.
    """
    
    # --- Part 1: Load and Merge BIOM Tables ---
    print("--- Step 0: Loading and Merging BIOM tables ---")
    all_dfs = []
    for path in biom_paths:
        try:
            table = biom.load_table(path)
            df = table.to_dataframe(dense=True)
            all_dfs.append(df)
            print(f"Loaded {path} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

    # Concatenate along columns (samples), filling missing OTUs with 0
    merged_df = pd.concat(all_dfs, axis=1).fillna(0).astype(np.int64)
    print(f"\nSuccessfully merged. Raw merged table shape (OTUs x Samples): {merged_df.shape}")

    # --- Part 2: MDeep Preprocessing Pipeline ---
    
    # (i) Removing outlier samples
    print("\n--- Step 1: Filtering outlier samples using Bray-Curtis distance ---")
    # beta_diversity expects Samples x Features, so we transpose
    df_for_beta = merged_df.loc[:, merged_df.sum(axis=0) > 0].T
    
    if df_for_beta.shape[0] < 2:
        print("Not enough samples to calculate distances. Skipping sample filtering.")
        df_step1 = merged_df.copy()
    else:
        bc_matrix = beta_diversity("braycurtis", df_for_beta.values, df_for_beta.index)
        bc_df = bc_matrix.to_data_frame()
        
        mj = bc_df.median(axis=1)
        median_mj = mj.median()
        oj = mj / median_mj
        
        outlier_samples = oj[oj > 2].index
        print(f"Found {len(outlier_samples)} outlier samples to remove.")
        
        df_step1 = merged_df.drop(columns=outlier_samples, errors='ignore')
    
    print(f"Shape after sample filtering: {df_step1.shape}")

    # (ii) Removing less informative and noisy OTUs
    print("\n--- Step 2: Filtering OTUs by prevalence and abundance ---")
    n_samples = df_step1.shape[1]
    
    # Filter 1: Prevalence >= 10%
    prevalence_mask = (df_step1 > 0).sum(axis=1) >= (n_samples * 0.10)
    
    # Filter 2: Median non-zero counts >= 10
    df_nonzero = df_step1.replace(0, np.nan)
    median_mask = df_nonzero.median(axis=1) >= 10
    
    final_otu_mask = prevalence_mask & median_mask
    df_step2 = df_step1[final_otu_mask]
    
    print(f"Kept {final_otu_mask.sum()} OTUs out of {len(final_otu_mask)} after filtering.")
    print(f"Shape after OTU filtering: {df_step2.shape}")

    # (iii) Normalizing using GMPR
    print("\n--- Step 3: Normalizing counts using GMPR ---")
    # gmpr function expects Samples x Features numpy array
    otu_table_for_gmpr = df_step2.T.values
    gmpr_size_factors = gmpr(otu_table_for_gmpr)
    
    # The paper states: "normalized counts were then divided by sj"
    df_step3 = df_step2.div(gmpr_size_factors, axis=1)
    print("Normalization complete.")

    # (iv) Replacing outlier counts using winsorization
    print("\n--- Step 4: Replacing outlier counts using Winsorization (97th percentile) ---")
    quantiles_97 = df_step3.quantile(0.97, axis=1)
    df_step4 = df_step3.clip(upper=quantiles_97, axis=0)
    print("Winsorization complete.")

    # (v) Reducing influence by square-root transformation
    print("\n--- Step 5: Applying square-root transformation ---")
    df_final = np.sqrt(df_step4)
    print("Transformation complete.")

    # --- Part 3: Save Final Outputs ---
    print(f"\nFinal processed matrix shape: {df_final.shape}")
    final_X_matrix = df_final.to_numpy()
    final_sample_ids = df_final.columns.to_numpy()
    final_feature_ids = df_final.index.to_numpy()

    # Save the processed abundance matrix X and corresponding IDs
    np.savez_compressed(
        output_npz_path,
        matrix=final_X_matrix,
        sample_ids=final_sample_ids,
        feature_ids=final_feature_ids
    )
    print(f"Successfully saved final X matrix to {output_npz_path}")

    # Save the list of surviving OTU IDs for QIIME 2 tree-building step
    with open(output_ids_path, "w") as f:
        f.write("#OTUID\n")  # This header is required by QIIME 2
        for feature_id in final_feature_ids:
            f.write(f"{feature_id}\n")
    print(f"Successfully saved list of final feature IDs to {output_ids_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge multiple BIOM tables and apply the MDeep preprocessing pipeline."
    )
    parser.add_argument(
        "-b", "--biom_files", 
        nargs='+',  # This allows multiple input files
        required=True, 
        help="Space-separated list of paths to input .biom files."
    )
    parser.add_argument(
        "-o", "--output_npz", 
        required=True, 
        help="Path for the output .npz file containing the final X matrix."
    )
    parser.add_argument(
        "-f", "--feature_ids_txt",
        required=True,
        help="Path for the output .txt file listing the final filtered feature IDs."
    )
    args = parser.parse_args()
    
    merge_and_preprocess(args.biom_files, args.output_npz, args.feature_ids_txt)