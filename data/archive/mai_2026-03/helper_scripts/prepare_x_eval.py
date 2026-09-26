import pandas as pd
import numpy as np
import logging
import os
import sys

# Configure logging to print everything clearly to the terminal
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def calculate_gmpr_size_factors(matrix):
    logging.debug("Starting GMPR calculation...")
    n_samples = matrix.shape[0]
    size_factors = np.zeros(n_samples)

    for i in range(n_samples):
        ratios = []
        for j in range(n_samples):
            if i == j: 
                continue
            valid_otus = (matrix[i, :] > 0) & (matrix[j, :] > 0)
            if np.sum(valid_otus) > 0:
                r = np.median(matrix[i, valid_otus] / matrix[j, valid_otus])
                ratios.append(r)
                
        if len(ratios) > 0:
            size_factors[i] = np.exp(np.mean(np.log(ratios))) 
        else:
            size_factors[i] = 1.0 
    
    logging.debug("Finished GMPR calculation.")
    return size_factors

def prepare_evaluation_data(test_data_path, features_list_path, output_npy_path):
    logging.info("--- Starting Preparation Pipeline ---")
    
    # --- Check if files exist ---
    if not os.path.exists(features_list_path):
        logging.error(f"Blueprint file missing: Could not find {features_list_path}")
        return
    if not os.path.exists(test_data_path):
        logging.error(f"Data file missing: Could not find {test_data_path}")
        return

    # --- 1. Load Blueprint ---
    logging.info(f"Loading feature blueprint from: {features_list_path}")
    with open(features_list_path, 'r') as f:
        target_features = [line.strip() for line in f if line.strip()]
    logging.info(f"Model expects exactly {len(target_features)} features.")

    # --- 2. Load Data ---
    logging.info(f"Loading raw Mai data from: {test_data_path}")
    # skiprows=1 ignores the "# Constructed from biom file" line
    test_df = pd.read_csv(test_data_path, sep='\t', skiprows=1, index_col=0) 
    
    # .T transposes the matrix so Samples are rows and OTUs are columns!
    test_df = test_df.T 
    logging.info(f"Original Mai Data Shape (Transposed): {test_df.shape}")

    # --- 3. Alignment ---
    logging.info("Aligning Mai columns to match the model blueprint...")
    test_aligned = test_df.reindex(columns=target_features, fill_value=0.0)
    logging.info(f"Aligned Mai Data Shape: {test_aligned.shape}")

    # --- 4. GMPR Normalization ---
    logging.info("Applying GMPR Normalization...")
    size_factors = calculate_gmpr_size_factors(test_aligned.values)
    test_norm = test_aligned.div(size_factors, axis=0).fillna(0.0)

    # --- 5. Winsorization ---
    logging.info("Applying 97% Winsorization...")
    quantiles_97 = test_norm.quantile(0.97)
    test_winsorized = test_norm.clip(upper=quantiles_97, axis=1)

    # --- 6. Square Root ---
    logging.info("Applying Square-Root Transformation...")
    test_final = np.sqrt(test_winsorized)

    # --- 7. Save ---
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_npy_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    logging.info(f"Saving final array to: {output_npy_path}")
    final_matrix = test_final.values.astype(np.float32)
    np.save(output_npy_path, final_matrix)
    
    logging.info("--- Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    try:
        logging.info("Script initialized.")
        
        # --- UPDATE THESE PATHS ---
        TEST_DATA_FILE = "data/Mai.tsv" 
        FEATURE_IDS_FILE = "data/3_countries_feature_ids.txt" 
        OUTPUT_NPY_FILE = "data/Mai_3c/X_eval.npy" 
        
        prepare_evaluation_data(TEST_DATA_FILE, FEATURE_IDS_FILE, OUTPUT_NPY_FILE)
        
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")