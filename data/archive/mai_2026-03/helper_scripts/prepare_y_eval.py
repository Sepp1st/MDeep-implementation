import pandas as pd
import numpy as np
import os

def generate_y_eval(test_data_path, output_npy_path):
    print(f"Reading sample order from: {test_data_path}")
    
    # 1. Load the transposed test data just to get the exact Sample IDs in the correct order
    test_df = pd.read_csv(test_data_path, sep='\t', skiprows=1, index_col=0).T
    sample_ids = test_df.index.tolist()
    n_samples = len(sample_ids)
    
    print(f"Found {n_samples} samples.")

    # 2. Create the empty one-hot encoded array (Shape: [24, 2])
    y_eval = np.zeros((n_samples, 2), dtype=np.float32)

    print("Generating 'ALL DISEASE' labels...")
    for i in range(n_samples):
        # Class 1 (Disease) = [0.0, 1.0]
        y_eval[i, 0] = 0.0
        y_eval[i, 1] = 1.0

    # 3. Save the final array
    output_dir = os.path.dirname(output_npy_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(output_npy_path, y_eval)
    print(f"\nSuccess! Saved labels to: {output_npy_path}")
    print(f"Final shape: {y_eval.shape}")

if __name__ == "__main__":
    TEST_DATA_FILE = "data/Mai.tsv" 
    OUTPUT_LABEL_FILE = "data/Mai_3c/Y_eval.npy" # Make sure the casing matches your script (Y vs y)
    
    generate_y_eval(TEST_DATA_FILE, OUTPUT_LABEL_FILE)