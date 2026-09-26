import argparse
import numpy as np
import skbio

def tree_to_c_matrix(newick_path, output_npy_path, rho):
    """
    Reads a Newick tree file, calculates the patristic distance matrix (D),
    transforms it to the correlation matrix C, and saves it.
    """
    print(f"Reading Newick tree from: {newick_path}")
    
    # Read the tree using scikit-bio
    tree = skbio.TreeNode.read(newick_path)
    
    # Calculate the patristic distance matrix (D)
    print("Calculating patristic distance matrix (D)...")
    distance_matrix_d = tree.tip_tip_distances()
    
    # Get the D matrix as a NumPy array and the corresponding OTU IDs
    d_matrix = distance_matrix_d.data
    feature_ids = np.array(distance_matrix_d.ids) # Get IDs in the correct order
    
    print(f"D matrix shape: {d_matrix.shape}")
    
    # --- The crucial calculation step ---
    print(f"Calculating C matrix with rho = {rho}")
    c_matrix = np.exp(-2 * rho * d_matrix)
    print(f"Successfully calculated C matrix with shape: {c_matrix.shape}")
    
    # Save the final C matrix
    np.save(output_npy_path, c_matrix)
    print(f"Saved NumPy C matrix to: {output_npy_path}")

    # Save the corresponding feature IDs to ensure alignment later
    ids_output_path = output_npy_path.replace('.npy', '_ids.npy')
    np.save(ids_output_path, feature_ids)
    print(f"Saved corresponding feature IDs to: {ids_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate MDeep C matrix from a Newick tree.")
    parser.add_argument("-i", "--input_newick", required=True, help="Path to the input tree.nwk file.")
    parser.add_argument("-o", "--output_npy", required=True, help="Path for the output C matrix .npy file.")
    parser.add_argument("--rho", type=float, default=2.0, help="The rho parameter. Default is 2.0.")
    args = parser.parse_args()
    
    tree_to_c_matrix(args.input_newick, args.output_npy, args.rho)