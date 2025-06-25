#!/usr/bin/env python3
'''
torus_nmf.py  - this script will attempt to utilize nonnegative matrix
                factorization to identify subtypes of torus data. Torus
                coordinate data is arrange in a matrix for NMF to categorize;
                resulting encoding matrix is clustered to determine if distinct
                subtypes can be determined
                
                Torus data is created/retrieved in the following ways:
                 -> .npy file exists:
                        load matrix T from file
                 -> no .npy, but .ply files found:
                        load all .ply files, arrange tori data into T matrix,
                        save T to .npy file
                 -> no .ply files: 
                        generate all .ply files using the torus_gen.py script,
                        repeat above
                
                Using tori data from .npy file (generates if none exist), this 
                script arranges torus data in a matrix and computes NMF:
                    
                        T is approximated by WH = V

                    T = original matrix; torus coordinate data represented in
                        each row via [x1, y1, z1, x2, y2, z2, ..., xm, ym, zm]
                        dimension n x 3m matrix, n tori, m coordinates in each
                    W = encoding matrix for torus data: deduced via NMF to
                        show latent subtype representation of each torus
                        dimension n x k, n tori, k optimal rank of encoding
                    H = component matrix of torus data: latent basis shape of 
                        coordinate data
                        dimension k x 3m
                    V = approximation of T matrix from derived W & H
                        dimension n x 3m, same as T
                    

                Note: credit for original torus generation can be given to Khoi
                and his work in continuous variation in torus generation:
                https://github.com/Khoi-Nguyen-Xuan/Torus_Bump_Generation

Authors:        Benji Lawrence
Last Modified:  Jun 25, 2025
'''
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.decomposition import NMF, PCA
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
import os
import csv
import torus_gen
import trimesh
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pyvista as pv
from collections import defaultdict
import pickle
from datetime import datetime
from opnmf.model import OPNMF

def compute_torus_nmf (nmf_mode='opnmf', optimal_r=None):
    '''
    Program functionality executed here:
        - reads files and/or generates torus data
            - saves generated matrix T if needed
        - TODO: computes rank k for decomposition matrix
        - TODO: initializes W matrix - multiple methods possible
        - computes NMF of matrix - saves W, H, & V
    '''
    # load data - check if files exist and load accordingly
    T_file = "./saved_data/T.npy"
    torus_labels = defaultdict(int)
    filenames = list()
    if (os.path.isfile(T_file)):
        T = np.load(T_file, allow_pickle=True)
        with open("./saved_data/filenames.npy", "rb") as f:
            filenames = pickle.load(f)
        with open("./saved_data/labels.npy", "rb") as f:
            torus_labels = pickle.load(f)
    else:
        T = create_T_matrix(T_file, torus_labels, filenames)
    
    print(f"T Matrix loaded:\n{T}\nShape: {T.shape}")
    # optimal matrix rank for encoding matrix W
    # TODO: algorithm for determining optimal rank?
    optimal_r = 3
 
    # run nmf algorithm with selected mode
    nmf_mode = "opnmf"
    W, H, V = run_nmf(T, optimal_r, mode=nmf_mode)
    
    print(f"W Matrix computed:\n{W}\nShape: {W.shape}")
    print(f"H Matrix computed:\n{H}\nShape: {H.shape}")
    print(f"V Matrix computed:\n{V}\nShape: {V.shape}")
    subtype_labels = W.argmax(axis=1) 
    
    print(f"T values - Min: {np.min(T)}, Max: {np.max(T)}, MeanL {np.mean(T)}")
    visualize_nmf_torus(W, H)
    evaluate_nmf_labels(T, W, H, torus_labels, filenames)

    
        
def create_T_matrix(matrix_name, labels, filenames):
    ''' 
    Creates T matrix using signed displacement values (relative to the first torus in the dataset).
    Saves matrix as .npy file for later use and returns it.
    '''
    # retrieve torus data - generate if does not exist
    torus_dir_str = os.path.expanduser("./torus_data/")
    matrix = None       # initialized - adjusted to np.array on first loop
    torus_dir = os.fsencode(torus_dir_str)
    if not (os.path.exists(os.path.join(torus_dir_str, "torus_000.ply"))):
        print("No torus files - generating...")
        torus_gen.generate_torus()
    
    ply_files = sorted([
        f for f in os.listdir(torus_dir)
        if f.endswith(b'.ply')
    ])

    # Iterate through nmf_ground_truth files to disregard for T matrix
    gt = True
    while(gt):
        ref_filepath = os.path.join(torus_dir_str, os.fsdecode(ply_files[0]))
        if "groundtruth_" not in ref_filepath:
            gt = False
        else:
            ply_files.pop(0)

    # Standard torus loaded from above action
    ref_filepath = os.path.join(torus_dir_str, os.fsdecode(ply_files[0]))
    print(f"Loading standard torus (reference): {ref_filepath}")
    try:
        ref_mesh = trimesh.load_mesh(ref_filepath)
        verts_standard = ref_mesh.vertices
        normals_standard = ref_mesh.vertex_normals
    except Exception as e:
        print(f"Could not load reference torus: {e}")
        return None
    
    # iterate through torus files, append to matrix
    for file in ply_files[1:]:
        filename = os.fsdecode(file)
        filepath = os.path.join(torus_dir_str, filename)
        print(f"Reading file: {filepath}")
        try:
            mesh = trimesh.load_mesh(filepath)
            verts_warped = mesh.vertices
            
            # normalize thickness to reduce influence
            # verts_warped = normalize_torus_thickness(verts_warped)

            # determines signed displacements - uses only positive values
            displacement_vectors = verts_warped - verts_standard
            signed_displacements = np.einsum('ij,ij->i', displacement_vectors, normals_standard)
            #print("Signed displacements:\n", signed_displacements)
            signed_displacements = np.clip(signed_displacements, 0, None)
            
            # assign vertex colours - for visualization
            mesh = colour_mesh_vertices(mesh, signed_displacements)
            mesh.export(filepath)
        
            # assign to matrix
            if matrix is None:
                matrix = signed_displacements.reshape(1, -1)  # reshape to 2D array
            else:
                matrix = np.vstack((matrix, signed_displacements))

            # log filename->label map in dictionary for later label comparison
            label = int(filename.split('_')[-1][0]) # isolate group value from filename
            labels[filename] = label
            filenames.append(filename)

        except Exception as e:
            print(f"Could not load/save {filepath}: {e}")

    if matrix is not None:
        os.makedirs("./saved_data/", exist_ok=True)
        np.save(matrix_name, matrix)
        with open("./saved_data/labels.npy", "wb") as f:
            pickle.dump(labels, f)
        with open("./saved_data/filenames.npy", "wb") as f:
            pickle.dump(filenames, f)
    return matrix


def colour_mesh_vertices(mesh, displacements):
    '''
    Assigns displacement colours to each mesh vertex - saves result to filepath
    '''
    norm = Normalize(vmin=displacements.min(), vmax=displacements.max())
    colourmap = cm.get_cmap('plasma') 
    colours = (colourmap(norm(displacements))[:, :3] * 255).astype(np.uint8)  # drop alpha
    mesh.visual.vertex_colors = colours
    return mesh


def visualize_nmf_torus(W, H, ref_path="./torus_data/torus_000.ply", out_path="./results/"):
    ''' 
    Visualizes NMF component heatmaps on the reference torus mesh without
    altering the geometry. Exports colored meshes for each component and
    an averaged heatmap across components.
    '''
    os.makedirs(out_path, exist_ok=True)
    
    ref_mesh = trimesh.load_mesh(ref_path)
    ref_verts = ref_mesh.vertices
    ref_normals = ref_mesh.vertex_normals
    faces = ref_mesh.faces
    num_components, n_vertices = H.shape

    assert len(ref_verts) == n_vertices, "Mesh vertex and H features MISMATCH"
    
    for i in range(num_components):
        displacements = H[i, :]  # shape: (n_vertices,)
        
        heatmap_mesh = ref_mesh.copy()
        heatmap_mesh = colour_mesh_vertices(heatmap_mesh, displacements)
        heatmap_mesh.export(os.path.join(out_path, f"nmf_component_{i}.ply"))

    # Combined component heatmap
    combined_displacements = np.max(H, axis=0)
    combined_mesh = ref_mesh.copy()
    combined_mesh = colour_mesh_vertices(combined_mesh, combined_displacements)
    combined_mesh.export(os.path.join(out_path, "nmf_combined.ply"))

def run_nmf(T, rank, mode="nmf", init="nndsvd", max_iter=1000):
    '''
    Computes NMF or OPNMF based on selected mode.
    Returns W, H, and reconstructed matrix V.
    '''
    if mode == "nmf":
        model = NMF(n_components=rank, init=init, max_iter=max_iter, random_state=0)
    elif mode == "opnmf":
        model = OPNMF(n_components=rank, init=init, max_iter=max_iter)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'nmf' or 'opnmf'.")

    W = model.fit_transform(T)
    H = model.components_
    V = np.dot(W, H)
    return W, H, V


def extract_ground_truth_masks(num_components, gt_dir="./torus_data/", ref_file="torus_000.ply", prefix="groundtruth_", threshold=0.01):
    """
    Extracts binary masks from ground truth tori by comparing vertex displacements to the reference torus.
    Returns binary masks for each ground truth component (same shape as H[i]).
    """
    masks = []

    # Load reference torus
    ref_path = os.path.join(gt_dir, ref_file)
    ref_mesh = trimesh.load_mesh(ref_path)
    ref_verts = ref_mesh.vertices
    ref_normals = ref_mesh.vertex_normals

    for i in range(num_components):
        gt_path = os.path.join(gt_dir, f"{prefix}{i}.ply")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Missing ground truth torus: {gt_path}")
        
        gt_mesh = trimesh.load_mesh(gt_path)
        gt_verts = gt_mesh.vertices

        # Compute signed displacements
        displacement_vectors = gt_verts - ref_verts
        signed_displacements = np.einsum('ij,ij->i', displacement_vectors, ref_normals)

        # Threshold to produce binary mask
        mask = (signed_displacements > threshold).astype(int)
        masks.append(mask)

    return masks


def evaluate_nmf_labels(T, W, H, labels, filenames, ground_truth_masks=None, vertex_positions=None):
    ''' 
    Compare ground truth labels derived from filenames against NMF-inferred labels
    Also evaluate reconstruction error, classification, spatial agreement, and component angular distance
    '''
    try:
        ground_truth = [int(labels[fn]) for fn in filenames]
    except KeyError as e:
        print(f"Error: Filename {e} not found!")
        return

    initial_labels = np.argmax(W, axis=1)

    # reorder predicted labels manually
    print("=== Mapping Predicted Labels ===\nPlease visually inspect the component tori" 
          " and map their original labels to their current position as determined" 
          " by the NMF:")
    component_to_label = defaultdict(int)
    k = H.shape[0]
    for i in range(k):
        component_to_label[i] = int(input(f"{i} -> "))

    predicted_labels = [component_to_label[initial] for initial in initial_labels]

    # one-hot encode labels (for AUC)
    onehot_gt = np.zeros((len(ground_truth), k))
    for i, label in enumerate(ground_truth):
        onehot_gt[i][label] = 1

    # Reconstruction error
    V = np.matmul(W, H)
    frobenius_err = np.linalg.norm(T - V, "fro") / np.linalg.norm(T, "fro")
    mse_err = np.mean((T - V) ** 2)

    # Classification
    acc = accuracy_score(ground_truth, predicted_labels)
    f1 = f1_score(ground_truth, predicted_labels, average="macro")
    try:
        auc = roc_auc_score(onehot_gt, W, multi_class='ovo')
    except ValueError:
        auc = None

    # Confusion matrix
    c_matrix = confusion_matrix(ground_truth, predicted_labels)
    
    # Get ground truth masks
    ground_truth_masks = extract_ground_truth_masks(H.shape[0])
    
    # Reorder H for Pearson and Dice scores
    # Invert the mapping: new_index -> old_index
    label_to_component = {v: k for k, v in component_to_label.items()}

    H_reordered = []

    for i in range(H.shape[0]):
        original_row = label_to_component[i]  # Which row in H corresponds to label i?
        H_reordered.append(H[original_row])

    H_reordered = np.array(H_reordered)

    # Surface group correlation - pearson threshold and dice scores
    if ground_truth_masks is not None:
        corrs = [pearsonr(H_reordered[i], ground_truth_masks[i])[0] for i in range(k)]
        dice_scores = []
        for i in range(k):
            pred_binary = (H_reordered[i] > 0.25).astype(int)
            true_binary = ground_truth_masks[i].astype(int)
            intersection = np.sum(pred_binary * true_binary)
            union = np.sum(pred_binary) + np.sum(true_binary)
            dice = 2 * intersection / (union + 1e-8)
            dice_scores.append(dice)
    else:
        corrs = None
        dice_scores = None

    # TODO: implement centroid angle (if needed)
    '''
    # Centroid angle distances
    if vertex_positions is not None:
        centroids = []
        for i in range(k):
            weights = H[i]
            centroid = np.average(vertex_positions, axis=0, weights=weights)
            centroids.append(centroid / np.linalg.norm(centroid))

        angles = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dot_product = np.clip(np.dot(centroids[i], centroids[j]), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot_product))
                angles.append(angle)
    else:
        angles = None
    '''
    '''
    # Output
    print("\n--- NMF Evaluation Summary ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    if auc is not None:
        print(f"AUC (OVO): {auc:.4f}")
    else:
        print("AUC: Not computable (check label variety or sample size).")
    print(f"Reconstruction Error (Frobenius norm): {frobenius_err:.4f}")
    print(f"Mean Squared Error: {mse_err:.4f}")
    print("Confusion Matrix:")
    print(c_matrix)

    if corrs is not None:
        print("Component-to-Mask Correlations:", ' '.join(f"{v:.2f}" for v in corrs))
    if dice_scores is not None:
        print("Dice Scores:", ' '.join(f"{v:.2f}" for v in dice_scores))
    if angles is not None:
        print(f"Centroid Angle Distances (degrees): {[f'{v:.1f}' for v in angles]}")
    '''
    # save evaluation data for output
    evaluation_data = {
        "Accuracy": acc,
        "F1 Score (macro)": f1,
        "AUC (OVO)": auc if auc is not None else "N/A",
        "Reconstruction Error": frobenius_err,
        "Mean Squared Error": mse_err,
        "Confusion Matrix": c_matrix.tolist(),
    }

    if corrs is not None:
        evaluation_data["Component-to-Mask Correlations"] = corrs
    if dice_scores is not None:
        evaluation_data["Dice Scores"] = dice_scores
    # TODO: implement centroid angle (if needed)
    # if angles is not None:
        # evaluation_data["Centroid Angle Distances"] = angles

    output_evaluation_summary(evaluation_data)


def output_evaluation_summary(evaluation_data, log_path="nmf_output_log.txt", csv_path="nmf_results_summary.csv"):
    '''
    Outputs evaluation data to:
    - stdout
    - a timestamped .txt log file
    - a cumulative .csv file (one row per run)

    Parameters:
        evaluation_data (dict): keys are metric names, values are floats, lists, or matrices
    '''

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- STDOUT ---
    print("\n--- NMF Evaluation Summary ---")
    for key, value in evaluation_data.items():
        if isinstance(value, list):
            if isinstance(value[0], list):  # e.g., Confusion Matrix
                print(f"{key}:")
                for row in value:
                    print("  " + ' '.join(f"{v:.0f}" for v in row))
            else:
                print(f"{key}: " + ' '.join(f"{v:.2f}" for v in value))
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # --- LOG FILE ---
    with open(log_path, "a") as f:
        f.write(f"\n--- NMF Evaluation Summary ({timestamp}) ---\n")
        for key, value in evaluation_data.items():
            if isinstance(value, list):
                if isinstance(value[0], list):
                    f.write(f"{key}:\n")
                    for row in value:
                        f.write("  " + ' '.join(f"{v:.0f}" for v in row) + "\n")
                else:
                    f.write(f"{key}: " + ' '.join(f"{v:.2f}" for v in value) + "\n")
            elif isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

    # --- CSV FILE ---
    csv_row = [timestamp]
    header = ["Timestamp"]
    for key, value in evaluation_data.items():
        header.append(key)
        if isinstance(value, list):
            if isinstance(value[0], list):
                csv_row.append(';'.join(','.join(str(v) for v in row) for row in value))
            else:
                csv_row.append(';'.join(f"{v:.4f}" for v in value))
        else:
            csv_row.append(f"{value:.4f}" if isinstance(value, float) else str(value))

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(csv_row)


if __name__ == "__main__": 
    compute_torus_nmf()
