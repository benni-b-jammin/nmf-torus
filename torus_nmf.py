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
Last Modified:  Jun 06, 2025
'''
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.decomposition import NMF, PCA
from scipy.optimize import linear_sum_assignment
import os
import torus_gen
import trimesh
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pyvista as pv
from collections import defaultdict
import pickle

def main ():
    '''
    Program functionality executed here:
        - reads files and/or generates torus data
            - saves generated matrix T if needed
        - TODO: computes rank k for decomposition matrix
        - TODO: initializes W matrix - multiple methods possible
        - computes NMF of matrix - saves W, H, & V
        - TODO: clusters matrices based on encoded W matrix
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
    
    #T = normalize(T, axis=1, norm="l1")
    print(f"T Matrix loaded:\n{T}\nShape: {T.shape}")
    # optimal matrix rank for encoding matrix W
    # TODO: algorithm for determining optimal rank?
    optimal_r = 3

    # run NMF algorithm - initialize with Nonnegative Double Singular Value
    # Decomposition (nndsvd) for better result convergence than random
    # TODO: test different initialization methods?
    model = NMF(n_components=optimal_r, init='nndsvd', random_state=0)
    W = model.fit_transform(T)
    H = model.components_
    V = np.matmul(W, H)
    
    print(f"W Matrix computed:\n{W}\nShape: {W.shape}")
    print(f"H Matrix computed:\n{H}\nShape: {H.shape}")
    print(f"V Matrix computed:\n{V}\nShape: {V.shape}")
    subtype_labels = W.argmax(axis=1) 
    pca = PCA(n_components=2)
    coords = pca.fit_transform(W)

    plt.scatter(coords[:, 0], coords[:, 1], c=subtype_labels, cmap='Set1')
    plt.title('Torus Subtyping via NMF')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(label='Subtype')
    #plt.show()
    
    #visualize_torus_results(W, H)
    print(f"T values - Min: {np.min(T)}, Max: {np.max(T)}, MeanL {np.mean(T)}")
    #visualize_reconstruction_error(T, V)
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
        print("No torus files - generating")
        torus_gen.generate_torus()
    
    ply_files = sorted([
        f for f in os.listdir(torus_dir)
        if f.endswith(b'.ply')
    ])

    # Load standard torus for use as reference
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

            # determines signed displacements - uses only positive values
            displacement_vectors = verts_warped - verts_standard
            signed_displacements = np.einsum('ij,ij->i', displacement_vectors, normals_standard)
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
            print(f"Could not load {filepath}: {e}")

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
    Creates more torus data based on factorization - displacement patterns of 
    each bump location in own mesh is produced, as well as a torus showing
    all displacement patterns determined
    '''
    os.makedirs(out_path, exist_ok=True)
    ref_mesh = trimesh.load_mesh(ref_path)
    ref_verts = ref_mesh.vertices
    ref_normals = ref_mesh.vertex_normals
    faces = ref_mesh.faces
    num_components, n_vertices = H.shape
    assert len(ref_verts) == n_vertices, "Mesh vertex and H features MISMATCH"
    
    # Create torus for each NMF result
    for i in range(num_components):
        displacements = H[i, :] 
        bump_verts = ref_verts + (ref_normals * displacements[:, np.newaxis])
        bump_mesh = trimesh.Trimesh(vertices=bump_verts, faces=faces)
        bump_mesh = colour_mesh_vertices(bump_mesh, displacements) 
        bump_mesh.export(os.path.join(out_path, f"nmf_component_{i}.ply"))

    # Generate combined torus (heatmap mean of all components)
    # TODO: fix to display all 3, not just the mean
    combined_displacements = np.mean(H, axis=0)
    bump_verts = ref_verts + (ref_normals * displacements[:, np.newaxis])
    bump_mesh = trimesh.Trimesh(vertices=bump_verts, faces=faces)
    bump_mesh = colour_mesh_vertices(bump_mesh, combined_displacements) 
    bump_mesh.export(os.path.join(out_path, "nmf__combined.ply"))



def evaluate_nmf_labels(T, W, H, labels, filenames):
    '''
    Compare ground truth labels derived from filenames against NMF-inferred labels

    NMF labels may be in a different order to those of input ground truth -
    manual visual inspection of component tori is required
    '''
    # obtain ground truth & predicted labels
    try:
        ground_truth = [int(labels[fn]) for fn in filenames]
    except KeyError as e:
        print(f"Error: Filename {e} not found!")
        return
    initial_labels = np.argmax(W, axis=1)
        
    # reorder predicted labels - map initial labels to actual labels
    print("=== Mapping Predicted Labels ===\nPlease visually inspect the component tori" \
          " and map their original labels to their current position as determined" \
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

    # metrics - reconstruction error, accuracy, f1, ROC AUC, confusion matrix
    # reconstruction error
    V = np.matmul(W, H)
    frobenius_err = np.linalg.norm(T-V, "fro") / np.linalg.norm(T, "fro")
    
    # accuracy & f1
    acc = accuracy_score(ground_truth, predicted_labels)
    f1 = f1_score(ground_truth, predicted_labels, average="macro")
    
    # ROC AUC - set to None if not valid
    try:
        auc = roc_auc_score(onehot_gt, W, multi_class='ovo')
    except ValueError:
        auc = None  # Not enough variation

    # confusion matrix 
    c_matrix = confusion_matrix(ground_truth, predicted_labels)

    print("\n--- NMF Evaluation Summary ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    if auc is not None:
        print(f"AUC (OVO): {auc:.4f}")
    else:
        print("AUC: Not computable (check label variety or sample size).")
    print(f"Relative Reconstruction Error (Frobenius norm): {frobenius_err:.4f}")
    print("Confusion Matrix:")
    print(c_matrix)

if __name__ == "__main__": 
    main()
