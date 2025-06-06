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
Last Modified:  May 30, 2025
'''
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF, PCA
import os
import torus_gen
import trimesh
import matplotlib.pyplot as plt
import pyvista as pv
import os

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
    T_file = "./T.npy"
    torus_filenames = []
    if (os.path.isfile(T_file)):
        T = np.load(T_file, allow_pickle=True)
    else:
        T = create_T_matrix(T_file, torus_filenames)
    
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
    
        
        
    
def create_T_matrix(matrix_name, filenames):
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
        filepath = os.path.join(torus_dir_str, os.fsdecode(file))
        print(f"Reading file: {filepath}")
        try:
            mesh = trimesh.load_mesh(filepath)
            verts_warped = mesh.vertices

            # determines signed displacements - uses only positive values
            displacement_vectors = verts_warped - verts_standard
            signed_displacements = np.einsum('ij,ij->i', displacement_vectors, normals_standard)
            signed_displacements = np.clip(signed_displacements, 0, None)
        
            # assign to matrix
            if matrix is None:
                matrix = signed_displacements.reshape(1, -1)  # reshape to 2D array
            else:
                matrix = np.vstack((matrix, signed_displacements))
            filenames.append(os.fsdecode(file)) # log for later clustering

        except Exception as e:
            print(f"Could not load {filepath}: {e}")

    if matrix is not None:
        np.save(matrix_name, matrix)
    return matrix

def visualize_torus_results(W, H, sample_index=0):
    '''
    Determines 2D and 3D heatmap visualizations for display and saves for future reference
    '''
    torus_dir = os.path.expanduser("./torus_data/")
    ref_path = os.path.join(torus_dir, "torus_000.ply")

    try:
        mesh = pv.read(ref_path)
    except Exception as e:
        print(f"Failed to load reference torus: {e}")
        return

    verts = mesh.points
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    # Compute torus angles for 2D projection
    theta = (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
    phi = (np.arctan2(np.sqrt(x**2 + y**2) - 1.0, z) + 2*np.pi) % (2*np.pi)

    num_vertices = verts.shape[0]
    num_subtypes = W.shape[1]

    for subtype_idx in range(num_subtypes):
        basis_vector = H[subtype_idx, :]
        if basis_vector.shape[0] != num_vertices:
            print(f"Skipping subtype {subtype_idx} due to shape mismatch.")
            continue

        contribution = W[sample_index, subtype_idx] * basis_vector
        mesh.point_data.clear()
        mesh.point_data["subtype_contrib"] = contribution

        # normalize to 0–255 for RGB colouring
        contrib_norm = 255 * (contribution - contribution.min()) / (contribution.ptp() + 1e-8)
        contrib_rgb = plt.cm.viridis(contrib_norm / 255.0)[:, :3]  # Drop alpha
        mesh.point_data['RGB'] = (contrib_rgb * 255).astype(np.uint8)

        # ---- Save colored 3D mesh ----
        ply_path = f"torus_colored_subtype_{subtype_idx}_sample_{sample_index}.ply"
        mesh.save(ply_path)
        print(f"Saved colored mesh: {ply_path}")

        # ---- Save static 3D view ----
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh, scalars="subtype_contrib", cmap="viridis", show_edges=False)
        plotter.view_isometric()
        plotter.add_scalar_bar(f"Subtype {subtype_idx}")
        plotter.screenshot(f"torus_3D_subtype_{subtype_idx}_sample_{sample_index}.png")
        plotter.close()

        # ---- Save 2D projection ----
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(theta, phi, c=contribution, cmap='viridis', s=1)
        plt.colorbar(sc, label="Subtype contribution")
        plt.title(f"2D Projection - Subtype {subtype_idx}")
        plt.xlabel("Theta (tube angle)")
        plt.ylabel("Phi (donut angle)")
        plt.savefig(f"torus_2D_subtype_{subtype_idx}_sample_{sample_index}.png", dpi=300)
        plt.close() 

if __name__ == "__main__": 
    main()
