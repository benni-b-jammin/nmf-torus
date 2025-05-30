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
                
                tori data from .npy file (generates if none exist), this 
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
Last Modified:  May 23, 2025
'''
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
import os
import torus_gen
import trimesh

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
    
    
        
        
    
def create_T_matrix(matrix_name, filenames):
    '''
    Creates T matrix using existing torus data - returns matrix and saves as 
    .npy file for later use if needed
    '''
    # retrieve torus data - generate if does not exist
    torus_dir_str = os.path.expanduser("./torus_data/")
    if not (os.path.exists(os.path.join(torus_dir_str, "torus_000.ply"))):
        print("no tori detected - generating")
        torus_gen.generate_torus()
    
    matrix = None       # initialized - adjusted to np.array on first loop
    torus_dir = os.fsencode(torus_dir_str)
    ply_files = sorted([
        f for f in os.listdir(torus_dir)
        if f.endswith(b'.ply')
    ])
    
    # iterate through torus files, append to matrix
    for file in ply_files:
        filepath = os.path.join(torus_dir_str, os.fsdecode(file))
        print(f"Reading file: {filepath}")
        try:
            mesh = trimesh.load_mesh(filepath)
            row = mesh.vertices.flatten()
            if matrix is None:
                matrix = row.reshape(1, -1)  # reshape to 2D array
            else:
                matrix = np.vstack((matrix, row))
            filenames.append(os.fsdecode(file)) # log for later clustering

        except Exception as e:
            print(f"Could not load {filepath}: {e}")

    if matrix is not None:
        np.save(matrix_name, matrix)
    return matrix

    
    

if __name__ == "__main__":
    main()
