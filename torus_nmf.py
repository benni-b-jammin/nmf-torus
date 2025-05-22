#!/usr/bin/env python3
'''
torus_gen.py  - this script will attempt to utilize nonnegative matrix
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

Authors:        Benji Lawrence, Khoi Nguyen Xuan
Last Modified:  May 21, 2025
'''
