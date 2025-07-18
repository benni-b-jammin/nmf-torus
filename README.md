# Categorizing Torus Data using Non-negative Matrix Factorization (NMF)


This project explores the use of **Nonnegative Matrix Factorization (NMF)** and **Orthogonal Projective NMF (OPNMF)** to discover latent spatial patterns and subtypes in **synthetic torus data**. This will be used as a basis to justify the use of (OP)NMF in the analysis of cortical thickness MRI brain data.

---

## Project Structure
```bash
.
├── torus_nmf.py              # Main NMF/OPNMF pipeline & utility functions (visualization, evaluation)
├── torus_gen.py              # Torus mesh generator for bump/noise patterns
├── torus_data/               # Contains .ply meshes and precomputed .npy matrix
├── results/                  # Exported colored meshes and evaluation output
└── README.md                 # You are here
```

---

## Purpose

This project provides an analog that simulates **geometric brain-like data** on a torus surface and applies NMF to uncover:

- Latent spatial components (regions)
- Subtypes of subjects (encoding weights)
- Ground truth alignment via Dice and Pearson correlation
- Regional interpretability via component heatmaps

---

## Functionality Overview

### Input Modes

The script intelligently loads data in one of the following ways:

- **`.npy` exists**: load displacement matrix directly
- **`.ply` files exist**: compute displacements and save `.npy`
- **No files exist**: generate tori with `torus_gen.py`, then process
- **Hard reset enabled**: regenerate everything from scratch

### Surface Types

Two distinct deformation types can be generated:

- **`bump`**: Gaussian-shaped protrusions at angular locations
- **`noise`**: Correlated angular band noise (non-Gaussian, localized)

---

## Matrix Factorization

Given a displacement matrix `T ∈ ℝⁿˣᵐ`, we factor it as:
```T ≈ W @ H```


Where:

- `T`: Displacement values (subjects × vertices)
- `W`: Latent encoding (subjects × components)
- `H`: Spatial components (components × vertices)
- `V`: Reconstructed data (`W @ H`)

Supported algorithms:

- **Traditional NMF** (via `sklearn.decomposition.NMF`)
- **OPNMF** (via [opnmf](https://github.com/juaml/opnmf))

---

## Ground Truth & Evaluation

### Ground Truth Masks

- **Bump mode**: Thresholded displacements from a reference torus
- **Noise mode**: Angular bands defined on the torus geometry

### Metrics

- **Dice Score** – spatial overlap between component and ground truth mask
- **Component-to-Mask Correlation** – Pearson correlation
- **Reconstruction Error**
- **Classification Accuracy**
- **AUC (OVO)** – from clustering assignments

---

## Visualization

Outputs include:

- Component heatmaps (`H`) visualized on torus surfaces
- Combined heatmap (maximum projection)
- Ground truth mask overlays
- Exported `.ply` files for 3D inspection

---

## Advanced Features (Planned)

- **Laplacian smoothing** of spatial components (`H`)
- **Manifold regularization** using graph Laplacians
- **Regression with clinical data** (e.g., age or synthetic variables)
- Post-hoc component filtering and regional compactness scoring

---

## Getting Started

### Dependencies

```bash
pip install numpy scipy scikit-learn trimesh opnmf matplotlib
```

### Run NMF or OPNMF
```
python torus_nmf.py --mode opnmf --surface noise --k 3
```
Optional arguments:
    `--Reset` to regenerate all data
    `--normalize` for global normalization, per_entry, and PLS

## Dataset Format

Each torus is a .ply mesh aligned in vertex order with a reference torus. Displacements are computed as:
```
displacement = (subject_verts - ref_verts) · ref_normals
```
These are stored as rows in matrix T, allowing voxel-wise comparison across the surface.

