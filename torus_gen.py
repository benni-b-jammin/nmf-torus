#!/usr/bin/env python3
'''
torus_gen.py  - simple script to generate torus data with variations in torus
                bump angle, height, width, and height - saves torus data as in
                .ply format for use in main NMF script
                
                Note: credit for original torus generation can be given to Khoi
                and his work in continuous variation in torus generation:
                https://github.com/Khoi-Nguyen-Xuan/Torus_Bump_Generation

Authors:        Benji Lawrence, Khoi Nguyen Xuan
Last Modified:  Jun 26, 2025
'''
# necessary packages for implementation - Khoi
import os, sys 
from math import ceil, pi # Round up
import numpy as np 
# import meshplot as mp # Visualize 3D meshes
# import ipywidgets # Interactive sliders/widgets
from skimage import measure #surface extraction (marching cubes)
from scipy.ndimage import zoom, gaussian_filter # Resize
from scipy.interpolate import interpn #interpolate a volume at given points
from IPython.display import display  #For display 
from einops import rearrange # Elegant tensor reordering
import trimesh
import random
import pickle
import hashlib
import time

### Utilities -----------------------------------------------------------------
# Dot product on the first dimension of n-dimensional arrays x and y
def dot(x, y):
    return np.einsum('i..., i... -> ...', x, y)

#SDF for Torus (formula: https://iquilezles.org/articles/distfunctions/)
#x[[0, 2]]: Get only XZ dimensions → projects onto XZ plane, centered at the origin
#use np.linalg.norm to computes the distance from the Y-axis
#minus the radius and thickness to calculate the SD 
#Stack with x[1]: Y component preserved → represents distance from torus tube.
def sdf_torus(x, radius, thickness):
    q = np.stack([np.linalg.norm(x[[0, 2]], axis=0) - radius, x[1]])
    return np.linalg.norm(q, axis=0) - thickness

# Crop an n-dimensional image with a centered cropping region (after zoom)
def center_crop(img, shape):
    start = [a // 2 - da // 2 for a, da in zip(img.shape, shape)] #computes center offset by minusing da (desired a)
    end = [a + b for a, b in zip(start, shape)] #start + desired shape
    slices = tuple([slice(a, b) for a, b in zip(start, end)]) #returns cropping slices for each axis
    return img[slices]

# Add noise to coordinates
def gradient_noise(x, scale, strength, seed=None):
    shape = [ceil(s / scale) for s in x.shape[1:]]
    if seed:
        np.random.seed(seed)
    scalar_noise = np.random.randn(*shape)
    scalar_noise = zoom(scalar_noise, zoom=scale)
    scalar_noise = center_crop(scalar_noise, shape=x.shape[1:])
    vector_noise = np.stack(np.gradient(scalar_noise))
    return vector_noise * strength

# Meshplot will left an annoying print statement in their code
# This function used to supress it.
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def compute_bump_field(x, angle, radius, bump_width, bump_height):
    center = np.array([np.sin(angle), 0., np.cos(angle)]) * radius
    dist = np.linalg.norm((x - center[:, None, None, None]), axis=0)
    return bump_height * np.exp(-1. / bump_width * dist**2)
'''
def compute_correlated_noise(x, radius, theta_min, theta_max, strength=4.0, sigma=1.5, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    res = x.shape[1]
    X, Y = x[0], x[1]  # shape = (res, res, res)
    theta = np.mod(np.arctan2(Y, X), 2 * np.pi)
    mask = (theta >= theta_min) & (theta < theta_max)

    # Make raw noise and apply spatial mask
    noise = np.zeros_like(X)
    noise[mask] = np.random.randn(np.sum(mask))
    # Smooth noise for spatial correlation
    noise = gaussian_filter(noise, sigma=sigma)
    # Create a 3-channel (vector field) noise aligned to the grid
    noise_field = np.stack([noise, noise, noise])  # shape (3, res, res, res)

    return noise_field * strength
'''
def compute_correlated_noise(x, center_angle, radius, angular_spread=np.pi/12, spatial_spread=0.2, strength=8.0, sigma=1.0, seed=None):
    '''
    Create localized correlated noise centered at a specific angle on the torus surface.

    Parameters:
    - x: 3D grid of shape (3, res, res, res)
    - center_angle: angle (in radians) on the torus for the center of the noise
    - radius: torus radius
    - spread: controls how wide the noisy region is (in Euclidean distance units)
    - strength: scaling of noise intensity
    - sigma: Gaussian smoothing parameter
    - seed: for reproducibility
    '''
    if seed is not None:
        np.random.seed(seed)
    # Compute 3D Gaussian centered at (sinθ, 0, cosθ) * radius
    center = np.array([np.sin(center_angle), 0., np.cos(center_angle)]) * radius  # shape (3,)

    # Compute distance from center for all grid points
    dist = np.linalg.norm(x - center[:, None, None, None], axis=0)  # shape (res, res, res)

    # Gaussian mask around the bump center
    gaussian_mask = np.exp(-0.5 * (dist / spatial_spread) ** 2)
    
    # Angular band mask
    X, Z = x[0], x[2]
    theta = np.mod(np.arctan2(Z, X), 2 * np.pi)
    theta_min = (center_angle - angular_spread) % (2 * np.pi)
    theta_max = (center_angle + angular_spread) % (2 * np.pi)
    # print(f"Angle values: theta_min={theta_min}, theta_max={theta_max}")
    
    if theta_min < theta_max:
        angular_mask = (theta >= theta_min) & (theta <= theta_max)
    else:
        angular_mask = (theta >= theta_min) | (theta <= theta_max)

    # Apply random noise and smooth
    # raw_noise = np.random.randn(*dist.shape) * gaussian_mask * angular_mask
    raw_noise = np.random.randn(*dist.shape) * angular_mask
    smoothed_noise = gaussian_filter(raw_noise, sigma=sigma)

    # Return as a 3D vector field (same noise for x/y/z directions)
    noise_vector_field = np.stack([smoothed_noise]*3) * strength
    return noise_vector_field


def deform_vertices(verts, x_warp, resolution):
    x_warp = rearrange(x_warp, 'v h w d -> h w d v')
    vertex_noise = interpn([np.arange(resolution)] * 3, x_warp, verts, bounds_error=False, fill_value=0)
    vertex_noise = np.nan_to_num(vertex_noise)
    return verts + vertex_noise

def generate_ground_truth(label, x, verts, faces, radius, surface="bump"):
    angle = (2 * pi / 3) * label
    if surface == "bump":
        bump = compute_bump_field(x, angle, radius, bump_width=0.01, bump_height=15.0)
        x_warp = -np.stack(np.gradient(bump))
    elif surface == "noise":
        x_warp = compute_correlated_noise(x, angle, radius, seed = 1)
        
    deformed_verts = deform_vertices(verts.copy(), x_warp, x.shape[-1])
    mesh = trimesh.Trimesh(vertices=deformed_verts, faces=faces, process=False)
    mesh.export(f"./torus_data/groundtruth_{label}.ply")
   
def create_batchID():
    # creates unique batch ID off of current timestamp
    return hashlib.sha256(str(time.time_ns()).encode('utf-8')).hexdigest()[:16]

## Torus Generation Function --------------------------------------------------
def generate_torus(num=99, variable=None, surface="bump"):
    '''
    Creates torus data and saves to .ply files
    Torus data is configured to be between 3 regions so the NMF algorithm can 
    cluster and group the tori; variation to bump thickness and height are also
    included for confirming NMF effectiveness
    '''
    # setup output directory - can be changed if needed
    out_dir = os.path.expanduser("./torus_data/")
    os.makedirs(out_dir, exist_ok=True)
    metadata = {}
    
    # batchID - for tracking purpose
    batchID = create_batchID()

    # stable parameters - thickness overridden if variable thickness enabled
    resolution = 100
    radius = 0.25
    seed = None

    # standard variables for metadata
    bump_width = None
    bump_height = None
    angle = None
    second_angle = None
    corrn_strength = None
    corrn_spread = None

    #Create an array of 100 points from -1 to 1 
    coords = np.linspace(-1, 1, resolution)
    #Create a 3D grid with coords above 
    x = np.stack(np.meshgrid(coords, coords, coords)) # x.shape = (3, 100, 100, 100)

    # Generate standard Torus for displacement comparison
    sdf_standard = sdf_torus(x, radius, 0.1)
    verts_standard, faces_standard, normals_standard, _ = measure.marching_cubes(sdf_standard, level=0)

    # Save refrence as PLY - make smaller to ensure all displacements >= 0
    shrink_factor = 0.5
    verts_ref = verts_standard + (normals_standard * shrink_factor)
    mesh = trimesh.Trimesh(vertices=verts_ref, faces=faces_standard, process=False)
    mesh.export(os.path.join(out_dir, f"torus_000.ply"))

    # Generate and save ground truth data - for correlation-based evaluation
    for label in range(3):
        generate_ground_truth(label, x, verts_ref, faces_standard, radius, surface)

    for i in range(num):
        # randomize noise, bump_size, thickness
        noise_scale = random.randint(18, 20)
        noise_strength = random.randint(6, 10)
        verts = verts_standard.copy()
        norms = normals_standard.copy()

        # Thickness displacement: move vertices along normals scaled by thickness difference
        if (variable == "thickness") or (variable == "both"):
            thickness = round(random.uniform(0.1, 0.2), 2)
            thickness_displacement = (thickness - 0.1)  # relative to base thickness 0.1
            verts += norms * thickness_displacement
        else:
            thickness = 0.1
        
        # Noise field
        x_warp = gradient_noise(x, noise_scale, noise_strength, seed)

        if (surface == "bump"):
            bump_stats = bumpy_surface(i, variable, verts, norms, x, radius, seed)
            x_bump_warp, bump_width, bump_height, angle, second_angle = bump_stats
            x_warp += x_bump_warp

        elif (surface == "noise"):
            corrn_stats = noisy_surface(i, variable, x, radius, seed)
            x_corrn_warp, corrn_strength, corrn_spread, angle, second_angle = corrn_stats
            x_warp += x_corrn_warp
        
        # Interpolate and deform
        warped_verts = deform_vertices(verts, x_warp, resolution)

        # Save as PLY
        mesh = trimesh.Trimesh(vertices=warped_verts, faces=faces_standard, process=False)
        new_mesh_filename = os.path.join(out_dir, f"torus_{(i+1):03d}_{i%3}.ply")
        mesh.export(new_mesh_filename)
        print(f"Torus {i+1} generated - saved to {new_mesh_filename}")

        # Log torus metadata
        metadata[f"torus_{(i+1):03d}_{i%3}"] = {
            "group": i % 3,
            "thickness": thickness if (variable in ["thickness", "both"]) else 0.1,
            "bump_height": bump_height,
            "bump_width": bump_width,
            "primary_angle": angle,
            "secondary_angle": second_angle if (variable in ["secondangle", "both"]) else None,
            "corr_noise_strength": corrn_strength,
            "corr_noise_spread": corrn_spread,
        }
    
    # Save batch ID & torus metadata
    md = [batchID, surface, variable, metadata]
    with open(os.path.join(out_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(md, f)
        print("Metadata saved to metadata.pkl")


def bumpy_surface(
    iteration,
    variable,
    verts,
    norms,
    x,
    radius,
    seed=None):
    
    # randomize bump_size, thickness
    bump_width = round(random.uniform(0.005, 0.015), 3)
    bump_height = round(random.uniform(5.0, 25.0), 1)
    
    # parameters for second bump - utilized if variable second bump enabled
    second_height = 5
    second_width = 0.01
    second_angle = None
    
    # calculate primary bump angle and compute
    angle_centre = (2*np.pi/3) * (iteration % 3)
    delta = np.pi/24
    angle = random.uniform(angle_centre - delta, angle_centre + delta)
    bump = compute_bump_field(x, angle, radius, bump_width, bump_height)
    x_bwarp = -np.stack(np.gradient(bump))

    # Extra bump: adds variability to prevent overfitting
    if (variable == "secondangle") or (variable == "both"):
        # place at random angle
        second_angle = random.uniform(0, 2*np.pi)
        bump2 = compute_bump_field(x, second_angle, radius, second_width, second_height)
        x_bwarp += -np.stack(np.gradient(bump2))
    else:
        second_angle = None
    
    return x_bwarp, bump_width, bump_height, angle, second_angle

'''
def noisy_surface(
    iteration,
    variable,
    x,
    radius,
    seed=None):
    
    # calculate zone angles based on iteration
    group = iteration % 3
    theta0 = (2*np.pi/3 * group) % (2*np.pi)
    theta1 = (2*np.pi/3 * (group + 1)) % (2*np.pi)
    if theta1 == 0:     theta1 = 2*np.pi
    noise_zone = [theta0, theta1]
    strength = 7
    second_zone = None
    x_cwarp = compute_correlated_noise(x, radius, noise_zone[0], noise_zone[1], strength=strength, seed=seed)
    
    # Extra noise zone: adds variability to prevent overfitting
    if (variable == "secondangle") or (variable == "both"):
        # random angle as start location
        second_angle = random.uniform(0, 2*np.pi)
        second_zone = [second_angle, (second_angle + 2*np.pi/3) % (2*np.pi)]
        second_warp = compute_correlated_noise(x, radius, second_zone[0], second_zone[1], strength=strength, seed=seed)
        x_cwarp += second_warp

    return x_cwarp, strength, noise_zone, second_zone
    '''
def noisy_surface(iteration, variable, x, radius, seed=None):
    center_angle = (2 * np.pi / 3) * (iteration % 3)
    s_spread = 0.1
    strength = 3.0
    sigma = 1.0
    a_spread = np.pi/12
    
    x_cwarp = compute_correlated_noise(x, center_angle, radius, angular_spread=a_spread, spatial_spread=s_spread, strength=strength, sigma=sigma, seed=seed)

    second_angle = None
    if variable in ["secondangle", "both"]:
        second_angle = random.uniform(0, 2*np.pi)
        second_warp = compute_correlated_noise(x, second_angle, radius, angular_spread=a_spread, spatial_spread=s_spread, strength=(strength/2), sigma=sigma, seed=seed)
        x_cwarp += second_warp

    return x_cwarp, strength, s_spread, center_angle, second_angle

    

if __name__ == "__main__":
    generate_torus(6, variable=None, surface="noise")
