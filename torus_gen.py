#!/usr/bin/env python3
'''
torus_gen.py  - simple script to generate torus data with variations in torus
                bump angle, height, width, and height - saves torus data as in
                .ply format for use in main NMF script

                Note: credit for original torus generation can be given to Khoi
                and his work in continuous variation in torus generation:
                https://github.com/Khoi-Nguyen-Xuan/Torus_Bump_Generation

Authors:        Benji Lawrence, Khoi Nguyen Xuan
Last Modified:  May 20, 2025
'''
# necessary packages for implementation - Khoi
import os, sys 
from math import ceil, pi # Round up
import numpy as np 
# import meshplot as mp # Visualize 3D meshes
# import ipywidgets # Interactive sliders/widgets
from skimage import measure #surface extraction (marching cubes)
from scipy.ndimage import zoom # Resize
from scipy.interpolate import interpn #interpolate a volume at given points
from IPython.display import display  #For display 
from einops import rearrange # Elegant tensor reordering
import trimesh
import random

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

## Torus Generation Function --------------------------------------------------
def generate_torus(num=99):
    '''
    Creates torus data and saves to .ply files
    Torus data is configured to be between 3 regions so the NMF algorithm can 
    cluster and group the tori; variation to bump thickness and height are also
    included for confirming NMF effectiveness
    '''
    # setup output directory - can be changed if needed
    out_dir = os.path.expanduser("./torus_data/")
    os.makedirs(out_dir, exist_ok=True)

    # some parameters; TODO: add more variability
    resolution = 100
    radius = 0.25
    thickness = 0.1
    seed = 42
    
    #Create an array of 100 points from -1 to 1 
    coords = np.linspace(-1, 1, resolution)
    #Create a 3D grid with coords above 
    x = np.stack(np.meshgrid(coords, coords, coords)) # x.shape = (3, 100, 100, 100)
    
    for i in range(num):#enumerate(np.linspace(-1.0, 1.0, num)):
        # randomize noise and bump size
        noise_scale = random.randint(18, 22)
        noise_strength = random.randint(6, 12)
        bump_width = round(random.uniform(0.005, 0.015), 3)
        print(bump_width)
        bump_height = round(random.uniform(20.0, 30.0), 1)
        print(bump_height)
        
        # randomize bump angle within interval
        # TODO: change to random selection?
        mod = i % 3
        if (mod == 0):      angle = random.uniform(0, 2*np.pi/3)
        elif (mod == 1):    angle = random.uniform(2*np.pi/3, 4*np.pi/3)
        else:               angle = random.uniform(2*np.pi/3, 6*np.pi/3)
        
        # Base torus SDF
        sdf = sdf_torus(x, radius, thickness)
        verts, faces, normals, values = measure.marching_cubes(sdf, level=0)
    
        # Noise field
        x_warp = gradient_noise(x, noise_scale, noise_strength, seed)
    
        # Bump field
        gaussian_center = np.array([np.sin(angle), 0., np.cos(angle)]) * radius
        x_dist = np.linalg.norm((x - gaussian_center[:, None, None, None]), axis=0)
        x_bump = bump_height * np.exp(-1. / bump_width * x_dist**2)
        x_warp += -np.stack(np.gradient(x_bump))

        # Interpolate and deform
        x_warp = rearrange(x_warp, 'v h w d -> h w d v')
        vertex_noise = interpn([np.arange(resolution)] * 3, x_warp, verts, bounds_error=False, fill_value=0)
        vertex_noise = np.nan_to_num(vertex_noise)
        warped_verts = verts + vertex_noise
            
        # Save as PLY
        mesh = trimesh.Trimesh(vertices=warped_verts, faces=faces, process=False)
        mesh.export(os.path.join(out_dir, f"torus_{i:03d}.ply"))


if __name__ == "__main__":
    generate_torus(6)
