# Author: Imraj Singh

# First version: 21st of May 2022

# CCP SyneRBI Synergistic Image Reconstruction Framework (SIRF).
# Copyright 2022 University College London.

# This is software developed for the Collaborative Computatiponal Project in Synergistic Reconstruction for Biomedical Imaging (http://www.ccpsynerbi.ac.uk/).

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import torch
import numpy as np
from .misc import random_phantom, shepp_logan, affine_transform_2D_image
from sirf.STIR import AcquisitionSensitivityModel, AcquisitionModelUsingRayTracingMatrix


import brainweb
from tqdm import tqdm
import random



import matplotlib.pyplot as plt

from scipy.ndimage import zoom

def generate_random_transform_values():
    theta = np.random.uniform(-np.pi/16, np.pi/16)
    tx, ty = np.random.uniform(-1, 1), np.random.uniform(-1, 5)
    sx, sy = np.random.uniform(0.95, 1.05), np.random.uniform(0.95, 1.05)
    return theta, tx, ty, sx, sy

# make average image value n
def make_max_n(image, n):
    image *= n/image.max()
    return image

def crop(img, cropx, cropy):
    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:, starty:starty+cropy,startx:startx+cropx]

## circular
def random_camouflage_array(shape, num_patches, min_size=2, max_size=10, colors=[0, 64, 128, 192, 255]):
    """
    Generate a random camouflage array
    Will possibly help with brains
    """

    canvas = np.zeros(shape)
    
    for _ in range(num_patches):
        # Generate random parameters for the patch
        center_x = np.random.randint(low=0, high=shape[1])
        center_y = np.random.randint(low=0, high=shape[0])
        size = np.random.randint(low=min_size, high=max_size)
        color = random.choice(colors)
        
        # Compute the patch boundaries
        x1 = center_x - size // 2
        y1 = center_y - size // 2
        x2 = center_x + size // 2
        y2 = center_y + size // 2
        
        # Clip the patch boundaries to the canvas boundaries
        x1 = np.clip(x1, 0, shape[1]-1)
        y1 = np.clip(y1, 0, shape[0]-1)
        x2 = np.clip(x2, 0, shape[1]-1)
        y2 = np.clip(y2, 0, shape[0]-1)
        
        # Fill the patch with the color
        canvas[y1:y2, x1:x2] = color
    
    return canvas

def random_polygons_array(shape, num_polygons, min_n=3, max_n=8, min_size=5, max_size=50):
    """
    Generate a random array of polygons outlines
    Will possibly help with edges
    """
    
    canvas = np.zeros(shape)

    for _ in range(num_polygons):
        # Generate random parameters for the polygon
        center_x = np.random.randint(low=0, high=shape[1])
        center_y = np.random.randint(low=0, high=shape[0])
        size = np.random.randint(low=min_size, high=max_size)
        n = np.random.randint(low=min_n, high=max_n)
        color = np.random.randint(low=1, high=255)
        
        # Generate the vertices of the polygon
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = center_x + size*np.cos(theta)
        y = center_y + size*np.sin(theta)
        
        # Convert the vertices to integer coordinates
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        
        # Clip the vertices to the canvas boundaries
        x = np.clip(x, 0, shape[1]-1)
        y = np.clip(y, 0, shape[0]-1)
        
        # Fill the polygon with the color
        for j in range(n):
            x1, y1 = x[j], y[j]
            x2, y2 = x[(j+1) % n], y[(j+1) % n]
            canvas = fill_line(canvas, x1, y1, x2, y2, color)
    
    return canvas

def fill_line(canvas, x1, y1, x2, y2, color):
    # Compute the line segment parameters
    dx = x2 - x1
    dy = y2 - y1
    d = max(abs(dx), abs(dy))
    sx = dx / d
    sy = dy / d
    
    # Iterate over the line segment pixels
    x = x1
    y = y1
    for _ in range(d+1):
        canvas[int(y), int(x)] = color
        x += sx
        y += sy
    
    return canvas
    
# 2D array so everything outside the circle is 0
def crop_to_circle(array, radius, centre = None):
    """
    Crop a 2D array to a circle
    """
    if centre is None:
        centre = [array.shape[0]//2, array.shape[1]//2]
    y, x = np.ogrid[-centre[0]:array.shape[0]-centre[0], -centre[1]:array.shape[1]-centre[1]]
    mask = x*x + y*y <= radius*radius
    array[~mask] = 0
    return array
##



# set up brainweb files
fname, url= sorted(brainweb.utils.LINKS.items())[0]
files = brainweb.get_file(fname, url, ".")
data = brainweb.load_file(fname)
class EllipsesDataset(torch.utils.data.Dataset):

    """ Pytorch Dataset for simulated ellipses

    Initialisation
    ----------
    radon_transform : `SIRF acquisition model`
        The forward operator
    image template : `SIRF image data`
        needed to project and to get shape
    n_samples : `int`
        Number of samples    
    mode : `string`
        Type of data: training, validation and testing
    seed : `int`
        The seed used for the random ellipses
    """

    def __init__(self, radon_transform, attenuation_image_template, sinogram_template, n_samples = 100, mode="train", seed = 1):
        self.radon_transform = radon_transform
        self.attenuation_image_template = attenuation_image_template.clone()
        self.n_samples = n_samples
        self.template = sinogram_template
        self.one_sino = sinogram_template.get_uniform_copy(1)
        self.tmp_acq_model = AcquisitionModelUsingRayTracingMatrix()

        self.primal_op_layer = radon_transform
        self.mode = mode
        np.random.seed(seed)


    def __get_sensitivity__(self, attenuation_image):
        # Forward project image then add noise
        asm_attn = AcquisitionSensitivityModel(attenuation_image, self.radon_transform)
        asm_attn.set_up(self.template)
        self.tmp_acq_model.set_acquisition_sensitivity(asm_attn)
        self.tmp_acq_model.set_up(self.template, attenuation_image)
        y = self.tmp_acq_model.backward(self.one_sino)
        return y

    def __len__(self):
        # Denotes the total number of iters
        return self.n_samples

    def __getitem__(self, index):
        # Generates one sample of data
        if self.mode == "train":
            random_phantom_array = make_max_n(random_polygons_array(self.attenuation_image_template.shape, 20, min_n=3, max_n=8, min_size=5, max_size=50), 0.096*2)
#             random_phantom_array = make_max_n(random_phantom(self.attenuation_image_template.shape, 20), 0.096*2)
            ct_image = self.attenuation_image_template.clone().fill(random_phantom_array) # random CT image
            sens_image = self.__get_sensitivity__(ct_image) # random sensitivity image
            theta, tx, ty, sx, sy = generate_random_transform_values()
            ct_image_transform = affine_transform_2D_image(theta, tx, ty, sx, sy, ct_image) # CT of transformed image
            ct_image_transform.move_to_scanner_centre(self.template)
            sens_image_transform = self.__get_sensitivity__(ct_image_transform) # sensitivity map of transformed image
   


        elif self.mode == "valid":
        ## instead of using a random phantom image, loads a brain image from the BrainWeb database.
            brainweb.seed(np.random.randint(500,1500))
            for f in tqdm([fname], desc="mMR ground truths", unit="subject"):
                vol = brainweb.get_mmr_fromfile(f, petNoise=1, t1Noise=0.75, t2Noise=0.75, petSigma=1, t1Sigma=1, t2Sigma=1)
            uMap_arr = vol['uMap'] # random CT brain image
        ## first zoomed and then cropped.
            umap_zoomed = crop(zoom(uMap_arr, 1, order=1), 155, 155)
            ct_image = self.attenuation_image_template.fill(np.expand_dims(umap_zoomed[50,:,:], axis=0)) # random CT image
            sens_image = self.__get_sensitivity__(ct_image) # sensitivity image 
            theta, tx, ty, sx, sy = generate_random_transform_values()
            ct_image_transform = affine_transform_2D_image(theta, tx, ty, sx, sy,ct_image) # CT of transformed image
            sens_image_transform = self.__get_sensitivity__(ct_image_transform.clone())

        else:
            NotImplementedError
        ## returning the sensitivity map of the original image and the transformed image, as well as the transformed CT image
        
        ## 3-1
#         return np.array([np.squeeze(sens_image.as_array()),np.squeeze(ct_image.as_array()), np.squeeze(ct_image_transform.as_array())]), sens_image_transform.as_array()
        
        
        ## 3-2
#         return np.array([np.squeeze(sens_image.as_array()),np.squeeze(ct_image.as_array())- np.squeeze(ct_image_transform.as_array())]), np.array([np.squeeze(sens_image.as_array())- np.squeeze(sens_image_transform.as_array())]) 

        ## 2-1
        return np.array([np.squeeze(sens_image.as_array()), np.squeeze(ct_image_transform.as_array())]), sens_image_transform.as_array()
    
    
#         return np.array([np.squeeze(sens_image.as_array())]), np.squeeze(ct_image.as_array()), np.squeeze(ct_image_transform.as_array()),np.squeeze(sens_image_transform.as_array())