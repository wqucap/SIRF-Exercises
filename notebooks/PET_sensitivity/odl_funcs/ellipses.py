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

import matplotlib.pyplot as plt

from scipy.ndimage import zoom

def generate_random_transform_values():
    theta = np.random.uniform(-np.pi/8, np.pi/8)
    tx, ty = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
    sx, sy = np.random.uniform(0.9, 1.0), np.random.uniform(0.9, 1.0)
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
            random_phantom_array = make_max_n(random_phantom(self.attenuation_image_template.shape, 20), 0.096*2)
            ct_image = self.attenuation_image_template.clone().fill(random_phantom_array) # random CT image
            sens_image = self.__get_sensitivity__(ct_image) # random sensitivity image
            theta, tx, ty, sx, sy = generate_random_transform_values()
            ct_image_transform = affine_transform_2D_image(theta, tx, ty, sx, sy, ct_image) # CT of transformed image
            ct_image_transform.move_to_scanner_centre(self.template)
            sens_image_transform = self.__get_sensitivity__(ct_image_transform) # sensitivity map of transformed image

        elif self.mode == "valid":
            brainweb.seed(np.random.randint(500,1500))
            for f in tqdm([fname], desc="mMR ground truths", unit="subject"):
                vol = brainweb.get_mmr_fromfile(f, petNoise=1, t1Noise=0.75, t2Noise=0.75, petSigma=1, t1Sigma=1, t2Sigma=1)
            uMap_arr = vol['uMap'] # random CT brain image
            umap_zoomed = crop(zoom(uMap_arr, 1, order=1), 155, 155)
            ct_image = self.attenuation_image_template.fill(np.expand_dims(umap_zoomed[50,:,:], axis=0)) # random CT image
            sens_image = self.__get_sensitivity__(ct_image) # sensitivity image 
            theta, tx, ty, sx, sy = generate_random_transform_values()
            ct_image_transform = affine_transform_2D_image(theta, tx, ty, sx, sy,ct_image) # CT of transformed image
            sens_image_transform = self.__get_sensitivity__(ct_image_transform.clone())

        else:
            NotImplementedError
        
        return np.array([np.squeeze(sens_image.as_array()), np.squeeze(ct_image_transform.as_array())]), sens_image_transform.as_array()