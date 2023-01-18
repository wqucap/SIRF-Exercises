# Author: Imraj Singh

# First version: 21st of May 2022

# CCP SyneRBI Synergistic Image Reconstruction Framework (SIRF).
# Copyright 2022 University College London.

# This is software developed for the Collaborative Computatiponal Project in Synergistic Reconstruction for Biomedical Imaging (http://www.ccpsynerbi.ac.uk/).

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import torch
import numpy as np
from .misc import random_phantom, shepp_logan, affine_transform_image, random_4x4_matrix
from sirf.STIR import AcquisitionSensitivityModel

from scipy.ndimage import affine_transform

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
        self.attenuation_image_template = attenuation_image_template
        self.n_samples = n_samples
        self.template = sinogram_template
        self.one_sino = sinogram_template.get_uniform_copy(1)

        if mode == 'valid':
            self.x = self.attenuation_image_template.clone().fill(shepp_logan(self.attenuation_image_template.shape))*0.05
            self.y = self.__get_sensitivity__(self.x)

        self.primal_op_layer = radon_transform
        self.mode = mode
        np.random.seed(seed)

    def __get_sensitivity__(self, attenuation_image):
        # Forward project image then add noise
        asm_attn = AcquisitionSensitivityModel(attenuation_image, self.radon_transform)
        asm_attn.set_up(self.template)
        self.radon_transform.set_acquisition_sensitivity(asm_attn)
        self.radon_transform.set_up(self.template, attenuation_image)
        y = self.radon_transform.backward(self.one_sino)
        return y

    def __len__(self):
        # Denotes the total number of iters
        return self.n_samples

    def __getitem__(self, index):
        # Generates one sample of data
        if self.mode == "train":
            x0 = self.attenuation_image_template.clone().fill(random_phantom(self.attenuation_image_template.shape))*0.05 # random CT image
            x1 = self.__get_sensitivity__(x0) # random sensitivity image
            x2 = affine_transform_image(x0, random_4x4_matrix()) # CT of transformed image
            y = self.__get_sensitivity__(x2) # sensitivity map of transformed image

        elif self.mode == "valid":
            x0 = self.x
            x1 = self.y # random sensitivity image
            x2 = affine_transform_image(x0, random_4x4_matrix()) # CT diff map
            y = self.__get_sensitivity__(x2)

        else:
            NotImplementedError

        
        
        return np.array([np.squeeze(x1.as_array()), np.squeeze(x2.as_array())]), y.as_array()