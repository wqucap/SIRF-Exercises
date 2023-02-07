#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import os

import numpy as np

from CNN import SimpleCNN
from UNet import UNet

from odl_funcs.ellipses import EllipsesDataset

from sirf.STIR import MessageRedirector, ImageData, AcquisitionData, AcquisitionModelUsingRayTracingMatrix
import sirf.STIR
from sirf.Utilities import examples_data_path
data_path = os.path.join(examples_data_path('PET'), 'thorax_single_slice')

msg = MessageRedirector()
sirf.STIR.set_verbosity(0)

attn_image = ImageData(os.path.join(data_path, 'attenuation.hv'))
template = AcquisitionData(os.path.join(data_path, 'template_sinogram.hs'))
radon_transform = AcquisitionModelUsingRayTracingMatrix()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

attn_image = ImageData(os.path.join(data_path, 'attenuation.hv'))
template = AcquisitionData(os.path.join(data_path, 'template_sinogram.hs'))
radon_transform = AcquisitionModelUsingRayTracingMatrix()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mini_batch_train = 2**6
mini_batch_valid = 2**2

train_dataloader = torch.utils.data.DataLoader( \
    EllipsesDataset(radon_transform, attn_image, template, mode="train", n_samples = 2**14) \
    , batch_size=mini_batch_train, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader( \
    EllipsesDataset(radon_transform, attn_image, template, mode="valid", n_samples = 2**3) \
    , batch_size=mini_batch_valid, shuffle=False)

net = UNet(in_ch = 2, out_ch = 1, chs = (8, 16, 32, 32, 32), n_latent = 2, k_size_latent=3, padding = 8)
net.to(device)

lr = 0.0001

#%%

loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_loss_history = []
valid_loss_history = []
valid_loss_history_mean = []

sum_valid_loss = 0
net.eval() # eval mode
with torch.no_grad():
    for valid in valid_dataloader:
        X, y = valid
        net.zero_grad() 
        optimizer.zero_grad()
        loss = loss_function(net(X.to(device)), y.to(device))
        valid_loss_history.append(float(loss))
        sum_valid_loss += float(loss)

    valid_loss_history_mean.append(sum_valid_loss/len(valid_dataloader))
for epoch in range(5): # 5 full passes over the data

    net.train() # prep model for training
    for data in train_dataloader:  # `data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
        optimizer.zero_grad()  # zero the gradient buffers
        output = net(X.to(device))  # pass in the reshaped batch
        loss = loss_function(output, y.to(device))  # calc and grab the loss value
        train_loss_history.append(float(loss))
        
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients

    sum_valid_loss = 0
    net.eval() # eval mode
    with torch.no_grad():
        for valid in valid_dataloader:
            X, y = valid
            net.zero_grad()
            optimizer.zero_grad()
            loss = loss_function(net(X.to(device)), y.to(device))
            valid_loss_history.append(float(loss))
            sum_valid_loss += float(loss)

        valid_loss_history_mean.append(sum_valid_loss/len(valid_dataloader))
    lr/=2
    optimizer = optim.Adam(net.parameters(), lr=lr)

    torch.cuda.empty_cache() #Clearing GPU cache

    print("Epoch: ", epoch, "Train loss: ", train_loss_history[-1], "Valid loss: ", valid_loss_history_mean[-1])

torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__), "model_unet.pt"))
    
plt.plot(train_loss_history, label = "train", color = "red")

x = np.arange(0, len(valid_loss_history_mean))
x*=len(train_loss_history)//(len(valid_loss_history_mean)-1)
plt.scatter(x, valid_loss_history_mean, label = "valid average", color = "blue")

plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "loss_history_UNet.png"))