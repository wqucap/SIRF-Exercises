import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import os

import pandas as pd

import numpy as np

from UNet import UNet
from CNN import SimpleCNN
from OriginalUNet import UNet as OriginalUNet

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

mini_batch_data = 10
mini_batch_valid = 1
train_dataloader = torch.utils.data.DataLoader( \
    EllipsesDataset(radon_transform, attn_image, template, mode="train", n_samples = 10000) \
    , batch_size=mini_batch_data, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader( \
    EllipsesDataset(radon_transform, attn_image, template, mode="valid", n_samples = 5) \
    , batch_size=mini_batch_valid, shuffle=False)

net = OriginalUNet(2,1)
net.to(device)

lr = 0.0001

loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
 
train_loss_history = []
valid_loss_history = []
valid_loss_history_sum = []


for epoch in range(5): # 5 full passes over the data
    for data in train_dataloader:  # `data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
        output = net(X.to(device))  # pass in the reshaped batch
        loss = loss_function(output, y.to(device))  # calc and grab the loss value
        train_loss_history.append(loss)
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    sum = []
    for validation in valid_dataloader: # pass through validation set for each epoch
        X_val, y_val = validation
        output = net(X_val.to(device))
        loss_valid = loss_function(output, y_val.to(device))
        valid_loss_history.append(loss_valid)
        sum.append(loss_valid.item())
    valid_loss_history_sum.append(np.sum(sum)/len(sum)) # average validation loss for epoch
    print(f"train loss: {loss} validation loss: {loss_valid}") # print loss. We hope loss (a measure of wrong-ness) declines!

torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__), "model_orig_unet.pt"))

loss_history = []
for i in train_loss_history:
    loss_history.append(i.item())

valid_history = []
for i in valid_loss_history:
    valid_history.append(i.item())
    
plt.plot(loss_history, label = "train", color = "red")

#x = np.arange(0, len(valid_history))
#x*=len(loss_history)//len(valid_history)
#plt.scatter(x, valid_history, label = "valid")

x = np.arange(1, len(valid_loss_history_sum)+1)
x*=len(loss_history)//len(valid_loss_history_sum)
plt.scatter(x, valid_loss_history_sum, label = "valid average", color = "blue")

#df = pd.DataFrame({"train": loss_history, "valid": valid_history})

plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "loss_history_Original_UNet.png"))