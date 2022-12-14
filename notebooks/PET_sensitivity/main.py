import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import os

from UNet import UNet

from odl_funcs.ellipses import EllipsesDataset

from sirf.STIR import *
import sirf.STIR
from sirf.Utilities import examples_data_path
data_path = os.path.join(examples_data_path('PET'), 'thorax_single_slice')

sirf.STIR.set_verbosity(0)

attn_image = ImageData(os.path.join(data_path, 'attenuation.hv'))
template = AcquisitionData(os.path.join(data_path, 'template_sinogram.hs'))
radon_transform = AcquisitionModelUsingRayTracingMatrix()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mini_batch = 10
train_dataloader = torch.utils.data.DataLoader( \
    EllipsesDataset(radon_transform, attn_image, template, mode="train", n_samples = 1000) \
    , batch_size=mini_batch, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader( \
    EllipsesDataset(radon_transform, attn_image, template, mode="valid", n_samples = 100) \
    , batch_size=mini_batch, shuffle=False)

net = UNet()
net.to(device)

lr = 0.0001

loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_loss_history = []
valid_loss_history = []

for epoch in range(5): # 5 full passes over the data
    for data, validation in zip(train_dataloader, valid_dataloader):  # `data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
        output = net(X.to(device))  # pass in the reshaped batch
        loss = loss_function(output, y.to(device))  # calc and grab the loss value
        train_loss_history.append(loss)
        #valid_loss_history.append(loss_function(net(validation[0].to(device)), validation[1].to(device)))
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
        optim_state=optimizer.state_dict()
    lr /= 2
    optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer.load_state_dict(optim_state)
    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 
    
torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__), "model.pt"))

loss_history = []
for i in train_loss_history:
    loss_history.append(i.item())
    
plt.plot(loss_history)
plt.savefig(os.path.join(os.path.dirname(__file__), "loss_history.png"))