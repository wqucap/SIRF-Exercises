import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import os

import numpy as np
from tqdm import tqdm
from CNN import SimpleCNN
from Unet import UNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.__version__)
print(device)

net = SimpleCNN()
#net = UNet()
net.to(device)
network_type = type(net).__name__

# ###### Load the state dictionary from the previous training
# state_dict = torch.load('model_SimpleCNN_10000.pt')

# # Apply the state dictionary to your model
# net.load_state_dict(state_dict)

lr = 0.00001

loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_loss_history = []
valid_loss_history = []
valid_loss_history_mean = []

############## 
# Trainset

# Load the data from the .pth file
X_loaded, y_loaded = torch.load('train_data_3-1_1000.pth')

# Create a TensorDataset from the loaded data
loaded_dataset = torch.utils.data.TensorDataset(X_loaded, y_loaded)

train_dataloader = torch.utils.data.DataLoader(loaded_dataset, batch_size=20, shuffle=True)

#Valid
# Load the arrays
X_valid = np.load('X_valid.npy')
y_valid = np.load('y_valid.npy')

# Convert arrays to tensors
X_valid_tensor = torch.from_numpy(X_valid)
y_valid_tensor = torch.from_numpy(y_valid)

# Create a TensorDataset and DataLoader
valid_dataset = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=20, shuffle=True)

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
for epoch in tqdm(range(5), desc='Epochs'): # 5 full passes over the data

    net.train() # prep model for training
    for data in tqdm(train_dataloader, desc='Training'):  # `data` is a batch of data
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

    print("\n Epoch: ", epoch, "Train loss: ", train_loss_history[-1], "Valid loss: ", valid_loss_history_mean[-1])

#torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__), "model_cnn_large2.pt"))
torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__), f"model_{network_type}_{len(train_dataloader.dataset)}.pt"))  

plt.plot(train_loss_history, label = "train", color = "red")

x = np.arange(0, len(valid_loss_history_mean))
x*=len(train_loss_history)//(len(valid_loss_history_mean)-1)
plt.scatter(x, valid_loss_history_mean, label = "valid average", color = "blue")

plt.legend()
#plt.savefig(os.path.join(os.path.dirname(__file__), "loss_history_CNN_large.png"))
plt.savefig(os.path.join(os.path.dirname(__file__), f"loss_history_{network_type}_{len(train_dataloader.dataset)}.png"))
plt.show()