import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SimpleCNN(nn.Module):
    """ 
    A CNN for PET sensitivity estimation.
    Consists of 3 convolutional layers with ReLU activation. 
    Kernel sizes are 5x5, 3x3, 3x3. 
    Input:
        2 images (sensitivity with no motion and attenuation correction map)
    Output:
        1 image (sensitivity with motion)
    """

    def __init__(self, in_ch=2, out_ch=1, image_size=(128,128)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 5, padding = (2,2), padding_mode = 'reflect')
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding = (1,1), padding_mode = 'reflect')
        self.conv3 = nn.Conv2d(128, out_ch, 3, padding = (1,1), padding_mode = 'reflect')

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


