import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SimpleCNN(nn.Module):
    """ 
    A CNN for PET sensitivity estimation.
    Consists of 3 convolutional layers with ReLU activation. 
    Kernel sizes are 15x15, 9x9, 5x5, 3x3, 7x7, 15x15, 3x3, 3x3.
    Kernel sizes are designed to start with a large kernel size and then gradually decrease the kernel size 
    in order to capture large scale features and then small scale features.
    The kernels size then ramps back up to preserve edge information.
    Input:
        2 images (sensitivity with no motion and attenuation correction map)
    Output:
        1 image (sensitivity with motion)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 15, padding=7)
        self.conv2 = nn.Conv2d(16, 32, 9, padding=4)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 7, padding=3)
        self.conv6 = nn.Conv2d(32, 16, 15, padding=7)
        self.conv7 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv8 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x[:,1,:,:]=x[:,1,:,:].mul(10) # multiply the attenuation correction map by the ratio of the sensitivity with no motion to the sensitivity with motion 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return x
