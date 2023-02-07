import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def pad_to(x, stride):
    """ 
    Pads the input tensor to the next multiple of stride.
    """
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    """
    Function to unpad the input tensor.
    """
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

class Block(nn.Module):
    """ 
    A block of two convolutional layers with ReLU activation. 
    """
    
    def __init__(self, in_ch, out_ch, leaky = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 5, stride = 1, padding=(2,2))
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size = 5, stride = 1, padding=(2,2))
        # leaky relu activation function
        if leaky:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
    
class DownConv(nn.Module):
    """
    A convolutional layer with stride 2.
    """
    
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size = 5, stride = 2, padding=(2,2))
        # leaky relu activation function
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.lrelu(self.conv(x))
        return x
    
class UpConv(nn.Module):
    """
    A transposed convolutional layer with stride 2
    """
    
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, in_ch, kernel_size = 5, stride = 2, padding = (2,2), output_padding=(1,1))
        # leaky relu activation function
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.lrelu(self.conv(x))
        return x
    
class ConvRelu(nn.Module):
    """
    A convolutional layer with stride 1.
    """
    
    def __init__(self, in_ch, out_ch, leaky = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding=(1,1))
        # relu activation function
        if leaky:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv(x))
        return x
    
class Encoder(nn.Module):
    """ 
    Encoder of the U-Net.
    Used Leakly ReLU activation function.
    """
    def __init__(self, chs):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(chs)-1):
            self.blocks.append(Block(chs[i], chs[i+1], leaky = True))
        self.down_convs = nn.ModuleList()
        for i in range(len(chs)-1):
            self.down_convs.append(DownConv(chs[i+1]))
    
    def forward(self, x):
        skip_connections = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            skip_connections.append(x)
            x = self.down_convs[i](x)
        return x, skip_connections
    
class Latent(nn.Module):
    """
    Latent layer of the U-Net.
    n convolutional layers with leaky ReLU activation.
    """
    
    def __init__(self, n, in_ch, k_size):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n):
            self.convs.append(nn.Conv2d(in_ch, in_ch, kernel_size = k_size, stride = 1, padding=(k_size//2,k_size//2)))
            
        
    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = nn.LeakyReLU(0.2, inplace=True)(x)
        return x
            
class Decoder(nn.Module):
    """
    Decoder of the U-Net.
    Uses Relu activation function.
    """
    def __init__(self, chs):
        super().__init__()
        self.up_convs = nn.ModuleList()
        for i in range(1, len(chs)):
            self.up_convs.append(UpConv(chs[-i]))
        self.blocks = nn.ModuleList()
        for i in range(1, len(chs)):
            self.blocks.append(Block(chs[-i], chs[-(i+1)], leaky=False))
        
    def forward(self, x, skip_connections):
        for i in range(1, len(self.blocks)+1):
            x = self.up_convs[i-1](x)
            x = x + skip_connections[-i]
            x = self.blocks[i-1](x)
        return x
    
class UNet(nn.Module):
    """
    U-Net model.
    """
    def __init__(self, in_ch, out_ch, chs, n_latent, k_size_latent, padding):
        super().__init__()
        self.inc = ConvRelu(in_ch, chs[0], leaky = True)
        self.encoder = Encoder(chs)
        self.latent = Latent(n_latent, chs[-1], k_size_latent)
        self.decoder = Decoder(chs)
        self.outc = ConvRelu(chs[0], out_ch, leaky = False)
        self.padding = padding
        
    def forward(self, x):
        x, pad = pad_to(x, self.padding)
        x = self.inc(x)
        x, skip_connections = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x, skip_connections)
        x = self.outc(x)
        x = unpad(x, pad)
        return x
        
            
    
