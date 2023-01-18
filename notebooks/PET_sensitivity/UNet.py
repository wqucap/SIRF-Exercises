import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Block(nn.Module):
    """ 
    A block of two convolutional layers with ReLU activation. 
    """
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 5, padding = (2,2), padding_mode = 'reflect')
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 5, padding = (2,2), padding_mode = 'reflect')
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class Encoder(nn.Module):
    """ 
    Encoder of the U-Net.
    """
    def __init__(self, chs=(2,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) # encoder blocks
        self.pool       = nn.MaxPool2d(2) # pooling layer
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x) 
            ftrs.append(x) 
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    """ 
    Decoder of the U-Net.
    """
    def __init__(self, chs=(256, 128, 64)):
        super().__init__()
        self.chs         = chs # number of channels
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)]) # upconvolutions
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) # decoder blocks
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1) # add skip connections
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        """ Crops the encoder features to the size of the decoder features. """
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
    
class UNet(nn.Module):
    """ 
    U-Net architecture.
    """
    def __init__(self, enc_chs=(2,64,128,256), dec_chs=(256, 128, 64), num_class=1, retain_dim=True, out_sz=(155,155)):
        super().__init__()
        self.encoder     = Encoder(enc_chs) # encoder
        self.decoder     = Decoder(dec_chs) # decoder
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1, padding= (0,0)) # head
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x) # encoder features
        dec_ftrs = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:]) # decoder features
        out      = self.head(dec_ftrs) # head
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out