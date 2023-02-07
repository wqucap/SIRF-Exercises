import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def pad_to(x, stride):
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

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

class Block(nn.Module):
    """ 
    A block of two convolutional layers with ReLU activation. 
    """
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 5, stride = 1, padding=(2,2))
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size = 5, stride = 1, padding=(2,2))
        # leaky relu activation function
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        return x
    
class Encoder(nn.Module):
    """ 
    Encoder of the U-Net.
    """
    def __init__(self, chs=(2,16,32,32,32,32)):
        super().__init__()
        self.chs         = chs # number of channels
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.down_convs = nn.ModuleList([nn.Conv2d(chs[i+1], chs[i+1], kernel_size = 5, stride = 2, padding=(2,2)) for i in range(len(chs)-1)])

    def forward(self, x):
        encoder_features = []
        for i in range(len(self.chs)-1):
            x = self.enc_blocks[i](x)
            encoder_features.append(x)
            x = self.down_convs[i](x)
        return encoder_features


class Decoder(nn.Module):
    """ 
    Decoder of the U-Net.
    """
    def __init__(self, chs=(32,32,32,32,16)):
        super().__init__()
        self.chs         = chs # number of channels
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size = 5, stride = 2, padding=(2,2), output_padding=(1,1)) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i+1], chs[i+1]) for i in range(len(chs)-1)]) # decoder blocks

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            x += encoder_features[i+1]
            x = self.dec_blocks[i](x)
        return x
    
class UNet(nn.Module):
    """ 
    U-Net architecture.
    """
    def __init__(self, enc_chs=(2,16,32,32,32,32), dec_chs=(32,32,32,32,16), out_chs=(16,16,3), num_class=1, retain_dim=True, out_sz=(155,155)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.conv = nn.Conv2d(dec_chs[-1], out_chs[0], kernel_size = 5, stride = 1, padding=(2,2))
        self.convs =  nn.ModuleList([nn.Conv2d(out_chs[i], out_chs[i+1], kernel_size = 5, stride = 1, padding=(2,2)) for i in range(len(out_chs)-1)])
        self.outconv = nn.Conv2d(out_chs[-1], num_class, kernel_size = 1, stride = 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        x[:,1,:,:]=x[:,1,:,:].mul(x[:,0,:,:]/x[:,1,:,:])
        x, pad = pad_to(x, 8)
        encoder_features = self.encoder(x)
        x = self.decoder(encoder_features[-1], encoder_features[::-1])
        x = self.conv(x)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
        x = self.outconv(x)
        x = unpad(x, pad)
        return x

