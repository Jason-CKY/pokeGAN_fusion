import torch.nn as nn
import torch.nn.functional as F
import torch
from models.spectral_normalization import SpectralNorm

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.ngpu = int(config['ngpu'])
        nz = int(config['nz'])
        ngf = int(config['ngf'])
        nc = int(config['nc'])
        label_nc = int(config['label_nc'])
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 16),
            # nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
    
        self.input_deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
            # state size. (ngf*8) x 4 x 4
        )
        self.label_deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( label_nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
            # state size. (ngf*8) x 4 x 4
        )
    
    def forward(self, input, label):
        x = self.input_deconv(input)
        y = self.label_deconv(label)
        x = torch.cat([x, y], 1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.ngpu = int(config['ngpu'])
        ndf = int(config['ndf'])
        nc = int(config['nc'])
        label_nc = int(config['label_nc'])
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            SpectralNorm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            SpectralNorm(nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
            # output: 1x1x1
        )
        self.input_conv = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d( nc, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf/2) x 64 x 64
        )
        self.label_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d( label_nc, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf/2) x 64 x 64
        )

    def forward(self, input, label):
        x = self.input_conv(input)
        y = self.label_conv(label)
        x = torch.cat([x, y], 1)
        return self.main(x)