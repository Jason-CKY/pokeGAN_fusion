import torch.nn as nn
import torch.nn.functional as F

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
        self.ngpu = config['ngpu']
        nz = config['nz']
        ngf = config['ngf']
        nc = config['nc']
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*32, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*32) x 4 x 4
            nn.ConvTranspose2d(in_channels=ngf*32, out_channels=ngf*16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),     
            # state size: (ngf*16) x 8 x 8
            nn.ConvTranspose2d(in_channels=ngf*16, out_channels=ngf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),     
            # state size: (ngf*8) x 16 x 16
            nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),     
            # state size: (ngf*4) x 32 x 32
            nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # state size: (nc*2) x 64 x 64
            nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # state size: (ngf) x 128 x 128
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size: (nc) x 256 x 256
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.ngpu = config['ngpu']
        ndf = config['ndf']
        nc = config['nc']
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            # state size: (ndf*2) x 64 x 64
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            # state size: (ndf*4) x 32 x 32
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 16 x 16
            nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 8 x 8
            nn.Conv2d(ndf*16, ndf*32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*32) x 4 x 4
            nn.Conv2d(ndf*32, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # output: 1x1x1
        )

    def forward(self, input):
        return self.main(input)