import configparser
import os
from utils import util
from data.PokemonDataset import PokemonDataset
from torchvision import transforms
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models.dcgan import Generator, Discriminator, weights_init
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import cv2

config = configparser.ConfigParser()
config.read(os.path.join("cfg", "config.ini"))
config = config['PARAMETERS']


def main():
    dataset = PokemonDataset(dataroot=config['dataroot'],
                            transform=transforms.Compose([
                                transforms.Resize(int(config['image_size'])),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]),
                            config=config)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size= int(config['batch_size']),
                                            shuffle=True, num_workers= int(config['workers']))

    device = torch.device("cuda:0" if (torch.cuda.is_available() and int(config['ngpu']) > 0) else "cpu")

    # Create the generator
    netG = Generator(config).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (int(config['ngpu']) > 1):
        netG = nn.DataParallel(netG, list(range(int(config['ngpu']))))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    # netG.eval()
    g = torch.load('output\\6499_netG.pt')
    g.copy()
    netG.load_state_dict(g)
    netG.eval()
    # Print the model
    print(netG)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, int(config['nz']), 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    netG.eval()
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake, padding=2, normalize=True)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save("output.png")


if __name__ == '__main__':
    main()