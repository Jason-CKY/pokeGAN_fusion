import configparser
import os
from utils import util
import torch
import numpy as np
from PIL import Image
from models.cdcgan import Generator
import torch.nn as nn
import torchvision.utils as vutils
import argparse

config = configparser.ConfigParser()
config.read(os.path.join("cfg", "config.ini"))
config = config['PARAMETERS']

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Images using cDCGAN')
    
    parser.add_argument('--primary_type',
                        help='desired pokemon primary type',
                        required=True,
                        type=str)
    parser.add_argument('--secondary_type',
                        help='desired pokemon primary type (optional)',
                        required=False,
                        type=str)
    parser.add_argument('-bs', '--batch_size',
                        help='batch size of output',
                        required=True,
                        type=int)
    parser.add_argument('--output',
                        help='output directory',
                        required=True,
                        type=str)
    parser.add_argument('--grid',
                        help='output directory',
                        action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pt_path = config['pretrained']+'_netG.pt'
    if not os.path.isfile(pt_path):
        print(f"{pt_path} pt file does not exist.")
        return

    device = torch.device("cuda:0" if (torch.cuda.is_available() and int(config['ngpu']) > 0) else "cpu")

    # Create the generator
    netG = Generator(config).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (int(config['ngpu']) > 1):
        netG = nn.DataParallel(netG, list(range(int(config['ngpu']))))

    netG.load_state_dict(torch.load(pt_path))
    netG.eval()
    # Print the model
    print(netG)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(args.batch_size, int(config['nz']), 1, 1, device=device)
    fixed_onehot = util.get_GAN_onehot_labels(args.primary_type, args.secondary_type, args.batch_size, int(config['label_nc'])).to(device)

    with torch.no_grad():
        fake = netG(fixed_noise, fixed_onehot).detach().cpu()
        if args.grid:
            grid = vutils.make_grid(fake, padding=2, normalize=True)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(os.path.join(args.output, "grid.png"))
        else:
            for idx, tensor in enumerate(fake):
                image = util.tensor2im(tensor, normalize=True)
                im = Image.fromarray(image)
                im.save(os.path.join(args.output, f"{args.primary_type}{'_' + args.secondary_type if args.secondary_type is not None else ''}_{idx}.png"))




if __name__ == '__main__':
    main()