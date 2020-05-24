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
import cv2


config = configparser.ConfigParser()
config.read(os.path.join("cfg", "config_dcgan.ini"))
config = config['PARAMETERS']


def main():
    dataset = PokemonDataset(dataroot=config['dataroot'],
                            transform=transforms.Compose([
                                transforms.Resize(int(config['image_size'])),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                                transforms.RandomRotation(10),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size= int(config['batch_size']),
                                         shuffle=False, num_workers= int(config['workers']))

    device = torch.device("cuda:0" if (torch.cuda.is_available() and int(config['ngpu']) > 0) else "cpu")
    # Plot some training images
    real_batch = next(iter(dataloader))
    images = []
    for tensor in real_batch[0][:25]:
        img = util.tensor2im(tensor)
        images.append(img)
        
    util.grid_image(images, (5, 5))
    
if __name__ == '__main__':
    main()