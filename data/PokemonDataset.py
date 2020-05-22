import os
from PIL import Image
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PokemonDataset(Dataset):
  def __init__(self, dataroot, transform=transforms.ToTensor()):
    self.paths = glob.glob(dataroot+'/**/*.png', recursive=True)
    self.transform = transform
  
  def __getitem__(self, index):
    image = Image.open(self.paths[index]).convert('RGB')
    image = self.transform(image)
    return (image, 1)

  def __len__(self):
    return len(self.paths)
