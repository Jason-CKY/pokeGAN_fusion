from PIL import Image
import glob
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn.functional as F
import math
import re

class PokemonDataset(Dataset):
  def __init__(self, dataroot, config, transform=transforms.ToTensor()):
    self.paths = glob.glob(dataroot+'/**/*.png', recursive=True)
    self.transform = transform
    self.df = pd.read_csv(os.path.join(dataroot, "pokemon.csv"))
    self.config = config
    self.label_dictionary = {
      'Grass' : 0, 
      'Fire' : 1, 
      'Water' : 2, 
      'Bug' : 3, 
      'Normal' : 4, 
      'Poison' : 5, 
      'Electric' : 6, 
      'Ground' : 7, 
      'Fairy' : 8, 
      'Fighting' : 9, 
      'Psychic' : 10, 
      'Rock' : 11, 
      'Ghost' : 12, 
      'Ice' : 13, 
      'Dragon' : 14, 
      'Dark' : 15, 
      'Steel' : 16, 
      'Flying' : 17
  }
  def __getitem__(self, index):
    image = Image.open(self.paths[index]).convert('RGB')
    image = self.transform(image)
    str_ = self.paths[index].split(os.path.sep)[-1].split('.')[0].split('-')[0]
    pokedex_entry = int(re.findall("\d+", str_)[0])
    # pokedex_entry = int(self.paths[index].split(os.path.sep)[-1].split('.')[0].split('-')[0])
      
    primary_type = self.df.iloc[pokedex_entry-1, 1]
    secondary_type = self.df.iloc[pokedex_entry-1, 2]
    try:
      primary_type = self.label_dictionary[primary_type] # if not math.isnan(primary_type) else -1
      secondary_type = self.label_dictionary[secondary_type] # if not math.isnan(secondary_type) else -1
    except KeyError:
      secondary_type = -1

    onehot_label = F.one_hot(torch.tensor(primary_type), num_classes=int(self.config['label_nc']))
    y_fill = torch.zeros(int(self.config['label_nc']), int(self.config['image_size']), int(self.config['image_size']))
    y_fill[primary_type] = 1
    if secondary_type != -1:     
      onehot_label += F.one_hot(torch.tensor(secondary_type), num_classes=int(self.config['label_nc']))
      y_fill[secondary_type] = 1
    
    return (image, onehot_label.view(int(self.config['label_nc']), 1, 1), y_fill)

  def __len__(self):
    return len(self.paths)
