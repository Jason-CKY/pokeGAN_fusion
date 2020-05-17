import os
import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset

class PokemonDataset(Dataset):
    def __init__(self, fnames, transforms=None):
        self.size = 50
        self.label_dict = {"Cat": 0, "Dog": 1}
        if fnames[-1] == '':
            fnames = fnames[:-1]
        self.fnames = fnames
        self.labels = [fname.split('\\')[-2] for fname in self.fnames]
        self.catCount = 0
        self.dogCount = 0
        for label in self.labels:
            if label == 'Cat':
                self.catCount += 1
            elif label == 'Dog':
                self.dogCount += 1

        self.labels = [np.eye(len(self.label_dict))[self.label_dict[label]] for label in self.labels]
        self.labels = [torch.Tensor(label) for label in self.labels]
        self.transforms = transforms

    def rescale_image(self, image):
        h, w = image.shape[:2]
        if h>w:
            scale = self.size / h
        else:
            scale = self.size / w
        if scale < 1:
            image = cv2.resize(image, (int(h*scale), int(w*scale)), cv2.INTER_AREA)
        else:
            image = cv2.resize(image, (int(h*scale), int(w*scale)), cv2.INTER_CUBIC)
        h, w = image.shape[:2]
        delta_w = self.size - w
        delta_h = self.size - h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        # top, bottom = 0, delta_h
        # left, right = 0, delta_w
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0.0])

        return image
        
    def __getitem__(self, index):
        image = cv2.imread(self.fnames[index], cv2.IMREAD_GRAYSCALE)
        image = self.rescale_image(image)
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)
            # image = self.transforms(self.images[index]).float()
        return (image, label)
        # return (image, self.labels[index])

    def __len__(self):
        return len(self.fnames)
        # return len(self.images)

    def describe(self):
        total = self.catCount + self.dogCount
        print("Cats: {}".format(round(self.catCount/total, 3)))
        print("Dogs: {}".format(round(self.dogCount/total, 3)))
