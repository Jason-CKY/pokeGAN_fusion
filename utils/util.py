import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def get_GAN_onehot_labels(primary_type, secondary_type, bs, label_nc):
    label_dictionary = {
    'grass' : 0, 
    'fire' : 1, 
    'water' : 2, 
    'bug' : 3, 
    'normal' : 4, 
    'poison' : 5, 
    'electric' : 6, 
    'ground' : 7, 
    'fairy' : 8, 
    'fighting' : 9, 
    'psychic' : 10, 
    'rock' : 11, 
    'ghost' : 12, 
    'ice' : 13, 
    'dragon' : 14, 
    'dark' : 15, 
    'steel' : 16, 
    'flying' : 17
    }
    
    onehot_label = F.one_hot(torch.tensor(label_dictionary[primary_type.lower()]), num_classes=label_nc)
    if secondary_type is not None:     
        onehot_label += F.one_hot(torch.tensor(label_dictionary[secondary_type.lower()]), num_classes=label_nc)

    output_label = []
    for _ in range(bs):
        output_label.append(list(onehot_label.numpy()))
    output_label = torch.tensor(output_label).float().view(bs, label_nc, 1, 1)

    return output_label

def get_random_labels(bs, num_classes, image_size, p=0.5):
    '''
        Generate random onehot vector of labels for generator class.
        Since pokemon can have 1-2 types, there is a <probability, default=0.5> for the 
        onehot vector to have 2 indices with value of 1.
        Args:
            bs: batch size of the data
            num_classes: number of label classes
            p: probability of having a second type
        Return:
            onehot: onehot label for G
            c_fill: onehot label for D
    '''
    fill = torch.zeros([num_classes, num_classes, image_size, image_size])
    onehot = (torch.rand(bs) * num_classes).type(torch.int64)
    for i in range(num_classes):
        fill[i, i, :, :] = 1

    c_fill = fill[onehot]
    onehot = F.one_hot(onehot, num_classes=num_classes)

    for x, fill in zip(onehot, c_fill):
        if random.random() <= p:
            indice = (x==1).nonzero()
            new_indice = random.randint(0, num_classes-1)
            while new_indice == indice:
                new_indice = random.randint(0, num_classes-1)
            fill[new_indice] = 1
            x[new_indice] = 1
    return onehot.float().view(bs, num_classes, 1, 1), c_fill

def flip_label(label, p):
    '''
        Randomly flip the labels based on probability p
        Args:
            label: tensor of labels
            p: probability
        Return:
            output: new labels tensor with flipped labels
    '''
    real = 0.9
    fake = 0
    output = []
    for num in label:
        if random.random() < p:
            if num > 0.5:
                output.append(fake)
            else:
                output.append(real)
        else:
            output.append(num)
    output = torch.tensor(output)
    return output

def tensor2im(image_tensor, imtype=np.uint8, normalize=False):  
    '''
        Converts a Tensor into a Numpy array
        Args:
            image_tensor: the tensor to be converted
            imtype: the desired type of the converted numpy array
            normalize: denormalize array if true
        Return:
            image_numpy: image numpy array or list of image numpy array
    '''
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def grid_image(image_list, shape, normalize=True, save_path = None):
    """Display a grid of images in a single figure with matplotlib.
    
    Parameters
    ---------
    image_list: List of np.arrays compatible with plt.imshow or pytorch 4D tensor with (B x C x H x W)
    
    shape: shape of the grid (5x5)/(8x8)
    """
    if not torch.is_tensor(image_list):
        images = image_list.copy()
    else:
        images = image_list
        
    images = images[:shape[0]*shape[1]]
    fig = plt.figure(figsize=(8,8))
    for n, image in enumerate(images):
        a = fig.add_subplot(shape[0], shape[1], n + 1)
        if image.ndim == 2:
            plt.gray()
        a.axis("off")
        if torch.is_tensor(image):
            image = tensor2im(image, normalize=normalize)
        a.imshow(image)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    del images