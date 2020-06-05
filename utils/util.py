import numpy as np
import torch
import random
import matplotlib.pyplot as plt

def randrange_1d(start, end, len):
    '''
        Create a 1d list of random float values from start <= x <= end
        Args:
            len: length of list created
        Return: 
            output: 1d list of random float values
    '''
    output = []
    for i in range(len):
        output.append(random.uniform(start, end))
    return output

def flip_label(label, p):
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