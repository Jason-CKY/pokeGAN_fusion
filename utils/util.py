import numpy as np
import torch
import random
import matplotlib.pyplot as plt

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):  
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

def grid_image(image_list, shape = None):
    """Display a grid of images in a single figure with matplotlib.
    
    Parameters
    ---------
    image_list: List of np.arrays compatible with plt.imshow.
    
    shape: shape of the grid (5x5)/(8x8)
    """
    images = image_list.copy()
    images = images[:shape[0]*shape[1]]
    n_images = len(images)
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    for n, image in enumerate(images):
        a = fig.add_subplot(shape[0], shape[1], n + 1)
        if image.ndim == 2:
            plt.gray()
        a.axis("off")
        a.imshow(image)
    plt.show()
    del images