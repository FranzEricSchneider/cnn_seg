import numpy
from skimage.transform import resize


def resize_mask(mask, size=(3036, 4024)):
    """
    Load a numpy (.npy) mask of unknown size, resize it to a fixed size, and
    cast as a bool

    Arguments:
        mask: ????
        size: (height, width) of the desired final (resized) mask in pixels

    Returns: bool numpy array of the desired size
    """
    return resize(
        mask, size, order=0, preserve_range=True, anti_aliasing=False
    ).astype(bool)


def mask_rgb(mask):
    """
    Arguments:
        mask: bool numpy array

    Returns: Numpy array RGB image of the same size as the mask, which is green
        where the mask was true
    """
    mask_im = numpy.ones(mask.shape + (3,)) * [0, 0.5, 0]
    mask_im[~mask] = 0
    return mask_im
