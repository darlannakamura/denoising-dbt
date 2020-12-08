import bm3d
import numpy as np
from denoising.utils import *

def BM3D(noisy_images: np.ndarray, noise_std_dev: float) -> np.ndarray:
    """
    Params:
    noisy_images: receive noisy_images with shape (IMG, M, N, C), 
    where IMG is the quantity of noisy images 
    with dimensions MxN and
    C represents the color dimensions, i.e. if the images are colored,
    C = 3 (RGB), otherwise if C = 1, is grayscale. 

    noise_std_dev: the standart deviation from noise.
    """ 
    validate_array_input(noisy_images)
    validate_if_noise_std_dev_is_a_float(noise_std_dev)

    filtered_images = []
    for i in range(noisy_images.shape[0]):
        filtered_images.append(
            bm3d.bm3d(
                noisy_images[i, :,:,:], 
                sigma_psd=noise_std_dev, 
                stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
            )
        )
    
    filtered_images = np.array(filtered_images)

    return filtered_images
