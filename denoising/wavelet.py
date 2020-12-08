import numpy as np
from denoising.utils import *
from skimage.restoration import denoise_wavelet
from denoising import _batch_algorithm_implementation
from tqdm import tqdm

def wavelet_soft_thresholding(noisy_images: np.ndarray, noise_std_dev: float, show_progress:bool = False) -> np.ndarray:
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

    if show_progress:
        for i in tqdm(range(noisy_images.shape[0])):
            filtered_images.append(
                denoise_wavelet(
                    noisy_images[i, :,:,0], 
                    sigma=noise_std_dev,
                    multichannel=False,
                    mode='soft'
                )
            )
    else:
        for i in range(noisy_images.shape[0]):
            filtered_images.append(
                denoise_wavelet(
                    noisy_images[i, :,:,0], 
                    sigma=noise_std_dev,
                    multichannel=False,
                    mode='soft'
                )
            )
    
    filtered_images = np.array(filtered_images)

    return filtered_images
