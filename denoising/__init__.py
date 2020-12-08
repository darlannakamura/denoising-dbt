import numpy as np
from denoising.utils import *

# from denoising.bm3d import BM3D
# from denoising.non_local_means import NLM
# from denoising.bilateral_filter import bilateral_filter
# from denoising.wavelet import wavelet_soft_thresholding, \
#     wavelet_hard_thresholding
# from denoising.wiener import wiener_filter

from tqdm import tqdm

def _batch_algorithm_implementation(single_image_denoising_algorithm, noisy_images: np.ndarray, noise_std_dev: float, show_progress:bool = False, *args, **kwargs):
    validate_array_input(noisy_images)
    validate_if_noise_std_dev_is_a_float(noise_std_dev)

    filtered_images = []

    if show_progress:
        for i in tqdm(range(noisy_images.shape[0])):
            filtered_images.append(
                single_image_denoising_algorithm(
                    noisy_images[i, :,:,:], 
                    *args,
                    **kwargs
                )
            )
    else:
        for i in range(noisy_images.shape[0]):
            filtered_images.append(
                single_image_denoising_algorithm(
                    noisy_images[i, :,:,:], 
                    *args,
                    **kwargs
                )
            )
    
    filtered_images = np.array(filtered_images)

    return filtered_images
