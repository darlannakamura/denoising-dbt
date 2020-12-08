import numpy as np
from denoising.utils import *
from skimage.restoration import denoise_nl_means

from tqdm import tqdm

def NLM(noisy_images: np.ndarray, noise_std_dev: float, show_progress=False) -> np.ndarray:
    validate_array_input(noisy_images)
    validate_if_noise_std_dev_is_a_float(noise_std_dev)

    filtered_images = []

    if show_progress:
        for i in tqdm(range(noisy_images.shape[0])):
                filtered_images.append(
                    denoise_nl_means(
                        noisy_images[i, :,:,0], 
                        sigma=noise_std_dev
                    )
                )
        
    else:
        for i in range(noisy_images.shape[0]):
            filtered_images.append(
                denoise_nl_means(
                    noisy_images[i, :,:,0], 
                    sigma=noise_std_dev
                )
            )
    
    filtered_images = np.array(filtered_images)

    return filtered_images
