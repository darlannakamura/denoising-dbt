import numpy as np

def KSVD(noisy_images: np.ndarray, noise_std_dev: float) -> np.ndarray:
    """
    Params:
    noisy_images: receive noisy_images with shape (IMG, M, N, C), 
    where IMG is the quantity of noisy images 
    with dimensions MxN and
    C represents the color dimensions, i.e. if the images are colored,
    C = 3 (RGB), otherwise if C = 1, is grayscale. 

    noise_std_dev: the standart deviation from noise.
    """
    print('Not implemented yet.')
