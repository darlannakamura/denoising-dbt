import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, \
    structural_similarity as ssim
from denoising.utils import *
from typing import List

def validate_inputs(array1: np.ndarray, array2: np.ndarray):
    validate_array_input(array1)
    validate_array_input(array2)

    dimensions = len(array1.shape)
    
    for dimension in range(dimensions):
        assert array1.shape[dimension] == array2.shape[dimension], \
            f'The inputs should have same dimensions.'

def need_to_normalize(array: np.ndarray) -> bool:
    if array.min() == 0 and array.max() == 255:
        return False
    
    return True

def PSNR(ground_truth_images: np.ndarray, noisy_images: np.ndarray) -> List[float]:
    """
    Calculate the medium PSNR over the ground truth images and noisy images.
    """
    validate_inputs(ground_truth_images, noisy_images)

    psnr_acumulated = []

    quantity_of_images = ground_truth_images.shape[0]

    if need_to_normalize(ground_truth_images):
        ground_truth_images = normalize(ground_truth_images, \
            interval=(0,255), data_type='int')
    
    if need_to_normalize(noisy_images):
        noisy_images = normalize(noisy_images, \
            interval=(0,255), data_type='int')
    
    for i in range(quantity_of_images):
        psnr_image = psnr(
            ground_truth_images[i,:,:,0], 
            noisy_images[i,:,:,0],
            data_range=256
        )
        psnr_acumulated.append(psnr_image)

    # psnr_acumulated = np.array(psnr_acumulated)

    # return psnr_acumulated.mean()
    return psnr_acumulated

def SSIM(ground_truth_images: np.ndarray, noisy_images: np.ndarray) -> List[float]:
    """
    Calculate the medium SSIM over the ground truth images and noisy images.
    """
    validate_inputs(ground_truth_images, noisy_images)

    ssim_accumulated = []

    quantity_of_images = ground_truth_images.shape[0]

    if need_to_normalize(ground_truth_images):
        ground_truth_images = normalize(ground_truth_images, \
            interval=(0,255), data_type='int')
    
    if need_to_normalize(noisy_images):
        noisy_images = normalize(noisy_images, \
            interval=(0,255), data_type='int')

    for i in range(quantity_of_images):
        ssim_image = ssim(
            ground_truth_images[i,:,:,0], 
            noisy_images[i,:,:,0],
            data_range=256
        )
        ssim_accumulated.append(ssim_image)
    
    return ssim_accumulated
    # ssim_accumulated = np.array(ssim_accumulated)

    # return ssim_accumulated.mean()
