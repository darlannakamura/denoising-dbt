import numpy as np
import cv2

def validate_array_input(noisy_images: np.ndarray):
    assert type(noisy_images) == np.ndarray, \
        'noisy_images should be a numpy array.'
    
    assert len(noisy_images.shape) == 4, \
        'noisy_images should have 4 dimensions. \
            Read the function documentation for more details.'

    assert noisy_images.shape[3] == 1 or noisy_images.shape[3] == 3, \
        'The color dimensions should be 1 for grayscale \
            or 3 for RGB colored images.'

def validate_if_noise_std_dev_is_a_float(noise_std_dev: float):
    assert type(noise_std_dev) == float, 'noise_std_dev should be float.'

def normalize(numpy_array: np.ndarray, interval=(0, 255), data_type: str = 'float') -> np.ndarray:
    assert data_type in ('float', 'int'), 'data_type should be "float" or "int".'
    
    out = cv2.normalize(numpy_array, None, alpha = interval[0], beta = interval[1], norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    if data_type == 'float':
        return out
    else:
        return out.astype('uint16')
