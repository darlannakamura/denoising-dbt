from typing import List
import numpy as np


def crop_image(image: np.ndarray, height: int, width: int, stride:int = 1) -> List[np.ndarray]:
    """
    Retorna os pedaços da imagem com dimensão height X width.
    """
    w, h = image.shape
    
    patches = []

    for row in range(0, w - width + 1, stride):
        for col in range(0, h - height + 1, stride):
            patches.append(image[row:row + width, col: col + height])
    
    patches = np.array(patches)

    return patches

if __name__ == '__main__':
    a = np.arange(180*180).reshape(180, 180)
    print(a.shape)

    patches = crop_image(a, height=40, width=40, stride=40)
    print(patches.shape)

    print(patches[0].shape)
    print(patches)
    import matplotlib.pyplot as plt

    plt.imshow(patches[0], cmap='gray', interpolation='nearest')
    plt.show()