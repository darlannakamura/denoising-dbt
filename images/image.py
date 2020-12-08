import cv2
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

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

def show(img):
    assert len(img.shape) == 2, 'A imagem deve ter apenas 2 dimensões.'

    plt.figure(figsize=(12,8))
    plt.imshow(img, cmap='gray', interpolation='nearest')

def mostrar_lado_a_lado(imagens: List[np.ndarray], titulos: List[str], figsize: Dict[int, int] = (12,8)):
    """
    Imprime as imagens que estiverem na lista com os títulos apresentados.
    """
    
    assert len(imagens) == len(titulos), 'imagens e titulos devem ter o mesmo tamanho.'
    assert len(imagens[0].shape) == 2, 'As imagens deve ter apenas 2 dimensões.'
    
    quantidade = len(imagens)
    
    fig, ax = plt.subplots(1, quantidade, figsize=figsize)
    
    for i in range(quantidade):
        ax[i].axis('off')
        ax[i].set_title(titulos[i])
        ax[i].imshow(imagens[i], cmap='gray', interpolation='nearest')

def adiciona_a_dimensao_das_cores(array:np.ndarray) -> np.ndarray:
    """
    Adiciona a dimensão das cores no array numpy, considerando a imagem sendo escala de cinza.
    """
    return array.reshape( array.shape + (1,) )

def histograma_colorido(imagem, intervalo=(0, 256)):
    """ 
    Histograma para imagem colorida (RGB).
    """
    
    color = ('b','g','r')
    
    fig, ax = plt.subplots(3,1, figsize=(12,8))
    
    for i,col in enumerate(color):
        histr = cv2.calcHist([imagem],[i],None,[intervalo[1]],[intervalo[0],intervalo[1]])
        ax[i].plot(histr, color = col)
        ax[i].set_xlim([intervalo[0],intervalo[1]])
#         plt.plot(histr,color = col)
#         plt.xlim([intervalo[0],intervalo[1]])
    plt.show()

def histograma(imagem, intervalo=(0, 256)):
    plt.figure(figsize=(12,8))
    
    histr = cv2.calcHist([imagem],[0],None,[intervalo[1]],[intervalo[0],intervalo[1]])
    plt.plot(histr, color = 'b')
    plt.xlim([intervalo[0],intervalo[1]])
    plt.show()

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
