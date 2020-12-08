import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import Sequential, layers, activations
from keras.models import Model, load_model
from images.image import * 
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from typing import List, Dict, Tuple
from copy import deepcopy

def show(img):
    plt.figure(figsize=(12,8))
    plt.imshow(img, cmap='gray', interpolation='nearest')

def reconstroi_projecao_a_partir_dos_patches(patches: np.ndarray, dimensao_patch=(40,40), stride=40, dimensao_imagem=(650, 1650)) -> np.ndarray:
    imagem_predita = np.zeros(shape=dimensao_imagem)
    i = 0
    for row in range(0, (dimensao_imagem[0] - dimensao_patch[0] +1), stride):
        for col in range(0, (dimensao_imagem[1] - dimensao_patch[1] + 1), stride):
            imagem_predita[row: row + dimensao_patch[0], col: col + dimensao_patch[1]] = patches[i].reshape(dimensao_patch[0],dimensao_patch[1])
            i += 1
    
    return imagem_predita

def verifica_se_todos_os_pixels_sao_iguais(imagem: np.ndarray) -> bool:
    """
    Verifica se um array numpy de formato (m,n) possui todos os pixels iguais. 
    """
    
    assert len(imagem.shape) == 2, 'A imagem deve ter apenas 2 dimensões.'
    
    (w, h) = imagem.shape
    
    return imagem.max() == imagem.min()

def pega_os_patches(projecoes:np.ndarray, dimensao:Tuple[int,int], stride:int, remover_patches_pretos = False) -> np.ndarray:
    """
    Retorna os patches das projeções `projecoes` de tamanho dimensao[0] x dimensao[1] com stride de `stride`.
    O parâmetro `remover_patches_pretos` delimita se deve incluir os patches que contém apenas preto.
    """
    
    (largura, altura) = dimensao
    
    projecoes = remove_a_dimensao_das_cores(projecoes)

    patches = []
    
    for p in range(projecoes.shape[0]): # percorre as projeções        
        patches_numpy = crop_image(projecoes[p], height=altura, width=largura, stride=stride)
        
        for i in range(patches_numpy.shape[0]): # percorre os patches extraídos da projeção
            if remover_patches_pretos:
                if verifica_se_todos_os_pixels_sao_iguais(patches_numpy[i]) == False:
                    patches.append(patches_numpy[i])
            else:
                patches.append(patches_numpy[i])
        
    # converte os patches para um array numpy.
    patches = np.array(patches)
    
    return patches

def remove_a_dimensao_das_cores(array_numpy: np.ndarray) -> np.ndarray:
    """
    Recebe um array numpy com shape (QUANTIDADE, LARGURA, ALTURA, COR) e 
    retorna um novo array numpy com shape (QUANTIDADE, LARGURA, ALTURA).
    """
    
    assert len(array_numpy.shape) == 4, 'O array deve ter shape = (QUANTIDADE, LARGURA, ALTURA, COR).'
    
    (qtde, w, h, _) = array_numpy.shape
    
    array_numpy = array_numpy.reshape(qtde, w, h)
    
    return array_numpy

def adiciona_a_dimensao_das_cores(array:np.ndarray) -> np.ndarray:
    """
    Adiciona a dimensão das cores no array numpy, considerando a imagem sendo escala de cinza.
    """
    return array.reshape( array.shape + (1,) )

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
        
def compara_imagens_em_relacao_ao_psnr_e_ssim(imagem_ground_truth: np.ndarray, imagem: np.ndarray, data_range: int):
    """
    Imprime o PSNR e o SSIM entre a imagem_ground_truth e a imagem.
    """
    print('PSNR: %.2f dB, e SSIM: %.2f' % (psnr(imagem, imagem_ground_truth, data_range=data_range), ssim(imagem, imagem_ground_truth, data_range=data_range)))
    
def compara_datasets_em_relacao_ao_psnr_e_ssim_medio(imagens_ground_truth: np.ndarray, imagens_filtradas: np.ndarray, data_range: int = 256):
    """
    Imprime o PSNR e o SSIM médio entre os datasets ground truth e as imagens filtradas.
    """
    
    assert imagens_ground_truth.shape == imagens_filtradas.shape, 'Os datasets devem ter as mesmas dimensões.'
    assert len(imagens_ground_truth.shape) == 3, 'Os datasets não devem ter a dimensão de cor.'
    
    psnr_acumulado = []
    ssim_acumulado = []
    
    for i in range(imagens_ground_truth.shape[0]):
        psnr_acumulado.append(psnr(imagens_ground_truth[i], imagens_filtradas[i], data_range=data_range))
        ssim_acumulado.append(ssim(imagens_ground_truth[i], imagens_filtradas[i], data_range=data_range))
    
    psnr_acumulado = np.array(psnr_acumulado)
    ssim_acumulado = np.array(ssim_acumulado)
    
    print('PSNR médio: %.2f dB e SSIM médio: %.2f' % (psnr_acumulado.mean(), ssim_acumulado.mean()))

    
def dividir_dataset_em_treinamento_e_teste(dataset: np.ndarray, divisao=(80,20)):
    """
    Divisão representa a porcentagem entre conj. de treinamento e conj. de teste.
    Ex: (80,20) representa 80% para treino e 20% para teste.
    """
    assert len(divisao) == 2, 'Divisão deve ser: % de conj. de treinamento e % de conj. de teste.'
    
    n_treino, n_teste = divisao
    
    assert n_treino + n_teste == 100, 'A soma da divisão deve ser igual a 100.'
    
    total = dataset.shape[0] 
    porcentagem_treino = n_treino/100 #0.8
    porcentagem_teste = n_teste/100 #0.2
    
    return dataset[:int(porcentagem_treino*total)], dataset[int(porcentagem_treino*total):]


def carrega_dataset(divisao: Tuple[int, int], embaralhar=True):
    DIRETORIO_DATASETS = 'dataset/patch-50x50-cada-projecao-200'
        
    x = np.load(os.path.join(DIRETORIO_DATASETS, 'noisy.npy'))
    y = np.load(os.path.join(DIRETORIO_DATASETS, 'original.npy'))
    
    if embaralhar:
        np.random.seed(42)
        np.random.shuffle(x)
        np.random.seed(42)
        np.random.shuffle(y)
    
    x_train, x_test = dividir_dataset_em_treinamento_e_teste(x, divisao=divisao)
    y_train, y_test = dividir_dataset_em_treinamento_e_teste(y, divisao=divisao)
    
    return (x_train, y_train, x_test, y_test)

def build_dncnn_model(nb_layers = 10, with_subtract=True):
  model = keras.Sequential()
  input = layers.Input(shape=(None, None, 1), name='input')

  output = layers.Conv2D(filters=64,kernel_size=(3,3), strides=(1,1), 
                        padding='same', name='conv1')(input)
  output = layers.Activation('relu')(output)

  for layer in range(2, nb_layers): # original é 19 em vez de 5
    output = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), 
                      padding='same', name='conv%d' %layer)(output)
    output = layers.BatchNormalization(axis=-1, epsilon=1e-3, name='batch_normalization%d' %layer)(output)
    output = layers.Activation('relu')(output)

  output = layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', strides=(1,1), name=f'conv{nb_layers}')(output)

  if with_subtract:
    output = layers.Subtract(name='subtract')([input, output])

  model = Model(inputs=input, outputs=output)
  return model
