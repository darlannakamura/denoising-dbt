import os, cv2
import numpy as np
from .utils import normalize
from typing import Tuple, Dict

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


def carrega_dataset(caminho_diretorio: str, divisao: Tuple[int, int], embaralhar=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Especifique o caminho do diretório em que os arquivos `noisy.npy`e `original.npy` estão. 

    Args:
        caminho_diretorio (str): caminho do diretório.
        divisao (Tuple[int, int]): como será a divisão entre treinamento e teste.
        embaralhar (bool, optional): se deseja embaralhar o dataset. Defaults to True.

    Returns:
        Tuple: retorna (x_train, y_train, x_test, y_test)
    """    
    if caminho_diretorio != '':
        x = np.load(os.path.join(caminho_diretorio, 'noisy.npy'))
        y = np.load(os.path.join(caminho_diretorio, 'original.npy'))
    else:
        x = np.load('noisy.npy')
        y = np.load('original.npy')

    if embaralhar:
        np.random.seed(42)
        np.random.shuffle(x)
        np.random.seed(42)
        np.random.shuffle(y)
    
    x_train, x_test = dividir_dataset_em_treinamento_e_teste(x, divisao=divisao)
    y_train, y_test = dividir_dataset_em_treinamento_e_teste(y, divisao=divisao)
    
    return (x_train, y_train, x_test, y_test)

def adiciona_a_dimensao_das_cores(array:np.ndarray) -> np.ndarray:
    """
    Adiciona a dimensão das cores no array numpy, considerando a imagem sendo escala de cinza.
    """
    return array.reshape( array.shape + (1,) )

def pre_processing(caminho_dataset: str, tamanho_patch: Dict[int,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Esta função executa a etapa de pré-processamento.

    Serão executado as seguintes etapas:
    1 - O dataset será carregado;
    2 - O dataset embaralhado;
    3 - O dataset será dividido 80/20 (treinamento/teste);
    4 - Os exames, projeções e patches formarão uma só dimensão;
    5 - Os valores serão normalizados entre [0,1];
    6 - Por fim, é adicionado a dimensão das cores.

    Args:
        caminho_dataset (str): diretório onde o dataset está.
        tamanho_patch (Dict[int,int]): tamanho do patch. Exemplo: 50x50, 100x100.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, y_train, x_test, y_test
    """    
    x_train, y_train, x_test, y_test = carrega_dataset(
        caminho_diretorio=caminho_dataset,
        divisao=(80,20),
        embaralhar=True
    )

    # junta exames, projeções e patches na mesma dimensão
    x_train = np.reshape(x_train, (-1, tamanho_patch[0], tamanho_patch[1]))
    y_train = np.reshape(y_train, (-1, tamanho_patch[0], tamanho_patch[1]))
    x_test = np.reshape(x_test, (-1, tamanho_patch[0], tamanho_patch[1]))
    y_test = np.reshape(y_test, (-1, tamanho_patch[0], tamanho_patch[1]))
    
    #normaliza entre [0,1]
    x_train = normalize(x_train, interval=(0,1))
    y_train = normalize(y_train, interval=(0,1))
    x_test = normalize(x_test, interval=(0,1))
    y_test = normalize(y_test, interval=(0,1))
    
    #adiciona a dimensão das cores
    x_train = adiciona_a_dimensao_das_cores(x_train)
    y_train = adiciona_a_dimensao_das_cores(y_train)

    x_test = adiciona_a_dimensao_das_cores(x_test)
    y_test = adiciona_a_dimensao_das_cores(y_test)


    return (x_train, y_train, x_test, y_test)
