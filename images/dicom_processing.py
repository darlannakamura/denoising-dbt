import os, pydicom, numpy
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pyplot, cm
from skimage.io import imread, imsave
from glob import glob

from typing import List, Dict, Tuple

def get_nparray_from_dicom(input_file: str) -> numpy.ndarray:
	ds = pydicom.read_file(input_file)

	# Load dimensions based on the number of rows and columns
	ConstPixelDims = (int(ds.Rows), int(ds.Columns))
	# print("Quantidade de Rows w Cols")
	# print(ConstPixelDims)

	# Load spacing values (in mm)
	ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
	# print("Valor do Pixel Spacing")
	# print(ConstPixelSpacing)

	# Lista começando em 0, finalizando em
	x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
	#x = 2048
	y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
	#y = 1792

	# The array is sized based on 'ConstPixelDims'
	ArrayDicom = numpy.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)

	ArrayDicom[:, :] = ds.pixel_array
	ArrayDicom = ArrayDicom.astype(numpy.float32)
	# print(ArrayDicom)

	# Transpõem a matriz
	ArrayDicom = numpy.transpose(ArrayDicom)
	# Inverte para a mama ficar em cima
	ArrayDicom = numpy.flip(ArrayDicom)

	return ArrayDicom

def get_max_in_all_projections(input_dir:str) -> numpy.float32:
    max_value = float("-Inf")

    files = glob(os.path.join(input_dir, '*.dcm'))

    #print('files:', files)
    for f in files:
        array = get_nparray_from_dicom(f)
        
        if array.max() > max_value:
            max_value = array.max()

    return max_value

def normalizacao_de_beer(imagem : np.ndarray, valor_maximo_nas_projecoes: float) -> np.ndarray:
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            imagem[i,j] = - np.log( imagem[i,j] / valor_maximo_nas_projecoes)
    
    return imagem

def dicom_to_png(input_dir, dicom_name, output_filename):
	array_dicom = get_nparray_from_dicom(os.path.join(input_dir, dicom_name))

	# Normalização de Beer. Questões da física ;)
	# MAX_MAX É O VALOR MÁXIMO DAS PROJEÇÕES. NÃO TÁ AQUI PQ ESSE É SÓ PARTE DO CÓDIGO
	#Tem que pegar o máximo de todas

	# MAX_IN_ALL_PROJECTIONS = numpy.amax(array_dicom)
	MAX_IN_ALL_PROJECTIONS = get_max_in_all_projections(input_dir)

	div = lambda t: -numpy.log( t / MAX_IN_ALL_PROJECTIONS )
	vfunc = numpy.vectorize(div)
	array_dicom = vfunc(array_dicom)


	array_dicom = scaling(array_dicom, numpy.amax(array_dicom))
	# print(ArrayDicom)
	imsave(output_filename, array_dicom)
