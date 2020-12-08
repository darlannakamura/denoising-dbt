import keras
from keras import Sequential, layers, activations
from keras.models import Model
from . import *

def dncnn(number_of_layers=10, learning_rate=0.001, optimizer_function='adam', loss_function='mse') -> Model:
    model = _build_dncnn(number_of_layers=number_of_layers)

    if loss_function == 'mse':
        curr_loss = 'mse'
    elif loss_function == 'ssim':
        curr_loss = loss_ssim

    if optimizer_function == 'adam':
        opt = keras.optimizers.Adam(learning_rate)
    elif optimizer_function == 'sgd':
        opt = keras.optimizers.SGD(learning_rate)
    elif optimizer_function == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate)
    elif optimizer_function == 'adagrad':
        opt = keras.optimizers.Adagrad(learning_rate)

    model.compile(optimizer=opt, loss=curr_loss, metrics=[psnr, ssim])

    return model

def _build_dncnn(number_of_layers) -> Model:
    model = keras.Sequential()
    input = layers.Input(shape=(None, None, 1), name='input')

    output = layers.Conv2D(filters=64,kernel_size=(3,3), strides=(1,1), 
                        padding='same', name='conv1')(input)
    output = layers.Activation('relu')(output)

    for layer in range(2, number_of_layers):
        output = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), 
                        padding='same', name='conv%d' %layer)(output)
        output = layers.BatchNormalization(axis=-1, epsilon=1e-3, name='batch_normalization%d' %layer)(output)
        output = layers.Activation('relu')(output)

    output = layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', strides=(1,1), name=f'conv{number_of_layers}')(output)

    output = layers.Subtract(name='subtract')([input, output])

    model = Model(inputs=input, outputs=output)

    return model
