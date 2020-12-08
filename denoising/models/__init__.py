import os, keras
import tensorflow as tf

def normalize_tensor(tensor, new_interval=(0,255.0)):
    # new_tensor = tf.div(
    #   tf.subtract(
    #       tensor, 
    #       tf.reduce_min(tensor)
    #   ), 
    #   tf.subtract(
    #       tf.reduce_max(tensor), 
    #       tf.reduce_min(tensor)
    #   )
    # )

    new_tensor = tf.multiply(tensor, new_interval[1])
    new_tensor = tf.dtypes.cast(new_tensor, tf.float32)
    # new_tensor = tf.to_float(new_tensor)

    return new_tensor

def psnr(y_val, y_pred):
    y_val, y_pred = normalize_tensor(y_val), normalize_tensor(y_pred)

    return tf.image.psnr(y_val, y_pred, max_val=255.0)

def ssim(y_val, y_pred):
    y_val, y_pred = normalize_tensor(y_val), normalize_tensor(y_pred)

    return tf.image.ssim(y_val, y_pred, max_val=255.0)

def loss_ssim(y_val, y_pred):
    y_val, y_pred = normalize_tensor(y_val), normalize_tensor(y_pred)

    return 1 - tf.image.ssim(y_val, y_pred, max_val=255.0)

def create_checkpoint(number_of_layers, lr, loss, optimizer, patch=(100,100)) -> keras.callbacks.ModelCheckpoint:
    CHECKPOINT_DIR = 'checkpoints'

    cktp_name = os.path.join(CHECKPOINT_DIR, f'{number_of_layers}l-{lr}-{loss}-{optimizer}-{patch[0]}x{patch[1]}.hdf5')

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    return keras.callbacks.ModelCheckpoint(
        cktp_name, 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False, 
        period=1
    )
    