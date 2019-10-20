import os

from keras.layers import Conv2D, BatchNormalization
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def conv2d_block(n_filters, kernel_size=3, batch_norm=True, input_tensor=None):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), activation='relu',
               padding='same')(input_tensor)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), activation='relu',
               padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)

    return x


def save(res_dir, model):
    # save model config to json
    model_json = model.to_json()
    with open(os.path.join(res_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(os.path.join(res_dir, "model.h5"))


def plot_history(res_dir, history):
    # Accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(res_dir, 'training_accuracy.pdf'))
    plt.close()

    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(res_dir, 'training_loss.pdf'))
