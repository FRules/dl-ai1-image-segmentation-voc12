from keras.models import *
from keras.layers import *
from keras.optimizers import *
from datetime import datetime
import matplotlib
import os
from keras.utils import plot_model
from config import VGG16_WEIGHTS_CIFAR_100_PATH, VGG16_WEIGHTS_CIFAR_10_PATH, VGG16_WEIGHTS_IMAGE_NET_PATH
from config import IMAGE_ORDERING, RESULTS_FOLDER, N_CLASSES

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_model(embedding, input_height=224, input_width=224, weights=None):
    # input_height and width must be dividable by 32 because maxpooling
    # with filter size = (2,2) is operated 5 times,
    # which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(
        x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(
        x)

    vgg = Model(img_input, pool5)
    vgg.load_weights(VGG16_WEIGHTS_IMAGE_NET_PATH)
    for layer in vgg.layers:
        layer.trainable = False

    o = (Conv2D(4096, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(4096, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

    # 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(N_CLASSES, kernel_size=(4, 4), strides=(4, 4), use_bias=False,
                              data_format=IMAGE_ORDERING)(
        conv7)

    # 2 times upsampling for pool411
    pool411 = (
        Conv2D(N_CLASSES, (1, 1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(
        pool4)
    pool411_2 = (
        Conv2DTranspose(N_CLASSES, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING))(
        pool411)

    pool311 = (
        Conv2D(N_CLASSES, (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(
        pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(N_CLASSES, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)
    if weights is not None:
        model.load_weights(weights.name, by_name=True)

    model.summary()

    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def get_embedding_path(embedding):
    if embedding == "imagenet":
        return VGG16_WEIGHTS_IMAGE_NET_PATH
    elif embedding == "cifar10":
        return VGG16_WEIGHTS_CIFAR_10_PATH
    elif embedding == "cifar100":
        return VGG16_WEIGHTS_CIFAR_100_PATH
    return None


def save(model, history, name):
    timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(RESULTS_FOLDER, timestamp + '_' + name)
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename + ".h5", overwrite=True)
    plot_history(history, filename)


def visualize_model(model, name):
    plot_model(model, to_file=name + '.png', show_shapes=True)


def plot_history(history, filename):
    matplotlib.use('Agg')
    # Accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename + '_accuracy.pdf')
    plt.close()
    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename + '_loss.pdf')
