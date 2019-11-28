import sys

from keras import Model, Input
from keras.applications import VGG16
from keras.layers import Conv2D, Conv2DTranspose, Add, Activation
from keras.optimizers import Adam


def get_model(n_classes, input_height=224, input_width=224, weights=None):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    # Encoder - Load VGG16
    encoder = VGG16(weights=None, input_tensor=img_input, include_top=False, pooling=None)
    encoder.load_weights("models/fcn8/vgg16_weights.h5", by_name=True)

    # Encoder -  Freeze layers
    for layer in encoder.layers:
        if "block" in layer.name:
            layer.trainable = False

    # Encoder -  Get intermediate VGG16 layers output
    pool3 = encoder.get_layer('block3_pool').output
    pool4 = encoder.get_layer('block4_pool').output
    pool5 = encoder.get_layer('block5_pool').output

    # Encoder - add two convolution layers
    conv6 = (Conv2D(filters=512, kernel_size=(7, 7), activation='relu', padding='same', name='block6_conv1'))(pool5)
    conv7 = (Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='same', name='block7_conv1'))(conv6)

    # Decoder - fcn32 - not used, but written for the learning purpose
    # conv8 = Conv2D(filters=n_classes, kernel_size=(1, 1), activation='relu', padding='same', name='block8_conv1')(conv7)
    # fcn32 = Conv2DTranspose(n_classes, kernel_size=(1, 1), strides=(32, 32), padding='valid', name='block8_deconv1)(conv8)

    # Decoder - fcn16 - not used, but written for the learning purpose
    # conv9 = Conv2D(filters=n_classes, kernel_size=(1, 1), activation='relu', padding='same', name='block9_conv1')(pool4)
    # deconv1 = Conv2DTranspose(filters=n_classes, kernel_size=(1, 1), strides=(2, 2), activation='relu', padding='same', name='block9_deconv1')(conv7)
    # add1 = Add(name='block9_add1')([conv9, deconv1])
    # fcn16 = Conv2DTranspose(n_classes, kernel_size=(1, 1), strides=(16, 16), padding='valid', name='block9_deconv2')(add1)

    # Decoder - fcn32
    conv10 = Conv2D(filters=n_classes, kernel_size=(1, 1), activation='relu', padding='same', name='block10_conv1')(pool3)
    conv11 = Conv2D(filters=n_classes, kernel_size=(1, 1), activation='relu', padding='same', name='block10_conv2')(pool4)
    deconv2 = Conv2DTranspose(filters=n_classes, kernel_size=(1, 1), strides=(2, 2), activation='relu', padding='same', name='block10_deconv1', use_bias=False)(conv11)
    deconv3 = Conv2DTranspose(filters=n_classes, kernel_size=(1, 1), strides=(4, 4), activation='relu', padding='same', name='block10_deconv2', use_bias=False)(conv7)
    add2 = Add(name='block10_add2')([conv10, deconv2, deconv3])
    fcn32 = Conv2DTranspose(n_classes, kernel_size=(1, 1), strides=(8, 8), padding='valid', name='block10_deconv3', use_bias=False)(add2)

    output = Activation('softmax')(fcn32)

    model = Model(img_input, output)

    if weights is not None:
        model.load_weights(weights.name, by_name=True)  # loading VGG weights for the encoder parts of FCN8

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1E-2),
                  metrics=['accuracy'])

    return model
