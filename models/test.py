import keras
from keras import Input
from keras.layers import Activation, Conv2D

from AbstractModel import AbstractModel
from config import N_CLASSES


class Model(AbstractModel):

    def get_model(self, **kwargs) -> keras.Model:
        img_input = Input(shape=(224, 224, 3))

        fcn32 = Conv2D(N_CLASSES, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(img_input)

        output = Activation('softmax')(fcn32)

        model = keras.Model(img_input, output)

        return model
