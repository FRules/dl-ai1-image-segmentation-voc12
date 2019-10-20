from keras.layers import *
from keras import Model

from models.common import conv2d_block


def get_model(input_img, n_filters=16, dropout=0.1, batch_norm=True):
    # Contracting Path
    c1 = conv2d_block(n_filters * 1, kernel_size=3, input_tensor=input_img)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(n_filters * 2, kernel_size=3, batch_norm=batch_norm, input_tensor=p1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(n_filters * 4, kernel_size=3, batch_norm=batch_norm, input_tensor=p2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(n_filters * 8, kernel_size=3, batch_norm=batch_norm, input_tensor=p3)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(n_filters=n_filters * 16, kernel_size=3, batch_norm=batch_norm, input_tensor=p4)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(n_filters * 8, kernel_size=3, batch_norm=batch_norm, input_tensor=u6)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(n_filters * 4, kernel_size=3, batch_norm=batch_norm, input_tensor=u7)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(n_filters * 2, kernel_size=3, batch_norm=batch_norm, input_tensor=u8)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(n_filters * 1, kernel_size=3, batch_norm=batch_norm, input_tensor=u9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
