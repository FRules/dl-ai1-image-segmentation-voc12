from keras import Input, Model
from keras.layers import MaxPooling2D, Conv2DTranspose, Concatenate, Dropout
from keras.optimizers import Adam
from numpy import concatenate

from models.common import conv2d_block


def get_model(n_classes, input_height=224, input_width=224, weights=None):
    n_filters = 64
    kernel_size = 3
    dropout = 0.1

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    # Start with 224x224x3. Apply 3x3x16 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 112x112x16
    c1 = conv2d_block(n_filters=n_filters * 1, kernel_size=kernel_size, input_tensor=img_input)
    p1 = MaxPooling2D((2, 2))(c1)
    d1 = Dropout(dropout)(p1)

    # 112x112x16. Apply 3x3x32 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 56x56x32
    c2 = conv2d_block(n_filters=n_filters * 2, kernel_size=kernel_size, input_tensor=d1)
    p2 = MaxPooling2D((2, 2))(c2)
    d2 = Dropout(dropout)(p2)

    # 56x56x32. Apply 3x3x64 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 28x28x64
    c3 = conv2d_block(n_filters=n_filters * 4, kernel_size=kernel_size, input_tensor=d2)
    p3 = MaxPooling2D((2, 2))(c3)
    d3 = Dropout(dropout)(p3)

    # 28x28x64. Apply 3x3x128 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 14x14x128
    c4 = conv2d_block(n_filters=n_filters * 8, kernel_size=kernel_size, input_tensor=d3)
    p4 = MaxPooling2D((2, 2))(c4)
    d4 = Dropout(dropout)(p4)

    # 14x14x128. Apply 3x3x256 Convolution, padding same. New dimensions: 14x14x256
    c5 = conv2d_block(n_filters=n_filters * 16, kernel_size=kernel_size, input_tensor=d4)

    # Upsampling part starts here
    # Start with dimensions 14x14x256
    u6 = Conv2DTranspose(n_filters * 8, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    d6 = Dropout(dropout)(u6)
    c6 = conv2d_block(n_filters * 8, kernel_size=3, input_tensor=d6)

    u7 = Conv2DTranspose(n_filters * 4, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    d7 = Dropout(dropout)(u7)
    c7 = conv2d_block(n_filters * 4, kernel_size=3, input_tensor=d7)

    u8 = Conv2DTranspose(n_filters * 2, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    d8 = Dropout(dropout)(u8)
    c8 = conv2d_block(n_filters * 2, kernel_size=3, input_tensor=d8)

    u9 = Conv2DTranspose(n_filters * 1, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    d9 = Dropout(dropout)(u9)
    c9 = conv2d_block(n_filters * 1, kernel_size=3, input_tensor=d9)

    # Apply 1x1 convolution
    outputs = Conv2DTranspose(n_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[img_input], outputs=[outputs])

    model.summary()

    if weights is not None:
        model.load_weights(weights)

    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
