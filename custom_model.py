from keras.models import *
from keras.layers import *
from keras.optimizers import *

VGG_Weights_path = "vgg16_weights.h5"

def custom_model_regularized_2(n_classes, input_height=224, input_width=224, weights=None):
    n_filters = 16
    kernel_size = 3

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    # Start with 224x224x3. Apply 3x3x16 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 112x112x16
    c1 = conv2d_block(n_filters=n_filters * 1, kernel_size=kernel_size, input_tensor=img_input)
    p1 = MaxPooling2D((2, 2))(c1)
    d1 = Dropout(0.25)(p1)

    # 112x112x16. Apply 3x3x32 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 56x56x32
    c2 = conv2d_block(n_filters=n_filters * 2, kernel_size=kernel_size, input_tensor=d1)
    p2 = MaxPooling2D((2, 2))(c2)
    d2 = Dropout(0.25)(p2)

    # 56x56x32. Apply 3x3x64 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 28x28x64
    c3 = conv2d_block(n_filters=n_filters * 4, kernel_size=kernel_size, input_tensor=d2)
    p3 = MaxPooling2D((2, 2))(c3)
    d3 = Dropout(0.25)(p3)

    # 28x28x64. Apply 3x3x128 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 14x14x128
    c4 = conv2d_block(n_filters=n_filters * 8, kernel_size=kernel_size, input_tensor=d3)
    p4 = MaxPooling2D((2, 2))(c4)
    d4 = Dropout(0.25)(p4)

    # 14x14x128. Apply 3x3x256 Convolution, padding same. New dimensions: 14x14x256
    c5 = conv2d_block(n_filters=n_filters * 16, kernel_size=kernel_size, input_tensor=d4)

    # Upsampling part starts here
    # Start with dimensions 14x14x256
    u6 = Conv2DTranspose(n_filters * 8, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    d6 = Dropout(0.25)(u6)
    c6 = conv2d_block(n_filters * 8, kernel_size=3, input_tensor=d6)

    u7 = Conv2DTranspose(n_filters * 4, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    d7 = Dropout(0.25)(u7)
    c7 = conv2d_block(n_filters * 4, kernel_size=3, input_tensor=d7)

    u8 = Conv2DTranspose(n_filters * 2, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    d8 = Dropout(0.25)(u8)
    c8 = conv2d_block(n_filters * 2, kernel_size=3, input_tensor=d8)

    u9 = Conv2DTranspose(n_filters * 1, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    d9 = Dropout(0.25)(u9)
    c9 = conv2d_block(n_filters * 1, kernel_size=3, input_tensor=d9)

    # Apply 1x1 convolution
    outputs = Conv2DTranspose(n_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[img_input], outputs=[outputs])
    model.summary()

    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if weights is not None:
        model.load_weights(weights)

    return model


def custom_model_regularized(n_classes, input_height=224, input_width=224, weights=None):
    n_filters = 16
    kernel_size = 3

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    # Start with 224x224x3. Apply 3x3x16 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 112x112x16
    c1 = conv2d_block(n_filters=n_filters * 1, kernel_size=kernel_size, input_tensor=img_input)
    p1 = MaxPooling2D((2, 2))(c1)
    d1 = Dropout(0.25)(p1)

    # 112x112x16. Apply 3x3x32 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 56x56x32
    c2 = conv2d_block(n_filters=n_filters * 2, kernel_size=kernel_size, input_tensor=d1)
    p2 = MaxPooling2D((2, 2))(c2)
    d2 = Dropout(0.25)(p2)

    # 56x56x32. Apply 3x3x64 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 28x28x64
    c3 = conv2d_block(n_filters=n_filters * 4, kernel_size=kernel_size, input_tensor=d2)
    p3 = MaxPooling2D((2, 2))(c3)
    d3 = Dropout(0.25)(p3)

    # 28x28x64. Apply 3x3x128 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 14x14x128
    c4 = conv2d_block(n_filters=n_filters * 8, kernel_size=kernel_size, input_tensor=d3)
    p4 = MaxPooling2D((2, 2))(c4)
    d4 = Dropout(0.25)(p4)

    # 14x14x128. Apply 3x3x256 Convolution, padding same. New dimensions: 14x14x256
    c5 = conv2d_block(n_filters=n_filters * 16, kernel_size=kernel_size, input_tensor=d4)

    # Upsampling part starts here
    # Start with dimensions 14x14x256
    u6 = Conv2DTranspose(n_filters * 8, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c5)
    concatenate([u6, c4])
    c6 = conv2d_block(n_filters * 8, kernel_size=3, input_tensor=u6)

    u7 = Conv2DTranspose(n_filters * 4, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c6)
    concatenate([u7, c3])
    c7 = conv2d_block(n_filters * 4, kernel_size=3, input_tensor=u7)

    u8 = Conv2DTranspose(n_filters * 2, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c7)
    concatenate([u8, c2])
    c8 = conv2d_block(n_filters * 2, kernel_size=3, input_tensor=u8)

    u9 = Conv2DTranspose(n_filters * 1, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c8)
    concatenate([u9, c1])
    c9 = conv2d_block(n_filters * 1, kernel_size=3, input_tensor=u9)

    # Apply 1x1 convolution
    outputs = Conv2DTranspose(n_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[img_input], outputs=[outputs])
    model.summary()

    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if weights is not None:
        model.load_weights(weights)

    return model


def conv2d_block(n_filters, kernel_size=3, input_tensor=None):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), activation='relu',
                   padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), activation='relu',
               padding='same')(x)
    x = BatchNormalization()(x)
    return x


def custom_model(n_classes, input_height, input_width, weights=None):
    n_filters = 16
    kernel_size = 3

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    # Start with 224x224x3. Apply 3x3x16 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 112x112x16
    c1 = conv2d_block(n_filters=n_filters * 1, kernel_size=kernel_size, input_tensor=img_input)
    p1 = MaxPooling2D((2, 2))(c1)

    # 112x112x16. Apply 3x3x32 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 56x56x32
    c2 = conv2d_block(n_filters=n_filters * 2, kernel_size=kernel_size, input_tensor=p1)
    p2 = MaxPooling2D((2, 2))(c2)

    # 56x56x32. Apply 3x3x64 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 28x28x64
    c3 = conv2d_block(n_filters=n_filters * 4, kernel_size=kernel_size, input_tensor=p2)
    p3 = MaxPooling2D((2, 2))(c3)

    # 28x28x64. Apply 3x3x128 Convolution, padding same and 2x2 Pooling. New dimensions:
    # 14x14x128
    c4 = conv2d_block(n_filters=n_filters * 8, kernel_size=kernel_size, input_tensor=p3)
    p4 = MaxPooling2D((2, 2))(c4)

    # 14x14x128. Apply 3x3x256 Convolution, padding same. New dimensions: 14x14x256
    c5 = conv2d_block(n_filters=n_filters * 16, kernel_size=kernel_size, input_tensor=p4)

    # Upsampling part starts here
    # Start with dimensions 14x14x256
    u6 = Conv2DTranspose(n_filters * 8, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c5)
    concatenate([u6, c4])
    c6 = conv2d_block(n_filters * 8, kernel_size=3, input_tensor=u6)

    u7 = Conv2DTranspose(n_filters * 4, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c6)
    concatenate([u7, c3])
    c7 = conv2d_block(n_filters * 4, kernel_size=3, input_tensor=u7)

    u8 = Conv2DTranspose(n_filters * 2, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c7)
    concatenate([u8, c2])
    c8 = conv2d_block(n_filters * 2, kernel_size=3, input_tensor=u8)

    u9 = Conv2DTranspose(n_filters * 1, kernel_size=(kernel_size, kernel_size),
                         strides=(2, 2), padding='same')(c8)
    concatenate([u9, c1])
    c9 = conv2d_block(n_filters * 1, kernel_size=3, input_tensor=u9)

    # Apply 1x1 convolution
    outputs = Conv2DTranspose(n_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[img_input], outputs=[outputs])


    model.summary()

    if weights is not None:
        model.load_weights(weights)

    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



    return model


def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def save(model, name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + ".h5", overwrite=True)


def vgg16_model(n_classes, input_height=224, input_width=224):
    # input_height and width must be dividable by 32 because maxpooling
    # with filter size = (2,2) is operated 5 times,
    # which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

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
    # vgg.load_weights(VGG_Weights_path)  # loading VGG weights for the encoder parts of FCN8

    o = (Conv2D(4096, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(4096, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

    # 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(4, 4), use_bias=False,
                              data_format=IMAGE_ORDERING)(
        conv7)

    # 2 times upsampling for pool411
    pool411 = (
        Conv2D(n_classes, (1, 1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(
        pool4)
    pool411_2 = (
        Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING))(
        pool411)

    pool311 = (
        Conv2D(n_classes, (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(
        pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)
    model.summary()

    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
