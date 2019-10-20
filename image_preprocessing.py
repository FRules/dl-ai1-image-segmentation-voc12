import os

import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

from config import SRC_DIR_TRAINING_X, FILE_TYPE_TRAINING_X, SRC_DIR_TRAINING_Y, FILE_TYPE_TRAINING_Y, N_CLASSES


def remove_color_map(image):
    return np.asarray(image)


def get_segmentation_array(y_image):
    seg_labels = np.zeros((y_image.shape[0], y_image.shape[1], N_CLASSES))
    y_image = np.array(y_image)
    # convert 255 to 21 to fit in array
    y_image[y_image == 255] = 21
    for c in range(N_CLASSES):
        seg_labels[:, :, c] = (y_image == c).astype(int)
    return seg_labels


def get_input(path, file_name, input_width, input_height):
    image_name = file_name.strip()  # remove whitespaces
    with Image.open(os.path.join(path, image_name + FILE_TYPE_TRAINING_X)) as x_image:
        x_image = x_image.resize((input_width, input_height))
        return img_to_array(x_image)


def get_output(path, file_name, input_width, input_height):
    image_name = file_name.strip()  # remove whitespaces
    with Image.open(os.path.join(path, image_name + FILE_TYPE_TRAINING_Y)) as y_image:
        y_image = y_image.resize((input_width, input_height))
        y_image = remove_color_map(y_image)
        y_image = get_segmentation_array(y_image)
        return y_image


def get_image_generator(file_with_image_names, batch_size=64, input_width=224, input_height=224):
    files = open(file_with_image_names, "r").readlines()
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a=files, size=batch_size)
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for file_name in batch_paths:
            input = get_input(SRC_DIR_TRAINING_X, file_name, input_width, input_height)
            output = get_output(SRC_DIR_TRAINING_Y, file_name, input_width, input_height)

            batch_input += [input]
            batch_output += [output]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)
