import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

file_with_image_names_for_training = "VOCdevkit Training/VOC2012/ImageSets/Segmentation/train.txt"
file_with_image_names_for_validation = "VOCdevkit Training/VOC2012/ImageSets/Segmentation/val.txt"
src_dir_training_x = "VOCdevkit Training/VOC2012/JPEGImages/"
file_type_training_x = ".jpg"
src_dir_training_y = "VOCdevkit Training/VOC2012/SegmentationClass/"
file_type_training_y = ".png"
n_classes = 22  # 20 actual classes but 2 extra classes for borders and background


def remove_color_map(image):
    return np.asarray(image)


def get_segmentation_array(y_image):
    seg_labels = np.zeros((y_image.shape[0], y_image.shape[1], n_classes))
    y_image = np.array(y_image)
    # convert 255 to 21 to fit in array
    y_image[y_image == 255] = 21
    for c in range(n_classes):
        seg_labels[:, :, c] = (y_image == c).astype(int)
    return seg_labels


def get_input(path, file_name, input_width, input_height):
    image_name = file_name.strip()  # remove whitespaces
    with Image.open(path + image_name + file_type_training_x) as x_image:
        x_image = x_image.resize((input_width, input_height))
        return img_to_array(x_image)


def get_output(path, file_name, input_width, input_height):
    image_name = file_name.strip()  # remove whitespaces
    with Image.open(path + image_name + file_type_training_y) as y_image:
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
            input = get_input(src_dir_training_x, file_name, input_width, input_height)
            output = get_output(src_dir_training_y, file_name, input_width, input_height)

            batch_input += [input]
            batch_output += [output]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)
