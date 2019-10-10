import shutil
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array

# f = open("VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", "r")
# for filename in f:
#     filename_train = filename.strip() + ".jpg"
#     destination_train = "data2/train_frames/val/" + filename_train
#     shutil.copy("VOCdevkit/VOC2012/JPEGImages/" + filename_train, destination_train)
#     filename_val = filename.strip() + ".png"
#     destination_val = "data2/train_masks/val/" + filename_val
#     shutil.copy("VOCdevkit/VOC2012/SegmentationClass/" + filename_val, destination_val)
# f = open("VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r")
# for filename in f:
#     filename_train = filename.strip() + ".jpg"
#     destination_train = "data2/val_frames/val/" + filename_train
#     shutil.copy("VOCdevkit/VOC2012/JPEGImages/" + filename_train, destination_train)
#     filename_val = filename.strip() + ".png"
#     destination_val = "data2/val_masks/val/" + filename_val
#     shutil.copy("VOCdevkit/VOC2012/SegmentationClass/" + filename_val, destination_val)
# sys.exit(0)
# f = open("VOCdevkit 2 TEST/VOC2012/ImageSets/Segmentation/test.txt", "r")
# for filename in f:
#        filename_test = filename.strip() + ".jpg"
#        destination_train = "data2/test_frames/" + filename_test
#        shutil.copy("VOCdevkit 2 TEST/VOC2012/JPEGImages/" + filename_test, destination_train)
# sys.exit(0)


src_dir = "data/training_data/"
file_with_image_names_for_training = src_dir + "train.txt"
file_with_image_names_for_validation = src_dir + "val.txt"
src_dir_training_x = src_dir + "x/"
src_dir_training_y_class_seg = src_dir + "y_segmentation_class/"
src_dir_training_y_obj_seg = src_dir + "y_segmentation_object/"

input_width, input_height = 224, 224
output_width, output_height = 224, 224
n_classes = 22


def __get_data(file_with_image_names):
    file = open(file_with_image_names, "r")
    x = []
    y = []
    for image_name in file:
        print(len(y))
        image_name = image_name.strip()  # remove whitespaces
        with Image.open(src_dir_training_x + image_name + ".jpg") as x_image:
            x_image = x_image.resize((input_width, input_height))
            x.append(img_to_array(x_image))
        with Image.open(src_dir_training_y_class_seg + image_name + ".png") as y_image:
            y_image = y_image.resize((input_width, input_height))
            y_image = remove_color_map(y_image)
            y_image = get_segmentation_array(y_image)
            y.append(y_image)
    return np.array(x), np.array(y)


def get_training_data():
    return __get_data(file_with_image_names_for_training)


def get_validation_data():
    return __get_data(file_with_image_names_for_validation)


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

    # old method, super slow though
    # for width in range(len(y_image)):
    #    for height in range(len(y_image)):
    #        val = y_image[width, height]
    #        if val != 255:
    #            seg_labels[width, height, val] = 1
    # return seg_labels

