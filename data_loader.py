import os

import numpy as np
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence

from config import SRC_DIR_TRAINING_X, FILE_TYPE_X, SRC_DIR_TRAINING_Y, FILE_TYPE_Y, N_CLASSES


def remove_color_map(image):
    return np.asarray(image)


def get_segmentation_array(y_image):
    seg_labels = np.zeros((y_image.shape[0], y_image.shape[1], N_CLASSES))
    y_image = get_segmentation_array_sparse(y_image)
    for c in range(N_CLASSES):
        seg_labels[:, :, c] = (y_image == c).astype(int)
    return seg_labels


def get_segmentation_array_sparse(y_image):
    y_image = np.array(y_image, dtype=int)
    y_image[y_image == 255] = 21
    return y_image


def get_input(path, file_name, input_width, input_height):
    image_name = file_name.strip()  # remove whitespaces
    with Image.open(os.path.join(path, image_name + FILE_TYPE_X)) as x_image:
        x_image = x_image.resize((input_width, input_height))
    return img_to_array(x_image)


def get_output(path, file_name, input_width, input_height):
    image_name = file_name.strip()  # remove whitespaces
    with Image.open(os.path.join(path, image_name + FILE_TYPE_Y)) as y_image:
        y_image = y_image.resize((input_width, input_height))
        y_image = remove_color_map(y_image)
        y_image = get_segmentation_array(y_image)
    return y_image


class PascalVOCSequence(Sequence):

    def __init__(self, file_with_image_names, batch_size=64, dataset_dir=None, shuffle=False):
        self.batch_size = batch_size
        with open(file_with_image_names, "r") as f:
            self.files = [line.strip() for line in f]

        self.dataset_dir = dataset_dir

        if 'TMPDIR' in os.environ and 'SLURM_JOBID' in os.environ:
            tmpdir_env_value = os.environ['TMPDIR']
            slurm_jobid_env_value = os.environ['SLURM_JOBID']
            if self.dataset_dir in tmpdir_env_value and slurm_jobid_env_value not in self.dataset_dir:
                self.dataset_dir = tmpdir_env_value

        self.shuffle = shuffle
        self.indexes = []
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        batch_files = [self.files[k] for k in indexes]

        batch = [(get_input(os.path.join(self.dataset_dir, SRC_DIR_TRAINING_X), file_name, 224, 224),
                  get_output(os.path.join(self.dataset_dir, SRC_DIR_TRAINING_Y), file_name, 224, 224)) for file_name in batch_files]
        batch_x, batch_y = list(zip(*batch))
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

