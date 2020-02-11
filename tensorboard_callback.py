import os
from glob import glob

from PIL import Image
import numpy as np
import keras
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array

from config import SRC_DIR_TESTING_X, FILE_TYPE_X, CUSTOM_COLOR_MAPPING


class MyTensorBoardCallback(keras.callbacks.TensorBoard):

    def __init__(self, args, log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                 write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None):
        super().__init__(log_dir, histogram_freq, batch_size, write_graph, write_grads, write_images, embeddings_freq,
                         embeddings_layer_names, embeddings_metadata)

        self.args = args
        self.testing_orig, self.testing_processed = self._load_testing_images()

    @staticmethod
    def _parse_args(args):
        header_row = 'Parameter | Value\n' \
                     '----------|------\n'

        args_dict = vars(args)

        table_body = ["{} | {}".format(key, value) for key, value in args_dict.items()]

        markdown = header_row + "\n".join(table_body)
        return markdown

    @staticmethod
    def _load_testing_images(input_width=224, input_height=224):
        testing_orig = []
        testing_processed = []

        for filepath in glob(os.path.join(SRC_DIR_TESTING_X, '*' + FILE_TYPE_X)):
            with Image.open(filepath) as x_image:
                x_image = x_image.resize((input_width, input_height))
                x_image = img_to_array(x_image)
                testing_orig.append(x_image)
                testing_processed.append(x_image)

        testing_processed = np.array(testing_processed)
        return testing_orig, testing_processed

    @staticmethod
    def _get_output_colored(segmented_masks):
        output_colored = np.zeros([*(segmented_masks.shape)[:3], 3])
        segmented_masks = np.argmax(segmented_masks, axis=-1)
        for index, class_index in np.ndenumerate(segmented_masks):
            sample = index[0]
            width = index[1]
            height = index[2]
            rgb_array = CUSTOM_COLOR_MAPPING[class_index]
            output_colored[sample, width, height, :] = rgb_array
        return output_colored

    def on_train_begin(self, logs=None):
        summary = tf.summary.text('run_settings', tf.convert_to_tensor(self._parse_args(self.args)))
        summary_str = summary.eval(session=self.sess)
        self.writer.add_summary(summary_str, 1)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        super(MyTensorBoardCallback, self).on_epoch_end(epoch, logs)

        predicted_segmented_masks = self.model.predict(np.array(self.testing_processed))

        summary_tensor = tf.convert_to_tensor(self._get_output_colored(predicted_segmented_masks))
        summary = tf.summary.image("Test", summary_tensor, max_outputs=len(self.testing_orig))
        summary_str = summary.eval(session=self.sess)
        self.writer.add_summary(summary_str, epoch)
        self.writer.flush()
