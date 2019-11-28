import keras
import tensorflow as tf


class MyTensorBoardCallback(keras.callbacks.TensorBoard):
    def __init__(self, args, log_dir='logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch',
                 profile_batch=2, embeddings_freq=0, embeddings_metadata=None, **kwargs):
        super().__init__(log_dir, histogram_freq, write_graph, write_images, update_freq, profile_batch,
                         embeddings_freq, embeddings_metadata, **kwargs)

        self.args = args

    @staticmethod
    def _parse_args(args):
        header_row = 'Parameter | Value\n' \
                     '----------|------\n'

        args_dict = vars(args)

        table_body = ["{} | {}".format(key, value) for key, value in args_dict.items()]

        markdown = header_row + "\n".join(table_body)
        return markdown

    def on_train_begin(self, logs=None):
        tf.summary.text('run_settings', tf.convert_to_tensor(self._parse_args(self.args)))
        tf.summary.merge_all()

