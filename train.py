import argparse
import glob
import os
import sys
from importlib import import_module
from datetime import datetime

from keras import utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import image_preprocessing
import models.common
from config import FILE_WITH_IMAGE_NAMES_FOR_TRAINING, FILE_WITH_IMAGE_NAMES_FOR_VALIDATION, N_CLASSES
from tensorboard_callback import MyTensorBoardCallback


def train(args):
    model_name, epochs, batch_size, weights, dataset_dir = args.model, args.epochs, args.batch_size, args.weights, args.dataset_dir

    no_of_training_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_TRAINING, "r").readlines())
    no_of_val_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_VALIDATION, "r").readlines())

    print("Number of training images:", no_of_training_images)
    print("Number of validation images:", no_of_val_images)

    model_module = import_module("models.{}.model".format(model_name))

    model = model_module.get_model(n_classes=N_CLASSES, weights=weights)

    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")

    res_model_dir = os.path.join("results", model_name)
    if not os.path.isdir(res_model_dir):
        os.mkdir(res_model_dir)

    res_dir = os.path.join("results", model_name, dt_string)
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    if 'pydot' in sys.modules and 'graphviz' in sys.modules:
        utils.plot_model(model, to_file=os.path.join(res_dir, 'model.png'), show_shapes=True)

    history = model.fit_generator(generator=image_preprocessing.PascalVOCSequence(FILE_WITH_IMAGE_NAMES_FOR_TRAINING,
                                                                                  batch_size, dataset_dir),
                                  epochs=epochs,
                                  steps_per_epoch=int(no_of_training_images // float(batch_size)),
                                  validation_data=image_preprocessing.PascalVOCSequence(FILE_WITH_IMAGE_NAMES_FOR_VALIDATION,
                                                                                        batch_size, dataset_dir),
                                  validation_steps=int(no_of_val_images // float(batch_size)),
                                  verbose=1,
                                  callbacks=[
                                      ModelCheckpoint(os.path.join(res_dir, "weights.hdf5"),
                                                      save_best_only=True, verbose=1, save_weights_only=True, period=1,
                                                      monitor='val_acc'),
                                      ReduceLROnPlateau(patience=3, factor=0.1, verbose=1, epsilon=1e-3,  monitor='loss'),
                                      MyTensorBoardCallback(args, res_dir, write_graph=False)
                                  ],
                                  use_multiprocessing=True,
                                  workers=8,
                                  max_queue_size=1
                                  )

    models.common.save(res_dir, model)
    models.common.plot_history(res_dir, history)


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


def main():
    parser = argparse.ArgumentParser(description='DL-MAI project #1 (FNN/CNN) training script.')

    available_models = [model_name.split("/")[1] for model_name in glob.glob("models/*/model.py")]
    parser.add_argument('model', choices=available_models)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--dataset-dir', type=str, default='.')
    parser.add_argument('--weights', type=lambda x: is_valid_file(parser, x))

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
