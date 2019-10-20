import argparse
import glob
import os
from importlib import import_module
from datetime import datetime

import image_preprocessing
import models.common
from config import FILE_WITH_IMAGE_NAMES_FOR_TRAINING, FILE_WITH_IMAGE_NAMES_FOR_VALIDATION, N_CLASSES


def get_train_and_val_generator(batch_size, input_width=224, input_height=224):
    train_generator = image_preprocessing.get_image_generator(
        file_with_image_names=FILE_WITH_IMAGE_NAMES_FOR_TRAINING,
        batch_size=batch_size,
        input_width=input_width)
    val_generator = image_preprocessing.get_image_generator(
        file_with_image_names=FILE_WITH_IMAGE_NAMES_FOR_VALIDATION,
        batch_size=batch_size,
        input_height=input_height)
    return train_generator, val_generator


def train(model_name, epochs, batch_size, input_height, input_width, weights):
    train_generator, val_generator = get_train_and_val_generator(batch_size=batch_size,
                                                                 input_width=input_width,
                                                                 input_height=input_height)

    no_of_training_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_TRAINING, "r").readlines())
    no_of_val_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_VALIDATION, "r").readlines())

    print("Number of training images:", no_of_training_images)
    print("Number of validation images:", no_of_val_images)

    model_module = import_module("models.{}.model".format(model_name))

    model = model_module.get_model(n_classes=N_CLASSES, input_height=input_height, input_width=input_width,
                                   weights=weights)

    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")

    res_model_dir = os.path.join("results", model_name)
    if not os.path.isdir(res_model_dir):
        os.mkdir(res_model_dir)

    res_dir = os.path.join("results", model_name, dt_string)
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    history = model.fit_generator(generator=train_generator, epochs=epochs,
                                  steps_per_epoch=(no_of_training_images // batch_size),
                                  validation_data=val_generator,
                                  validation_steps=(no_of_val_images // batch_size),
                                  verbose=2)

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
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--input-height', default=224, type=int)
    parser.add_argument('--input-width', default=224, type=int)
    parser.add_argument('--weights', type=lambda x: is_valid_file(parser, x))

    args = parser.parse_args()

    train(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size,
          input_height=args.input_height, input_width=args.input_width, weights=args.weights)


if __name__ == "__main__":
    main()
