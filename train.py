import argparse
from glob import glob
import os
import sys
from datetime import datetime

from keras import utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

import data_loader
import common
from common import is_valid_file
from config import FILE_WITH_IMAGE_NAMES_FOR_TRAINING, FILE_WITH_IMAGE_NAMES_FOR_VALIDATION
from tensorboard_callback import MyTensorBoardCallback


def train(args):
    model_name, epochs, batch_size, weights, dataset_dir = args.model, args.epochs, args.batch_size, args.weights, args.dataset_dir

    no_of_training_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_TRAINING, "r").readlines())
    no_of_val_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_VALIDATION, "r").readlines())

    print("Number of training images:", no_of_training_images)
    print("Number of validation images:", no_of_val_images)

    model = common.get_model_class(model_name).get_model()

    dt_string = datetime.now().strftime('%d%m%Y_%H%M%S')
    res_dir = os.path.join("results", model_name, dt_string)
    os.makedirs(res_dir)

    if 'pydot' in sys.modules and 'graphviz' in sys.modules:
        utils.plot_model(model, to_file=os.path.join(res_dir, 'model.png'), show_shapes=True)

    callbacks = [
        ModelCheckpoint(os.path.join(res_dir, "weights.hdf5"),
                        save_best_only=True, verbose=1, save_weights_only=True, period=1,
                        monitor='val_categorical_accuracy'),
        MyTensorBoardCallback(args, res_dir, write_graph=False)
    ]

    if args.reduce_on_plateu:
        callbacks.append(
            ReduceLROnPlateau(patience=args.reduce_on_plateu_patience,
                              factor=args.reduce_on_plateu_factor, verbose=1,
                              epsilon=args.reduce_on_plateu_epsilon,  monitor='loss')
        )

    common.save_model_architecture(res_dir, model)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=args.base_lr),
                  metrics=['categorical_accuracy'])

    model.fit_generator(generator=data_loader.PascalVOCSequence(FILE_WITH_IMAGE_NAMES_FOR_TRAINING,
                                                                batch_size, dataset_dir),
                        epochs=1 if args.test else epochs,
                        steps_per_epoch=1 if args.test else int(no_of_training_images // float(batch_size)),
                        validation_data=data_loader.PascalVOCSequence(FILE_WITH_IMAGE_NAMES_FOR_VALIDATION,
                                                                      batch_size, dataset_dir),
                        validation_steps=1 if args.test else int(no_of_val_images // float(batch_size)),
                        verbose=1,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=2,
                        max_queue_size=2
                        )


def main():
    parser = argparse.ArgumentParser(description='DL-MAI project #1 (FNN/CNN) training script.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_models = [model_name.split("/")[1].split(".")[0] for model_name in glob("models/*.py") if "__init__.py" not in model_name]
    parser.add_argument('model', choices=available_models, help='Model name')
    parser.add_argument('--comment', type=str, help='Comment to write to TensorBoard')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of batch per training step')
    parser.add_argument('--base-lr', default=0.01, type=float, help='Base learning rate')
    parser.add_argument('--reduce-on-plateu', action='store_true', help='If provided, will reduce learning rate when no improvement')
    parser.add_argument('--reduce-on-plateu-patience', default=5, type=int, help='After how many epochs without improvement learning rate should be decreased')
    parser.add_argument('--reduce-on-plateu-factor', default=0.5, type=float, help='The factor that learning rate should be moultiplied when plateu')
    parser.add_argument('--reduce-on-plateu-epsilon', default=0.005, type=float, help='The value of change that make callback think there is no plateu')
    parser.add_argument('--dataset-dir', type=str, default='.', help='Path to directory of dataset')
    parser.add_argument('--weights', type=lambda x: is_valid_file(parser, x), help='Path to weights of pretrained model')
    parser.add_argument('--test', action='store_true', help='Whether run training and evaluation test or not (1 epoch, 1 step, 1 example')

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
