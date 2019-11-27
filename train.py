import os
import argparse
import image_preprocessing
import model as mo
from config import FILE_WITH_IMAGE_NAMES_FOR_TRAINING, FILE_WITH_IMAGE_NAMES_FOR_VALIDATION
from config import INPUT_WIDTH, INPUT_HEIGHT, AVAILABLE_EMBEDDINGS


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


def train(name, embedding, epochs, batch_size, input_height, input_width, weights):
    train_generator, val_generator = get_train_and_val_generator(batch_size=batch_size,
                                                                 input_width=input_width,
                                                                 input_height=input_height)

    no_of_training_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_TRAINING, "r").readlines())
    no_of_val_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_VALIDATION, "r").readlines())

    print("Number of training images:", no_of_training_images)
    print("Number of validation images:", no_of_val_images)

    model = mo.get_model(embedding, input_height=input_height, input_width=input_width, weights=weights)
    # mo.visualize_model(model, name)

    history = model.fit_generator(generator=train_generator, epochs=epochs,
                                  steps_per_epoch=(no_of_training_images // batch_size),
                                  validation_data=val_generator,
                                  validation_steps=(no_of_val_images // batch_size),
                                  verbose=2)
    mo.save(model, history, name)


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


def main():
    parser = argparse.ArgumentParser(description='DL-MAI project #1 and #3 (FNN/CNN) training script.')

    parser.add_argument('--name', type=str)
    parser.add_argument('--embedding', choices=AVAILABLE_EMBEDDINGS)
    parser.add_argument('--epochs', default=35, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--weights', type=lambda x: is_valid_file(parser, x))

    args = parser.parse_args()

    train(name=args.name, embedding=args.embedding, epochs=args.epochs, batch_size=args.batch_size,
          input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, weights=args.weights)


if __name__ == "__main__":
    main()
