import image_preprocessing
import model as mo
from config import FILE_WITH_IMAGE_NAMES_FOR_TRAINING, FILE_WITH_IMAGE_NAMES_FOR_VALIDATION
from config import INPUT_WIDTH, INPUT_HEIGHT


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


def train(epochs, batch_size, n_classes, input_height, input_width):
    train_generator, val_generator = get_train_and_val_generator(batch_size=batch_size,
                                                                 input_width=input_width,
                                                                 input_height=input_height)

    no_of_training_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_TRAINING, "r").readlines())
    no_of_val_images = len(open(FILE_WITH_IMAGE_NAMES_FOR_VALIDATION, "r").readlines())

    print("Number of training images:", no_of_training_images)
    print("Number of validation images:", no_of_val_images)

    model = mo.get_model(n_classes=n_classes, input_height=input_height, input_width=input_width)

    history = model.fit_generator(generator=train_generator, epochs=epochs,
                                  steps_per_epoch=(no_of_training_images // batch_size),
                                  validation_data=val_generator,
                                  validation_steps=(no_of_val_images // batch_size),
                                  verbose=2)
    mo.save(model, history, "fcn8")


if __name__ == "__main__":
    train(epochs=1, batch_size=16, n_classes=22, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH)
