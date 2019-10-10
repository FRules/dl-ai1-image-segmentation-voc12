import image_preprocessing
import custom_model as mo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

file_with_image_names_for_training = "VOCdevkit Training/VOC2012/ImageSets/Segmentation/train.txt"
file_with_image_names_for_validation = "VOCdevkit Training/VOC2012/ImageSets/Segmentation/val.txt"


def get_train_and_val_generator(batch_size, input_width=224, input_height=224):
    train_generator = image_preprocessing.get_image_generator(
        file_with_image_names=file_with_image_names_for_training,
        batch_size=batch_size,
        input_width=input_width)
    val_generator = image_preprocessing.get_image_generator(
        file_with_image_names=file_with_image_names_for_validation,
        batch_size=batch_size,
        input_height=input_height)
    return train_generator, val_generator


def plot_history(history):
    matplotlib.use('Agg')
    # Accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('training_accuracy.pdf')
    plt.close()
    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('training_loss.pdf')


def train(epochs, batch_size, n_classes, input_height, input_width):
    train_generator, val_generator = get_train_and_val_generator(batch_size=batch_size,
                                                                 input_width=input_width,
                                                                 input_height=input_height)

    no_of_training_images = len(open(file_with_image_names_for_training, "r").readlines())
    no_of_val_images = len(open(file_with_image_names_for_validation, "r").readlines())

    print("Number of training images:", no_of_training_images)
    print("Number of validation images:", no_of_val_images)

    model = mo.custom_model_regularized_2(n_classes=n_classes, input_height=input_height, input_width=input_width)

    history = model.fit_generator(generator=train_generator, epochs=epochs,
                                  steps_per_epoch=(no_of_training_images // batch_size),
                                  validation_data=val_generator,
                                  validation_steps=(no_of_val_images // batch_size),
                                  verbose=2)
    mo.save(model, "custom_model_regularized")
    plot_history(history)


if __name__ == "__main__":
    train(epochs=100, batch_size=16, n_classes=22, input_height=224, input_width=224)
