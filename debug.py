import numpy as np
# Setup constants
input_width = 224
input_height = 224
channels = 3
testing_directory = "VOCdevkit Testing/VOC2012/JPEGImages/"



def get_test_images(image_names):
    x_test = []
    for image_name in image_names:
        x_image = testing_directory + image_name + ".jpg"
        test_image = cig.get_input(testing_directory, image_name, input_width, input_height)
        print(test_image)
        x_test.append(test_image)
    return np.array(x_test)


def predict_test_images(model, x_test):
    return model.predict(x_test, verbose=1)


