AVAILABLE_EMBEDDINGS = ["imagenet"]
VGG16_WEIGHTS_IMAGE_NET_PATH = "pretrained_vgg_weights/vgg16_weights_imagenet.h5"
VGG16_WEIGHTS_CIFAR_10_PATH = "pretrained_vgg_weights/vgg16_weights_cifar10.h5"
VGG16_WEIGHTS_CIFAR_100_PATH = "pretrained_vgg_weights/vgg16_weights_cifar100.h5"
RESULTS_FOLDER = "results"
FILE_WITH_IMAGE_NAMES_FOR_TRAINING = "VOCdevkit Training/VOC2012/ImageSets/Segmentation/train.txt"
FILE_WITH_IMAGE_NAMES_FOR_VALIDATION = "VOCdevkit Training/VOC2012/ImageSets/Segmentation/val.txt"
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
N_CLASSES = 22
IMAGE_ORDERING = "channels_last"

SRC_DIR_TRAINING_X = "VOCdevkit Training/VOC2012/JPEGImages/"
FILE_TYPE_TRAINING_X = ".jpg"
SRC_DIR_TRAINING_Y = "VOCdevkit Training/VOC2012/SegmentationClass/"
FILE_TYPE_TRAINING_Y = ".png"
