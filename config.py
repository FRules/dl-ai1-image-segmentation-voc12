import os

DIR_PATH_TO_VOC_TESTING_2012 = os.path.join("VOCdevkit Testing", "VOC2012")
DIR_PATH_TO_VOC_TRAINING_2012 = os.path.join("VOCdevkit Training", "VOC2012")

DIR_PATH_TO_SEGMENTATION_FILES = os.path.join(DIR_PATH_TO_VOC_TRAINING_2012, "ImageSets", "Segmentation")
FILE_WITH_IMAGE_NAMES_FOR_TRAINING = os.path.join(DIR_PATH_TO_SEGMENTATION_FILES, "train.txt")
FILE_WITH_IMAGE_NAMES_FOR_VALIDATION = os.path.join(DIR_PATH_TO_SEGMENTATION_FILES, "val.txt")
SRC_DIR_TRAINING_X = os.path.join(DIR_PATH_TO_VOC_TRAINING_2012, "JPEGImages")
FILE_TYPE_X = ".jpg"
SRC_DIR_TRAINING_Y = os.path.join(DIR_PATH_TO_VOC_TRAINING_2012, "SegmentationClass")
FILE_TYPE_Y = ".png"
SRC_DIR_TESTING_X = os.path.join(DIR_PATH_TO_VOC_TESTING_2012, "JPEGImages")

N_CLASSES = 22  # 20 classes + background + border

CUSTOM_COLOR_MAPPING = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [0, 0, 128],
    4: [0, 128, 128],
    5: [128, 128, 128],
    6: [255, 255, 0],
    7: [0, 255, 0],
    8: [0, 255, 255],
    9: [64, 64, 64],
    10: [128, 255, 255],
    11: [64, 255, 255],
    12: [128, 32, 32],
    13: [32, 32, 32],
    14: [255, 32, 32],
    15: [255, 255, 128],
    16: [40, 255, 128],
    17: [255, 0, 128],
    18: [200, 200, 200],
    19: [0, 200, 200],
    20: [0, 200, 0],
    21: [255, 255, 0]
}
