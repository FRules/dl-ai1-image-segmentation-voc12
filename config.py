import os

DIR_PATH_TO_VOC_2012 = os.path.join("VOCdevkit Training", "VOC2012")

DIR_PATH_TO_SEGMENTATION_FILES = os.path.join(DIR_PATH_TO_VOC_2012, "ImageSets", "Segmentation")
FILE_WITH_IMAGE_NAMES_FOR_TRAINING = os.path.join(DIR_PATH_TO_SEGMENTATION_FILES, "train.txt")
FILE_WITH_IMAGE_NAMES_FOR_VALIDATION = os.path.join(DIR_PATH_TO_SEGMENTATION_FILES, "val.txt")
SRC_DIR_TRAINING_X = os.path.join(DIR_PATH_TO_VOC_2012, "JPEGImages")
FILE_TYPE_TRAINING_X = ".jpg"
SRC_DIR_TRAINING_Y = os.path.join(DIR_PATH_TO_VOC_2012, "SegmentationClass")
FILE_TYPE_TRAINING_Y = ".png"

N_CLASSES = 22  # 20 actual classes but 2 extra classes for borders and background
