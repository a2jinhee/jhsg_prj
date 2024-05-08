# FOLDER SETTING SECTION
#-----------------------------------------------

# for train
# User check list
## work path for modeling
CONST_WORK_PATH = "~/dncskin_segmentation/workspace/data/cat"
## class
CONST_CLASS = ["A2"]
## hyperparameter
# learning rate - CONST_CLASS 리스트 내 클래스별 learning rate 별도 지정
LEARNING_RATE = [1e-4]

BATCH_SIZE = 8      # Batch size

EPOCHS = 100        # epoch

#########################################################################################
# set folder name
FOLDER_TFRC = 'tfrc_segm'
FOLDER_RESULT = 'result_segm'
FOLDER_LOG = "logs"
FOLDER_TENSORBOARD = "tensorboard"

import os
CONST_TFRC_PATH = os.path.join(os.path.expanduser(CONST_WORK_PATH), FOLDER_TFRC)
CONST_RESULT_PATH = os.path.join(os.path.expanduser(CONST_WORK_PATH), FOLDER_RESULT)

CONST_TRAIN = "train"
CONST_VAL = "validation"
CONST_TEST = "test"
CONST_SCORE = ['loss', 'acc', 'precision', 'recall', 'miou', 'iou']
#########################################################################################
