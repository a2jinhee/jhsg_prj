# Path
# tfrecord 및 결과 저장 상위 경로
CONST_ROOT_PATH = '~/dncskin_classification/workspace/data/dog'

# tfrecord 경로 (하위폴더 구조 : (train / val)/*.tfrecords) 
CONST_DIR_PATH = f"{CONST_ROOT_PATH}/tfrc_cla/tfrecord_A1_A3_A6"

# tensorboard, model, console log 결과 저장 경로
CONST_SAVE_DIR = f"{CONST_ROOT_PATH}/result_cla"

# binary or multi
TRAIN_TYPES = 'multi' 

# hyperparameter
EPOCH = 3
LEARNING_RATE = 0.000213388
BATCH_SIZE = 16
IMAGE_SIZE = 224

#shuffle True or Flase
SHUFFLE = True

# Tfrecords train,val path
# tfrecord train / val / test directory name
TARIN_TFRECORDS_NAME = 'train'
VAL_TFRECORDS_NAME = 'val'

import os
CONST_TRAIN_PATH = os.path.join(os.path.expanduser(CONST_DIR_PATH), TARIN_TFRECORDS_NAME)
CONST_VAL_PATH = os.path.join(os.path.expanduser(CONST_DIR_PATH), VAL_TFRECORDS_NAME)

# model
MODEL_NAME = 'inception_v4'