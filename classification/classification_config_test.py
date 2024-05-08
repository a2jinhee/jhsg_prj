TEST_BATCH_SIZE = 16
TEST_IMAGE_SIZE = 224
TEST_LEARNING_RATE = 0.000213388 

TEST_CLASS = ['1', '3', '6', '7']

# multi or binary
TEST_TYPE= 'multi' 

# model 저장된 경로
MODEL_PATH = '~/dncskin_classification/workspace/data/dog/result_cla/multi_A1_A3_A6/model'

# test 결과를 저장할 경로의 상위 경로
TEST_RESULT_ROOT = '~/dncskin_classification/workspace/data/dog'

# test tfrecord 저장된 경로
TEST_PATH = f"{TEST_RESULT_ROOT}/tfrc_cla/tfrecord_A1_A3_A6/test"

# test log, confusion matrix 저장할 폴더 경로
TEST_FOLDER = f"{TEST_RESULT_ROOT}/result_cla/multi_A1_A3_A6"

# log 저장 파일명
LOG_FNAME = 'classification_test_log' 

# confusion matrix 저장 파일명
CONFUSION_MATRIX_FNAME = 'classification_confusion_matrix' 