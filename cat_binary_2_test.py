TEST_BATCH_SIZE = 32
TEST_IMAGE_SIZE = 224
TEST_LEARNING_RATE = 0.00017641 

TEST_CLASS = ['2', '7']

# multi or binary
TEST_TYPE= 'binary' 

# model 저장된 경로
MODEL_PATH = '~/dncskin_classification/workspace/data/cat/result_cla/binary_A2/model'

# test 결과를 저장할 경로의 상위 경로
TEST_RESULT_ROOT = '~/dncskin_classification/workspace/data/cat'

# test tfrecord 저장된 경로
TEST_PATH = f"{TEST_RESULT_ROOT}/tfrc_cla/tfrecord_A2/test"

# test log, confusion matrix 저장할 폴더 경로
TEST_FOLDER = f"{TEST_RESULT_ROOT}/result_cla/binary_A2"

# log 저장 파일명
LOG_FNAME = 'classification_test_log' 

# confusion matrix 저장 파일명
CONFUSION_MATRIX_FNAME = 'classification_confusion_matrix' 