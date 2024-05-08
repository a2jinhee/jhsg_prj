import os

# 이미지 및 TFRecord 상위 경로
CONST_ROOT_PATH = os.path.expanduser('~/dncskin_classification/workspace/data/dog')

# 이미지 경로 지정 (하위 폴더 구조 : (유증상/무증상)/class/_bbox/이미지)
CONST_IMG_PATH = f'{CONST_ROOT_PATH}/bbox'

# TFRecord 저장할 폴더 경로 지정
CONST_WORK_PATH = f'{CONST_ROOT_PATH}/tfrc_cla'

# True or False
SHUFFLE = False

# ex) [ "A1", "A2", "A3", "A4", "A5", "A6" ]
CONST_CLASS = ["A4", "A5", "A6"] 

LABEL_NAME = "_".join(CONST_CLASS)

# set folder name
FOLDER_BBOX = '_bbox'
FOLDER_TFRC = f'tfrecord_{LABEL_NAME}'
FOLDER_RESULT = 'result'
FOLDER_SYMPTOMATIC = "유증상"
FOLDER_ASYMPTOMATIC = "무증상"
FOLDER_LOG = "logs"

CONST_TFRC_PATH = os.path.join(CONST_WORK_PATH, FOLDER_TFRC)

EXTENSION_IMAGE = '.jpg'

EXTENSION_TFRECORDS = '.tfrecords'

CONST_CSV_TFRCLIST = "tfrecord_list.csv"
CONST_CSV_SPLITTED = "splited_df.csv"

CONST_TFRECORD = "tfrecord"
CONST_RESULT_LOSS = "result_loss.jpg"

NUM_IMAGE = 1000