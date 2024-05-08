# FOLDER SETTING SECTION
#-----------------------------------------------

# for make_tfrecord.py
# 전처리 프로그램을 통해 생성된 이미지 path
CONST_IMG_PATH = '/home/gazzilabs/dncskin_segmentation/workspace/data/cat/mask'

# for train
# mask 이미지 분류에 따른 tfrc 별도 생성이 발생하기 때문에 tfrc별 저장 및 결과 저장 work path를 별도 지정
CONST_OUTPUT_PATH = "/home/gazzilabs/dncskin_segmentation/workspace/data/cat"

# set folder name
FOLDER_IMAGE = "_input"
FOLDER_MASK = "_mask"
FOLDER_TFRC = 'tfrc_segm'
FOLDER_SYMPTOMATIC = "유증상"
FOLDER_ASYMPTOMATIC = "무증상"
FOLDER_LOG = "logs"

import os
CONST_TFRC_PATH = os.path.join(CONST_OUTPUT_PATH, FOLDER_TFRC)

EXTENSION_IMAGE = '.jpg'
EXTENSION_JSON = ".json"
EXTENSION_MASK = '_mask'
EXTENSION_TFRECORDS = '.tfrecords'

CONST_CSV_TFRCLIST = "tfrecord_segm_list.csv"

CONST_TFRECORD = "tfrecord"
NUM_IMAGE = 1000
