# FOLDER SETTING SECTION
#-----------------------------------------------
import os
# tfrc폴더와 result폴더가 같이 있는 경로 지정
CONST_WORK_PATH = "~/dncskin_segmentation/workspace/data/cat"

CONST_MODEL_PATH = os.path.expanduser("~/dncskin_segmentation/workspace/data/cat/result_segm")

CONST_IOU = 0.5	# 정확도 측정 시 IoU 기준

# set folder name
FOLDER_TFRC = 'tfrc_segm'
FOLDER_RESULT = 'result_segm'
FOLDER_LOG = "logs"

CONST_TFRC_PATH = os.path.join(os.path.expanduser(CONST_WORK_PATH), FOLDER_TFRC)
CONST_RESULT_PATH = os.path.join(os.path.expanduser(CONST_WORK_PATH), FOLDER_RESULT)

CONST_TEST = "test"
