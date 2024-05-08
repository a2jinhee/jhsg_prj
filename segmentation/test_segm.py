import os
from pyexpat.errors import XML_ERROR_NOT_STANDALONE
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

import matplotlib.pyplot as plt
import pandas as pd

from test_segm_conf import *
from function import *
from print_log import *


CONST_LOG_TAG = "test_segm"

def logout(msg, stdout = True, force_flush = False):
    print_log(level = 'i', msg = msg, tag = CONST_LOG_TAG, on_screen_display = (stdout or is_debug()), force_flush = force_flush)

# get best weight & class_id
def check_model_path(model_path):
    latest = tf.train.latest_checkpoint(os.path.join(model_path, 'epochs'))
    if latest == None:
        raise Exception("There isn't model check point. You should check ")
    else:
        model_info = os.path.basename(model_path).split('_')
        class_id = model_info[1]
        lr = model_info[4]
    return latest, class_id, lr

# tranform tfrecord
def parse_tfrecord2(tfrecord):
    def parse_image_function(tfrecord):
        features = {
                    'id': tf.io.FixedLenFeature([], tf.int64),
                    'label_id': tf.io.FixedLenFeature([], tf.int64),
                    'image_raw': tf.io.FixedLenFeature([], tf.string),
                    'image_mask': tf.io.FixedLenFeature([], tf.string)}

        records = tf.io.parse_single_example(tfrecord, features)
        return records

    def transform_record2(records):
        image_id = records['id']
        image = transform_image(records['image_raw']) / 255.
        mask = transform_mask(records['image_mask'], opt=True)

        return image_id, image, mask
    
    dataset = tfrecord.map(parse_image_function, num_parallel_calls=tf.data.AUTOTUNE).map(transform_record2, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# prepare test data
def prepareTestData(class_id):
    with tf.device('/cpu:0'):
        logout(f"start make test from {CONST_TFRC_PATH}")
        test_ds = read_tfrc(tfrc_list(CONST_TFRC_PATH, split=CONST_TEST, label=class_id, asymp=False))
        test_ds = parse_tfrecord2(test_ds)
    image_id = []
    X_test = []
    y_test = []
    num = 0
    total_num = len(list(test_ds))
    logout(f"the number of test dataset is {total_num}")
    for data in test_ds:
        num += 1
        print(f"processing...({num}/{total_num})")
        image_id.append(data[0])
        X_test.append(data[1])
        y_test.append(data[2])
    return image_id, X_test, y_test

# load model
def load_model(latest, lr):
    model = build_model(lr)
    model.load_weights(latest)
    return model

def cal_iou(y_true, y_pred):
    y_true = np.argmax(y_true, axis=-1)[:,:,np.newaxis]
    y_pred = np.argmax(y_pred, axis=-1)[:,:,np.newaxis]
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    x = (intersection + 1e-15) / (union + 1e-15)
    x = x.astype(np.float32)
    return x, y_true, y_pred

def eval_model(X_test, y_test, model, image_id, IoU_threshold, class_id):
    tp = 0
    positive = 0
    total = 0
    iou_list = []
    total_num = len(X_test)
    for i in range(total_num):
        img_id = image_id[i]
        x_test = X_test[i]
        y_true = y_test[i]
        if np.argmax(y_true, axis=-1).sum() == 0:
            pass
        elif np.argmax(y_true, axis=-1).sum() < 0:
            raise Exception(f"the value in mask image have negative value, image_id={img_id}")
        else:
            positive += 1
        total += 1
        # predict
        y_pred = model.predict(x_test[tf.newaxis, ...])[0]
        
        # calculate IoU
        iou_value, y_true, y_pred = cal_iou(y_true, y_pred)
        iou_list.append(iou_value)
        
        # save image
        if iou_value >= IoU_threshold:
            image_name = os.path.join(base_path, "predictions", f"TRUE_{class_id}_prediction_{IoU_threshold}_IoU_{i+1}_of_{total_num}_{img_id}.png")
            show_prediction(x_test, y_true, model=None, model_pred=y_pred, opt='total', save_path=image_name, show=False)
            logout(f"{img_id}_processing...({i+1}/{total_num})...TRUE...{iou_value}...model_accuracy...{(tp/total)*100}%")
            tp += 1
        else:
            image_name = os.path.join(base_path, "predictions", f"FALSE_{class_id}_prediction_{IoU_threshold}_IoU_{i+1}_of_{total_num}_{img_id}.png")
            show_prediction(x_test, y_true, model=None, model_pred=y_pred, opt='total', save_path=image_name, show=False)
            logout(f"{img_id}_processing...({i+1}/{total_num})...FALSE...{iou_value}...model_accuracy...{(tp/total)*100}%")
    
    logout(f"test data prediction saved to {os.path.join(base_path, 'predictions')}")

    acc = tp/total
    recall = tp/positive
    logout(f"complete evaluating model")
    logout(f"result: accuracy = {acc*100}%, recall = {recall*100}% at IoU {IoU_threshold*100}%")
    return iou_list

if __name__ == "__main__":
    set_debug(False)
    set_flush_counter(0)

    result_path = os.path.join(os.path.expanduser(CONST_WORK_PATH), FOLDER_RESULT)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    check_model_list = os.listdir(CONST_MODEL_PATH)

    for i in range(len(check_model_list)):
        base_path = os.path.join(result_path, check_model_list[i])
        if not os.path.exists(base_path):
    	    os.makedirs(base_path)
        model_path = os.path.join(CONST_MODEL_PATH, check_model_list[i])
        
        log_path = os.path.join(base_path, FOLDER_LOG)
        if not os.path.exists(log_path):
    	    os.makedirs(log_path)
        set_log_path(log_path, CONST_LOG_TAG)
        
        pred_path = os.path.join(base_path, "predictions")
        if not os.path.exists(pred_path):
    	    os.makedirs(pred_path)

        logout("Evaluating process being started...", force_flush=True)

        latest, class_id, lr = check_model_path(model_path)
        logout(f"detect test model info")

        logout(f"start train for class {class_id}")

        image_id, X_test, y_test = prepareTestData(class_id)
        logout("preparing datasets done")

        model = load_model(latest, lr)
        logout("complete load model")

        iou_list = eval_model(X_test, y_test, model, image_id, CONST_IOU, class_id)
        id_df = pd.DataFrame(image_id, columns=['id'])
        iou_df = pd.DataFrame(iou_list, columns=['iou'])
        df = pd.concat((id_df, iou_df), axis=1)
        df['tp'] = 0
        df['tp'][df['iou']>=CONST_IOU] = 1
        iou_path = os.path.join(base_path,f"{class_id}_iou.csv")
        df.to_csv(iou_path, index=False)
        logout(f"result of test saved to {iou_path}")
        logout(f"test for class {class_id} done", force_flush=True)

    logout("process done.", force_flush=True)
    
