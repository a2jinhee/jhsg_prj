import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from time import sleep
from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd

from train_segm_conf import *
from function import *
from print_log import *


CONST_LOG_TAG = "train_segm"

def logout(msg, stdout = True, force_flush = False):
    print_log(level = 'i', msg = msg, tag = CONST_LOG_TAG, on_screen_display = (stdout or is_debug()), force_flush = force_flush)


# set path
MODEL_NAME = ""

# set root path
MODEL_PATH = ""

# set save path for prediction images
PREDICTION_PATH = ""

# set save path for model per epochs
EPOCH_PATH = ""


def checkResultFolder():
    global MODEL_NAME
    global MODEL_PATH
    global PREDICTION_PATH
    global EPOCH_PATH

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if not os.path.exists(PREDICTION_PATH):
        os.makedirs(PREDICTION_PATH)

    if not os.path.exists(EPOCH_PATH):
        os.makedirs(EPOCH_PATH)

# Read TFRecord
def parse_tfrecord(tfrecord):
    def parse_image_function(tfrecord):
        features = {
            'id': tf.io.FixedLenFeature([], tf.int64),
            'label_id': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'image_mask': tf.io.FixedLenFeature([], tf.string)
            }

        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(tfrecord, features)

    def transform_record(records):
        image = transform_image(records['image_raw']) / 255.
        mask = transform_mask(records['image_mask'], opt=True)

        return image, mask
    
    dataset = tfrecord.map(parse_image_function, num_parallel_calls=tf.data.AUTOTUNE).map(transform_record, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def prepareDataset(class_index):
    with tf.device('/cpu:0'):
        logout(f"start make train from {CONST_TFRC_PATH}")
        train_ds = read_tfrc(tfrc_list(CONST_TFRC_PATH, split=CONST_TRAIN, label=CONST_CLASS[class_index], asymp=False))
        train_ds = parse_tfrecord(train_ds)
        train_ds = train_ds.batch(BATCH_SIZE) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        logout(f"success make train")

        logout(f"start make validation {CONST_TFRC_PATH}")
        val_ds = read_tfrc(tfrc_list(CONST_TFRC_PATH, split=CONST_VAL, label=CONST_CLASS[class_index], asymp=False))
        val_ds = parse_tfrecord(val_ds)
        val_ds = val_ds.batch(BATCH_SIZE)
        logout(f"success make validation")

    return train_ds, val_ds


def getOneExample(dataset, class_index):
    sample_image = None
    sample_mask = None

    for images, masks in dataset.take(1):
        sample_image, sample_mask = images[0], masks[0]

        h, w, c = sample_mask.shape
        sample_mask = tf.argmax(sample_mask, axis=-1)
        sample_mask = tf.reshape(sample_mask, (h, w, 1))

        save_name = os.path.join(EPOCH_PATH, f"{CONST_CLASS[class_index]}_sample.png")
        show_prediction(sample_image, sample_mask, model=None, opt='total', save_path=save_name, show=False)
        sleep(1.)

    return sample_image, sample_mask

def doFit(model, train_dataset, val_dataset, sample_image, sample_mask, class_index):
    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            clear_output(wait=True)

            save_name = os.path.join(EPOCH_PATH, f"{CONST_CLASS[class_index]}_model_{epoch:04d}.png")
            show_prediction(sample_image, sample_mask, model=model, opt='total', save_path=save_name, show=False)
            logout('Sample Prediction after epoch {}\n'.format(epoch+1))

    checkPointCallback = ModelCheckpoint(EPOCH_PATH + os.sep + 'model-{epoch:03d}-{acc:03f}-{val_acc:03f}-{loss:03f}-{val_loss:03f}-{iou:03f}-{val_iou:03f}', verbose=1,
                                monitor = 'val_loss', save_best_only = False, save_weights_only = True, mode = 'min', save_freq = "epoch")

    tb_log_dir = os.path.join(MODEL_PATH, FOLDER_LOG, FOLDER_TENSORBOARD, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tbCallback = TensorBoard(log_dir = tb_log_dir, histogram_freq = 1)

    callback_list = [DisplayCallback(), checkPointCallback, tbCallback]

    history = model.fit(train_dataset,
                    batch_size = BATCH_SIZE, epochs = EPOCHS,
                    validation_data=val_dataset,
                    callbacks=callback_list)

    return history

def displayHistory(history, ray_class_id, metrix):
    plt.figure()

    plt.plot(history.epoch, history.history[f'{metrix}'], 'r', label=f'Training {metrix}')
    plt.plot(history.epoch, history.history[f'val_{metrix}'], 'm.', label=f'Validation {metrix}')
    
    plt.title(f'Training and Validation {metrix}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.savefig(os.path.join(MODEL_PATH, f"{ray_class_id}_{metrix}.png"))
    plt.cla()
    
if __name__ == "__main__":
    set_debug(False)
    set_flush_counter(0)

    for class_index in range(len(CONST_CLASS)):
        # set path
        MODEL_NAME = f"U-Net_{CONST_CLASS[class_index]}_{EPOCHS}_{BATCH_SIZE}_{LEARNING_RATE[class_index]}_{IMAGE_SHAPE}_{MASK_SHAPE}"

        # set root path
        MODEL_PATH = os.path.join(CONST_RESULT_PATH, MODEL_NAME)

        # set save path for prediction images
        PREDICTION_PATH = os.path.join(MODEL_PATH, "predictions")

        # set save path for model per epochs
        EPOCH_PATH = os.path.join(MODEL_PATH, "epochs")
        
        checkResultFolder()
        
        set_log_path(os.path.join(MODEL_PATH, FOLDER_LOG), CONST_LOG_TAG)

        logout("Modeling process being started...", force_flush=True)
        
        # checkResultFolder()
        logout("check folders for results done")

        logout(f"start train for class {CONST_CLASS[class_index]}")

        train_ds, val_ds = prepareDataset(class_index)
        logout("preparing datasets done and show one eample")

        sample_image, sample_mask = getOneExample(train_ds, class_index)

        model = build_model(lr=LEARNING_RATE[class_index])
        logout("building model done and start training")

        history = doFit(model, train_ds, val_ds, sample_image, sample_mask, class_index)

        logout("training model done and show history")
        scores = CONST_SCORE
        for metrix in scores:
            displayHistory(history, CONST_CLASS[class_index], metrix)

        # save history
        history_path = os.path.join(MODEL_PATH, f"{CONST_CLASS[class_index]}_history.csv")
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(history_path, index=False)
        logout(f"training history saved to {history_path}")

        save_name = os.path.join(EPOCH_PATH, f"{CONST_CLASS[class_index]}_model_result.png")
        show_prediction(sample_image, sample_mask, model=model, opt='total', save_path=save_name, show=False)

        logout(f"train for class {CONST_CLASS[class_index]} done", force_flush=True)

    logout("process done.", force_flush=True)
