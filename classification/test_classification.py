import tensorflow as tf
import os
import numpy as np
import pandas as pd
import seaborn as sns

from pytz import timezone
from datetime import datetime
from classification_config_test import *
from tfrecord_convert import tfrecord
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

import sys
if not os.path.exists(os.path.expanduser(TEST_FOLDER)):
    os.makedirs(os.path.expanduser(TEST_FOLDER))
sys.stdout = open(f'{os.path.expanduser(TEST_FOLDER)}/{LOG_FNAME}.txt', 'a', encoding='utf8')

today = datetime.now(timezone('Asia/Seoul'))
print(f"test start : {today}")

if TEST_TYPE == 'multi':
    CLASS_COUNT = 7
    ACTIVATION = 'softmax'
    LOSS = 'categorical_crossentropy'
elif TEST_TYPE == 'binary':
    CLASS_COUNT = 2
    ACTIVATION = 'sigmoid'
    LOSS = 'binary_crossentropy'
else:
    print(f"wrong KINDS : {TEST_TYPE}")


class Load_model:
    def __init__(self, model_name):
        self.num_class = CLASS_COUNT
        self.model_name = model_name
    
    def inception_v4(self):
        network = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(TEST_IMAGE_SIZE, TEST_IMAGE_SIZE, 3),
        pooling='avg')
        return network 
    
    def build_network(self):
        if self.model_name == 'inception_v4':
            network = self.inception_v4() 
        model = Sequential()
        model.add(network)
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(self.num_class, activation=ACTIVATION))
        model.summary()
        return model

def create_model():
    optimizer = tf.keras.optimizers.SGD(learning_rate = TEST_LEARNING_RATE, decay = 1e-5, momentum = 0.999, nesterov = True) 
    model = Load_model("inception_v4").build_network() 
    model.compile(loss=LOSS,
            optimizer=optimizer,
            metrics=['acc', tf.keras.metrics.Precision(name = 'precision'), tf.keras.metrics.Recall(name = 'recall')])
    return model

with tf.device('/cpu:0'):
    checkpoint = tf.train.latest_checkpoint(os.path.expanduser(MODEL_PATH))
    model = create_model()
    model.load_weights(checkpoint)

with tf.device('/cpu:0'):
    dataset = tfrecord(os.path.expanduser(TEST_PATH), TEST_TYPE, TEST_IMAGE_SIZE).batch(TEST_BATCH_SIZE)

test = model.evaluate(dataset, batch_size = TEST_BATCH_SIZE)

p = model.predict(dataset)

DATASET_COUNT = len(list(dataset))

y_true = []
y_pred = []

i=0
for image, label in dataset.take(DATASET_COUNT):
    i+=1
    label_len = len(label)
    for q in range(label_len):
        y_true.append(np.argmax(label[q])+1)
        y_pred.append(np.argmax(p[(i-1) * TEST_BATCH_SIZE + q])+1)

conf_mat = confusion_matrix(y_true, y_pred)
df_conf_mat = pd.DataFrame(conf_mat, columns = TEST_CLASS, index = TEST_CLASS)
sns_heatmap = sns.heatmap(data = df_conf_mat, annot = True, fmt = '', linewidths = .5, cmap = 'BuGn_r')
sns_heatmap.get_figure().savefig(f"{os.path.expanduser(TEST_FOLDER)}/{CONFUSION_MATRIX_FNAME}.png")

print(f'test_loss : {test[0]}, test_acc : {test[1]}, test_precision : {test[2]}, test_recall : {test[3]}')

print(classification_report(y_true, y_pred, digits = 5, target_names = TEST_CLASS))

today1 = datetime.now(timezone('Asia/Seoul'))
print(f"test end : {today1}")