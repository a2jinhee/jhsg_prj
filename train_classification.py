import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

from classification_config_train import *
from tfrecord_convert import tfrecord

from pytz import timezone
from datetime import datetime
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

'''#GPU 선택 / GPU 하나일 경우 아래 두 코드 주석 처리
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"'''

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if TRAIN_TYPES == 'multi':
    CLASS_COUNT = 7
    ACTIVATION = 'softmax'
    LOSS = 'categorical_crossentropy'
elif TRAIN_TYPES == 'binary':
    CLASS_COUNT = 2 
    ACTIVATION = 'sigmoid'
    LOSS = 'binary_crossentropy'
else :
    print("TRAIN_TYPES ERROR!")

class Load_model:
    def __init__(self, MODEL_NAME):
        self.num_class = CLASS_COUNT
        self.model_name = MODEL_NAME

    def inception_v4(self):
        network = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                    pooling='avg')
        return network


    def build_network(self):
        if self.model_name == 'inception_v4':
            network = self.inception_v4()

        model = Sequential()
        model.add(network)
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(self.num_class, activation = ACTIVATION))
        model.summary()

        return model

class Fine_tunning:
    def __init__(self, train_path, val_path, model_name):
        with tf.device('/cpu:0'):
            train_dataset = tfrecord(train_path, TRAIN_TYPES, IMAGE_SIZE)

            val_dataset = tfrecord(val_path, TRAIN_TYPES, IMAGE_SIZE)

            if SHUFFLE == True :
                train_dataset = train_dataset.shuffle(buffer_size = 100000) \
                .cache() \
                .batch(BATCH_SIZE) \
                .prefetch(tf.data.experimental.AUTOTUNE)
            else :
                train_dataset = train_dataset.batch(BATCH_SIZE) \
                .prefetch(tf.data.experimental.AUTOTUNE)   

            val_dataset = val_dataset.batch(BATCH_SIZE)

        self.train_data =  train_dataset
        self.val_data =  val_dataset
        self.load_model = Load_model(model_name)
        self.epoch = EPOCH
        self.model_name = model_name
        self.train_path = train_path

    def training(self):
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, decay=1e-5, momentum=0.999, nesterov=True)

        model = self.load_model.build_network()
   
        check_point = ModelCheckpoint(CHECKPOINT_FILE, verbose=1,
                                    monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True)
        tbd_callback = TensorBoard(log_dir = SAVE_FOLDER_TSBOARD, histogram_freq = 1)

        model.compile(loss= LOSS,
                      optimizer=optimizer,
                      metrics=['acc', tf.keras.metrics.Precision(name = 'precision'), tf.keras.metrics.Recall(name = 'recall')])
        
        history = model.fit(
            self.train_data,
            epochs = self.epoch,
            validation_data = self.val_data,
            callbacks = [check_point, tbd_callback]
            )

        print("Train_Accuracy : ", history.history['acc'][-1])
        print("Train_Precision : ", history.history['precision'][-1])
        print("Train_Recall : ", history.history['recall'][-1])

        print("Val_Accuracy : ", history.history['val_acc'][-1])
        print("Val_Precision : ", history.history['val_precision'][-1])
        print("Val_Recall : ", history.history['val_recall'][-1])

        today1 = datetime.now(timezone('Asia/Seoul'))
        print(f"classification end : {today1}")
        
        return history

    def save_accuracy(self, history):
        acc = history.history['acc']
        loss = history.history['loss']
        precision = history.history['precision']
        recall = history.history['recall']

        val_acc = history.history['val_acc']
        val_loss = history.history['val_loss']
        val_precision = history.history['val_precision']
        val_recall = history.history['val_recall']

        epochs = range(len(acc))
        epoch_list = list(epochs)

        df = pd.DataFrame({'epoch': epoch_list, 'train_accuracy': acc, 'train_loss': loss, 'train_precision': precision, 'train_recall': recall, 'validation_accuracy': val_acc, 'validation_loss': val_loss, 'validation_precision': val_precision, 'validation_recall': val_recall},
                          columns=['epoch', 'train_accuracy', 'train_loss', 'train_precision', 'train_recall', 'validation_accuracy', 'validation_loss', 'validation_precision', 'validation_recall'])
        df_save_path = os.path.join(os.path.expanduser(RESULT_FILE_DIR), 'accuracy.csv')
        df.to_csv(df_save_path, index=False, encoding='euc-kr')

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        save_path = os.path.join(os.path.expanduser(RESULT_FILE_DIR), 'accuracy.png')
        plt.savefig(save_path)
        plt.cla()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        save_path = os.path.join(os.path.expanduser(RESULT_FILE_DIR), 'loss.png')
        plt.savefig(save_path)
        plt.cla()
        
today = datetime.now(timezone('Asia/Seoul'))
print(f"classification start : {today}")

if __name__ == '__main__':
    # tensorboard, model_saved path
    file_name = CONST_DIR_PATH.split('/')[-1]
    TRAIN_LABEL = file_name.strip('tfrecord_')

    MODEL_SAVE_DIR = TRAIN_TYPES + "_" + TRAIN_LABEL
    RESULT_FILE_DIR = os.path.join(os.path.expanduser(CONST_SAVE_DIR), MODEL_SAVE_DIR)

    # tensorboard
    SAVE_FOLDER_TSBOARD = os.path.join(os.path.expanduser(RESULT_FILE_DIR), "tensorboard")
    if not os.path.exists(SAVE_FOLDER_TSBOARD):
        os.makedirs(SAVE_FOLDER_TSBOARD)

    # model
    SAVE_FOLDER_MODEL = os.path.join(os.path.expanduser(RESULT_FILE_DIR), "model")
    if not os.path.exists(SAVE_FOLDER_MODEL):
        os.makedirs(SAVE_FOLDER_MODEL)
    CHECKPOINT_FILE = os.path.join(os.path.expanduser(SAVE_FOLDER_MODEL), 'model-{epoch:03d}-{acc:03f}-{val_acc:03f}')

    # console log
    LOG_DIR = os.path.join(os.path.expanduser(RESULT_FILE_DIR), 'log.txt')
    
    #console log save
    sys.stdout = open(LOG_DIR, 'a', encoding='UTF-8')

    fine_tunning = Fine_tunning(train_path = CONST_TRAIN_PATH,
                                val_path = CONST_VAL_PATH,
                                model_name = MODEL_NAME)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)
