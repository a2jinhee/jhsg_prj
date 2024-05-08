import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

import os
import numpy as np
import matplotlib.pyplot as plt

from common_conf import *

# input image String to Tensor
def transform_image(image_string):
    image = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


# mask image String to Tensor
def transform_mask(image_string, opt=True):
    mask = tf.io.decode_jpeg(image_string)
    mask = tf.image.resize(mask, IMAGE_SIZE)
    if opt:
        mask = tf.one_hot(tf.cast(mask, dtype=tf.int32), depth=2, axis=-1)
        mask = tf.reshape(mask, MASK_SHAPE)
    else:
        pass
    return mask

def tfrc_list(tfrc_path, split='train', label=None, asymp=False):
    if split not in ['train', 'validation', 'val', 'test']:
        raise Exception("you should select parameter 'split' in 'train', 'validation', 'val', 'test'")

    if label == None:
        print("choose all classes")
        label = '_'
    
    if asymp == True:
        print("include asymptomatic")
        asymptomatic = "_"
    elif asymp == False:
        print("only symptomatic")
        asymptomatic = "유증상"
    else:
        raise Exception("you should select parameter 'asymp' between True or False")
        
    file_list = f'*{label}*{asymptomatic}*{split}*{EXTENSION_TFRECORDS}'
    return os.path.join(tfrc_path, file_list)

# Generate TFRecord File List
def read_tfrc(path):
    tfrecord_files = tf.data.Dataset.list_files(path, shuffle=False)
    tfrecord = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    tfrecord = tfrecord.with_options(ignore_order)

    return tfrecord

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def build_model(lr):
    input_size = IMAGE_SHAPE

    inputs = Input(input_size, name="INPUT")
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(2, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(learning_rate = lr), loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics = ['acc', tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.MeanIoU(num_classes=2, name='miou'), iou])
    return model

# define function for drawing polygon in image
def show_comb(image, mask, pred=None):
    # draw polygon line in image
    mask = np.where(mask==0, image, [1, 1, 0])    # yellow

    plt.imshow(image)
    plt.imshow(mask, alpha=0.2)
    plt.title("ground_truth(yello)")
    if pred is not None:
        pred = np.where(pred == 0, image, [0, 0, 1])  # blue
        plt.imshow(pred, alpha=0.2)
        plt.title("ground_truth(yellow), prediction(blue)")
    plt.axis('off')
    

def model_predict(image, model):
    pred = model.predict(image[tf.newaxis, ...])
    b, h, w, c = pred.shape
    pred = tf.argmax(pred, axis=-1)
    pred = tf.reshape(pred, (h, w, 1))
    return pred

def show_prediction(image, mask, model=None, model_pred=None, opt='base', save_path=None, show=True):
    plt.clf()

    title = ['Input Image', 'True Mask', 'Predicted', 'Mask+Pred']
    
    if model != None:
        pred = model_predict(image, model)
        display_list = [image, mask, pred]
    elif model_pred is not None:
        pred = model_pred
        display_list = [image, mask, pred]
    else:
        pred = None
        display_list = [image, mask]

    if opt == 'base':
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        if save_path != None:
            plt.savefig(save_path)
        if show == True:
            plt.show()
    
    elif opt == 'comb':
        show_comb
        if save_path != None:
            plt.savefig(save_path)
        if show == True:
            plt.show()

    elif opt == 'total':
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list)+1, i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off') 
        plt.subplot(1, len(display_list)+1, len(display_list)+1)
        height, width, channels = image.shape

        mask = tf.image.resize(mask, (height, width), method="nearest")
        mask = np.where(mask==0, image, [0, 1, 0])    # green
        
        plt.imshow(image)
        plt.imshow(mask, alpha=0.2)
        plt.title("ground_truth(green)")
        if pred is not None:
            pred = np.where(pred == 0, image, [0, 0, 1])  # blue
            plt.imshow(pred, alpha=0.2)
            plt.title(title[3])
        plt.axis('off')
        if save_path != None:
            plt.savefig(save_path)
        if show == True:
            plt.show()
    else:
        raise Exception("fucntion show_pred: Wrong Option, You must choose one of ['base', comb', 'total']")
