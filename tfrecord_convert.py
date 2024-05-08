import tensorflow as tf
import os

def tfrecord(path, classification_type, image_size):
    tfrecord_files = tf.data.Dataset.list_files(os.path.join(path, "*.tfrecords"), shuffle=False)
    
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_files)

    def transform_image(image_string):
        image = tf.io.decode_jpeg(image_string, channels = 3)
        image = tf.image.resize(image,(image_size, image_size))
        return image

    def parse_image_function(raw_image_dataset):
        # Create a dictionary describing the features.
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        records = tf.io.parse_single_example(raw_image_dataset, image_feature_description)
        return records

    def transform_record(raw_image_dataset):
        image = transform_image(raw_image_dataset['image_raw']) / 255.
        if classification_type == 'multi':
            label = tf.reshape(tf.one_hot(raw_image_dataset['label']-1, 7), [7])
        elif classification_type == 'binary': 
            label = tf.where(raw_image_dataset['label']==7, [1, 0], [0, 1])
        else:
            print("CLASSIFICATION TYPE ERROR")
        return image, label

    dataset = raw_image_dataset.map(parse_image_function, num_parallel_calls = tf.data.AUTOTUNE).cache().map(transform_record, num_parallel_calls = tf.data.AUTOTUNE)
    return dataset