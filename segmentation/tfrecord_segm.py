from time import sleep
import tensorflow as tf
import os
import pandas as pd
import math

from IPython.display import clear_output
from tensorflow.python.util.deprecation import deprecated

from tfrc_segm_conf import *
from print_log import *

set_debug(False)
set_log_path(os.path.join(CONST_OUTPUT_PATH, FOLDER_LOG), FOLDER_TFRC)

def logout(msg, stdout = True, force_flush = False):
    print_log(level = 'i', msg = msg, tag = "tfrecord_segm", on_screen_display = (stdout or is_debug()), force_flush = force_flush)


def make_input_list(start_path : str, target_folder : str = None):
    '''
    make_input_list
    =============

    make input and target(mask) list.
    #### make_image_list() and make_mask_list() has deprecated!
    
    Arguments:
    ------------
    start_path : str
        root path to search
    target_folder : str
        target sub-folder name to be collected.
        if pass None, whole sub-folders to be collected.
        you can pass FOLDER_IMAGE or FOLDER_MASK. (see segmconf.py)

    Returns:
    ------------
    - image_label_list: collected sub-folder list as labels
    - image_list: collected files list
    - except_list: excepted files list
    '''
    # make list
    image_list = []
    image_label_list = []
    except_list = []

    # start traverse directories
    for (path, dirs, files) in os.walk(start_path):
        logout(f"search path: {path}")

        if len(files) == 0 or (target_folder is not None and target_folder not in path):
            logout("skip current folder")
            continue
        else:
            logout(f"{len(dirs)} dirs:\n\t{dirs}\n\tand {len(files)} files found")

        folder_label = os.path.basename(path.split(os.sep + target_folder)[0])
        for file in files:
            if os.path.splitext(file)[-1] in EXTENSION_IMAGE:
                image_label_list.append(folder_label)
                image_list.append(file)
            else: 
                except_list.append(file)

    logout(f"# of image label = {len(image_label_list)}")

    return image_label_list, image_list, except_list


# 생성된 image, mask 이미지 정보 결합을 통한 학습 가능 데이터 확인
def make_df(img_path):
    # 경로 지정
    label, image, exp = make_input_list(img_path, FOLDER_IMAGE)
    mask_label, mask, mask_exp = make_input_list(img_path, FOLDER_MASK)

    # make image df
    label_df = pd.DataFrame(label, columns=['folder_name'])
    image_df = pd.DataFrame(image, columns=['image_name'])
    
    image_df = pd.concat((image_df, label_df), axis=1)
    image_df['id'] = image_df.image_name.str.split('.').str[0]
    image_df.reset_index(drop=True, inplace=True)

    # make mask df
    mask_label_df = pd.DataFrame(mask_label, columns=['folder_name'])
    mask_df = pd.DataFrame(mask, columns=['mask_name'])
    mask_df = pd.concat((mask_df, mask_label_df), axis=1)
    mask_df['id'] = mask_df.mask_name.str.split('.').str[0]   # for crop mask images
    mask_df.reset_index(drop=True, inplace=True)

    # merge image and mask df
    df = pd.merge(image_df, mask_df, how='left', on=['id', 'folder_name'])
    logout("존재하지 않는 mask 이미지 개수: ", df['mask_name'].isnull().sum())
    df = df.fillna('0')
    return df


def split_data(df):
    group = list(df['folder_name'].unique())
    ds = pd.DataFrame({'image_id':[0], 'label': [0], 'set': ['non']})
    for symp in group:
        group_df = df[df['folder_name']==symp].reset_index(drop=True, inplace=False)
        group_df = group_df.sample(frac=1).reset_index(drop=True)

        symptomatic = group_df[['image_id', 'label']][group_df['label']!=7].reset_index(drop=True,inplace=False)
        asymptomatic = group_df[['image_id', 'label']][group_df['label']==7].reset_index(drop=True,inplace=False)
        
        symptomatic['set']='train'
        symptomatic['set'].loc[len(symptomatic)*0.8-1e-15:len(symptomatic)*0.9-1e-15]='validation'
        symptomatic['set'].loc[len(symptomatic)*0.9-1e-15:] = 'test'
        ds = pd.concat((ds, symptomatic), ignore_index=True)

        asymptomatic['set']='train'
        asymptomatic['set'].loc[len(asymptomatic)*0.8-1e-15:len(asymptomatic)*0.9-1e-15]='validation'
        asymptomatic['set'].loc[len(asymptomatic)*0.9-1e-15:] = 'test'
        ds = pd.concat((ds, asymptomatic), ignore_index=True)

    df = pd.merge(df, ds, how='left', on=['image_id', 'label'])
    return df


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_id, label, image_string, mask_string):

    feature = {
        'id': _int64_feature(image_id),
        'label_id': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
        'image_mask': _bytes_feature(mask_string)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def tfrecord_write(df, img_path, tfrc_path):
    # 메타데이터 내  mask 이미지가 존재하지 않는 데이터 제거 ('mask_name'이 0 또는 '0'으로 표시)
    df = df[(df['mask_name']!='0')&(df['mask_name']!=0)]
    folder_list = list(df['folder_name'].unique())
    set_list = list(df['set'].unique())

    # class (폴더명) 분리
    for folder in folder_list:
        clear_output(wait=True)
        folder_df = df[df['folder_name']==folder]
        logout(f"start to make tfrecord...class: {folder}")
        
        # train / validation / test dataset 분리
        for set in set_list:
            #logout(f"start to make {set} data tfrecord...class: {folder}")
            set_df = folder_df[folder_df['set']==set]
            # set_df.reset_index(drop=True, inplace=True)
            label_list = list(set_df['label'].unique())

            # 유증상, 무증상 분리
            for label in label_list:
                if label == 7:
                    symp = FOLDER_ASYMPTOMATIC
                else:
                    symp = FOLDER_SYMPTOMATIC
                label_df = set_df[set_df['label']==label]
                label_df.reset_index(drop=True, inplace=True)

                total_image = len(label_df)
                num_file = math.ceil(total_image/NUM_IMAGE)

                logout(f"start to make tfrecord for {set} (class: {folder} in {symp})")

                # group(class, 유/무증상, train/validation/test)의 tfrecord 분리
                for i in range(num_file):
                    start_index = i*NUM_IMAGE
                    end_index = (i+1)*NUM_IMAGE
                    file_list = label_df[start_index:end_index]
                    file_list.reset_index(drop=True, inplace=True)

                    folder_name = file_list['folder_name'][i]
                    record_file = CONST_TFRECORD+f"_label_{folder_name}_{symp}_{set}_set_"+str(i+1)+f"_of_{num_file}"+EXTENSION_TFRECORDS
                    logout(f"process {i+1}/{num_file}th tfrecord for {NUM_IMAGE} images [{start_index}:{end_index}] from {folder_name} with label {label}")

                    with tf.io.TFRecordWriter(os.path.join(tfrc_path, record_file)) as writer:
                        for j in range(len(file_list)):
                            #logout(f"class {label} {symp} {set} data set...{i+1}/{num_file} tfrecord processing...({j+1}/{NUM_IMAGE})")
                            image_name = file_list['image_name'][j]
                            mask_name = file_list['mask_name'][j]
                            label_id = file_list['label'][j]
                            image_id = file_list['image_id'][j]
                            
                            image_file_path = os.path.join(img_path, symp, folder_name, FOLDER_IMAGE, image_name)
                            mask_file_path = os.path.join(img_path, symp, folder_name, FOLDER_MASK, mask_name)
                            image_string = open(image_file_path, 'rb').read()    
                            mask_string = open(mask_file_path, 'rb').read()
                            
                            tf_example = image_example(image_id, label_id, image_string, mask_string)
                            writer.write(tf_example.SerializeToString())
                        sleep(1.)

            logout(f"end to make {num_file} {symp} {set} data tfrecord...class: {label}")
        logout(f"end to make tfrecord...class: {label}")


if __name__ == '__main__':
    logout("Start progress...")

    # output path for make_tfrecord.py
    if not os.path.exists(CONST_TFRC_PATH):
        os.makedirs(CONST_TFRC_PATH)
        logout(f"output folder {CONST_TFRC_PATH} created")
    logout(f"Created TFRecord files will be stored in {CONST_TFRC_PATH}")

    logout(f"start to make {FOLDER_SYMPTOMATIC} image list")
    meta_symptomatic = make_df(os.path.join(CONST_IMG_PATH, FOLDER_SYMPTOMATIC))
    logout(f"finish to make {FOLDER_SYMPTOMATIC} image list")

    logout(f"start to make {FOLDER_ASYMPTOMATIC} image list")
    meta_asymptomatic = make_df(os.path.join(CONST_IMG_PATH, FOLDER_ASYMPTOMATIC))
    logout(f"finish to make {FOLDER_ASYMPTOMATIC} image list")

    logout("start organizing list of image files")
    meta_symptomatic['label'] = 0
    for i in range(len(meta_symptomatic['folder_name'].unique())):
        meta_symptomatic['label'][meta_symptomatic['folder_name']==meta_symptomatic.sort_values('folder_name')['folder_name'].unique()[i]] = i+1
    meta_asymptomatic['label'] = 7
    meta = pd.concat((meta_symptomatic, meta_asymptomatic), ignore_index=True)
    meta.reset_index(drop=True, inplace=True)

    # path_csv = os.path.join(CONST_OUTPUT_PATH, CONST_CSV_TFRCLIST)
    # meta.to_csv(path_csv, index=False, encoding='utf-8-sig')
    logout(f"finish organizing list of image files ")

    # check statistics of mask image
    mask_df = meta[(meta['mask_name']!=0)&(meta['mask_name']!='0')]
    mask_df.reset_index(drop=True, inplace=True)
    file_name = []
    rate = []
    height = []
    width = []

    logout(f"start calculation...(total: {len(mask_df)})")

    i = 0
    for i in range(len(mask_df)):
        label_name = mask_df['folder_name'][i]
        mask_name = mask_df['mask_name'][i]
        image_name = mask_df['image_name'][i]
        if mask_df['label'][i] != 7:
            mask_file_path = os.path.join(CONST_IMG_PATH, FOLDER_SYMPTOMATIC, label_name, EXTENSION_MASK, mask_name)
        else:
            mask_file_path = os.path.join(CONST_IMG_PATH, FOLDER_ASYMPTOMATIC, label_name, EXTENSION_MASK, mask_name)
        mask_string = open(mask_file_path, 'rb').read()
        mask = tf.io.decode_jpeg(mask_string)
        h, w, d= mask.shape
        mask_rate = mask.numpy().sum() / (h*w)
        file_name.append(image_name)
        rate.append(mask_rate)
        height.append(h)
        width.append(w)

        if (i + 1) % 1000 == 0:
            logout(f"{(i + 1)} / {len(mask_df)} done")
            sleep(1.)
    logout(f"calculating {i + 1} / {len(mask_df)} files done")

    file_df = pd.DataFrame(file_name, columns=['image_name'])
    rate_df = pd.DataFrame(rate, columns=['mask_rate'])
    height_df = pd.DataFrame(height, columns=['height'])
    width_df = pd.DataFrame(width, columns=['width'])
    mask_rate = pd.concat((file_df, rate_df, height_df, width_df), axis=1)

    rate_df = pd.merge(meta, mask_rate, how='left', on='image_name')
    rate_df.fillna(0, inplace=True)
    rate_df = rate_df.sample(frac=1).reset_index(drop=True)
    col_names = rate_df.columns
    rate_df.reset_index(drop=False, inplace=True)
    new_col = ['image_id']
    new_col = new_col + list(col_names)
    rate_df.columns = new_col
    rate_df = rate_df.sample(frac=1).reset_index(drop=True)

    # path_csv = os.path.join(CONST_OUTPUT_PATH, CONST_CSV_MASKAREA)
    # rate_df.to_csv(path_csv, index=False, encoding='utf-8-sig')
    # logout(f"area of labeled region saved in {path_csv}")

    splited_df = split_data(rate_df)

    path_csv = os.path.join(CONST_OUTPUT_PATH, CONST_CSV_TFRCLIST)
    splited_df.to_csv(path_csv, index=False, encoding='utf-8-sig')
    logout(f"splitted dataset info saved in {path_csv}")
    
    logout("writing tfrecrd files being started")
    tfrecord_write(splited_df, CONST_IMG_PATH, CONST_TFRC_PATH)

    logout("Done.", force_flush=True)
