from time import sleep
import tensorflow as tf
import os
import pandas as pd
import math

from IPython.display import clear_output

from dog_multi_456_tfrecord import *

from print_log import *

set_debug(False)
set_log_path(os.path.join(CONST_WORK_PATH, FOLDER_LOG), FOLDER_TFRC)

def logout(msg, stdout = True, force_flush = False):
    print_log(level = 'i', msg = msg, tag = "make_tfrecord", on_screen_display = (stdout or is_debug()), force_flush = force_flush)


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
        for a in range(len(CONST_CLASS)):
            if CONST_CLASS[a] in path:
                if len(files) == 0 or (target_folder is not None and target_folder not in path):
                    logout("skip current folder")
                    continue
                else:
                    logout(f"{len(dirs)} dirs:\n\t{dirs}\n\tand {len(files)} files found")

                folder_label = path.split(os.path.sep)[-2]
                for file in files:
                    if os.path.splitext(file)[-1] in EXTENSION_IMAGE:
                        image_label_list.append(folder_label)
                        image_list.append(file)
                    else: 
                        except_list.append(file)
            else:
                pass

    logout(f"# of image label = {len(image_label_list)}")

    return image_label_list, image_list, except_list


# 생성된 image, mask 이미지 정보 결합을 통한 학습 가능 데이터 확인
def make_df(img_path):
    # 경로 지정
    '''
    label, image, exp = make_image_list(img_path)
    mask_label, mask, mask_exp = make_mask_list(mask_path)
    '''
    label, image, exp = make_input_list(img_path, FOLDER_BBOX)

    # make image df
    label_df = pd.DataFrame(label, columns=['folder_name'])
    image_df = pd.DataFrame(image, columns=['image_name'])
    image_df = pd.concat((image_df, label_df), axis=1)
    image_df['id'] = image_df.image_name.str.split('.').str[0]
    image_df.reset_index(drop=True, inplace=True)

    return image_df


def split_data(df):
    group = list(df['folder_name'].unique())
    ds = pd.DataFrame({'image_id':[0], 'label': [0], 'set': ['non']})
    for symp in group:
        group_df = df[df['folder_name']==symp].reset_index(drop=True, inplace=False)
        
        if SHUFFLE == True:
            group_df = group_df.sample(frac=1).reset_index(drop=True)
        elif SHUFFLE == False:
            pass

        symptomatic = group_df[['image_id', 'label']][group_df['label']!=7].reset_index(drop=True,inplace=False)
        asymptomatic = group_df[['image_id', 'label']][group_df['label']==7].reset_index(drop=True,inplace=False)
        
        symptomatic['set']='train'
        symptomatic['set'].loc[len(symptomatic)*0.8-0.000001:len(symptomatic)*0.9-0.000001]='val'
        symptomatic['set'].loc[len(symptomatic)*0.9-0.000001:] = 'test'
        ds = pd.concat((ds, symptomatic), ignore_index=True)

        asymptomatic['set']='train'
        asymptomatic['set'].loc[len(asymptomatic)*0.8-0.000001:len(asymptomatic)*0.9-0.000001]='val'
        asymptomatic['set'].loc[len(asymptomatic)*0.9-0.000001:] = 'test'
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


def image_example(image_id, image_string, label):
    image_shape = tf.io.decode_jpeg(image_string).shape

    feature = {
        'id': _int64_feature(image_id),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def tfrecord_write(df, img_path, tfrc_path):
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
                            label_id = file_list['label'][j]
                            image_id = file_list['image_id'][j]
                            
                            image_file_path = os.path.join(img_path, symp, folder_name, FOLDER_BBOX, image_name)
                            image_string = open(image_file_path, 'rb').read()    
                            
                            tf_example = image_example(image_id, image_string, label_id)
                            writer.write(tf_example.SerializeToString())
                        sleep(1.)

            logout(f"end to make {num_file} {symp} {set} data tfrecord...class: {label}")
        logout(f"end to make tfrecord...class: {label}")

def tfrecord_write_group(df, img_path, tfrc_path):
    set_list = list(df['set'].unique())
        
    # train / validation / test dataset 분리
    for set in set_list:
        #logout(f"start to make {set} data tfrecord...class: {folder}")
        set_df = df[df['set']==set]

        num_file = math.ceil(len(set_df) / NUM_IMAGE)

        for i in range(num_file):
            start_index = i*NUM_IMAGE
            end_index = (i+1)*NUM_IMAGE
            file_list = set_df[start_index:end_index]
            file_list.reset_index(drop=True, inplace=True)
            
            tfr_path = os.path.join(tfrc_path, set)
            if not os.path.exists(tfr_path):
                os.makedirs(tfr_path)
            
            record_file = f"{set}_{str(i+1)}_{LABEL_NAME}"+EXTENSION_TFRECORDS
            logout(f"process {i+1}/{num_file}th tfrecord for {NUM_IMAGE} images [{start_index}:{end_index}]")

            with tf.io.TFRecordWriter(os.path.join(tfr_path, record_file)) as writer:
                for j in range(len(file_list)):
                    #logout(f"class {label} {symp} {set} data set...{i+1}/{num_file} tfrecord processing...({j+1}/{NUM_IMAGE})")
                    image_name = file_list['image_name'][j]
                    label_id = file_list['label'][j]
                    image_id = file_list['image_id'][j]
                    if label_id == 7:
                        symp = FOLDER_ASYMPTOMATIC
                    else:
                        symp = FOLDER_SYMPTOMATIC
                    folder_name = file_list['folder_name'][j]
                    
                    image_file_path = os.path.join(img_path, symp, folder_name, FOLDER_BBOX, image_name)
                    image_string = open(image_file_path, 'rb').read()    
                    
                    tf_example = image_example(image_id, image_string, label_id)
                    writer.write(tf_example.SerializeToString())
                sleep(1.)

        logout(f"end to make {num_file} {set} data tfrecord")
    logout(f"end to make tfrecord")


if __name__ == '__main__':
    logout("Start progress...")

    # output path for make_tfrecord.py
    if not os.path.exists(CONST_TFRC_PATH):
        os.makedirs(CONST_TFRC_PATH)
        logout(f"output folder {CONST_TFRC_PATH} created")
    logout(f"Created TFRecord files will be stored in {CONST_TFRC_PATH}")

    logout(f"start to make {FOLDER_SYMPTOMATIC} image list")
    meta_symptomatic = make_df(os.path.join(CONST_IMG_PATH, FOLDER_SYMPTOMATIC))
    meta_symptomatic.to_csv(os.path.join(CONST_TFRC_PATH, FOLDER_SYMPTOMATIC+'_'+CONST_CSV_TFRCLIST), index=False, encoding='utf-8-sig')
    logout(f"finish to make {FOLDER_SYMPTOMATIC} image list")

    logout(f"start to make {FOLDER_ASYMPTOMATIC} image list")
    meta_asymptomatic = make_df(os.path.join(CONST_IMG_PATH, FOLDER_ASYMPTOMATIC))
    meta_asymptomatic.to_csv(os.path.join(CONST_TFRC_PATH, FOLDER_ASYMPTOMATIC+'_'+CONST_CSV_TFRCLIST), index=False, encoding='utf-8-sig')
    logout(f"finish to make {FOLDER_ASYMPTOMATIC} image list")

    logout("start organizing list of TFRecord files")
    meta_symptomatic['label'] = 0
    folder_list = meta_symptomatic.sort_values('folder_name')['folder_name'].unique()
    for i in range(len(folder_list)):
        if 'A1' in folder_list[i]:
           meta_symptomatic['label'][meta_symptomatic['folder_name']==meta_symptomatic.sort_values('folder_name')['folder_name'].unique()[i]] = 1

        elif 'A2' in folder_list[i]:
           meta_symptomatic['label'][meta_symptomatic['folder_name']==meta_symptomatic.sort_values('folder_name')['folder_name'].unique()[i]] = 2
        
        elif 'A3' in folder_list[i]:
           meta_symptomatic['label'][meta_symptomatic['folder_name']==meta_symptomatic.sort_values('folder_name')['folder_name'].unique()[i]] = 3
        
        elif 'A4' in folder_list[i]:
           meta_symptomatic['label'][meta_symptomatic['folder_name']==meta_symptomatic.sort_values('folder_name')['folder_name'].unique()[i]] = 4
        
        elif 'A5' in folder_list[i]:
           meta_symptomatic['label'][meta_symptomatic['folder_name']==meta_symptomatic.sort_values('folder_name')['folder_name'].unique()[i]] = 5
        
        elif 'A6' in folder_list[i]:
           meta_symptomatic['label'][meta_symptomatic['folder_name']==meta_symptomatic.sort_values('folder_name')['folder_name'].unique()[i]] = 6
        
    meta_asymptomatic['label'] = 7
    meta = pd.concat((meta_symptomatic, meta_asymptomatic))
    meta.reset_index(drop=True, inplace=True)

    # add image id
    col_names = meta.columns
    new_col = ['image_id']
    new_col = new_col + list(col_names)
    # shuffle
    if SHUFFLE == True:
        meta = meta.sample(frac=1).reset_index(drop=True)
        meta.reset_index(drop=False, inplace=True)
        meta.columns = new_col        
    elif SHUFFLE == False:
        meta.reset_index(drop=False, inplace=True)
        meta.columns = new_col        

    
    path_csv = os.path.join(CONST_TFRC_PATH, CONST_CSV_TFRCLIST)
    meta.to_csv(path_csv, index=False, encoding='utf-8-sig')
    logout(f"list of TFRecord files saved in {path_csv}")

    splited_df = split_data(meta)

    path_csv = os.path.join(CONST_TFRC_PATH, CONST_CSV_SPLITTED)
    splited_df.to_csv(path_csv, index=False, encoding='utf-8-sig')
    logout(f"splitted dataset info saved in {path_csv}")

    logout("writing tfrecrd files being started")
    tfrecord_write_group(splited_df, CONST_IMG_PATH, CONST_TFRC_PATH)

    logout("Done.", force_flush=True)