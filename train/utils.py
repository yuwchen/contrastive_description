import os
import pandas as pd
import shutil
from tqdm import tqdm
import torch

from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


def get_object2dirname():
        # construct object2dirname mapping from old geode dataset
    object2dirname = {}
    # iterate directories over '/local2/data/xuanming/geode/images/' to get object2index
    for dirname in os.listdir('/local2/data/xuanming/geode/images/'):
        idx = dirname.split('.')[0]
        obj = dirname.split('.')[1]
        object2dirname[obj] = dirname
    
    return object2dirname

def get_id2label(train_ds):
    return {id:label for id, label in enumerate(train_ds.features['label'].names)}

def get_label2id(id2label):
    return {label:id for id,label in id2label.items()}

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def convert_train_test_from_geode():
### load geode dataset
# convert paths in filename_list_train/test.txt to paths in geode_country

    df = pd.read_csv('/local2/data/xuanming/geode/index.csv', index_col=False)
    class_list = list(df['object'].unique())  # class list in Geo-DE
    root_path = '/local/data/xuanming/datasets_difficult_images/'
    src_root_path = '/local2/data/xuanming/geode_country/'  # root path of Geo-DE

    # create a folder for each class in train and test
    for split in ['train', 'test']:
        for class_ in class_list:
            if not os.path.exists(os.path.join(root_path, split, class_)):
                os.makedirs(os.path.join(root_path, split, class_))

    # get object2dirname mapping
    object2dirname = get_object2dirname()

    # construct training set
    with open('data/filename_list_train.txt', 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            filename = line.strip()
            region = filename.split('/')[0].lower()
            country = filename.split('/')[1]
            img_fname = filename.split('/')[2]
            if len(img_fname.split(country)[1].split('_')) == 3:
                class_ = img_fname.split(country)[1].split('_')[1]
            else:
                class_ = '_'.join(img_fname.split(country)[1].split('_')[1:3])
            # print(class_)
            src_img_path = os.path.join(src_root_path, f'geode_{region.lower()}_{country.lower()}', 'images', object2dirname[class_], img_fname)
            # print(src_img_path)
            dst_img_path = os.path.join(root_path, 'train', class_, img_fname)
            shutil.copy(src_img_path, dst_img_path)
            # os.system('cp /local2/data/xuanming/geode/' + filename + ' ' + new_filename)

    # construct test set
    with open('data/filename_list_test.txt', 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            filename = line.strip()
            region = filename.split('/')[0].lower()
            country = filename.split('/')[1]
            img_fname = filename.split('/')[2]
            if len(img_fname.split(country)[1].split('_')) == 3:
                class_ = img_fname.split(country)[1].split('_')[1]
            elif len(img_fname.split(country)[1].split('_')) == 4:
                class_ = '_'.join(img_fname.split(country)[1].split('_')[1:3])
            else:
                class_ = '_'.join(img_fname.split(country)[1].split('_')[1:4])
            # print(class_)
            src_img_path = os.path.join(src_root_path, f'geode_{region.lower()}_{country.lower()}', 'images', object2dirname[class_], img_fname)
            # print(src_img_path)
            dst_img_path = os.path.join(root_path, 'test', class_, img_fname)
            shutil.copy(src_img_path, dst_img_path)
            # os.system('cp /local2/data/xuanming/geode/' + filename + ' ' + new_filename)


