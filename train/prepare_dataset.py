import pandas as pd
import os
import shutil
from tqdm import tqdm

### load geode dataset
# convert paths in filename_list_train/test.txt to paths in geode_country

df = pd.read_csv('/local2/data/xuanming/geode/index.csv', index_col=False)
class_list = list(df['object'].unique())  # class list in Geo-DE
root_path = '/local/data/xuanming/datasets_geode_all_images/'
src_root_path = '/local2/data/xuanming/geode_country/'  # root path of Geo-DE

# create a folder for each class in train and test
for split in ['train', 'test']:
    for class_ in class_list:
        if not os.path.exists(os.path.join(root_path, split, class_)):
            os.makedirs(os.path.join(root_path, split, class_))

# construct object2dirname mapping from old geode dataset
object2dirname = {}
# iterate directories over '/local2/data/xuanming/geode/images/' to get object2index
for dirname in os.listdir('/local2/data/xuanming/geode/images/'):
    idx = dirname.split('.')[0]
    obj = dirname.split('.')[1]
    object2dirname[obj] = dirname
    # object2index[os.path.basename(dir)] = len(object2index)

with open('data/all/all_file_train.txt', 'r') as f:
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
        dst_img_path = os.path.join(root_path, 'train', class_, img_fname)
        shutil.copy(src_img_path, dst_img_path)
        # os.system('cp /local2/data/xuanming/geode/' + filename + ' ' + new_filename)


with open('data/all/all_file_test.txt', 'r') as f:
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