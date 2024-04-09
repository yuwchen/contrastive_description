import os
import torch
import pickle
import load
import pandas as pd
from PIL import Image
from torchvision import datasets
import torchvision.transforms as transforms

class GEODEDataset(datasets.ImageFolder):
    """
    Wrapper for the Geo-DE dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    Dataset per https://geodiverse-data-collection.cs.princeton.edu/
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):

        self.img_root = os.path.join(root, 'images')
        meta_path = os.path.join(root, 'index.csv')
        meta_data = pd.read_csv(meta_path, index_col=False)
        #print('columns:', meta_data.columns) # 'object'
        self.labels = {}
        self.samples = meta_data['file_path'].to_numpy()
        
        for index, row in meta_data.iterrows():
            self.labels[row['file_path']] = row['object']
            

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        

    def __getitem__(self, index):
 

        filepath = self.samples[index]
        label = self.labels[filepath]
        sample =  Image.open(os.path.join(self.img_root, filepath))
        sample = self.transform_(sample)
 
        return sample, label, filepath
    

def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
