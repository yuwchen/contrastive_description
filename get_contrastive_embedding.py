import torch
import pickle
import os
import clip
import itertools
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.nn import functional as F
from PIL import Image
from sklearn.metrics import accuracy_score
from collections import Counter
from tqdm import tqdm

def load_pkl(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file

def get_performance(labels, predictions):
    descr_predictions = predictions.argmax(dim=1)
    accuracy = accuracy_score(labels, descr_predictions)
    print(accuracy)
    for topk in range(1,6):
        _, y_pred = predictions.topk(topk, dim=1) 
        top_k_accuracy = (y_pred == labels.view(-1, 1)).sum().item() / len(labels)
        print("Top-",topk," Accuracy:", top_k_accuracy)

def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    else: raise ValueError("Unknown aggregate_similarity")

def img_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def compute_paired_description_encodings(model, paired_description):

    description_encodings = OrderedDict()
    for k, v in paired_description.items():
        tokens = clip.tokenize(v).to(device)
        description_encodings[k] = F.normalize(model.encode_text(tokens))

    return description_encodings

def transform_des_format(model, contrastive_des):
    encoded_des_all = {}
    confused_class = []
    for k, v in contrastive_des.items():

        idx_pair = [k[0], k[1]]
        idx_pair.sort()
        idx_pair = tuple(idx_pair)
        confused_class.append(idx_pair)
        encoded_des = compute_paired_description_encodings(model, v)
        encoded_des_all[idx_pair] = encoded_des
    
    return encoded_des_all, confused_class

def most_frequent_number(numbers):
    counter = Counter(numbers)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

img_root = '../geode/images'
description_path = './descriptors/gpt4_contrastive_des_top10.pkl'
features_dir = './Results/descriptors_geode_gpt3'

labels = torch.load(os.path.join(features_dir, 'labels_all.pt'), map_location=torch.device('cpu'))
filename_list = torch.load(os.path.join(features_dir,'filename_list.pt'), map_location=torch.device('cpu'))
predictions = torch.load(os.path.join(features_dir,'predictions.pt'), map_location=torch.device('cpu'))
label_to_classname =  torch.load(os.path.join(features_dir,'label_to_classname.pt'), map_location=torch.device('cpu'))


predictions = predictions.to(torch.float32)
filename_list = list(itertools.chain.from_iterable(filename_list))
print("Number of files:", len(filename_list))


print('Performance of the first stage:')
get_performance(labels,  predictions)

clip_model = "ViT-B/32"

if clip_model=="ViT-B/32":
    img_size=224
elif clip_model=="ViT-L/14@336px":
    img_size= 336
elif clip_model=="RN50x4":
    img_size= 288
elif clip_model=="RN50x16":
    img_size= 384
elif clip_model=="RN50x64":
    img_size= 448
else:
    print('unsupport clip model:', clip_model)

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cuda'
model, preprocess = clip.load(clip_model , device=device, jit=False)
model.eval()
model.requires_grad_(False)
print('finish loading model ...with device:', device)

tfms = img_transform(img_size)

print("Load contrastive description")
contrastive_des = load_pkl(description_path)
"""
format of the description:
data:{(o_id1, o_id2):{o_id1:[des1, des2, des3, ...],
                      o_id2:[des1, des2, des3, ...]},
       ...}
==> transform to the description enconding.
data:{(o_id1, o_id2):{o_id1:[clip_encoded_des1, clip_encoded_des2, clip_encoded_des3, ...],
                      o_id2:[clip_encoded_des1, clip_encoded_des2, clip_encoded_des3, ...]},
       ...}
"""

#confused_class =  [(2,26) , (33,38) , (4,16), (3,14), (22,24), (11,16), (24,28), (23,28), (1,14), (1,11)]


print('encode description')
encoded_contrastive_des, confused_class = transform_des_format(model, contrastive_des)
print('two stage predicrtion classes:', confused_class)

print('finish encode description')

new_predictions = []
selected_idx = []

for sample_idx, filepath in tqdm(enumerate(filename_list), total=len(filename_list)):

    images =  Image.open(os.path.join(img_root, filepath))
    images = tfms(images)
    images = images.unsqueeze(0)
    images = images.to(device)    
    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)

    the_prediction = predictions[sample_idx].unsqueeze(0)
    _, y_pred = the_prediction.topk(2, dim=1) 
    y_pred = y_pred.squeeze(0)
    top_1_idx = y_pred[0]
    top_2_idx = y_pred[1]

    idx_pair = [top_1_idx.item(), top_2_idx.item()]
    idx_pair.sort()
    idx_pair = tuple(idx_pair)

    if idx_pair in confused_class:
        the_description_encodings = encoded_contrastive_des[idx_pair]
    else:
        continue
    ## format of the_description_encodings={o_id1:[clip_encoded_des1, clip_encoded_des2, clip_encoded_des3, ...],
    #                                       o_id2:[clip_encoded_des1, clip_encoded_des2, clip_encoded_des3, ...]}
    collected_doc_matrix = {}
    
    for k, v in the_description_encodings.items(): 
        dot_product_matrix = image_encodings @ v.T   
        collected_doc_matrix[k] = dot_product_matrix[0]
    
    new_predictions.append(collected_doc_matrix)
    selected_idx.append(sample_idx)


des_name = os.path.basename(description_path).split('.')[0]
torch.save(new_predictions, os.path.join(features_dir,des_name+'_two_stage_scores.pt'))
torch.save(selected_idx, os.path.join(features_dir,des_name+'_selected_idx.pt'))
