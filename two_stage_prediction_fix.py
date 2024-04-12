import json
import torch
import pickle
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.metrics import accuracy_score
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier


def select_samples(index, *args):
    selected_values = []
    for arg in args:
        selected_values.append(arg[index])
    return selected_values

def get_contrastive_labels(description_scores):

    contrastive_labels = []
    for sample in description_scores:
        score_mean = {}
        for key, value in sample.items():
            score_mean[key] = torch.mean(value).item()
        the_label = max(score_mean, key=score_mean.get)
        contrastive_labels.append(the_label)

    contrastive_labels = np.asarray(contrastive_labels)
    return contrastive_labels


def get_description_embed(description_scores):

    description_embed = []
    top2_pair = []

    for sample in description_scores:
        the_embed = []
        pair = []
        for key, value in sample.items():
            the_embed.extend(value.tolist())
            pair.append(key)
        pair.sort()
        description_embed.append(the_embed)   
        top2_pair.append(tuple(pair))
    description_embed = np.asarray(description_embed, dtype=object)


    return description_embed, top2_pair

def get_index_of_target_pairs(top2_pair_list):

    index_dict = defaultdict(list)
    for idx, top2_pair in enumerate(top2_pair_list):
        index_dict[tuple(top2_pair)].append(idx)
    return index_dict

def description_prediction(embed_train, labels_train, embed_test):

    clf = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=100)
    
    the_des_embed_train = [line for line in embed_train]
    the_des_embed_train = np.asarray(the_des_embed_train)
    the_des_embed_test = [line for line in embed_test]
    the_des_embed_test = np.asarray(the_des_embed_test)    

    clf.fit(the_des_embed_train, labels_train)
    test_pred = clf.predict(the_des_embed_test)

    return test_pred

def write_list_to_file(input_list, outputpath):
    
    f = open(outputpath,'w')

    for line in input_list:
        f.write(line+'\n')

    f.close()


def get_match_index(filelist, selected_filepath):
    
    matched_idx = []
    f = open(selected_filepath,'r').read().splitlines()
    for i, path in enumerate(filelist):
         if path in f:
            matched_idx.append(i)

    return matched_idx

labels = torch.load('./Results/descriptors_geode_gpt3/labels_all.pt', map_location=torch.device('cpu'))
filename_list = torch.load('./Results/descriptors_geode_gpt3/filename_list.pt', map_location=torch.device('cpu'))
predictions = torch.load('./Results/descriptors_geode_gpt3/predictions.pt', map_location=torch.device('cpu'))
label_to_classname = torch.load('./Results/descriptors_geode_gpt3/label_to_classname.pt', map_location=torch.device('cpu'))

filename_list = list(itertools.chain.from_iterable(filename_list))


# index of samples that top2 is in the top confused pairs
selected_idx = torch.load('./Results/descriptors_geode_gpt3/gpt4_contrastive_des_top10_selected_idx.pt')
#load description scores, the length of the embedding should match the length of selectd index
description_scores = torch.load('./Results/descriptors_geode_gpt3/gpt4_contrastive_des_top10_two_stage_scores.pt')


## first stage prediction scores for all samples.
descr_predictions = predictions.argmax(dim=1)
accuracy = accuracy_score(labels, descr_predictions)
print(accuracy)
for topk in range(1,5):
    cumulative_tensor = predictions.to(torch.float32)
    _, y_pred = cumulative_tensor.topk(topk, dim=1) 
    top_k_accuracy = (y_pred == labels.view(-1, 1)).sum().item() / len(labels)
    print("Top-",topk," Accuracy:", top_k_accuracy)

labels = np.asarray(labels)
predictions = np.asarray(predictions)
filename_list = np.asarray(filename_list)



## now we only focus on the samples that top2 belong to the selected pairs.

selected_label, selected_filename_list, selected_predictions = select_samples(selected_idx, labels, filename_list, descr_predictions)

## calculate the contrastive prediction label and obtain the description embedding.
description_embed, top2_pair = get_description_embed(description_scores)
contrastive_labels = get_contrastive_labels(description_scores)

acc_ori_all = 0
acc_con_all = 0
acc_des_all = 0
total_samples = 0

#loop over all pair for two stage training.
for pair in list(set(top2_pair)):
    
    trainfile = './train_test_split/{}-{}-filename_list_train.txt'.format(label_to_classname[pair[0]],label_to_classname[pair[1]])
    testfile = './train_test_split/{}-{}-filename_list_test.txt'.format(label_to_classname[pair[0]],label_to_classname[pair[1]])
    train_idx = get_match_index(selected_filename_list, trainfile)
    test_idx = get_match_index(selected_filename_list, testfile)

    the_embed_train = description_embed[train_idx] 
    the_embed_test = description_embed[test_idx]

    the_labels_train = selected_label[train_idx]   
    the_labels_test = selected_label[test_idx]    

    pred = description_prediction(the_embed_train, the_labels_train, the_embed_test)

    num_of_sample = len(test_idx)
    print(pair[0], label_to_classname[pair[0]], pair[1], label_to_classname[pair[1]], 'Num of sample:', num_of_sample)
    acc_ori = accuracy_score(the_labels_test, selected_predictions[test_idx])
    acc_con = accuracy_score(the_labels_test, contrastive_labels[test_idx])
    acc_des = accuracy_score(the_labels_test, pred)

    # print('One stage:', round(acc_ori, 4))
    # print('Gpt4 - contrastive:',round(acc_contrastive_gpt4, 4))
    # print('Gpt4 - description prediction:',round(acc_des_gpt4, 4))
    # print('Gpt35 - contrastive:', round(acc_contrastive_gpt35, 4))
    # print('Gpt35 - description prediction:', round(acc_des_gpt35, 4))

    print(round(acc_ori, 4))
    print(round(acc_con, 4))
    print(round(acc_des, 4))
 
    acc_ori_all+=acc_ori*num_of_sample
    acc_con_all+=acc_con*num_of_sample
    acc_des_all+=acc_des*num_of_sample
    total_samples+=num_of_sample
    
    

print()
print('All testing data score')
print('Number of testing data:', total_samples)
print('one stage prediction:', round(acc_ori_all/total_samples,4))
print('two stage contrastive prediction:', round(acc_con_all/total_samples,4))
print('two stage training prediction:', round(acc_des_all/total_samples,4))
