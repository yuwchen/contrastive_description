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

    top2_pair = np.asarray(top2_pair)

    return description_embed, top2_pair

def get_index_of_target_pairs(top2_pair_list):

    index_dict = defaultdict(list)
    for idx, top2_pair in enumerate(top2_pair_list):
        index_dict[tuple(top2_pair)].append(idx)
    return index_dict
    
labels = torch.load('./Results/gpt3_labels_all.pt', map_location=torch.device('cpu'))
filename_list = torch.load('./Results/gpt3_filename_list.pt', map_location=torch.device('cpu'))
predictions = torch.load('./Results/gpt3_predictions.pt', map_location=torch.device('cpu'))
label_to_classname = torch.load('./Results/gpt3_label_to_classname.pt', map_location=torch.device('cpu'))

filename_list = list(itertools.chain.from_iterable(filename_list))


# index of samples that top2 is in the top confused pairs
selected_idx = torch.load('/Users/yuwen/Desktop/CV/code/Results/ori_gpt3_gpt4_selected_idx.pt')
#the length of the embedding should match the length of selectd index
description_scores_gpt4 = torch.load('/Users/yuwen/Desktop/CV/code/Results/ori_gpt3_gpt4_two_stage_scores.pt')
description_scores_gpt35 = torch.load('/Users/yuwen/Desktop/CV/code/Results/ori_gpt3_gpt35_two_stage_scores.pt')


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
print('length should be the same:', len(selected_idx), len(selected_label), len(selected_filename_list), len(selected_predictions), len(description_scores_gpt4), len(description_scores_gpt35))


## calculate the contrastive prediction label and formulate  description embedding.
contrastive_labels_gpt4 = get_contrastive_labels(description_scores_gpt4)
description_embed_gpt4, top2_pair = get_description_embed(description_scores_gpt4)

contrastive_labels_gpt35 = get_contrastive_labels(description_scores_gpt35)
description_embed_gpt35, top2_pair = get_description_embed(description_scores_gpt35)

## apply train test split to the samples
random_idx = np.arange(len(selected_label))
np.random.shuffle(random_idx)

num_of_train = int(len(random_idx)*0.10)
train_idx = random_idx[:num_of_train]
test_idx = random_idx[num_of_train:]

labels_train, filename_list_train, predictions_train, gpt4_con_train, gpt4_embed_train, top2_pair_train, gpt35_con_train, gpt35_embed_train = select_samples(train_idx, selected_label, selected_filename_list, selected_predictions, contrastive_labels_gpt4, description_embed_gpt4, top2_pair, contrastive_labels_gpt35, description_embed_gpt35)
labels_test, filename_list_test, predictions_test, gpt4_con_test, gpt4_embed_test, top2_pair_test, gpt35_con_test, gpt35_embed_test = select_samples(test_idx, selected_label, selected_filename_list, selected_predictions, contrastive_labels_gpt4, description_embed_gpt4, top2_pair, contrastive_labels_gpt35, description_embed_gpt35)

#target_pairs
index_dict_train = get_index_of_target_pairs(top2_pair_train)
index_dict_test = get_index_of_target_pairs(top2_pair_test)

def description_prediction(embed_train, labels_train, embed_test):

    clf = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=100)
    
    the_des_embed_train = [line for line in embed_train]
    the_des_embed_train = np.asarray(the_des_embed_train)
    the_des_embed_test = [line for line in embed_test]
    the_des_embed_test = np.asarray(the_des_embed_test)    

    clf.fit(the_des_embed_train, labels_train)
    test_pred = clf.predict(the_des_embed_test)

    return test_pred
    
#loop over all pair for two stage training.
acc_all_gpt4 = 0
acc_all_gpt35 = 0

for pair in list(index_dict_train.keys()):

    the_embed_train_gpt4 = gpt4_embed_train[index_dict_train[pair]] 
    the_embed_test_gpt4 = gpt4_embed_test[index_dict_test[pair]]

    the_embed_train_gpt35 = gpt35_embed_train[index_dict_train[pair]] 
    the_embed_test_gpt35 = gpt35_embed_test[index_dict_test[pair]]

    the_labels_train = labels_train[index_dict_train[pair]]   
    the_labels_test = labels_test[index_dict_test[pair]]    

    pred_gpt4 = description_prediction(the_embed_train_gpt4, the_labels_train, the_embed_test_gpt4)
    pred_gpt35 = description_prediction(the_embed_train_gpt35, the_labels_train, the_embed_test_gpt35)


    num_of_sample = len(index_dict_test[pair])
    print("")
    print(pair[0], label_to_classname[pair[0]], pair[1], label_to_classname[pair[1]], 'Num of sample:', num_of_sample)
    acc_ori = accuracy_score(the_labels_test, predictions_test[index_dict_test[pair]])
    acc_contrastive_gpt4 = accuracy_score(the_labels_test, gpt4_con_test[index_dict_test[pair]])
    acc_des_gpt4 = accuracy_score(the_labels_test, pred_gpt4)

    acc_contrastive_gpt35 = accuracy_score(the_labels_test, gpt35_con_test[index_dict_test[pair]])
    acc_des_gpt35 = accuracy_score(the_labels_test, pred_gpt35)

    # print('One stage:', round(acc_ori,4))
    # print('Contrastive:', round(acc_contrastive,4))
    # print('Description training:', round(acc_des,4))
    print('One stage:', round(acc_ori, 4))
    print('Gpt4 - contrastive:',round(acc_contrastive_gpt4, 4))
    print('Gpt4 - description prediction:',round(acc_des_gpt4, 4))
    print('Gpt35 - contrastive:', round(acc_contrastive_gpt35, 4))
    print('Gpt35 - description prediction:', round(acc_des_gpt35, 4))
    acc_all_gpt4+=acc_des_gpt4*num_of_sample
    acc_all_gpt35+=acc_des_gpt35*num_of_sample

    

##calculate the upper bound.
#labels_test top2_pair_test predictions_test
upper_bound_acc = []
for i, gt_label in enumerate(labels_test):
    if gt_label in top2_pair_test[i]:
        upper_bound_acc.append(gt_label)
    else:
        upper_bound_acc.append(predictions_test[i])
    

acc_ori_test = accuracy_score(labels_test, predictions_test)
acc_con_test_gpt4 = accuracy_score(labels_test, gpt4_con_test)
acc_uppr_test = accuracy_score(labels_test, upper_bound_acc)
acc_con_test_gpt35 = accuracy_score(labels_test, gpt35_con_test)


print()
print('All testing data score')
print('Number of training data:', len(labels_train))
print('Number of testing data:', len(labels_test))
print('one stage prediction:', round(acc_ori_test,4))
print('GPT4 two stage contrastive prediction:', round(acc_con_test_gpt4,4))
print('GPT4 two stage training prediction:', round(acc_all_gpt4/len(labels_test),4))
print('GPT35 two stage contrastive prediction:', round(acc_con_test_gpt35,4))
print('GPT35 two stage training prediction:', round(acc_all_gpt35/len(labels_test),4))

print('upper bound of two stage prediction:', round(acc_uppr_test,4))
