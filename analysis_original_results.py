import json
import torch
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.metrics import accuracy_score
from collections import defaultdict


def load_pkl(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file


labels = torch.load('./Results/labels_all.pt', map_location=torch.device('cpu'))
filename_list = torch.load('./Results/filename_list.pt', map_location=torch.device('cpu'))
predictions = torch.load('./Results/predictions.pt', map_location=torch.device('cpu'))
label_to_classname = torch.load('./Results/label_to_classname.pt', map_location=torch.device('cpu'))

print(Counter(labels.numpy()))

filename_list = list(itertools.chain.from_iterable(filename_list))

print(labels.shape)
print(predictions.shape)
print(len(filename_list))

descr_predictions = predictions.argmax(dim=1)

accuracy = accuracy_score(labels, descr_predictions)
print(accuracy)
for topk in range(1,5):
    cumulative_tensor = predictions.to(torch.float32)
    
    _, y_pred = cumulative_tensor.topk(topk, dim=1) 
    print("###")
    top_k_accuracy = (y_pred == labels.view(-1, 1)).sum().item() / len(labels)
    print("Top-",topk," Accuracy:", top_k_accuracy)

#cm = confusion_matrix(descr_predictions, labels, normalize='pred')
cm = confusion_matrix(descr_predictions, labels)

confused_classes = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:
            i_label = label_to_classname[i]
            j_label = label_to_classname[j]
            confused_classes.append((i_label, j_label, cm[i, j]))

#print(confused_classes)
confused_classes = sorted(confused_classes, key=lambda x: x[-1])
#print("#######")
print(confused_classes)


plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(40)
plt.xticks(tick_marks, label_to_classname, rotation=90)
plt.yticks(tick_marks, label_to_classname)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


selected_class_idx = [1,2, 3,4,11,14,16,22,23,24,26,28,33,38]
label_to_classname =np.asarray(label_to_classname)
selected_classes = cm[selected_class_idx][:, selected_class_idx]

plt.imshow(selected_classes, interpolation='nearest', cmap=plt.cm.Blues)
tick_marks = np.arange(len(selected_class_idx))
plt.xticks(tick_marks, label_to_classname[selected_class_idx], rotation=90)
plt.yticks(tick_marks, label_to_classname[selected_class_idx])
plt.show()


merged_tuples = defaultdict(int)

# Merge tuples with the same first two elements (sorted)
for tup in confused_classes:
    key = tuple(sorted(tup[:2]))
    merged_tuples[key] += tup[2]

# Convert the dictionary back to a list of tuples
result = [(key[0], key[1], value) for key, value in merged_tuples.items()]

#print(result)

confused_classes = {}
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:
            i_label = label_to_classname[i]
            j_label = label_to_classname[j]
            idx = [i_label, j_label]
            idx.sort()
            idx = tuple(idx)
            try:
                confused_classes[idx] += cm[i, j]
            except:
                confused_classes[idx] = cm[i, j]


top_keys = sorted(confused_classes, key=lambda x: confused_classes[x], reverse=True)[:10]
class_to_label = {}
for idx in range(len(label_to_classname)):
    class_to_label[label_to_classname[idx]] = idx

for key in top_keys:
    print(key, class_to_label[key[0]], class_to_label[key[1]],':::', confused_classes[key])
