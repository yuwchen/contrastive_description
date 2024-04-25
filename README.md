# Geographically Diverse Image Classification via Visual Descriptive Representations


## Step 0 : Prepare description file

descriptions of origianl model (.json):  
{"object":
[des1, des2, ...],
...}
  
descriptions to get contrastive descriptive representation (.pkl):  
{(obj_id_a, obj_id_b):  
 {obj_id_a:[des_a_1, des_a_2, ...], obj_id_b:[des_b_1, des_b_2]}, ...}

## Step 1 : Get object description scores

Modify the path in load.py
- Change the dataset path to /path/to/your/geode/images
- Change the description path to /path/to/your/obj_description.json
  
Get the object description scores by runing
```
python main_save_results.py
```
- The results will be saved in 'Results/{name of your description file}' directory  
(1) filename_list.pt: path of all images  
(2) labels_all.pt: ground-truth labels of all images  
(3) predictions.pt: prediction scores of all images of all classes  
(4) label_to_classname.pt: mapping between object name and index  

## Step 2 : Get contrastive descrption embedding

Modify the path in get_contrastive_embedding.py
- Change img_root to '/path/to/your/geode/images'
- Change description_path to '/path/to/your/contrastive_descrption.pkl'
- Change features_dir to '/path/to/Results/{name of your description file}' in previous step

Get the contrastive description scores embedding by runing
```
python get_contrastive_embedding.py
```
- The results will be save in 'Results/{name of your description file}' directory  
(1) {name of your description file}+'_two_stage_scores.pt':  contrastive description scores embedding of the selected samples    
(2) {name of your description file}+'_selected_idx.pt': indexs of the selectes samples (respect to the original file list)    
Note: selected sample means the top2 prediction pair is in the contrastive_descrption file.  

## Step 3: second stage prediction

Using random train test split
```
two_stage_prediction_random.py
```
Train test on predefined train test split
```
two_stage_prediction_fix.py
```
