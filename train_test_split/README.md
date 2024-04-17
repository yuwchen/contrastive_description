


- all_file.txt (61940 images): all files in the geode dataset

- difficult_images.txt (9739 images): among 61940 images, the top two predicion of 9739 samples are in the selected confused pairs. These images are difficult images because the first stage model often confused about the top 2 choices. The prediction of these samples will be reconsidered with the second stage process.

  The selected confused pairs are:  
  ('dustbin', 'waste_container') ,  
  ('stall', 'storefront') ,  
  ('toothpaste_toothpowder', 'medicine'),  
  ('toothbrush', 'cleaning_equipment'),  
  ('tree', 'backyard'),   
  ('spices', 'medicine'),  
  ('backyard', 'fence'),  
  ('house', 'fence'),  
  ('hand_soap', 'cleaning_equipment'),  
  ('hand_soap', 'spices')  


The difficult_images are split into the train and test set. 10% of data are used for training, whereas 90% of data are used for testing.   
- difficult_file_train.txt (973 images): 10% of samples in files_for_second_stage.txt that used for the training    
- difficult_file_test.txt (8766 images): 90% of samples in files_for_second_stage.txt that used for the testing    


Train/Test for each confused pairs:  

e.g. samples that top2 prediction of first stage model are dustbin and waste container:  
dustbin-waste_container-filename_list_train.txt  
dustbin-waste_container-filename_list_test.txt  
...
...
