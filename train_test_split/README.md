
- all_file.txt (61940 images): all files in the geode dataset
  
- files_for_second_stage.txt (9739 images): among 61940 images, the top two predicion of 9739 samples are in the selected confused pairs. The prediction of these samples will be reconsidered with the second stage process.

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

- filename_list_train.txt (973 images): 10% of samples in files_for_second_stage.txt are used for the training  
- filename_list_test.txt (8766 images): 90% of samples in files_for_second_stage.txt are used for the testing

