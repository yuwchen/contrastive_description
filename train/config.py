
hparams = {}

# path hyperparameters
hparams['dataset_root_path'] = '/local/data/xuanming/'
hparams['data_dir'] = 'datasets_geode_all_images'
hparams['output_dir'] = '/local/data/xuanming/vit_base_32_in21k_all'
hparams['save_best_model_path'] = hparams['output_dir'] + '/best_model_0.8951612903225806'
# hparams['save_best_model_path'] = '/local/data/xuanming/vit_base_16_all/best_model_0.8838709677419355'

# /local/data/xuanming/vit_base_32_in21k_all/best_model_0.8951612903225806

# gpu hyperparameters
hparams['use_cpu'] = False
hparams['device'] = 'cuda:4'

# model hyperparameters
hparams['model_name'] = "google/vit-base-patch32-224-in21k"

# training hyperparameters
hparams['metric_name'] = "accuracy"
hparams['per_device_train_batch_size'] = 8
hparams['per_device_eval_batch_size'] = 8
hparams['num_train_epochs'] = 10
hparams['learning_rate'] = 2e-4
hparams['weight_decay'] = 0.01
hparams['logging_steps'] = 50
hparams['run_name'] = "vit-base-patch32-in21k-geodeall-10epochs-2e-4lr-8bsz"

# other
hparams['do_train'] = False