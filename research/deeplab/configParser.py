# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:44:17 2019

@author: wayne.kuo
"""

import configparser
config = configparser.ConfigParser(allow_no_value=True)    # 注意大小寫

path = 'C:/tensorflow/models/research/deeplab/config.ini'
#config.read(path)   # 配置檔案的路徑
#


config.sections()



#%%
#train
config.add_section('deeplab')
config.set('deeplab', 'train_logdir', './deeplab/datasets/pascal_voc_seg/ckpt')
config.set('deeplab', 'dataset_dir', './deeplab/datasets/pascal_voc_seg/tfrecord')
config.set('deeplab', 'dataset', 'wood')
config.set('deeplab', 'model_variant', 'nas_hnasnet')
config.set('deeplab', 'training_number_of_steps', '100000')
config.set('deeplab', 'tf_initial_checkpoint', './deeplab/datasets/pascal_voc_seg/ckpt2/model.ckpt-100000')
config.set('deeplab', 'atrous_rates', '6,12,18')
config.set('deeplab', 'output_stride', '16')
config.set('deeplab', 'decoder_output_stride', '4')
config.set('deeplab', 'train_crop_size', '513,513')
config.set('deeplab', 'train_batch_size', '4')
config.set('deeplab', 'fine_tune_batch_norm', 'False')
config.set('deeplab', 'drop_path_keep_prob', '1.0')
config.set('deeplab', 'hard_example_mining_step', '20000')
config.set('deeplab', 'top_k_percent_pixels', '0.25')
config.set('deeplab', 'save_summaries_images', 'False')
config.set('deeplab', 'train_splits', '68')
config.set('deeplab', 'profile_logdir', ) #'./deeplab/datasets/pascal_voc_seg/profile'

#vis
config.set('deeplab', 'vis_logdir', './deeplab/datasets/pascal_voc_seg/vis')
config.set('deeplab', 'vis_split', 'train')
config.set('deeplab', 'predict_crop_size', '1285,1285')
config.set('deeplab', 'colormap_type', 'wood')
config.set('deeplab', 'also_save_raw_predictions', 'False')

#eval
config.set('deeplab', 'eval_logdir', './deeplab/datasets/pascal_voc_seg/eval')
config.set('deeplab', 'eval_split', 'train')

#export
config.set('deeplab', 'checkpoint_dir', './deeplab/datasets/pascal_voc_seg/ckpt/model.ckpt-100000')
config.set('deeplab', 'export_path', './deeplab/datasets/pascal_voc_seg/frozen_inference_graph.pb')
config.set('deeplab', 'inference_scales', '1.0')
config.set('deeplab', 'num_classes', '6')


config.write(open(path, 'w'))    # 一定要寫入才生效
