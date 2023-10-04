#!/usr/bin/env bash

# train on kitti_v
python train.py --dataset_name FluidFlow --nb_points 2048 \
    --batch_size_train 4 --batch_size_val 10 --nb_epochs 100 --nb_workers 8 \
    --backward_dist_weight 0.0 --use_corr_conf 1 --corr_conf_loss_weight 0.1 \
    --add_model_suff 1 --save_model_epoch 25 --log_dir GOT_extractor_1300_add_div_test_lattice_2nb  --nb_train_examples 1300 --nb_val_examples 1 \
    --use_smooth_flow 1 --use_div_flow 1 --div_flow_loss_weight 0.01 --div_neighbor 2 --lattice_steps 10

