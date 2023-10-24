#!/usr/bin/env bash

python train.py --dataset_name FluidFlow --path2data ../demo/training_set/PTVflow3D_norm/ --nb_points 2048 \
    --batch_size_train 4 --batch_size_val 10 --nb_epochs 100 --nb_workers 8 \
    --backward_dist_weight 0.0 --use_corr_conf 1 --corr_conf_loss_weight 0.1 \
    --add_model_suff 1 --save_model_epoch 25 --log_dir demo  --nb_train_examples -1 --nb_val_examples 1 \
    --use_smooth_flow 1 --use_div_flow 1 --div_flow_loss_weight 0.01 --div_neighbor 2 --lattice_steps 10


python evaluate.py --dataset_name FluidFlow --path2data ../demo/test_data/mhd1024_norm/ --mode test --nb_points 2048 \
    --path2ckpt ./../experiments/demo/model_e100.tar --backward_dist_weight 0.0\
    --use_test_time_refinement 1 --test_time_num_step 1000 --test_time_update_rate 0.01 --use_smooth_flow 0 --nb_neigh_smooth_flow 32 --smooth_flow_loss_weight 1.0 \
    --use_div_flow 0 --nb_neigh_div_flow 1 --div_flow_loss_weight 0.0001 \
    --log_fname demo.txt   --test_time_verbose 0 --save_metrics 0 