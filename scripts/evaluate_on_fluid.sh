#!/usr/bin/env bash

# evaluate on FluidFlow with all the points
python evaluate.py --dataset_name FluidFlow --mode test --nb_points 2048 \
    --path2ckpt ./../experiments/GOT_extractor_130_add_div_test_lattice_2nb/model_e300.tar --backward_dist_weight 0.0\
    --use_test_time_refinement 1 --test_time_num_step 1000 --test_time_update_rate 0.01 --use_smooth_flow 0 --nb_neigh_smooth_flow 32 --smooth_flow_loss_weight 1.0 \
    --use_div_flow 0 --nb_neigh_div_flow 1 --div_flow_loss_weight 0.0001 \
    --log_fname norm.txt   --test_time_verbose 0 --save_metrics 0 --metrics_fname metrics_results_beltrami.npz

