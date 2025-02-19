# Overview
Code for NeurIPS 2024 paper: **Dual-frame Fluid Motion Estimation with Test-time Optimization and Zero-divergence Loss**

# System Requirements
- operating system: tested on Ubuntu 22.04
- software dependencies:
  - Python 3.8.10
  - CUDA 11.3
  - cudNN 8.3.0
- Hardware: tested on a single NVIDIA RTX4070ti, with RAM 32G
- Python dependencies:
  - PyTorch 1.12.0
  - tensorboard 2.4.1
  - tqdm 4.63.1
  - scipy
  - imageio
  - torch-scatter 2.1.1
  - ChamferDistancePytorch

# Installation Guide
The code has been tested with Python 3.8.10, PyTorch 1.12.0, CUDA 11.3, and cuDNN 8.3.0 on Ubuntu 22.04.

Clone this repository:

```
  git clone https://github.com/Forrest-110/DECROB.git 
  cd DECROB/
```

Install [Pytorch](https://pytorch.org/) ,[KNN_cuda](https://github.com/unlimblue/KNN_CUDA), and other required packages:

```
  pip install tensorboard==2.4.1 --no-cache-dir
  pip install tqdm==4.63.1 --no-cache-dir
  pip install scipy
  pip install imageio
  pip install torch-scatter==2.1.1 -f https://pytorch-geometric.com/whl/torch-1.12.0+cu113.html
  pip install Ninja
```


Compile the Chamfer Distance op, implemented by [Groueix _et al._](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) The op is located under `auxiliary/ChamferDistancePytorch/chamfer3D` folder. The following compilation script uses a CUDA 11.3 path. If needed, modify script to point to your CUDA path. Then, use:
 ```bash
sh compile_chamfer_distance_op.sh
```

The compilation results should be created under `auxiliary/ChamferDistancePytorch/chamfer3D/build` folder.

The typical install time on a normal desktop computer should be no more than half an hour.

# Demo

## Instructions to run on data
We provide a small train and test dataset at the folder 'demo'. To run on demo data, follow the instructions below.
```
cd scripts
sh demo.sh
```

The demo will train and evaluate on the demo data.

## Expected output
To provide reproduction ability, we set every random seed to 42. So the following result should be presented if your follow the instructions above.

```
EPE: 1.3995, NEPE: 12.8406, ACC3DS: 0.0039, ACC3DR: 0.0040, Outlier: 0.9961
```

## Expected run time for demo
About 3 minutes on a normal desktop computer.

# Required Data
To evaluate/train DECROB, you will need to download the required datasets [FluidFlow3D-family](https://github.com/JiamingSkGrey/FluidFlow3D-family). We also generate a new dataset FluidFlow3D-cases using FluidFlow3D-norm in FluidFlow3D-family, and we provide it [here](https://drive.google.com/file/d/1JWGYtn9fADccVere9oC_UnueaFEG-t_j/view?usp=drive_link).

It would be best if you arrange your data like this:

```
.
├── FluidFlow3D-cases
│   ├── beltrami
│   ├── channel
│   ├── isotropic
│   ├── mhd
│   ├── transition
│   └── uniform
├── FluidFlow3D-noise
│   ├── 000
│   ├── 002
│   ├── 004
│   ├── 006
│   ├── 008
│   └── 010
├── FluidFlow3D-norm
└── FluidFlow3D-ratio
    ├── 080
    ├── 090
    ├── 100
    ├── 110
    ├── 120
    └── 130
```

# Train & Evaluate

The train and evaluate scripts are located under `scripts/` folder. The scripts are named as `train_*.sh` and `eval_*.sh` respectively. Or you can modify the scripts to train and evaluate your own model.

```
# train DECROB

python train.py --dataset_name FluidFlow --nb_points 2048 \
    --batch_size_train 4 --batch_size_val 10 --nb_epochs 100 --nb_workers 8 \
    --backward_dist_weight 0.0 --use_corr_conf 1 --corr_conf_loss_weight 0.1 \
    --add_model_suff 1 --save_model_epoch 25 --log_dir \your\log\dir  --nb_train_examples 1300 --nb_val_examples -1 \
    --use_smooth_flow 1 --use_div_flow 1 --div_flow_loss_weight 0.01 --div_neighbor 2 --lattice_steps 10

# test DECROB
python evaluate.py --dataset_name FluidFlow --mode test --nb_points 2048 \
    --path2ckpt /your/ckpt/path --backward_dist_weight 0.0\
    --use_test_time_refinement 1 --test_time_num_step 1000 --test_time_update_rate 0.01 --use_smooth_flow 0 --nb_neigh_smooth_flow 32 --smooth_flow_loss_weight 1.0 \
    --use_div_flow 0 --nb_neigh_div_flow 1 --div_flow_loss_weight 0.0001 \
    --log_fname norm.txt   --test_time_verbose 0 --save_metrics 0 --metrics_fname metrics_results_beltrami.npz



# all the parameters are interpreted in the code

```

# Instruction For Use
We provide checkpoints which are trained with 13621 samples, 1300 samples and 130 samples. All of them are put under 'ckpt' direction. To use our code on your own data or reproduce our results, you can **train or evaluate** on your custom data, following the instructions above. Or you can simply use our trained checkpoint for evaluation.

## How to run the software on your data
Custom dataset should follow our dataset structure or you could write a dataloader yourself. Follow the examples at 'datasets/fluidflow.py'.

## Reproduction instructions
The following script reproduces our main result.

```
python evaluate.py --dataset_name FluidFlow --path2data /path/to/FluidFlow3D-family/FluidFlow3D-norm --mode test --nb_points 2048 \
    --path2ckpt ../ckpt/model_all.tar --backward_dist_weight 0.0\
    --use_test_time_refinement 1 --test_time_num_step 1000 --test_time_update_rate 0.01 --use_smooth_flow 0 --nb_neigh_smooth_flow 32 --smooth_flow_loss_weight 1.0 \
    --use_div_flow 0 --nb_neigh_div_flow 1 --div_flow_loss_weight 0.0001 \
    --log_fname modelall.txt   --test_time_verbose 0 --save_metrics 0 --save_pc_res 0 

python evaluate.py --dataset_name FluidFlow --path2data /path/to/FluidFlow3D-family/FluidFlow3D-norm --mode test --nb_points 2048 \
    --path2ckpt ../ckpt/model_1300.tar --backward_dist_weight 0.0\
    --use_test_time_refinement 1 --test_time_num_step 1000 --test_time_update_rate 0.01 --use_smooth_flow 0 --nb_neigh_smooth_flow 32 --smooth_flow_loss_weight 1.0 \
    --use_div_flow 0 --nb_neigh_div_flow 1 --div_flow_loss_weight 0.0001 \
    --log_fname modelall.txt   --test_time_verbose 0 --save_metrics 0 --save_pc_res 0 

python evaluate.py --dataset_name FluidFlow --path2data /path/to/FluidFlow3D-family/FluidFlow3D-norm --mode test --nb_points 2048 \
    --path2ckpt ../ckpt/model_130.tar --backward_dist_weight 0.0\
    --use_test_time_refinement 1 --test_time_num_step 1000 --test_time_update_rate 0.01 --use_smooth_flow 0 --nb_neigh_smooth_flow 32 --smooth_flow_loss_weight 1.0 \
    --use_div_flow 0 --nb_neigh_div_flow 1 --div_flow_loss_weight 0.0001 \
    --log_fname modelall.txt   --test_time_verbose 0 --save_metrics 0 --save_pc_res 0 
```

The Output should be
```
EPE: 0.0046, NEPE: 0.0188, ACC3DS: 0.9869, ACC3DR: 0.9877, Outlier: 0.0131

EPE: 0.0064, NEPE: 0.0251, ACC3DS: 0.9827, ACC3DR: 0.9837, Outlier: 0.0174

EPE: 0.0085, NEPE: 0.0317, ACC3DS: 0.9761, ACC3DR: 0.9776, Outlier: 0.0240
```
