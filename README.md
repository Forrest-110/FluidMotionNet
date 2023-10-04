# DECROB
Code for Test-time Self Supervision for "*Test-time Self Supervision for **D**ata **E**fficient and **C**ross-domain **Rob**ust Particle Tracking in Turbulent Flow*"

# Installation
The code has been tested with Python 3.8.10, PyTorch 1.12.0, CUDA 11.3, and cuDNN 8.3.0 on Ubuntu 22.04.

Clone this repository:

```
  git clone https://github.com/Forrest-110/DECROB.git 
  cd DECROB/
```

Install [Pytorch](https://pytorch.org/) and other required packages:

```
  pip install tensorboard==2.4.1 --no-cache-dir
  pip install tqdm==4.63.1 --no-cache-dir
  pip install scipy
  pip install imageio
  pip install torch-scatter==2.1.1 -f https://pytorch-geometric.com/whl/torch-1.12.0+cu113.html
```


Compile the Chamfer Distance op, implemented by [Groueix _et al._](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) The op is located under `auxiliary/ChamferDistancePytorch/chamfer3D` folder. The following compilation script uses a CUDA 10.1 path. If needed, modify script to point to your CUDA path. Then, use:
 ```bash
sh compile_chamfer_distance_op.sh
```

The compilation results should be created under `auxiliary/ChamferDistancePytorch/chamfer3D/build` folder.


# Required Data
To evaluate/train DEPT, you will need to download the required datasets [FluidFlow3D-family](https://github.com/JiamingSkGrey/FluidFlow3D-family). We also generate a new dataset FluidFlow3D-cases using FluidFlow3D-norm in FluidFlow3D-family, and we provide it [here](https://drive.google.com/file/d/1JWGYtn9fADccVere9oC_UnueaFEG-t_j/view?usp=drive_link).

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
