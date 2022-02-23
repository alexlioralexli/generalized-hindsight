#!/bin/bash

#SBATCH --job-name=pointTr
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G


## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.

singularity \
    exec --nv \
    --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    --overlay /home/yb1025/pyenv/overlay-7.5GB-300K.ext3:ro \
    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "
source /ext3/env.sh
export LD_LIBRARY_PATH=/home/yb1025/.mujoco/mujoco200_linux/bin:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/.singularity.d/libs:\$LD_LIBRARY_PATH
python train.py env=cheetah_run
"

