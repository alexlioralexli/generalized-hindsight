#!/bin/bash
#SBATCH --job-name=pointTr
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 2

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.

singularity \
    exec --nv \
    --overlay /home/yb1025/pyenv/overlay-7.5GB-300K.ext3:ro \
    --overlay /scratch/work/public/singularity/openmpi4.1.1-ubuntu20.04.sqf:ro \
    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif  \
    /bin/bash -c "
source /ext3/env.sh
source /ext3/openmpi.sh
source activate pytorch_sac
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yb1025/.mujoco/mjpro150/bin
export LD_PRELOAD=/share/apps/local/lib/libGL.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yb1025/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yb1025/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/nvidia-<xxx>:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/nvidia-410/libEGL.so.1:$LD_PRELOAD
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
python train.py env=cheetah_run

"