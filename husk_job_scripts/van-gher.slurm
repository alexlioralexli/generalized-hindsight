#!/bin/bash
#SBATCH --job-name=gher
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 2

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.

singularity \
    exec --nv \
    --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    --overlay /home/yb1025/pyenv/overlay-7.5GB-300K.ext3:ro \
    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "
source /ext3/env.sh
#source /ext3/pytorch_sac
## FOR GYM USE 210, for HYDRA use 200
export LD_LIBRARY_PATH=/home/yb1025/.mujoco/mujoco210/bin:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/.singularity.d/libs:\$LD_LIBRARY_PATH
python dummy_gpu.py > dummy_gpu.log 2>&1 &
python /home/yb1025/Research/GRAIL/HUSK/accelerate-skillDiscovery/generalized-hindsight/launch_gher.py --env AntEnv --epochs 2 --relabel --n_sampled_latents 100  --cache --irl > train2.log 2>&1 
wait
"

