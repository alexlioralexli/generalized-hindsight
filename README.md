# Generalized Hindsight for Reinforcement Learning


## Installation

1. Install and use the included Ananconda environment
```
$ conda env create -f environment/[linux-cpu|linux-gpu|mac]-env.yml
$ source activate rlkit
(rlkit) $ python examples/ddpg.py
```
Choose the appropriate `.yml` file for your system.
You may face installation issues, in which case you can try the following (which may go out of date) 
```
conda config --append channels conda-forge
conda env create -f environment/temp.yml
```
These Anaconda environments use MuJoCo 1.5 and gym 0.10.5.
You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

2. Add this repo directory to your `PYTHONPATH` environment variable or simply
run:
```
pip install -e .
```

3. (Optional) Copy `conf.py` to `conf_private.py` and edit to override defaults:
```
cp rlkit/launchers/conf.py rlkit/launchers/conf_private.py
```

DISCLAIMER: the mac environment has only been tested without a GPU.

For an even more portable solution, try using the docker image provided in `environment/docker`.
The Anaconda env should be enough, but this docker image addresses some of the rendering issues that may arise when using MuJoCo 1.5 and GPUs.
The docker image supports GPU, but it should work without a GPU.
To use a GPU with the image, you need to have [nvidia-docker installed](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

## Example of training a policy
No relabeling:
```
python launch_gher.py --epochs 1000 --env pointmass2
```

Random relabeling:
```
python launch_gher.py --epochs 1000 --relabel --n_sampled_latents 1 --env pointmass2
```

Advantage relabeling:
```
python launch_gher.py --epochs 1000 --relabel --n_sampled_latents 100 --use_advantages --env pointmass2
```


AIR:
```
python launch_gher.py --epochs 1000 --relabel --n_sampled_latents 100 --use_advantages --env pointmass2 --cache --irl
```



## Visualizing a policy and seeing results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `rlkit.launchers.config.LOCAL_LOG_DIR`. Default name is 'output'.
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

```
(rlkit) $ python scripts/run_multitask_policy.py LOCAL_LOG_DIR/<exp_prefix>/<foldername>/params.pkl
```

If you have rllab installed, you can also visualize the results
using `rllab`'s viskit, described at
the bottom of [this page](http://rllab.readthedocs.io/en/latest/user/cluster.html)


This codebase is based on `rlkit`. You can see the original [here](https://github.com/vitchyr/rlkit).
