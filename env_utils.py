"""
This file should basically provide a unified interface for making generalized hindsight environments
"""
from rlkit.envs.point_robot_new import PointEnv as PointEnv2
from rlkit.envs.fetch_reach import FetchReachEnv

env_ids = {
    'pointmass2',
    'pointmass2-v0',
    'fetchreach'
}

def make_gher_env(env_id, env_kwargs):
    assert env_id in env_ids
    if env_id == 'pointmass2' or env_id == 'pointmass2-v0':
        env = PointEnv2(**env_kwargs)
    elif env_id == 'fetchreach':
        env = FetchReachEnv(**env_kwargs)
    else:
        raise NotImplementedError
    return env

