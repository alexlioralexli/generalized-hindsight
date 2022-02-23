from gym.envs.registration import register

register(
    id='PointMassDense-v0',
    entry_point='rlkit.envs:PointEnv',
    max_episode_steps=15,
    reward_threshold=50.0,
)
#
# register(
#     id='FetchReachDense',
#     entry_point='envs:FetchReachEnv',
#     max_episode_steps=50,
#     reward_threshold=75.0,
# )


# what are all the envs I have

#


