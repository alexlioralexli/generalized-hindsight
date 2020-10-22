import numpy as np
from rlkit.torch.multitask.gym_relabelers import ReacherRelabelerWithGoal, ReacherRelabelerWithFixedGoal, ReacherRelabelerWithGoalSimple
from rlkit.torch.multitask.fetch_reach_relabelers import FetchReachRelabelerWithGoalAndObs
from rlkit.torch.multitask.hand_reach_relabelers import HandRelabeler, FingerRelabeler
from rlkit.envs.reacher_3dof import ReacherEnv
from rlkit.envs.point_reacher_env import PointReacherEnv
from rlkit.envs.point_reacher_env_3d import PointReacherEnv3D


def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    if hasattr(agent, 'eval'):
        agent.eval()
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def multitask_rollout_with_relabeler(
        env,
        agent,
        relabeler,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        latent=None,
        calculate_r_d=True,
        hide_latent=False,
        fast_rgb=True
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    latents = []
    qpos = []
    next_qpos = []
    rgb_array = []
    path_length = 0
    agent.reset()

    if hasattr(env, 'sim') and 'fixed' in env.sim.model.camera_names:
        camera_name = 'fixed'
    else:
        camera_name = None
    if latent is None:
        latent = relabeler.sample_task()
        if render:
            print("Current latent:", latent)
    if render and (isinstance(relabeler, ReacherRelabelerWithGoal)
                   or isinstance(relabeler, ReacherRelabelerWithFixedGoal)
                   or isinstance(relabeler, ReacherRelabelerWithGoalSimple)
                   or isinstance(relabeler, FetchReachRelabelerWithGoalAndObs)
                   or isinstance(relabeler, HandRelabeler)
                   or isinstance(relabeler, FingerRelabeler)):
        print(relabeler.interpret_latent(latent))
        goal = relabeler.get_goal(latent)
        o = env.reset(goal_pos=goal)
    elif isinstance(env, PointReacherEnv) or isinstance(env, PointReacherEnv3D):
        goal = relabeler.get_goal(latent)
        o = env.reset(goal)
    else:
        o = env.reset()
    if render:
        if render_kwargs['mode'] == 'rgb_array':
            if not fast_rgb:
                rgb_array.append(env.wrapped_env.sim.render(500, 500, camera_name=camera_name))
            # else:
            #     rgb_array.append(np.zeros((500, 500, 3), dtype=np.uint8))
        else:
            env.render(**render_kwargs)
    if hasattr(agent, 'eval'):
        agent.eval()
    while path_length < max_path_length:
        dict_obs.append(o)
        if hasattr(env, 'env'):
            qpos.append(env.env.sim.data.qpos.flat[:2])
        if observation_key:
            o = o[observation_key]
        if hide_latent:
            latent_input = np.zeros_like(latent)
        else:
            latent_input = latent
        a, agent_info = agent.get_action(o, latent_input, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if calculate_r_d:
            r, d_new = relabeler.reward_done(o, a, latent, env_info)
        d = d or d_new
        rewards.append(r)
        if hasattr(env, 'env'):
            next_qpos.append(env.env.sim.data.qpos.flat[:2])
        if render:
            if render_kwargs['mode'] == 'rgb_array':
                if path_length % 3 == 0 or not fast_rgb:
                    rgb_array.append(env.wrapped_env.sim.render(500, 500, camera_name=camera_name))
                else:
                    rgb_array.append(np.zeros((500, 500, 3), dtype=np.uint8))
            else:
                env.render(**render_kwargs)
        # print(o, a, next_o, r, latent, np.array_equal(o, latent))
        observations.append(o)
        latents.append(latent)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    latents = np.array(latents)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    result = dict(
        observations=observations,
        latents=latents,
        actions=actions,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=dict_obs,
        rewards=np.array(rewards).reshape(-1, 1),
        qpos=np.array(qpos),
        next_qpos=np.array(next_qpos)
    )
    if len(rgb_array) > 0 and rgb_array[0] is not None:
        result['rgb_array'] = np.array(rgb_array)
    return result


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        fast_rgb=True
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    rgb_array = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0

    if hasattr(env, 'sim') and 'fixed' in env.sim.model.camera_names:
        camera_name = 'fixed'
    else:
        camera_name = None

    if render:
        # import ipdb; ipdb.set_trace(context=10)
        if render_kwargs['mode'] == 'rgb_array':
            if not fast_rgb:
                rgb_array.append(env.sim.render(500, 500, camera_name=camera_name))
            else:
                rgb_array.append(np.zeros((500, 500, 3), dtype=np.uint8))
        else:
            env.render(**render_kwargs)

    # print("###############################")
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        # print(a)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            if render_kwargs['mode'] == 'rgb_array':
                if path_length % 3 == 0 or not fast_rgb:
                    rgb_array.append(env.sim.render(500, 500, camera_name=camera_name))
                else:
                    rgb_array.append(np.zeros((500, 500, 3), dtype=np.uint8))
            else:
                env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    result = dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
    if len(rgb_array) > 0 and rgb_array[0] is not None:
        result['rgb_array'] = np.array(rgb_array)
    return result
