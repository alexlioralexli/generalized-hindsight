DEFAULT_PARAMS = {
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 2,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values

    'bc_loss': 0, # whether or not to use the behavior cloning loss as an auxilliary loss
    'q_filter': 0, # whether or not a Q value filter should be used on the Actor outputs
    'num_demo': 100, # number of expert demo episodes
    'demo_batch_size': 128, #number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
    'prm_loss_weight': 0.001, #Weight corresponding to the primary loss
    'aux_loss_weight':  0.0078, #Weight corresponding to the auxilliary loss also called the cloning loss
}
DEFAULT_VARIANT = dict(
        algorithm_kwargs=dict(
            num_epochs= 10,
            num_eval_steps_per_epoch=500,  # want to replace with num_trajs
            num_trains_per_train_loop=40,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=10000,
            num_train_loops_per_epoch=10,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=1e-2,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        replay_buffer_size=int(1E6),
)
