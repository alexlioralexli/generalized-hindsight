import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector

class DIAYNBatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            agent,
            exploration_env,
            evaluation_env,
            cfg, 
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            log_exploration=True,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            log_exploration
        )
        self.cfg = cfg 
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training


    #EVALUATE FUNCTION FROM DIAYN:

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode < 3))
            done = False
            episode_reward = 0
            skill = self.agent.skill_dist.sample()

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, skill, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}_{episode}_skill_{skill.argmax().cpu().item()}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)




    
    def _train(self):

        """
            50 EPOCHS
            NUM TRAIN LOOPS PER EPOCH = 1
            num trains per loop = 100

            WITHIN COLLECT PATHS
            num steps collected < num_steps:

                num_steps = 1000
            WITHIN ROLLOUTS:
            path_length < max_path_length

        """
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            # print(f"KEYS OF ORIGINAL init paths are: {len(init_expl_paths)}")
            # print(f"the first index is a : {type(init_expl_paths)}")
            # print(f"THE KEYS OF THE FIRST INDEX ARE: {init_expl_paths[0].keys()}")
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            print(f"NUM EPOCHS IS: {self.num_epochs}")

            """
                MAX PATH LENGTH IN ALGO KWARGS IS 15.

            """
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                print(f"MAX PATH LENGTH IN BATCH RL is : {self.max_path_length}")
                gt.stamp('exploration sampling', unique=False)
                print(f"IN epoch NUMBER: {epoch}")

                #THE ADD_PATHS, should return the new data points in the replay_buffer, : skills, not_done, not_done_no_max.
                # print(f"new_expl_paths keys are: {new_expl_paths.keys()}")
                # print(f"Type of paths after path collector : {new_expl_paths}, length is:  {len(new_expl_paths) if isinstance(new_expl_paths, list) else None}")
                """

                    *WARNING:

                    MIN NUMBER OF PATHS IS 15, 

                    However, it is coming out to be 1 all the time.

                """

                print(f"Num train loops per epoch is: {self.num_train_loops_per_epoch}")
                
                
                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                # self.training_mode(True)
                self.trainer.trainParamSet(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)

                        # THE NETWORKS ARE UPDATED HERE. SO DIAYN WILL BE TIED UP HERE.
                    self.trainer.train(train_data)
                # if hasattr(self.trainer, '_base_trainer'):
                #     self.trainer._base_trainer._update_target_networks()
                # else:
                #     self.trainer._update_target_networks()  #added 10/17
                # print("Reminder: changed the update target networks functionality")
                self.trainer.trainParamSet(False)
                gt.stamp('training', unique=False)
                # self.training_mode(False)


            self._end_epoch(epoch)


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            log_exploration=True,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            log_exploration
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')
            print(f"MAX PATH LENGTH IN GHER BATCH RL is : {self.max_path_length}")

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                # self.trainer.trainParamSet()
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)

                        # THE NETWORKS ARE UPDATED HERE. SO DIAYN WILL BE TIED UP HERE.
                    self.trainer.train(train_data)
                # if hasattr(self.trainer, '_base_trainer'):
                #     self.trainer._base_trainer._update_target_networks()
                # else:
                #     self.trainer._update_target_networks()  #added 10/17
                # print("Reminder: changed the update target networks functionality")
                gt.stamp('training', unique=False)
                # self.trainer.trainParamSet(False)

                self.training_mode(False)

            self._end_epoch(epoch)
