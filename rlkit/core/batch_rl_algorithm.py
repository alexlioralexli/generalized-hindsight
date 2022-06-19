import os
import abc
import torch 

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
            agent,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            log_exploration
        )
        self.agent = agent
        self.cfg = cfg 
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.work_dir = os.getcwd()
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
        self.recordEpoch = [10, 100, 500, 1000, 2500, 5000, 10000]
        # print(f"min num steps before training: {self.min_num_steps_before_training}")
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                rollType="exploration", 
            )
            # print(f"KEYS OF ORIGINAL init paths are: {len(init_expl_paths)}")
            # print(f"the first index is a : {type(init_expl_paths)}")
            # print(f"THE KEYS OF THE FIRST INDEX ARE: {init_expl_paths[0].keys()}")
            epoch_dummy = -1
            
            """ 
                UN COMMENT THIS LATER.

            """


            self.replay_buffer.add_paths(init_expl_paths, epoch_dummy)
            self.expl_data_collector.end_epoch(-1)

        # REMOVING 1 EPOCH, REMOVE THIS LINE LATER.
        # self.num_epochs -= 1
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
                rollType="evaluation",
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):


                # THIS SHOULD BE A 1000.
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                    rollType="exploration",
                )
                # print(f"MAX PATH LENGTH IN BATCH RL is : {self.max_path_length}")
                # gt.stamp('exploration sampling', unique=False)
                print(f"IN epoch NUMBER: {epoch}")

                #THE ADD_PATHS, should return the new data points in the replay_buffer, : skills, not_done, not_done_no_max.
                # print(f"new_expl_paths keys are: {new_expl_paths.keys()}")
                # print(f"Type of paths after path collector : {new_expl_paths}, length is:  {len(new_expl_paths) if isinstance(new_expl_paths, list) else None}")
                """

                    *WARNING:

                    MIN NUMBER OF PATHS IS 15, 

                    However, it is coming out to be 1 all the time.

                """

        
                print(f"Len of expl paths in DIAYN-HUSK: {len(new_expl_paths)}")
                self.replay_buffer.add_paths(new_expl_paths, epoch)

                """
                    ADD PATHS 

                    

                """
                gt.stamp('data storing', unique=False)

                # self.training_mode(True)
                self.trainer.trainParamSet(True)
                
                #NUM TRAINS PER TRAIN LOOP IS 100! 
                #WITH BATCH SIZE OF 128, 1000 IN ORIGINAL ROLLOUTS
                #THERE IS AN EPOCH
                for step in range(self.num_trains_per_train_loop):

                    #print(f"Num trains per loop is: {self.num_trains_per_train_loop}")
                    """
                        RANDOM BATCH -> REPLAY BUFFER -> RANDOM_BATCH

                    """
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    

                    
                    #print(f"The keys of train data is: {train_data.keys()}")
                    # obs = train_data["observations"]
                    # next_obs = train_data["next_observations"]
                    # print(f"Shape of obs is: {obs.shape}")
                    # print(f"The obs data is: {obs}")
                    # print(f"Next_obs shape is: {next_obs.shape}")
                    # print(f"The obs data is: {next_obs}")
                        # THE NETWORKS ARE UPDATED HERE. SO DIAYN WILL BE TIED UP HERE.
                    self.trainer.train(train_data, step, epoch)
                # if hasattr(self.trainer, '_base_trainer'):
                #     self.trainer._base_trainer._update_target_networks()
                # else:
                #     self.trainer._update_target_networks()  #added 10/17
                # print("Reminder: changed the update target networks functionality")
                self.trainer.trainParamSet(False)
                gt.stamp('training', unique=False)
                # self.training_mode(False)
            self._end_epoch(epoch)
        # Already taken care of in _end_epoch
        #     if epoch in self.recordEpoch:
        #         print(f"Pickling: {epoch}")
        #         filePathSave = self.work_dir + "/" + str(epoch) + ".pkl"
        #         torch.save(self.agent,filePathSave)    

        # filePathSave = self.work_dir + "/finalModel.pkl"
        # torch.save(self.agent,filePathSave)


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
            self.replay_buffer.add_paths(init_expl_paths, -1)
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
                print(f"Len of expl paths in GHER: {len(new_expl_paths)}")
                self.replay_buffer.add_paths(new_expl_paths, epoch)
                gt.stamp('data storing', unique=False)
                # if self.alg == "SAC":
                #     self.training_mode(True)
                # self.trainer.trainParamSet()
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    print(f"The train data keys in GHER are: {train_data.keys()}")
                        # THE NETWORKS ARE UPDATED HERE. SO DIAYN WILL BE TIED UP HERE.
                    self.trainer.train(train_data)
                # if hasattr(self.trainer, '_base_trainer'):
                #     self.trainer._base_trainer._update_target_networks()
                # else:
                #     self.trainer._update_target_networks()  #added 10/17
                # print("Reminder: changed the update target networks functionality")
                gt.stamp('training', unique=False)
                # self.trainer.trainParamSet(False)
                # if self.alg == "SAC":
                #     self.training_mode(False)

            self._end_epoch(epoch)
