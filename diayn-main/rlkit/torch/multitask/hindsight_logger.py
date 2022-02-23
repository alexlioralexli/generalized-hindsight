




class HindsightLogger(object):
    def __init__(self, n_epochs, save_freq=None):
        self.n_epochs = n_epochs
        self.save_freq = save_freq
        self.path_features = []
        self.original_lats = []
        self.relabeled_lats = []

    def log_video(self, video):
        raise NotImplementedError

    # log traj features, traj original latent, traj original
    def log_path_features(self, path_summary_info):
        self.path_features.append(path_summary_info['features'])
        self.original_lats.append(path_summary_info['original_lat'])
        self.relabeled_lats.append(path_summary_info['relabeled_lat'])

    def save(self):
        # get path
        # convert everything to npy
        raise NotImplementedError

    def end_epoch(self, epoch):
        if self.save_freq is None:
            if epoch == self.n_epochs:  #todo: double check the indexing
                raise NotImplementedError
        elif self.epoch % self.save_freq == 0:
            raise NotImplementedError

