   def add_single_sample(self, latent, observation, action, reward, next_observation, terminal=None, skill=None, originalLatent=None,**kwargs):
        print(f"In single sample")
        print(f"The latent is: {latent}, observation: {observation}, next_observation is: {next_observation}")

        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]

        if self.alg == "DIAYN":
            if (torch.is_tensor(latent)):
                latent = latent.cpu().detach().numpy()
            reward = reward.cpu().detach().numpy()
            # print(f"Latent received is : {latent}, the type is : {type(latent)}")  
            self._skills[self._top] = latent
            self._observations[self._top] = observation
            self._actions[self._top] = action
            # self._latents[self._top] = latent
            self._rewards[self._top] = reward
            self._next_obs[self._top] = next_observation
            # self._pureSkills[self._top] = originalLatent
        elif self.alg == "SAC":
            self._latents[self._top] = latent
            self._terminals[self._top] = terminal
            self._rewards[self._top] = reward
            self._observations[self._top] = observation
            self._actions[self._top] = action
            self._next_obs[self._top] = next_observation
      
        self._advance()