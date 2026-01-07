from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def calculate_discounted_rewards(self, rewards, gamma):
        n = len(rewards)
        discounted_rewards = np.zeros(n)  # 初始化存储折扣奖励的二维数组

        for t in range(n):
            cumulative_reward = 0  # 初始化累计奖励

            for k in range(t, n):
                cumulative_reward += rewards[k] * (gamma ** (k - t))
            discounted_rewards[t] = cumulative_reward

        return discounted_rewards

    def run(self, test_mode=False, diff_return=1.0):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        reward_list = []

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            reward_list.append(reward)
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "next_state": [self.env.get_state()],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)


        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        discounted_rewards = self.calculate_discounted_rewards(reward_list, self.args.gamma)
        for i in range(self.t):
            self.batch.update({"return":[(discounted_rewards[i],)]}, ts=i)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if self.args.evaluate:
            self.test_phi(self.batch["state"], self.batch["terminated"])

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
    
    def test_phi(self, states, terminated):
        
        terminated = terminated[:, :-1].float()
        avil_state_range = torch.nonzero(terminated[0])[0][0]

        states = states.squeeze()[:avil_state_range, :]
        phi1, phi2 = self.mac.embedding.encode(states)
        last_phi_1 = phi1[-1,:].reshape(1,phi1.shape[-1])
        last_phi_1 = last_phi_1.expand(phi1.shape[0], -1)
        last_phi_2 = phi2[-1,:].reshape(1,phi2.shape[-1])
        last_phi_2 = last_phi_2.expand(phi2.shape[0], -1)

        squared_dist_1 = ((phi1 - last_phi_1) ** 2).sum(axis=-1)
        squared_dist_2 = ((phi2 - last_phi_2) ** 2).sum(axis=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist_1, min=1e-6))
        v2 = -torch.sqrt(torch.clamp(squared_dist_2, min=1e-6))
        v_t = (v1 + v2) / 2
        print("v1:", v1)
        print("v2:", v2)
        print("v_avg:", v_t)

