import pickle
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch
from torch.distributions import Categorical
from collections import deque
import random
import torch.nn.functional as F
import math
import os

def to_numpy(input_data):
    # 判断输入类型是否为 tensor
    if isinstance(input_data, torch.Tensor):
        return input_data.cpu().numpy()  # 将 tensor 转换为 numpy 数组
    else:
        return input_data  # 如果不是 tensor，原样返回
    
def to_tensor(input_data):
    # 判断输入类型是否为 numpy 数组
    if isinstance(input_data, np.ndarray):
        return torch.from_numpy(input_data).cuda()  # 将 numpy 数组转换为 tensor
    else:
        return input_data  # 如果不是 numpy 数组，原样返回


def is_dir_empty(directory):
    return not any(True for _ in os.scandir(directory))

def clear_non_empty_directory(directory):
    if not is_dir_empty(directory):
        # 清空文件夹
        for root, dirs, files in os.walk(directory):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))

def normalize_states(states: np.ndarray, max: np.ndarray, min: np.ndarray):
    states=torch.tensor((states - min) / (max-min))
    states=torch.where(torch.isnan(states), torch.full_like(states, 0), states)
    return states
        
def non_normalized(states:np.ndarray,max:np.ndarray,min:np.ndarray):
    states_new=[]
    for s in states:
        b=s*(max-min)+min
        states_new.append(b.tolist())
    return np.array(states_new)

def linear_decay(initial_value, lower_bound, decay_rate, current_time):
    decayed_value = max(lower_bound, initial_value - decay_rate * current_time / 1e4)
    return decayed_value

# 计算状态-目标的希尔伯特距离
def phi_distance(mac,cur_states,goal_states):
    with torch.no_grad():
        cur_phi_1, cur_phi_2 = mac.embedding.encode(cur_states) 
        cur_phi = (cur_phi_1 + cur_phi_2)/2
        goal_phi_1, goal_phi_2 = mac.embedding.encode(goal_states) 
        goal_phi = (goal_phi_1 + goal_phi_2)/2
        squared_dist = ((cur_phi - goal_phi) ** 2).sum(axis=-1)
        dist = torch.sqrt(torch.clamp(squared_dist, min=1e-6))
    return dist

class Diff_Exp_Runner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1


        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.log_goal_time = 0
        self.goals = []
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.last_return = 0

        # Log the first run
        self.log_train_stats_t = -1000000
        self.log_goal_t = 0
        self.log_goal_period = 0

        # 定义超参数初始值、下界和衰减速率
        self.upper_bound_thed = args.upper_bound_thed
        self.lower_bound_thed = args.lower_bound_thed
        self.decay_rate_thed = args.decay_rate_thed
        self.get_state_limit()
            # 要操作的文件夹路径
        folder_path = "/home/songshoucheng/GUF_2025/log_goals"

        # 判断文件夹是否为空，不为空则清空文件夹
        if not is_dir_empty(folder_path):
            clear_non_empty_directory(folder_path)
            print(f"文件夹 {folder_path} 已清空。")
        else:
            print(f"文件夹 {folder_path} 为空。")


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        

    def get_env_info(self):
        return self.env.get_env_info()
    
    def get_state_limit(self):
        self.args.states_max, self.args.states_min = self.env.get_state_limit()
        self.args.max_reward = self.env.max_reward
        self.args.reward_scale_rate = self.env.reward_scale_rate
        self.args.state_index = self.env.get_state_seg()
        self.args.teammate_dim = int(self.args.state_index[0].item())
        self.args.enemy_dim = int((self.args.state_index[self.env.n_agents] - self.args.state_index[self.env.n_agents - 1]).item())

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.last_return = 0
        self.t = 0

    def compute_influence(self, state, next_state, goal, factor_reward, ally_alive, enemy_alive):
        influences = []

        for i in range(self.args.n_agents + self.args.n_enemy):
            if i < self.args.n_agents:
                start_index = i * self.args.teammate_dim
                end_index = (i+1) * self.args.teammate_dim
                alive = ally_alive[i]
            else:
                start_index = (i - self.args.n_agents) * self.args.enemy_dim + self.args.n_agents * self.args.teammate_dim
                end_index = (i - self.args.n_agents + 1) * self.args.enemy_dim + self.args.n_agents * self.args.teammate_dim
                alive = enemy_alive[i - self.args.n_agents]

            org_state = state.clone()
            state[0, start_index:end_index] = next_state[0, start_index:end_index]
            mod_state = state

            org_dist = phi_distance(self.mac, org_state, goal)
            mod_dist = phi_distance(self.mac, mod_state, goal)

            influence = (org_dist - mod_dist) * alive
            influences.append(influence.item())

        total_influence = sum(influences)
        bias = (factor_reward - total_influence)/(self.args.n_agents + self.args.n_enemy)
        if(isinstance(bias, torch.Tensor)):
            bias = bias.item()
        influences = np.array(influences).squeeze()

        exp_reward = influences + bias
        exp_reward = exp_reward[:self.args.n_agents]
        return exp_reward, exp_reward.sum().item()

    def split_global_vector(self, global_vector, N, M, teammate_dim, enemy_dim):
        """
        Split the global input vector into teammate and enemy feature tensors.

        :param global_vector: Tensor of shape (batch_size, total_length), the global input vector.
        :param N: Number of teammates.
        :param M: Number of enemies.
        :param teammate_dim: Dimensionality of each teammate's feature vector.
        :param enemy_dim: Dimensionality of each enemy's feature vector.
        :return: Tuple (teammates, enemies)
            - teammates: Tensor of shape (batch_size, N, teammate_dim).
            - enemies: Tensor of shape (batch_size, M, enemy_dim).
        """
        batch_size = global_vector.size(0)
        total_length = global_vector.size(1)
        
        # Calculate expected length
        expected_length = N * teammate_dim + M * enemy_dim
        if total_length != expected_length:
            raise ValueError(f"Global vector length ({total_length}) does not match the expected length ({expected_length}).")
        
        # Reshape the global vector
        teammate_features = global_vector[:, :N * teammate_dim].view(batch_size, N, teammate_dim)
        
        return teammate_features


    def swap_dim(self, arr, dims, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        bs, N, dim = arr.shape
        x = dims  # 计算交换的元素个数
        
        swapped_arr = arr.copy()  # 避免原地修改
        
        for i in range(bs):  # 遍历每个 bs
            # 生成当前 bs 下的所有可能交换对 (j, k)
            indices = np.arange(N)
            np.random.shuffle(indices)  # 随机打乱
            if N%2 == 0:
                indices = indices.reshape(-1, 2)  # 两两配对
            else:
                indices = indices[:-1].reshape(-1, 2)
            
            # 交换前 x 个元素
            for j, k in indices:
                if k >= N:  # 防止越界
                    continue
                swapped_arr[i, j, :x], swapped_arr[i, k, :x] = (
                    swapped_arr[i, k, :x], swapped_arr[i, j, :x]
                )
        
        return swapped_arr

    def run(self, test_mode=False, diff_return=1.0, v_thed=1.0):
        self.reset()
        terminated = False
        episode_return = 0
        stop = 1
        self.mac.init_hidden(batch_size=self.batch_size)
        # thed = linear_decay(0.3, 0, 0.015, self.t_env)
        min_dis = 1000
        while not terminated:
            if self.t == 0:
                cur_state = np.array([self.env.get_state()])
                cur_state = torch.from_numpy(cur_state).cuda()
                last_state = cur_state

                start_state = np.array([self.env.get_state()])
                norm_state = normalize_states(start_state, self.args.states_max, self.args.states_min)
                norm_state_th = torch.tensor(norm_state).cuda()
                returns = torch.ones(norm_state_th.shape[0], 1).cuda() * diff_return

                with torch.no_grad():
                    goal_states_th = self.mac.diffusion_agent.forward(cond=norm_state_th, returns=returns) # sample (s_0, s_sg) from diffusion model
                    goal_states = non_normalized(goal_states_th[:,:self.args.state_shape].to('cpu').numpy(),self.args.states_max, self.args.states_min)
                    goal_states = torch.tensor(goal_states).cuda().float()

                last_dis = phi_distance(self.mac, cur_state, goal_states)
                init_dis = last_dis

            pre_transition_data = {
                "state": [self.env.get_state()],
                "goal": [goal_states.detach().cpu().numpy()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)
            
            #epsilon greedy action of each agent
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,test_mode=test_mode)
            reward, terminated, env_info = self.env.step(actions[0])
            
            episode_return += reward

            cur_state = np.array([self.env.get_state()])
            cur_state = torch.from_numpy(cur_state).cuda()

            cur_dis = phi_distance(self.mac, cur_state, goal_states)

            # if cur_dis < min_dis:
            #     min_dis = cur_dis
            #     t = self.t

            # if terminated and self.t_env > 5e4:
            #     factor_reward = (init_dis - min_dis)* t / 100
            # else:
            #     factor_reward = 0
            
            factor_reward = last_dis - cur_dis
            factor_reward = torch.clip(factor_reward,min=0) * 0.5

            if (self.env.death_tracker_ally).sum() > 0 or terminated:
                death = 1
            else:
                death = 0
            
            last_dis = cur_dis

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "cur_return": [(episode_return,)],
                "factor_reward": factor_reward,
                "death": [(death,)]
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "goal": [goal_states.detach().cpu().numpy()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

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
            states, goals, dis, exp_reward = self.test_phi(self.batch["state"], self.batch["terminated"], self.batch["goal"], self.batch["actions"], self.batch["factor_reward"])
            return states, goals, dis, exp_reward
        
        if self.t_env - self.log_goal_period >= 1000:
            self.goals.append(self.batch["goal"][0][0].detach().cpu().numpy())
            self.log_goal_period = self.t_env

        if self.t_env - self.log_goal_t >= 50000:
            self.log_goal()
            self.goals = []
            self.log_goal_t = self.t_env

        return self.batch

    def load_agent(self, diffusion):
        self.diffusion_agent = diffusion

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_median", np.median(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
    
    def test_phi(self, states, terminated, goals, actions, rewards):
        
        terminated = terminated[:, :-1].float()
        avil_state_range = torch.nonzero(terminated[0])[0][0]
        states = states.squeeze()[:avil_state_range, :]
        actions = actions.squeeze()[:avil_state_range, :]
        goals = goals.squeeze()[:avil_state_range, :]
        rewards = rewards.squeeze()[:avil_state_range, :]
        actions = actions.reshape(-1, self.args.n_agents)

        # phi1, phi2 = self.mac.embedding.encode(states)
        # goal_phi1, goal_phi2 = self.mac.embedding.encode(goals)
        # phi = (phi1 + phi2)/2
        # goal_phi = (goal_phi1 + goal_phi2)/2

        # squared_dist = ((phi - goal_phi) ** 2).sum(axis=-1)
        # v_t = torch.sqrt(torch.clamp(squared_dist, min=1e-6))
        split_goals = self.split_global_vector(goals, self.args.n_agents, self.args.n_enemy, self.args.teammate_dim, self.args.enemy_dim)
        split_state = self.split_global_vector(states, self.args.n_agents, self.args.n_enemy, self.args.teammate_dim, self.args.enemy_dim)
        v_t = self.split_distance(to_numpy(split_state), to_numpy(split_goals))
        
        return states.detach().cpu().numpy(), goals.detach().cpu().numpy(), v_t, rewards.detach().cpu().numpy()
    
    def log_goal(self):
        goals = np.array(self.goals)
        log_name = "/home/songshoucheng/GUF_2025/log_goals/goals_"+ str(int(self.t_env / 50000)) + ".pkl"
        with open(log_name, 'wb') as f:
            pickle.dump(goals, f)