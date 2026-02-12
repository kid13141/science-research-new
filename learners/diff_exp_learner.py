import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.exp_qmix import Exp_QMixer
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
import utils.trajectory_encoder as tra_enc
import torch.nn.functional as F
import torch.nn as nn
from modules.team_encoder import TeamEncoder
from modules.history_buffer import HistoryBuffer
import matplotlib.pyplot as plt

def expectile_loss(adv, diff, expectile=0.7):
    weight = th.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

def normalize_states(states: np.ndarray, max: np.ndarray, min: np.ndarray):
    states=th.tensor((states - min) / (max-min+1e-5))
    states=th.where(th.isnan(states), th.full_like(states, 0), states)
    return states.cpu().numpy()
        
def non_normalized(states:np.ndarray,max:np.ndarray,min:np.ndarray):
    states_new=[]
    for s in states:
        b=s*(max-min)+min
        states_new.append(b.tolist())
    return np.array(states_new)

def linear_decay(initial_value, lower_bound, decay_rate, current_time):
    decayed_value = max(lower_bound, initial_value - decay_rate * current_time / 1e4)
    return decayed_value

class IAU(th.nn.Module):
    def __init__(self, input_dim, n_action):
        super().__init__()
        self.fc1 = th.nn.Linear(input_dim, 128)
        self.fc2 = th.nn.Linear(128, n_action)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        return y

def compute_infonce_loss(c_t, d):

    logits = th.matmul(c_t, d.T)
    exp_logits = th.exp(logits)

    positive_pairs = th.diagonal(exp_logits, dim1=-2, dim2=-1)
    denominator = exp_logits.sum(dim=-1)
    info_nce_loss = -th.log(positive_pairs / denominator).mean()

    return info_nce_loss



class Diff_Exp_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.params = list(mac.parameters())
        self.diff_params = mac.imagine_net.parameters()
        self.last_target_update_episode = 0
        self.mixer = None
        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # ---初始化team多样性模块 ---
        self.team_encoder = TeamEncoder(args).to(self.device)
        self.params += list(self.team_encoder.parameters())
        
        self.history_buffer = HistoryBuffer(
            capacity=args.buffer_capacity, 
            k=args.knn_k, 
            latent_dim=args.team_latent_dim,
            device=self.device
        )

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.diff_optimiser = Adam(params=self.diff_params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_diff_return = 1

    
    def train_hilp(self, batch, t_env):
            hilp_state, hilp_next_state, hilp_goal, hilp_reward, hilp_mask = self.rebuild_batch(batch)
            value_loss, value_info = self.compute_value_loss(hilp_state, hilp_next_state, hilp_goal, hilp_reward, hilp_mask)
            self.mac.embedding.optim_net.zero_grad()
            value_loss.backward()
            self.mac.embedding.optim_net.step()
            self.logger.log_stat("v_max", value_info["v max"].item(), t_env)
            self.logger.log_stat("v_min", value_info["v min"].item(), t_env)
            self.logger.log_stat("v_avg", value_info["v mean"].item(), t_env)
            self.logger.log_stat("value_loss", value_loss.item(), t_env)
    
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
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        max_seq_length = batch.max_seq_length
        # Get the relevant quantities
        state = batch["state"][:,:-1]
        rewards = batch["reward"][:, :-1] # (32,200,1)
        factor_rewards = batch["factor_reward"].squeeze(-1)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        filled = batch["filled"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        goal_states = batch["goal"][:,:-1]

        # Get the threshold of first phase
        alpha = linear_decay(self.args.alpha, 0, 0.01, t_env)

        # Diffusion Training
        if self.args.pre_train_diff:
            diffusion_losses = []
            for i in range(self.args.diff_num):
                diff_state, diff_goal, diff_return = self.rebuild_diff_batch(batch)
                x = th.cat([diff_goal, diff_state], dim=-1)
                loss = self.mac.diffusion_agent.loss(x, diff_state, diff_return)
                self.diff_optimiser.zero_grad()
                loss.backward()
                self.diff_optimiser.step()
                diffusion_losses.append(loss.item())
            diffusion_loss = np.array(diffusion_losses).mean(-1).item()
            diff_return = diff_return.detach().cpu().max()
            self.last_diff_return = diff_return

        # Calculate estimated Q-Values
        mac_out = []
        q_nav_out = []
        q_act_out = []
        alpha_out = []
        hidden_states_list = [] # 存储每一步的隐状态
        self.mac.init_hidden(state.shape[0])

        for t in range(max_seq_length):
            agent_inputs = self.mac._build_inputs(batch, t=t)
            factor_reward = factor_rewards[:,t]
            agent_outs, q_nav, q_act, alpha, _ = self.mac.agent.get_dual_q(
                agent_inputs, self.mac.hidden_states, factor_reward
            )    
            agent_outs = agent_outs.view(batch.batch_size,self.n_agents,-1)
            q_nav = q_nav.view(batch.batch_size,self.n_agents,-1)
            q_act = q_act.view(batch.batch_size,self.n_agents,-1)
            alpha = alpha.view(batch.batch_size,self.n_agents,-1)

            mac_out.append(agent_outs)
            q_nav_out.append(q_nav)
            q_act_out.append(q_act)
            alpha_out.append(alpha)
            hidden_states_list.append(self.mac.hidden_states.clone())

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        q_nav_out = th.stack(q_nav_out, dim=1)
        q_act_out = th.stack(q_act_out, dim=1)
        hidden_states = th.stack(hidden_states_list, dim=1)[:, :-1]

        # Behavior Q (用于 QMIX)
        chosen_action_qvals = th.gather(q_act_out[:, :-1], dim=3, index=actions).squeeze(3)  
        # Navigation Q (用于独立 DQN)
        chosen_action_q_nav = th.gather(q_nav_out[:, :-1], dim=3, index=actions).squeeze(3)

        # exp_reward
        z_team = self.team_encoder(hidden_states) # [Batch, Seq-1, Latent]

        with th.no_grad():
            r_diversity = self.history_buffer.compute_entropy_reward(z_team)
            # 间隔更新 History Buffer
            if episode_num % self.args.buffer_update_interval == 0:
                # 展平并存入
                flat_z = z_team.reshape(-1, self.args.team_latent_dim)
                # 随机采样一部分存入，而不是全部 
                stride = self.args.buffer_stride
                self.history_buffer.add(flat_z[::stride].detach())
        beta = max(0, self.args.diversity_beta * (1 - t_env / self.args.t_max))
        

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_q_nav_out = []
        self.target_mac.init_hidden(state.shape[0])

        for t in range(max_seq_length):
            # target_agent_outs,_= self.target_mac.forward(batch, t=t)
            target_inputs = self.target_mac._build_inputs(batch, t=t)
            target_factor_reward = factor_rewards[:,t]
            tgt_out, tgt_nav, _, _, _ = self.target_mac.agent.get_dual_q(
                                target_inputs, self.target_mac.hidden_states, target_factor_reward
                            )
            tgt_out = tgt_out.view(batch.batch_size,self.n_agents,-1)
            tgt_nav = tgt_nav.view(batch.batch_size,self.n_agents,-1)
            target_mac_out.append(tgt_out)
            target_q_nav_out.append(tgt_nav)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  
        target_q_nav_out = th.stack(target_q_nav_out[1:], dim=1) 
        
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        target_q_nav_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1] #the index of max q action 
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3) # (32,200,5)

            q_nav_out_detach = q_nav_out.clone().detach()
            q_nav_out_detach[avail_actions == 0] = -9999999
            q_nav_cur_max_actions = q_nav_out_detach[:, 1:].max(dim=3, keepdim=True)[1] #the index of max q action 
            target_max_q_nav = th.gather(target_q_nav_out, 3, q_nav_cur_max_actions).squeeze(3) # (32,200,5)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_max_q_nav = target_q_nav_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Action Stream Loss 
        act_targets = rewards + beta * r_diversity + self.args.gamma * (1 - terminated) * target_max_qvals           
        td_error_act = (chosen_action_qvals - act_targets.detach())
        loss_act = (td_error_act ** 2 * mask).sum() / mask.sum()

        # Q_nav
        nav_targets = factor_rewards[:, :-1] + self.args.gamma * (1 - terminated).expand_as(target_max_q_nav) * target_max_q_nav
        td_error_nav = (chosen_action_q_nav - nav_targets.detach()) 
        loss_nav = (td_error_nav ** 2).sum() / mask.expand_as(td_error_nav).sum()

        lambda_nav = getattr(self.args, "lambda_nav", 0.5) # 导航 Loss 权重
        
        # 计算对比学习 Loss (Auxiliary Loss)
        noise = th.randn_like(hidden_states) * 0.1
        z_team_aug = self.team_encoder(hidden_states + noise)
        sim_matrix = th.bmm(z_team, z_team_aug.transpose(1, 2)) # [Batch, Seq, Seq]
        labels = th.arange(z_team.shape[1]).to(self.device).unsqueeze(0).repeat(batch.batch_size, 1)
        loss_cl = th.nn.CrossEntropyLoss()(sim_matrix / self.args.cl_temp, labels)

        # total loss
        loss = loss_act + lambda_nav * loss_nav
        loss  = loss + self.args.cl_weight * loss_cl

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if self.args.target_update_interval > 1.0 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval <= 1.0:
            self._update_targets_soft(self.args.target_update_interval)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            if self.args.pre_train_diff:
                self.logger.log_stat("diff_loss", diffusion_loss, t_env)
                self.logger.log_stat("diff_return", diff_return.item(), t_env)
            # self.logger.log_stat("prior_exp_reward", prior_exp_reward.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            # self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
        return diff_return.item()
    
    def _get_IAU_input(self, size, action_values, encoded, goals):
        IAU_inputs = th.cat((action_values,encoded, goals),dim=-1)
        return IAU_inputs
    
    def _get_OAU_input(self, size, action_values, encoded):
        OAU_inputs = th.cat((action_values,encoded),dim=-1)
        return OAU_inputs

    def _build_inputs(self, batch, t):

        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        return inputs
    
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_targets_soft(self,tau):
        # self.target_mac.load_state(self.mac)
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            # self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
    
    def load_diff_model(self, path):
        self.mac.load_diff_model(path)
    
    def rebuild_batch(self, batch):
        state = batch["state"][:,:-1].float()
        terminated = batch["terminated"][:, :-1].float()
        bs = state.shape[0]
        hilp_state = []
        hilp_next_state = []
        hilp_goal = []
        hilp_reward = []
        hilp_mask = []
        for i in range(state.shape[0]):
            avil_state_range = th.nonzero(terminated[i])[0][0] # 返回轨迹中的终止状态index
            all_integers = np.arange(0, avil_state_range.cpu())   # 生成范围 [0, avil_state_range] 内的所有整数
            selected_integers = np.random.choice(all_integers, size=self.args.hilp_sample_size, replace=True) # 从中随机选择 n 个可重复的整数
            states = state[i][selected_integers] # 从轨迹中随机选择 n 个状态

            rand = np.random.choice(all_integers, size=self.args.hilp_sample_size, replace=True)
            next_goal_index = np.clip(selected_integers + 1, a_min=None, a_max=avil_state_range.item()) # 以下一时刻状态为goal的index
            next_states = state[i][next_goal_index]
            rand_goal_index = np.clip(selected_integers + rand, a_min=None, a_max=avil_state_range.item()) # 以往后任意时刻状态为goal的index
            goal_indx = np.where(np.random.rand(self.args.hilp_sample_size) < self.args.p_nextgoal, next_goal_index, rand_goal_index) # 按概率选择goal模式
            goals = state[i][goal_indx]

            success = (selected_integers == goal_indx)
            success_rewards = success.astype(float) * self.args.reward_scale + self.args.reward_shift
            masks = (1.0 - success.astype(float))

            hilp_state.append(states.cpu().detach().numpy())
            hilp_next_state.append(next_states.cpu().detach().numpy())
            hilp_goal.append(goals.cpu().detach().numpy())
            hilp_reward.append(success_rewards)
            hilp_mask.append(masks)
        
        hilp_state = th.tensor(hilp_state).reshape(bs * self.args.hilp_sample_size, -1).cuda()
        hilp_next_state = th.tensor(hilp_next_state).reshape(bs * self.args.hilp_sample_size, -1).cuda()
        hilp_goal = th.tensor(hilp_goal).reshape(bs * self.args.hilp_sample_size, -1).cuda()
        hilp_reward = th.tensor(hilp_reward).reshape(bs * self.args.hilp_sample_size, -1).cuda()
        hilp_mask = th.tensor(hilp_mask).reshape(bs * self.args.hilp_sample_size, -1).cuda()

        return hilp_state, hilp_next_state, hilp_goal, hilp_reward, hilp_mask

    def rebuild_diff_batch(self, batch, good_batch=False):
        
        state = batch["state"][:, :-1].float()
        death = batch["death"][:, :-1].float().squeeze()
        rewards = batch["reward"][:, :-1].float()
        returns = batch["cur_return"][:, :-1].float()
        terminated = batch["terminated"][:, :-1].float().squeeze()

        # end_state_index = []
        end_return = []
        for i in range(state.shape[0]):
            avil_state_range2 = th.nonzero(terminated[i])[0][0]
            end_return.append(returns[i][avil_state_range2].detach().cpu().numpy())
        end_return = np.array(end_return).squeeze()
        top_indices = np.argsort(end_return)[::-1][:8]

        bs = state.shape[0]
        diff_state = []
        diff_goal = []
        diff_return = []
        over_num = 0

        for i in range(state.shape[0]):
            if i not in top_indices:
                continue
            avil_state_range = th.nonzero(death[i])[0] + 1 # 返回轨迹中的终止状态index
            avil_state_range2 = th.nonzero(terminated[i])[0][0]
            start_index = int(avil_state_range * self.args.start_ratio)
            if start_index < 2:
                over_num += self.args.sample_size
                continue
            end_index = int(avil_state_range * self.args.end_ratio)
            all_integers = np.arange(0, start_index)   # 生成范围 [0, start_index] 内的所有整数
            selected_integers = np.random.choice(all_integers, size=self.args.sample_size, replace=True) # 从中随机选择 n 个可重复的整数
            states = state[i][selected_integers] # 从轨迹中随机选择 n 个状态

            # all_integers = np.arange(end_index, avil_state_range.cpu())   # 生成范围 [0, start_index] 内的所有整数
            begin_index = int(avil_state_range * self.args.begin_ratio)
            all_integers = np.arange(begin_index, end_index)
            goal_index = np.random.choice(all_integers, size=self.args.sample_size, replace=True) # 从中随机选择 n 个可重复的整数

            avil_state_range = th.nonzero(terminated[i])[0][0]+1

            for j in range(self.args.sample_size):
                discount_return = 0
                reward_array = rewards[i][goal_index[j]-1:avil_state_range2].cpu().numpy().squeeze()
                discount_array = np.array([self.args.discount ** t for t in range(reward_array.shape[0])]).squeeze()
                discount_return = (reward_array * discount_array).sum()
                norm_return = discount_return * 5
                diff_return.append(norm_return) 
            
            goals = state[i][goal_index]
            states = normalize_states(states.cpu().detach().numpy(), self.args.states_max,  self.args.states_min)
            goals = normalize_states(goals.cpu().detach().numpy(), self.args.states_max,  self.args.states_min)

            diff_state.append(states)
            diff_goal.append(goals)

        
        diff_state = th.tensor(diff_state).reshape(8 * self.args.sample_size - over_num, -1).cuda().float()
        diff_goal = th.tensor(diff_goal).reshape(8 * self.args.sample_size - over_num, -1).cuda().float()
        diff_return = th.tensor(diff_return).reshape(8 * self.args.sample_size - over_num, 1).cuda().float()/80

        return diff_state, diff_goal, diff_return

    def compute_value_loss(self, hilp_state, hilp_next_state, hilp_goal, hilp_reward, hilp_mask):
        # masks are 0 if terminal, 1 otherwise
        masks = 1.0 - hilp_reward
        # rewards are 0 if terminal, -1 otherwise
        hilp_reward = hilp_reward - 1.0
        #to do
        (next_v1, next_v2) = self.target_mac.embedding(hilp_next_state, hilp_goal)
        next_v = th.min(next_v1, next_v2)
        q = hilp_reward + self.args.discount * masks * next_v

        #to do
        (v1_t, v2_t) = self.target_mac.embedding(hilp_state, hilp_goal)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = hilp_reward + self.args.discount * masks * next_v1
        q2 = hilp_reward + self.args.discount * masks * next_v2
        (v1, v2) = self.mac.embedding(hilp_state, hilp_goal)
        v = (v1 + v2) / 2

        value_loss1 = expectile_loss(adv, q1 - v1, self.args.expectile).mean()
        value_loss2 = expectile_loss(adv, q2 - v2, self.args.expectile).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'v max': v.max(),
            'v min': v.min(),
            'v mean': v.mean()
            }
    
def concatenate_dicts(dict1, dict2):
    """
    纵向拼接两个字典中对应 key 的内容
    :param dict1: 第一个字典
    :param dict2: 第二个字典
    :return: 纵向拼接后的新字典
    """
    # 初始化一个新的字典来存储拼接后的内容
    result = {}
    keys = {"state", "obs", "actions", "avail_actions", "reward", "terminated", "actions_onehot", "filled", "cur_return"}
    for key in keys:
            value1 = dict1[key]
            value2 = dict2[key]
            result[key] = th.cat((value1, value2), dim=0)
    
    return result
