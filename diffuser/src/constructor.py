import numpy as np
import torch
import math
import einops
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.distributions import Bernoulli
# from models.tf_dynamics_models.fc import FC
# from models.tf_dynamics_models.bnn import BNN

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = 'cuda'
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
		
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Imagine_Net(nn.Module):
	def __init__(self, state_dim, goal_dim, t_dim = 16, hidden_dim=200, session=None,):
		super().__init__()
		self.input_dim = state_dim + goal_dim + t_dim + 1
		print('[ Imagine-diffuser-model ] | State dim {} | Goal dim: {} | Hidden dim: {} | Input dim: {}'.format(state_dim, goal_dim, hidden_dim, self.input_dim))
		self.state_dim = state_dim
		self.goal_dim = goal_dim
		self.out_dim = state_dim + goal_dim + 1
		self.t_dim = t_dim
		self.time_mlp = nn.Sequential(
			SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            Mish(),
            nn.Linear(t_dim * 2, t_dim),
		)
		self.model = nn.Sequential(
			nn.Linear(self.input_dim, hidden_dim),
			Mish(),
			nn.Linear(hidden_dim, hidden_dim*2),
			Mish(),
			nn.Linear(hidden_dim*2, hidden_dim*2),
			Mish(),
			nn.Linear(hidden_dim*2, self.out_dim)
		)

	def forward(self, goal, state, time, return_go):
		if type(state) != torch.Tensor:
			state = torch.from_numpy(state).float().cuda()
		if type(goal) != torch.Tensor:
			goal = torch.from_numpy(goal).float().cuda()
		if type(time) != torch.Tensor:
			time = torch.from_numpy(time).float().cuda()
		if type(return_go) != torch.Tensor:
			return_go = torch.from_numpy(return_go).float().cuda()

		t = self.time_mlp(time)
		x = torch.cat([state, goal, t, return_go],dim=1)
		x = self.model(x)
		return x

class MLPnet(nn.Module):
    def __init__(
        self,
        goal_dim,
        cond_dim,
        dim=8,
        dim_mults=(1, 2, 4, 8),
        horizon=1,
        returns_condition=True,
        condition_dropout=0.1,
        calc_energy=False,
    ):
        super().__init__()

        if calc_energy:
            act_fn = nn.SiLU()
        else:
            act_fn = Mish().float()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        ).float()

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        self.transition_dim = goal_dim + cond_dim

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    ).float()
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
        else:
            embed_dim = dim

        self.mlp = nn.Sequential(
                        nn.Linear(embed_dim + self.transition_dim, 1024),
                        act_fn,
                        nn.Linear(1024, 1024),
                        act_fn,
                        nn.Linear(1024, self.transition_dim),
                    ).float()

    def forward(self, x, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x action ]
            cond: [batch x state]
            returns : [batch x 1]
        '''
        # Assumes horizon = 1
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        inp = torch.cat([t, x], dim=-1).float()
        out  = self.mlp(inp)

        if self.calc_energy:
            energy = ((out - x) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x, create_graph=True)
            return grad[0]
        else:
            return out

def format_samples_for_training(samples):
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	rew = samples['rewards']
	delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((rew, delta_obs), axis=-1)
	return inputs, outputs

def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))



# import numpy as np
# import torch
# import math
# import einops
# import torch.nn as nn
# import torch.nn.functional as F
# import pdb

# # from models.tf_dynamics_models.fc import FC
# # from models.tf_dynamics_models.bnn import BNN

# class Mish(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self,x):
#         x = x * (torch.tanh(F.softplus(x)))
#         return x

# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         device = 'cuda'
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
		
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb

# class Backward_Net(nn.Module):
# 	def __init__(self, observation_dim, action_dim, input_dim, out_dim, t_dim = 16, hidden_dim=200, session=None,):
# 		super().__init__()
# 		print('[ Backward-diffuser-model ] | Observation dim {} | Action dim: {} | Hidden dim: {}'.format(observation_dim, action_dim, hidden_dim))
# 		self.observation_dim = observation_dim
# 		self.action_dim = action_dim
# 		self.out_dim = out_dim
# 		self.t_dim = t_dim
# 		self.time_mlp = nn.Sequential(
# 			SinusoidalPosEmb(t_dim),
#             nn.Linear(t_dim, t_dim * 2),
#             Mish(),
#             nn.Linear(t_dim * 2, t_dim),
# 		)
# 		self.model = nn.Sequential(
# 			nn.Linear(input_dim, hidden_dim),
# 			Mish(),
# 			nn.Linear(hidden_dim, hidden_dim*2),
# 			Mish(),
# 			nn.Linear(hidden_dim*2, hidden_dim*2),
# 			Mish(),
# 			nn.Linear(hidden_dim*2, out_dim)
# 		)

# 	def forward(self, x, cond, time, is_cond = True):
# 		if type(x) != torch.Tensor:
# 			x = torch.from_numpy(x).float().cuda()
# 		if type(time) != torch.Tensor:
# 			time = torch.from_numpy(time).float().cuda()

# 		if is_cond and type(cond) != torch.Tensor:
# 			cond = torch.from_numpy(cond).float().cuda()
# 		# print(time)
# 		t = self.time_mlp(time)
# 		# print(t)
# 		# if use_dropout:
#         #     mask = self.mask_dist.sample(sample_shape=(self.observation_dim, 1)).cuda()
#         #     returns_embed = mask*returns_embed
#         # if force_dropout:
#         #     returns_embed = 0*returns_embed
#         # t = torch.cat([t, returns_embed], dim=-1)
# 		if is_cond:
# 			x[:,:40] = cond
# 			x = torch.cat([x,t],dim=1)
# 		else:
# 			cond = 0 * cond
# 			x[:,:40] = cond
# 			x = torch.cat([x,t],dim=1)
# 		x = self.model(x)
		
# 		return x

# class Reward_Net(nn.Module):
# 	def __init__(self, observation_dim, action_dim, rew_dim=1, hidden_dim=200,):
# 		super().__init__()
# 		print('[ reward-model ] | Observation dim {} | Action dim: {} | Hidden dim: {}'.format(observation_dim, action_dim, hidden_dim))
# 		self.observation_dim = observation_dim
# 		self.action_dim = action_dim
# 		self.rew_dim = rew_dim
# 		self.model = nn.Sequential(
# 			nn.Linear(observation_dim + action_dim, hidden_dim),
# 			Mish(),
# 			nn.Linear(hidden_dim, rew_dim),
# 		)

# 	def forward(self, x):
# 		if type(x) != torch.Tensor:
# 			x = torch.from_numpy(x).float().cuda()

# 		output = self.model(x)
# 		return output

# class Dynamics_Net(nn.Module):
# 	def __init__(self, observation_dim, action_dim, hidden_dim=200,):
# 		super().__init__()
# 		print('[ MLP-model ] | Observation dim {} | Action dim: {} | Hidden dim: {}'.format(observation_dim, action_dim, hidden_dim))
# 		self.observation_dim = observation_dim
# 		self.action_dim = action_dim
# 		self.model = nn.Sequential(
# 			nn.Linear(observation_dim + action_dim, hidden_dim),
# 			Mish(),
# 			nn.Linear(hidden_dim, hidden_dim),
# 			Mish(),
# 			nn.Linear(hidden_dim, observation_dim),
# 		)

# 	def forward(self, s, a):
# 		x = torch.cat((s,a),1)
# 		output = self.model(x)
# 		output += s
# 		return output
	
# 	def loss(self, s, a, s_):
# 		s_pr = self.forward(s,a)
# 		return F.mse_loss(s_, s_pr)
		

# class Backward_Policy_Net(nn.Module):
# 	def __init__(self, observation_dim, action_dim, hidden_dim=200, session=None,):
# 		super().__init__()
# 		print('[ Backward-diffuser-policy ] | Observation dim {} | Action dim: {} | Hidden dim: {}'.format(observation_dim, action_dim, hidden_dim))
# 		self.observation_dim = observation_dim
# 		self.action_dim = action_dim
# 		t_dim = 16
# 		self.time_mlp = nn.Sequential(
# 			SinusoidalPosEmb(t_dim),
#             nn.Linear(t_dim, t_dim * 2),
#             Mish(),
#             nn.Linear(t_dim * 2, t_dim),
# 		)
# 		self.model = nn.Sequential(
# 			nn.Linear(action_dim + t_dim + observation_dim, hidden_dim),
# 			Mish(),
# 			nn.Linear(hidden_dim, hidden_dim*2),
# 			Mish(),
# 			nn.Linear(hidden_dim*2, hidden_dim*2),
# 			Mish(),
# 			nn.Linear(hidden_dim*2, action_dim)
# 		)

# 	def forward(self, x, cond, time):
# 		if type(x) != torch.Tensor:
# 			x = torch.from_numpy(x).float().cuda()
# 		if type(cond) != torch.Tensor:
# 			cond = torch.from_numpy(cond).float().cuda()
# 		if type(time) != torch.Tensor:
# 			time = torch.from_numpy(time).float().cuda()
# 		t = self.time_mlp(time)
# 		x = torch.cat([x,t,cond],dim=1)
# 		x = self.model(x)
		
# 		return x

# class Generater_Net(nn.Module):
# 	def __init__(self, observation_dim, hidden_dim=200, session=None,):
# 		super().__init__()
# 		print('[ Generater-diffuser-net ] | Observation dim {} | Hidden dim: {}'.format(observation_dim, hidden_dim))
# 		self.observation_dim = observation_dim
# 		t_dim = 16
# 		self.time_mlp = nn.Sequential(
# 			SinusoidalPosEmb(t_dim),
#             nn.Linear(t_dim, t_dim * 2),
#             Mish(),
#             nn.Linear(t_dim * 2, t_dim),
# 		)
# 		self.model = nn.Sequential(
# 			nn.Linear(observation_dim + t_dim , hidden_dim),
# 			Mish(),
# 			nn.Linear(hidden_dim, hidden_dim*2),
# 			Mish(),
# 			nn.Linear(hidden_dim*2, hidden_dim*2),
# 			Mish(),
# 			nn.Linear(hidden_dim*2, observation_dim)
# 		)

# 	def forward(self, x, cond, time):
# 		if type(x) != torch.Tensor:
# 			x = torch.from_numpy(x).float().cuda()
		
# 		if type(time) != torch.Tensor:
# 			time = torch.from_numpy(time).float().cuda()

# 		t = self.time_mlp(time)
# 		x = torch.cat([x,t],dim=1)
# 		x = self.model(x)
		
# 		return x

# class Generater_state_Net(nn.Module):
# 	def __init__(self, observation_dim, hidden_dim=200, session=None,):
# 		super().__init__()
# 		print('[ Generater-diffuser-net ] | Observation dim {} | Hidden dim: {}'.format(observation_dim, hidden_dim))
# 		self.observation_dim = observation_dim
# 		t_dim = 16
# 		self.time_mlp = nn.Sequential(
# 			SinusoidalPosEmb(t_dim),
#             nn.Linear(t_dim, t_dim * 2),
#             Mish(),
#             nn.Linear(t_dim * 2, t_dim),
# 		)
# 		self.model = nn.Sequential(
# 			nn.Linear(observation_dim + t_dim + observation_dim, hidden_dim),
# 			Mish(),
# 			nn.Linear(hidden_dim, hidden_dim*2),
# 			Mish(),
# 			nn.Linear(hidden_dim*2, hidden_dim*2),
# 			Mish(),
# 			nn.Linear(hidden_dim*2, observation_dim)
# 		)

# 	def forward(self, x, cond, time):
# 		if type(x) != torch.Tensor:
# 			x = torch.from_numpy(x).float().cuda()
# 		if type(cond) != torch.Tensor:
# 			cond = torch.from_numpy(cond).float().cuda()
# 		if type(time) != torch.Tensor:
# 			time = torch.from_numpy(time).float().cuda()
# 		t = self.time_mlp(time)
# 		x = torch.cat([x,t,cond],dim=1)
# 		x = self.model(x)
		
# 		return x

# def format_samples_for_training(samples):
# 	obs = samples['observations']
# 	act = samples['actions']
# 	next_obs = samples['next_observations']
# 	rew = samples['rewards']
# 	delta_obs = next_obs - obs
# 	inputs = np.concatenate((obs, act), axis=-1)
# 	outputs = np.concatenate((rew, delta_obs), axis=-1)
# 	return inputs, outputs

# def reset_model(model):
# 	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
# 	model.sess.run(tf.initialize_vars(model_vars))

