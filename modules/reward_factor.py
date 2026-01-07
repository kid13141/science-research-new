import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

class RMixer(nn.Module):
    def __init__(self, args):
        super(RMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim * 3, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim * 3, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim * 3, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim  * 3, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim * 3, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim * 3, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, inputs):
        bs = agent_qs.size(0)
        states = inputs.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

class Factoring(nn.Module):
    def __init__(self, state_shape, action_shape, args):
        super(Factoring, self).__init__()
        self.args = args
        self.state_embed_net = nn.Sequential(nn.Linear(state_shape * 2 + args.n_agents, args.vae_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(args.vae_hidden_dim, args.vae_hidden_dim ),
                                            nn.ReLU(),                                            
                                            nn.Linear(args.vae_hidden_dim, args.n_agents)).to(self.args.device)
        self.loss_func = nn.MSELoss(reduction="mean")
        
    def forward(self, inputs):
        return self.state_embed_net(inputs)
    
    def update(self, inputs, total_reward):
        reward_f = self.forward(inputs)
        reward_f = reward_f.sum(-1)
        loss = self.loss_func(reward_f, total_reward)
        return loss
    


