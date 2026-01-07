import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionNetwork(nn.Module):
    def __init__(self, teammate_dim, enemy_dim, hidden_dim, output_dim, num_heads=4):
        """
        Initialize the CrossAttentionNetwork.

        :param teammate_dim: Dimensionality of teammate feature vectors.
        :param enemy_dim: Dimensionality of enemy feature vectors.
        :param hidden_dim: Dimensionality of the hidden representations.
        :param output_dim: Dimensionality of the final output vector.
        :param num_heads: Number of attention heads for multi-head attention.
        """
        super(CrossAttentionNetwork, self).__init__()
        
        # Linear layers for teammate and enemy feature projections
        self.teammate_proj = nn.Linear(teammate_dim, hidden_dim)
        self.enemy_proj = nn.Linear(enemy_dim, hidden_dim)
        
        # Cross attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Feed-forward layers after attention
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, teammates, enemies):
        """
        Forward pass of the network.

        :param teammates: Tensor of shape (batch_size, N, teammate_dim), teammate feature vectors.
        :param enemies: Tensor of shape (batch_size, M, enemy_dim), enemy feature vectors.
        :return: Tensor of shape (batch_size, output_dim), encoded output vector.
        """
        # Project teammate and enemy features to the same hidden dimension
        teammates_proj = self.teammate_proj(teammates)  # Shape: (batch_size, N, hidden_dim)
        enemies_proj = self.enemy_proj(enemies)        # Shape: (batch_size, M, hidden_dim)
        
        # Concatenate teammates and enemies along the sequence dimension
        all_features = torch.cat([teammates_proj, enemies_proj], dim=1)  # Shape: (batch_size, N+M, hidden_dim)
        
        # Apply cross-attention
        attention_output, _ = self.cross_attention(all_features, all_features, all_features)  # Shape: (batch_size, N+M, hidden_dim)
        
        # Pooling over the sequence dimension to get a single vector
        pooled_output = attention_output.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        # Feed-forward network to produce final output
        output = self.fc(pooled_output)  # Shape: (batch_size, output_dim)
        
        return output

class Embedding(nn.Module):
    def __init__(self, input_shape, args):
        super(Embedding, self).__init__()
        self.args = args
        self.dual_slack = 1e-3

        self.state_embed_net = CrossAttentionNetwork(args.teammate_dim, args.enemy_dim, args.vae_hidden_dim, args.latent_dim)

        # self.state_embed_net = nn.Sequential(nn.Linear(input_shape, args.vae_hidden_dim),
        #                                     nn.ReLU(),
        #                                     nn.Linear(args.vae_hidden_dim, args.vae_hidden_dim ),
        #                                     nn.ReLU(),                                            
        #                                     nn.Linear(args.vae_hidden_dim, args.latent_dim )).to(self.args.device)
        self.rec_emb = nn.Linear(args.latent_dim, args.vae_hidden_dim).to(self.args.device)
        self.rec_S = nn.Linear(args.vae_hidden_dim, args.state_shape).to(self.args.device)
        self.rec_H = nn.Linear(args.vae_hidden_dim, 1).to(self.args.device)

        self.loss_func = nn.MSELoss(reduction="mean")

        self.dual_lam = torch.tensor([0]).cuda()
        self.optim_lam = Adam(params=[self.dual_lam],  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))

        self.optim_net = Adam(params=[
            {'params': self.state_embed_net.parameters()},
            {'params': self.rec_emb.parameters()},
            {'params': self.rec_S.parameters()},
            {'params': self.rec_H.parameters()},
        ],  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
    
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
        enemy_features = global_vector[:, N * teammate_dim:].view(batch_size, M, enemy_dim)
        
        return teammate_features, enemy_features

    def encode(self, inputs):
        teammate_features, enemy_features = self.split_global_vector(inputs, self.args.n_agents, self.args.n_enemy, self.args.state_index[1], self.args.state_index[self.args.n_agents] - self.args.state_index[self.args.n_agents-1])
        return self.state_embed_net(teammate_features, enemy_features)
    
    def decode(self, z):
        rec_hidden = F.relu(self.rec_emb(z)).view(-1, self.args.vae_hidden_dim) 
        rec_s = self.rec_S(F.relu(rec_hidden))
        rec_h = self.rec_H(F.relu(rec_hidden))
        return rec_s, rec_h
    
    def _update_loss_dual_lam(self, lag_penalty):
        log_dual_lam = self.dual_lam
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * lag_penalty.mean()
        self.optim_lam.zero_grad()
        loss_dual_lam.backward()
        self.optim_lam.step()
        return loss_dual_lam

    def update(self, batch, t):
        losses = {}
        dual_lam = self.dual_lam.detach().exp()

        states = batch["state"][:,t]
        next_states = batch["next_state"][:,t]
        z = batch["skills"][:,t]
        returns = batch["return"][:,t]

        phi_s = self.encode(states)
        l2_norm = torch.norm(phi_s, p=2, dim=1)
        target_norm = torch.ones_like(l2_norm)
        loss_norm = self.loss_func(l2_norm, target_norm)
        # rec_s, rec_h = self.decode(phi_s)
        # loss_s = self.loss_func(rec_s, next_states)
        # loss_h = self.loss_func(rec_h, returns)

        phi_next_s = self.encode(next_states)
        delta_phi = phi_next_s - phi_s
        loss_g = (delta_phi * z).sum(dim=1).mean()

        one_dist = torch.ones_like(phi_s[:,0])
        phi_dist = torch.norm(phi_next_s - phi_s, dim=1)
        lag_penalty = one_dist - phi_dist
        # lag_penalty = torch.clamp(lag_penalty, max=self.dual_slack)
        lag_penalty = torch.abs(lag_penalty).mean()
        loss_lag = lag_penalty
        # loss_lag = (dual_lam * lag_penalty).mean()

        # loss = -(loss_g - loss_s - loss_h + loss_lag)
        loss = -(loss_g - loss_lag - loss_norm)
        self.optim_net.zero_grad()
        loss.backward()
        self.optim_net.step()
        losses["loss_g"] = loss_g
        losses["loss_embed"] = loss
        losses["loss_norm"] = loss_norm
        losses["loss_lag"] = loss_lag
        losses["dual_lam"] = dual_lam

        # phi_s = self.encode(states)
        # phi_next_s = self.encode(next_states)
        # one_dist = torch.ones_like(phi_s[:,0])
        # lag_penalty = one_dist - torch.square(phi_next_s - phi_s).mean(dim=1)
        # lag_penalty = torch.clamp(lag_penalty, max=self.dual_slack)
        # self._update_loss_dual_lam(lag_penalty)
        return losses


def compute_value_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['rewards'] = batch['rewards'] - 1.0
    #to do
    # (next_v1, next_v2) = agent.network(batch['next_observations'], batch['goals'], method='target_value')
    (next_v1, next_v2) = agent.network(batch['next_state'], batch['goals'], method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    # (v1_t, v2_t) = agent.network(batch['observations'], batch['goals'], method='target_value')
    #to do
    (v1_t, v2_t) = agent.network(batch['state'], batch['goals'], method='target_value')
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    # (v1, v2) = agent.network(batch['observations'], batch['goals'], method='value', params=network_params)
    (v1, v2) = agent.network(batch['state'], batch['goals'], method='value', params=network_params)
    v = (v1 + v2) / 2

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['expectile']).mean()
    value_loss = value_loss1 + value_loss2

    return value_loss, {
        'value_loss': value_loss,
        'v max': v.max(),
        'v min': v.min(),
        'v mean': v.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
    }