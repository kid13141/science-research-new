import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

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

class Hilp_Embedding(nn.Module):
    def __init__(self, input_shape, args):
        super(Hilp_Embedding, self).__init__()
        self.args = args

        # self.state_embed_net_1 = nn.Sequential(nn.Linear(input_shape, args.vae_hidden_dim),
        #                                     nn.ReLU(),
        #                                     nn.Linear(args.vae_hidden_dim, args.vae_hidden_dim ),
        #                                     nn.ReLU(),                                            
        #                                     nn.Linear(args.vae_hidden_dim, args.latent_dim )).to(self.args.device).float()
        # self.state_embed_net_2 = nn.Sequential(nn.Linear(input_shape, args.vae_hidden_dim),
        #                                     nn.ReLU(),
        #                                     nn.Linear(args.vae_hidden_dim, args.vae_hidden_dim ),
        #                                     nn.ReLU(),                                            
        #                                     nn.Linear(args.vae_hidden_dim, args.latent_dim )).to(self.args.device).float()
        self.state_embed_net_1 = CrossAttentionNetwork(args.teammate_dim, args.enemy_dim, args.vae_hidden_dim, args.latent_dim)
        self.state_embed_net_2 = CrossAttentionNetwork(args.teammate_dim, args.enemy_dim, args.vae_hidden_dim, args.latent_dim)
        self.optim_net = Adam(params=[
            {'params': self.state_embed_net_1.parameters()},
            {'params': self.state_embed_net_2.parameters()},
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

        batch_size = global_vector.shape[0]
        total_length = global_vector.shape[1]
        
        # Calculate expected length
        expected_length = N * teammate_dim + M * enemy_dim
        if total_length != expected_length:
            raise ValueError(f"Global vector length ({total_length}) does not match the expected length ({expected_length}).")
        
        # Reshape the global vector
        teammate_features = global_vector[:, :N * teammate_dim].reshape(batch_size, N, teammate_dim)
        enemy_features = global_vector[:, N * teammate_dim:].reshape(batch_size, M, enemy_dim)
        
        return teammate_features, enemy_features

    def encode(self, inputs):
        teammate_features, enemy_features = self.split_global_vector(inputs, self.args.n_agents, self.args.n_enemy, self.args.teammate_dim, self.args.enemy_dim)
        teammate_features = to_tensor(teammate_features)
        enemy_features = to_tensor(enemy_features)
        return self.state_embed_net_1(teammate_features, enemy_features), self.state_embed_net_2(teammate_features, enemy_features)

    
    def forward(self, state, goal):
        z_s_1, z_s_2 = self.encode(state)
        z_g_1, z_g_2 = self.encode(goal)
        squared_dist_1 = ((z_s_1 - z_g_1) ** 2).sum(axis=-1)
        squared_dist_2 = ((z_s_2 - z_g_2) ** 2).sum(axis=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist_1, min=1e-6))
        v2 = -torch.sqrt(torch.clamp(squared_dist_2, min=1e-6))
        return v1, v2


class Hilp_Multi_Embedding(nn.Module):
    def __init__(self, input_shape, args):
        super(Hilp_Multi_Embedding, self).__init__()
        self.args = args

        self.state_embed_net_1 = CrossAttentionNetwork(args.teammate_dim, args.enemy_dim, args.vae_hidden_dim, args.latent_dim)
        self.state_embed_net_2 = CrossAttentionNetwork(args.teammate_dim, args.enemy_dim, args.vae_hidden_dim, args.latent_dim)
        self.state_embed_net_3 = CrossAttentionNetwork(args.teammate_dim, args.enemy_dim, args.vae_hidden_dim, args.latent_dim)
        self.state_embed_net_4 = CrossAttentionNetwork(args.teammate_dim, args.enemy_dim, args.vae_hidden_dim, args.latent_dim)
        self.optim_net = Adam(params=[
            {'params': self.state_embed_net_1.parameters()},
            {'params': self.state_embed_net_2.parameters()},
            {'params': self.state_embed_net_3.parameters()},
            {'params': self.state_embed_net_4.parameters()},
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
        teammate_features, enemy_features = self.split_global_vector(inputs, self.args.n_agents, self.args.n_enemy, self.args.teammate_dim, self.args.enemy_dim)
        return self.state_embed_net_1(teammate_features, enemy_features), self.state_embed_net_2(teammate_features, enemy_features), self.state_embed_net_3(teammate_features, enemy_features), self.state_embed_net_4(teammate_features, enemy_features)
    
    def forward(self, state, goal):
        z_s_1, z_s_2, z_s_3, z_s_4 = self.encode(state)
        z_g_1, z_g_2, z_g_3, z_g_4 = self.encode(goal)
        squared_dist_1 = ((z_s_1 - z_g_1) ** 2).sum(axis=-1)
        squared_dist_2 = ((z_s_2 - z_g_2) ** 2).sum(axis=-1)
        squared_dist_3 = ((z_s_3 - z_g_3) ** 2).sum(axis=-1)
        squared_dist_4 = ((z_s_4 - z_g_4) ** 2).sum(axis=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist_1, min=1e-6))
        v2 = -torch.sqrt(torch.clamp(squared_dist_2, min=1e-6))
        v3 = -torch.sqrt(torch.clamp(squared_dist_3, min=1e-6))
        v4 = -torch.sqrt(torch.clamp(squared_dist_4, min=1e-6))
        return v1, v2, v3, v4


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        标准的 MLP 网络，用于将单一智能体的状态映射到潜在空间。
        """
        super(MLPNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Hilp2_Embedding(nn.Module):
    def __init__(self, input_shape, args):
        super(Hilp2_Embedding, self).__init__()
        self.args = args
        
        # 输入维度仅为智能体自身的特征维度
        self.net_1 = MLPNetwork( self.args.teammate_dim, args.vae_hidden_dim, args.latent_dim).to(args.device)
        self.net_2 = MLPNetwork( self.args.teammate_dim, args.vae_hidden_dim, args.latent_dim).to(args.device)

        self.optim_net = Adam(params=[
            {'params': self.net_1.parameters()},
            {'params': self.net_2.parameters()},
        ], lr=args.lr, weight_decay=getattr(args, "weight_decay", 1e-5))


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
        """
        编码过程：
        1. 提取自我特征
        2. 通过 MLP 映射到 latent space
        """
        # 1. 提取自我特征 (Batch, ..., Self_Dim)
        teammate_features, enemy_features = self.split_global_vector(inputs,self.args.n_agents, self.args.n_enemy, self.args.teammate_dim, self.args.enemy_dim)
        
        # 2. 这里的 inputs 可能是 (Batch, N, Dim) 也可能是 (Batch*N, Dim)
        # MLP 可以直接处理多维 Input，只要最后一维对得上
        
        z1 = self.net_1(teammate_features)
        z2 = self.net_2(teammate_features)
        
        return z1, z2
    
    def forward(self, state, goal):
        """
        计算 HILP 势能
        """
        # 编码 (自动内部截取 Self 特征)
        z_s_1, z_s_2 = self.encode(state)
        z_g_1, z_g_2 = self.encode(goal)
        
        # 计算欧氏距离平方 ||z_s - z_g||^2
        squared_dist_1 = ((z_s_1 - z_g_1) ** 2).sum(dim=-1)
        squared_dist_2 = ((z_s_2 - z_g_2) ** 2).sum(dim=-1)
        
        # 计算势能 V = -distance
        # 加 1e-6 防止梯度爆炸
        v1 = -torch.sqrt(torch.clamp(squared_dist_1, min=1e-6))
        v2 = -torch.sqrt(torch.clamp(squared_dist_2, min=1e-6))
        
        return v1, v2