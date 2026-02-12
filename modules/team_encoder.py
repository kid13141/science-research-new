import torch
import torch.nn as nn
import torch.nn.functional as F

class TeamEncoder(nn.Module):
    def __init__(self, args):
        super(TeamEncoder, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.team_latent_dim = args.team_latent_dim # 需要在yaml中定义，如 32
        
        # 1. 特征提取层
        self.fc1 = nn.Linear(self.rnn_hidden_dim, self.team_latent_dim * 2)
        self.fc2 = nn.Linear(self.team_latent_dim * 2, self.team_latent_dim)
        
        # 2. 聚合层 (这里使用 Attention 机制会更好，但 Mean Pooling 是最稳健的基线)
        # 如果需要更高级的，可以换成 Multi-Head Attention
        
    def forward(self, hidden_states):
        """
        Input: hidden_states [Batch, Seq_Len, N_Agents, RNN_Dim]
        Output: team_embedding [Batch, Seq_Len, Latent_Dim]
        """
        # [B, T, N, H] -> [B, T, N, Latent]
        x = F.relu(self.fc1(hidden_states))
        x = self.fc2(x)
        
        # 聚合：对 Agent 维度求平均，得到团队表征
        # [B, T, N, Latent] -> [B, T, Latent]
        team_embedding = x.mean(dim=2) 
        
        # 归一化，这对计算余弦相似度和欧氏距离都很重要
        team_embedding = F.normalize(team_embedding, p=2, dim=-1)
        
        return team_embedding