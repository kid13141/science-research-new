import torch
import numpy as np

class HistoryBuffer:
    def __init__(self, capacity, k, latent_dim, device):
        self.capacity = capacity
        self.k = k
        self.device = device
        self.latent_dim = latent_dim
        
        # 初始化 Buffer，预分配显存
        self.buffer = torch.zeros(capacity, latent_dim).to(device)
        self.ptr = 0
        self.size = 0
        self.full = False
        
    def add(self, z_team):
        """
        z_team: [Batch * Seq_Len, Latent_Dim]
        注意：传入前需要 detach() 并且 reshape 成二维
        """
        n_samples = z_team.shape[0]
        if n_samples == 0:
            return

        # 简单的 FIFO 循环覆盖
        indices = torch.arange(self.ptr, self.ptr + n_samples) % self.capacity
        self.buffer[indices] = z_team
        
        self.ptr = (self.ptr + n_samples) % self.capacity
        if self.size < self.capacity:
            self.size = min(self.size + n_samples, self.capacity)
            if self.size == self.capacity:
                self.full = True

    def compute_entropy_reward(self, z_query):
        """
        计算 z_query 在 Buffer 中的 k-近邻平均距离
        z_query: [Batch, Seq_Len, Latent_Dim]
        Return: reward [Batch, Seq_Len, 1]
        """
        b, t, d = z_query.shape
        flat_query = z_query.view(-1, d) # [B*T, D]
        
        if self.size < self.k:
            # 库还没填满，返回 0 奖励
            return torch.zeros(b, t, 1).to(self.device)

        # 获取当前有效的 Buffer 内容
        current_buffer = self.buffer[:self.size] if not self.full else self.buffer

        # 计算距离矩阵 [B*T, Buffer_Size]
        # 注意：如果 Buffer 很大 (如 10000)，直接算 cdist 可能会爆显存
        # 建议分块计算，或者使用 faiss (如果允许引入新库)。这里演示 PyTorch 原生实现。
        
        # 优化：为了速度，只随机采样 Buffer 的一部分作为参考 (例如 1000 个)
        # 这是一种常用的近似技巧
        if self.size > 1000:
            indices = torch.randint(0, self.size, (1000,)).to(self.device)
            ref_buffer = current_buffer[indices]
        else:
            ref_buffer = current_buffer

        # 计算欧氏距离
        dists = torch.cdist(flat_query, ref_buffer, p=2) # [B*T, Sample_Size]
        
        # 取最近的 k 个
        topk_dists, _ = dists.topk(self.k, largest=False, dim=1)
        
        # 计算平均距离作为熵的估计
        avg_dist = topk_dists.mean(dim=1, keepdim=True) # [B*T, 1]
        
        # Log 缩放
        reward = torch.log(avg_dist + 1.0)
        
        return reward.view(b, t, 1)