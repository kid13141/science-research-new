import torch
import numpy as np

class HistoryBuffer:
    def __init__(self, capacity, k, latent_dim, device):
        self.capacity = capacity
        self.k = k
        self.device = device
        self.latent_dim = latent_dim
        
        self.buffer = torch.zeros(capacity, latent_dim).to(device)
        self.size = 0  # 丢弃了 self.ptr，因为不再是 FIFO 队列
        self.full = False
        
    def add(self, z_team, epsilon=0.05):
        """
        z_team: [N, Latent_Dim]
        epsilon: 距离阈值，决定新样本是否足够“新颖”
        """
        if z_team.shape[0] == 0:
            return

        for i in range(z_team.shape[0]):
            z = z_team[i]
            
            # 如果 Buffer 为空，直接加入第一个
            if self.size == 0:
                self.buffer[0] = z
                self.size = 1
                continue
                
            current_buf = self.buffer[:self.size]
            
            # ==========================================
            # 1. 新奇度检查 (Novelty Check)
            # ==========================================
            # 计算当前样本 z 到已有 Buffer 所有样本的欧式距离
            dists = torch.norm(current_buf - z, dim=1) # [size]
            min_dist = dists.min()
            
            if min_dist < epsilon:
                # 距离小于阈值，说明该特征已被现有样本代表，直接丢弃
                continue
                
            # ==========================================
            # 2. 如果 Buffer 未满，直接追加
            # ==========================================
            if self.size < self.capacity:
                self.buffer[self.size] = z
                self.size += 1
                if self.size == self.capacity:
                    self.full = True
            
            # ==========================================
            # 3. Buffer 已满，基于密度替换 (Density-based)
            # ==========================================
            else:
                # 为了计算效率，随机抽取一部分样本（如 256 个）来评估密度
                eval_size = min(self.size, 256)
                idx = torch.randperm(self.size, device=self.device)[:eval_size]
                subset = self.buffer[idx]
                
                # 计算这 eval_size 个样本与整个 Buffer 的距离矩阵 [eval_size, size]
                sub_dists = torch.cdist(subset.unsqueeze(0), current_buf.unsqueeze(0)).squeeze(0)
                
                # 排除自己到自己的距离 (将 0 替换为无穷大，防止被误认为是最小距离)
                sub_dists.fill_diagonal_(float('inf'))
                
                # 找到 subset 中，与其最近邻距离最小的那个样本 
                # (距离最小，意味着它处于最拥挤/密度最高的冗余区域)
                min_sub_dists, _ = sub_dists.min(dim=1)
                victim_subset_idx = min_sub_dists.argmin()
                victim_idx = idx[victim_subset_idx]
                
                # 用新颖的样本 z 替换掉这个拥挤区域的冗余样本
                self.buffer[victim_idx] = z

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