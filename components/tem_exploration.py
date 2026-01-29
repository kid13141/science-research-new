import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim=64):
        super(ContrastiveTrajectoryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. 特征提取器 (GRU)
        # input_dim = obs_dim + action_dim (or state_dim + action_dim)
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # 2. 投影头 (Projection Head) - 对比学习的关键
        # 将 hidden state 映射到用于计算相似度的空间
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x, h_in=None):
        """
        x: [Batch, Seq_Len, Input_Dim]
        h_in: [1, Batch, Hidden_Dim]
        """
        # GRU 输出 output: [Batch, Seq_Len, Hidden], h_n: [1, Batch, Hidden]
        output, h_n = self.rnn(x, h_in)
        
        # 我们对每一个时间步的 output 都做投影，以便计算 Dense Reward
        # projected: [Batch, Seq_Len, Proj_Dim]
        projected = self.projection(output)
        
        # 对 Embedding 进行归一化 (Cosine Similarity 需要)
        projected = F.normalize(projected, p=2, dim=-1)
        
        return projected, h_n

class TEMExplorer:
    def __init__(self, args, input_dim):
        self.args = args
        self.device = args.device
        self.hidden_dim = args.rnn_hidden_dim
        self.proj_dim = 64
        self.temperature = 0.1  # InfoNCE 的温度系数
        self.k = 3              # k-NN 的 k
        
        # 初始化 Encoder
        self.encoder = ContrastiveTrajectoryEncoder(input_dim, self.hidden_dim, self.proj_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.lr)
        
        # 维护一个小的 History Buffer 用于计算 k-NN (可选，也可以直接用当前 Batch 计算)
        self.history_embeddings = None

    def get_aug_views(self, traj_batch):
        """
        数据增强：生成两个视角的轨迹
        方法：随机 Masking + 高斯噪声 (时序上保持一致，特征上扰动)
        """
        B, T, D = traj_batch.shape
        
        # View 1: 添加噪声
        noise1 = torch.randn_like(traj_batch) * 0.05
        view1 = traj_batch + noise1
        
        # View 2: 随机 Mask 掉部分特征 (模拟传感器失效)
        mask = torch.rand_like(traj_batch) > 0.1 # 10% 的特征被丢弃
        view2 = traj_batch * mask.float()
        
        return view1, view2

    def train_encoder(self, batch):
        """
        修正版：加入 Mask 机制，过滤 Padding，防止 Loss 坍塌和显存爆炸
        """
        # 1. 准备数据
        obs = batch["obs"]
        actions = batch["actions_onehot"]
        # 获取 mask: [Batch, Max_Seq, 1] -> 1 表示有效数据, 0 表示填充
        mask = batch["filled"].float() 
        
        inputs = torch.cat([obs, actions], dim=-1)
        bs, max_t, n_agents, feat_dim = inputs.shape
        
        # Flatten agents
        inputs = inputs.reshape(bs * n_agents, max_t, feat_dim)
        
        # 这里的 mask 也要扩展到 n_agents 维度并展平
        # mask: [Batch, Max_Seq, 1] -> [Batch, n_agents, Max_Seq, 1] -> [Batch*Agents, Max_Seq]
        mask = mask.unsqueeze(1).repeat(1, n_agents, 1, 1).reshape(bs * n_agents, max_t)

        # 2. 数据增强 (只对 valid 数据增强其实更省资源，但这里为了代码简单，对全量做也行)
        view1, view2 = self.get_aug_views(inputs)

        # 3. 前向传播
        z1, _ = self.encoder(view1) # [Batch*Agents, Max_Seq, Proj_Dim]
        z2, _ = self.encoder(view2)

        # 4. 关键修正：利用 Mask 筛选有效数据
        # ------------------------------------------------------
        # 我们只取 mask == 1 的时间步来计算 Loss
        # mask.bool() shape: [Batch*Agents, Max_Seq]
        valid_indices = mask.bool()
        
        # 选出有效数据的 embedding
        # z1_valid shape: [Total_Valid_Steps, Proj_Dim]
        # Total_Valid_Steps << Batch*Agents*Max_Seq (大大节省显存)
        z1_valid = z1[valid_indices]
        z2_valid = z2[valid_indices]
        
        # 如果当前 batch 全是 padding (极少见但要防备)，直接返回 0
        if z1_valid.shape[0] < 2:
            return 0.0
            
        # ------------------------------------------------------

        # 5. 计算 InfoNCE Loss (仅针对有效数据)
        
        # 相似度矩阵: [Total_Valid, Total_Valid]
        # 这里的矩阵会比原来小很多，且全是真实数据
        logits = torch.matmul(z1_valid, z2_valid.T) / self.temperature
        
        # 减去最大值以提高数值稳定性 (LogSumExp Trick)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Labels: 对角线
        labels = torch.arange(logits.shape[0]).to(self.device)
        
        loss = F.cross_entropy(logits, labels)

        # 6. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 加上梯度裁剪 (可选，防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        
        self.optimizer.step()

        return loss.item()

    def compute_intrinsic_reward(self, batch):
        """
        计算 k-NN 探索奖励
        """
        with torch.no_grad():
            obs = batch["obs"]
            actions = batch["actions_onehot"]
            inputs = torch.cat([obs, actions], dim=-1)
            
            bs, max_t, n_agents, feat_dim = inputs.shape
            inputs_flat = inputs.reshape(-1, max_t, feat_dim)
            
            # 获取当前 Batch 的 Embedding
            # embeddings: [Batch*Agents, Max_Seq, Proj_Dim]
            embeddings, _ = self.encoder(inputs_flat)
            
            # Reshape 回来以便计算: [Batch, n_agents, Max_Seq, Proj_Dim]
            embeddings = embeddings.reshape(bs, n_agents, max_t, -1)
            
            # --- 计算 k-NN 奖励 ---
            # 策略：对于每个样本 (b, a, t)，计算它与 batch 内其他样本的距离
            # 注意：实际应用中，应该维护一个外部的 memory buffer，这里为了 demo 简化为 batch 内互算
            
            # [Total_Samples, Proj_Dim]
            all_points = embeddings.reshape(-1, self.proj_dim)
            
            # 计算欧氏距离矩阵: [Total, Total]
            # 注意：这在大 Batch 下显存消耗大，可以使用 chunk 计算
            dists = torch.cdist(all_points, all_points, p=2)
            
            # 获取第 k+1 近的距离 (第1近是自己，距离为0，所以取 k+1)
            # topk 返回的是最大的，所以我们取负距离求 topk，或者用 kthvalue
            # values: [Total, k+1]
            k_dists, _ = torch.topk(dists, k=self.k + 1, dim=-1, largest=False)
            
            # 取第 k 个邻居的距离
            kth_dist = k_dists[:, -1] 
            
            # 奖励 = log(distance + epsilon)
            intrinsic_reward = torch.log(kth_dist + 1e-6)
            
            # Reshape 回 PyMARL 格式: [Batch, Max_Seq, n_agents]
            intrinsic_reward = intrinsic_reward.reshape(bs, n_agents, max_t).permute(0, 2, 1)
            
            # 归一化奖励 (可选，但推荐)
            intrinsic_reward = (intrinsic_reward - intrinsic_reward.mean()) / (intrinsic_reward.std() + 1e-8)
            
            return intrinsic_reward