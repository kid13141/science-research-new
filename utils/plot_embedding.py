import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path

def plot_tsne_static(encoder, batch, save_path):
    encoder.eval()
    
    # 1. 处理保存路径 (确保文件夹存在)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # 2. 准备数据并移动到正确的设备 (GPU/CPU)
        device = next(encoder.parameters()).device
        
        obs = batch["obs"].to(device)
        actions = batch["actions_onehot"].to(device)
        # 获取 mask，用于判断哪些步是填充的 (1为有效，0为填充)
        # shape: [Batch, Max_T, 1] -> [Batch, Max_T]
        mask = batch["filled"].float().to(device).squeeze(-1) 
        
        inputs = torch.cat([obs, actions], dim=-1)
        bs, max_t, n_agents, feat_dim = inputs.shape
        
        # Reshape 为 [Batch * Agents, Max_T, Feat_Dim]
        inputs_flat = inputs.reshape(-1, max_t, feat_dim)
        
        # 3. 获取 Embedding
        # embeddings: [Batch * Agents, Max_T, Proj_Dim]
        embeddings, _ = encoder(inputs_flat) 
    
        
        # 将 mask 扩展到 agent 维度并展平: [Batch * Agents, Max_T]
        mask_flat = mask.unsqueeze(2).repeat(1, 1, n_agents).reshape(-1, max_t)
        
        # 计算每个样本的真实长度 (sum mask) - 1 得到索引
        # lengths: [Batch * Agents]
        lengths = mask_flat.sum(dim=1).long() - 1
        # 防止长度为-1 (空序列保护)
        lengths = torch.clamp(lengths, min=0)
        
        # 利用 gather 取出每个序列最后有效的一步
        # indices: [Batch * Agents, 1, Proj_Dim]
        indices = lengths.view(-1, 1, 1).repeat(1, 1, embeddings.shape[-1])
        final_embeddings = torch.gather(embeddings, 1, indices).squeeze(1)
        
        # 转为 Numpy: [Batch * Agents, Proj_Dim]
        data_points = final_embeddings.cpu().numpy()

    # --- 4. t-SNE 降维 ---
    print(f"Running t-SNE on {data_points.shape[0]} points...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    data_2d = tsne.fit_transform(data_points)

    # --- 5. 绘图 ---
    plt.figure(figsize=(10, 8))
    
    # 构造标签：Agent ID
    agent_ids = np.tile(np.arange(n_agents), bs)
    
    sns.scatterplot(
        x=data_2d[:, 0], 
        y=data_2d[:, 1],
        hue=agent_ids, 
        palette="deep",
        s=60, 
        alpha=0.8,
        legend="full"
    )

    plt.title("Trajectory Embeddings t-SNE (Last Valid Step)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Agent ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout() # 防止图例被切掉

    plt.savefig(str(save_path), dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")