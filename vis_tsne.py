# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# from os.path import dirname, abspath
# import os

# def plot_tsne_with_return_tiers(data_dir="results/tsne_data", save_path="tsne_clustering.pdf"):
#     # 1. 加载数据
#     features = np.load(f"{data_dir}/traj_reprs.npy") # 形状: [N, 32]
#     returns = np.load(f"{data_dir}/traj_returns.npy")  # 形状: [N]

#     # 2. 动态计算分位数阈值 (P33 和 P67)
#     p33 = np.percentile(returns, 33)
#     p67 = np.percentile(returns, 67)
    
#     print(f"Return 划分阈值: Low(<= {p33:.2f}), Medium({p33:.2f} - {p67:.2f}), High(> {p67:.2f})")

#     # 3. 根据范围分配标签
#     label_categories = []
#     for ret in returns:
#         if ret <= p33:
#             label_categories.append("Low Return")
#         elif ret <= p67:
#             label_categories.append("Medium Return")
#         else:
#             label_categories.append("High Return")

#     # 4. 执行 t-SNE 降维
#     print("正在运行 t-SNE 降维...")
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
#     features_2d = tsne.fit_transform(features)

#     # 5. 学术绘图设置
#     sns.set_theme(style="whitegrid")
#     plt.rcParams.update({'font.family': 'serif', 'font.size': 14, 'pdf.fonttype': 42})
#     plt.figure(figsize=(8, 6))
    
#     # 定义色带 (红:低, 黄:中, 绿:高)
#     palette = {"Low Return": "#e74c3c", "Medium Return": "#f1c40f", "High Return": "#2ecc71"}
    
#     sns.scatterplot(
#         x=features_2d[:, 0], 
#         y=features_2d[:, 1],
#         hue=label_categories,
#         hue_order=["High Return", "Medium Return", "Low Return"], # 图例显示顺序
#         palette=palette,
#         s=80, alpha=0.85, edgecolor='w', linewidth=0.5
#     )

#     plt.title("Trajectory Representations Clustered by Episodic Return", fontsize=16, pad=15)
#     plt.xlabel("t-SNE Dimension 1", fontsize=14)
#     plt.ylabel("t-SNE Dimension 2", fontsize=14)
#     plt.legend(title="Return Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     sns.despine()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"可视化图像已保存至 {save_path}")
#     plt.show()

# if __name__ == "__main__":
#     data_path = os.path.join(dirname(dirname(abspath(__file__))), "results", "tsne_data")
#     plot_tsne_with_return_tiers(data_dir=data_path)



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from os.path import dirname, abspath

def plot_evolution_tsne(data_dir="results/tsne_data", save_path="tsne_clustering.pdf"):
    # 1. 确保数据存在并加载
    if not os.path.exists(f"{data_dir}/evo_traj_reprs.npy"):
        raise FileNotFoundError(f"找不到数据文件，请检查 {data_dir} 路径是否正确。")
        
    features = np.load(f"{data_dir}/evo_traj_reprs.npy") # 形状: [N, 32]
    step_labels = np.load(f"{data_dir}/evo_step_labels.npy") # 形状: [N]
    
    unique_steps = np.sort(np.unique(step_labels))
    print(f"已加载数据: 共 {len(features)} 条轨迹。")
    print(f"包含的训练节点: {unique_steps}")

    # 2. t-SNE 降维
    print("正在运行 t-SNE 降维 (这可能需要一两分钟)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features)

    # 3. 学术绘图设置 (使用 Seaborn 的底层样式，但用 Matplotlib 绘制)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'serif', 
        'font.size': 14, 
        'pdf.fonttype': 42  # 确保导出的 PDF 字体在 Illustrator 中可编辑
    })
    
    fig, ax = plt.subplots(figsize=(10, 6.5)) # 稍微加宽一点以容纳 Colorbar

    # 4. 绘制渐变色散点图 (使用 plt.scatter 完美支持连续色映射)
    # cmap 推荐: 'viridis' (紫->绿->黄), 'plasma' (紫->红->黄), 或 'coolwarm' (冷暖对比)
    scatter = ax.scatter(
        features_2d[:, 0], 
        features_2d[:, 1],
        c=step_labels,           # 关键：直接传入数值型的步数作为颜色映射基准
        cmap="viridis",          # 连续渐变色谱
        alpha=0.85,              # 透明度，防止点重叠时看不清
        s=60,                    # 点的大小
        edgecolors='white',      # 给点加个白边，提升质感
        linewidths=0.4
    )

    # 5. 添加并格式化连续颜色条 (Colorbar)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Training Environment Steps', rotation=270, labelpad=25, fontsize=14, fontweight='bold')
    
    # 将 Colorbar 的刻度精确对齐到你收集的 Checkpoints，并格式化为 k/M 结尾
    cbar.set_ticks(unique_steps)
    
    formatted_labels = []
    for val in unique_steps:
        if val >= 1e6:
            formatted_labels.append(f"{val/1e6:.1f}M")
        elif val >= 1e3:
            formatted_labels.append(f"{val/1e3:.0f}k")
        else:
            formatted_labels.append(str(int(val)))
            
    cbar.set_ticklabels(formatted_labels)
    cbar.ax.tick_params(labelsize=12)

    # 6. 图表标题与坐标轴美化
    plt.title("Evolution of Team Behavior Trajectories in Latent Space", fontsize=16, pad=15, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    
    # 移除顶部和右侧的边框线条
    sns.despine()
    plt.tight_layout()
    
    # 保存与展示
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"演化可视化图像已成功保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    data_path = os.path.join(dirname(dirname(abspath(__file__))), "results", "tsne_data")
    plot_evolution_tsne(data_dir=data_path, save_path="tsne_clustering.pdf")