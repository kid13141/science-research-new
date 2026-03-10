import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ===== 全局配置 =====
LOG_DIR = '/home/songshoucheng/GUF_2025/log_goals' 
SAVE_PNG_DIR = '/home/songshoucheng/GUF_2025/png_state'
DPI = 300
FIG_SIZE_SINGLE = (8, 8)

N_AGENTS = 5 # 已更新为你需要的数量
N_ENEMYS = 1   # 假设有3个敌人，如果只有1个请改为 1
AGENT_FEATURE_DIM = 7
ENEMY_FEATURE_DIM = 5

# 颜色配置
AGENT_GOAL_FACE = 'blue'
AGENT_GOAL_EDGE = 'darkblue'
ENEMY_GOAL_COLOR = 'red'

# 扩展了足够的颜色以支持 11 个 Agent (用于预览图)
AGENT_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'lime'] 
ENEMY_COLOR = 'black'

# 确保图片保存目录存在
os.makedirs(SAVE_PNG_DIR, exist_ok=True)

# 计算 enemy 起始索引
enemy_start_idx = N_AGENTS * AGENT_FEATURE_DIM

# 只获取 goals 的 pkl 文件，以此为基准去寻找对应的 starts 文件
goal_files = glob.glob(os.path.join(LOG_DIR, 'goals_*.pkl'))
goal_files.sort()

if not goal_files:
    print(f"在 {LOG_DIR} 中未找到任何 goals_*.pkl 文件！请检查路径。")
    raise SystemExit(1)

print(f"共找到 {len(goal_files)} 对 PKL 文件，开始批量生成可视化图像...")

# ========== 遍历每个 goals_*.pkl 文件 ==========
for goal_path in goal_files:
    # 解析出当前的 idx，例如从 'goals_1.pkl' 中提取 '1'
    base_name = os.path.basename(goal_path)
    idx_str = base_name.replace('goals_', '').replace('.pkl', '')
    
    # 构建对应的 starts 文件路径
    start_path = os.path.join(LOG_DIR, f'starts_{idx_str}.pkl')
    
    # 加载文件
    try:
        with open(goal_path, 'rb') as f:
            goals = pickle.load(f)
        with open(start_path, 'rb') as f:
            starts = pickle.load(f)
    except Exception as e:
        print(f"读取阶段 {idx_str} 的数据失败，请检查配套的 starts pkl 文件是否存在。错误：{e}")
        continue

    # 兼容不同的数据形状
    if hasattr(goals, 'shape'):
        n_episodes = goals.shape[0]
    else:
        n_episodes = len(goals)

    file_name = f"phase_{idx_str}"
    print(f"\n正在处理 [{file_name}] - 包含 {n_episodes} 个 episodes...")

    # 为当前训练阶段创建一个子文件夹
    current_save_dir = os.path.join(SAVE_PNG_DIR, file_name)
    os.makedirs(current_save_dir, exist_ok=True)

    # 遍历当前 pkl 中的每一条数据
    for ep_idx in range(n_episodes):
        try:
            if len(np.array(goals[ep_idx]).shape) > 1:
                goal_state = np.array(goals[ep_idx][0]).flatten()
                start_state = np.array(starts[ep_idx][0]).flatten()
            else:
                goal_state = np.array(goals[ep_idx]).flatten()
                start_state = np.array(starts[ep_idx]).flatten()
        except IndexError:
            goal_state = np.array(goals[ep_idx]).flatten()
            start_state = np.array(starts[ep_idx]).flatten()

        # ========== 坐标提取核心逻辑 (重塑为矩阵，永不越界) ==========
        # 1. 提取 Agent (Start & Goal)
        agent_goal_2d = goal_state[:enemy_start_idx].reshape(N_AGENTS, AGENT_FEATURE_DIM)
        goal_agent_x = agent_goal_2d[:, 2] * 28 + 16
        goal_agent_y = agent_goal_2d[:, 3] * 28 + 16

        agent_start_2d = start_state[:enemy_start_idx].reshape(N_AGENTS, AGENT_FEATURE_DIM)
        start_agent_x = agent_start_2d[:, 2] * 28 + 16
        start_agent_y = agent_start_2d[:, 3] * 28 + 16

        # 2. 提取 Enemy (Start & Goal)
        enemy_end_idx = enemy_start_idx + N_ENEMYS * ENEMY_FEATURE_DIM
        
        enemy_goal_2d = goal_state[enemy_start_idx:enemy_end_idx].reshape(N_ENEMYS, ENEMY_FEATURE_DIM)
        goal_enemy_x = enemy_goal_2d[:, 1] * 28 + 16
        goal_enemy_y = enemy_goal_2d[:, 2] * 28 + 16

        enemy_start_2d = start_state[enemy_start_idx:enemy_end_idx].reshape(N_ENEMYS, ENEMY_FEATURE_DIM)
        start_enemy_x = enemy_start_2d[:, 1] * 28 + 16
        start_enemy_y = enemy_start_2d[:, 2] * 28 + 16

        # ========== 开始绘图 ==========
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
        ax.set_xlim(0, 32)      
        ax.set_ylim(0, 32)     
        ax.set_title(f'Start vs Goal Positions - {file_name} (Episode {ep_idx})', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)

        # 绘制指向箭头 (从 Start 指向 Goal)
        for i in range(N_AGENTS):
            ax.annotate('', xy=(goal_agent_x[i], goal_agent_y[i]), 
                        xytext=(start_agent_x[i], start_agent_y[i]),
                        arrowprops=dict(arrowstyle='->', color=AGENT_GOAL_FACE, alpha=0.3))

        # === Agent Start: 添加 alpha=0.3 降低饱和度，变淡 ===
        ax.plot(
            start_agent_x, start_agent_y,
            marker='o', markersize=10, linestyle='None',
            markerfacecolor=AGENT_GOAL_FACE,
            markeredgecolor=AGENT_GOAL_EDGE,
            markeredgewidth=1.5,
            alpha=0.3, 
            label='Agent Start'
        )

        # === Agent Goal: 不加 alpha，全饱和度深色 ===
        ax.plot(
            goal_agent_x, goal_agent_y,
            marker='o', markersize=10, linestyle='None',
            markerfacecolor=AGENT_GOAL_FACE,
            markeredgecolor=AGENT_GOAL_EDGE,
            markeredgewidth=1.5,
            label='Agent Goal'
        )

        # === Enemy Start & Goal (同样利用 alpha 区分) ===
        ax.plot(
            start_enemy_x, start_enemy_y,
            marker='*', markersize=14, linestyle='None',
            color=ENEMY_GOAL_COLOR,
            alpha=0.3,
            label='Enemy Start'
        )
        ax.plot(
            goal_enemy_x, goal_enemy_y,
            marker='*', markersize=14, linestyle='None',
            color=ENEMY_GOAL_COLOR,
            markeredgecolor='darkred',
            markeredgewidth=1.0,
            label='Enemy Goal'
        )

        # 图例
        ax.legend(loc='upper right', fontsize=10)

        # 保存
        png_name = f'ep_{ep_idx:04d}.png'
        png_path = os.path.join(current_save_dir, png_name)
        plt.savefig(png_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    print(f"✅ {file_name} 的图片已保存至: {current_save_dir}")

# ================= 预览最后加载的文件中的最后一张图 =================
if n_episodes > 0:
    last_ep = n_episodes - 1
    
    if len(np.array(goals[last_ep]).shape) > 1:
        goal_state = np.array(goals[last_ep][0]).flatten()
        start_state = np.array(starts[last_ep][0]).flatten()
    else:
        goal_state = np.array(goals[last_ep]).flatten()
        start_state = np.array(starts[last_ep]).flatten()

    # 彻底使用 reshape 修复预览越界 bug
    agent_goal_2d = goal_state[:enemy_start_idx].reshape(N_AGENTS, AGENT_FEATURE_DIM)
    goal_agent_x = agent_goal_2d[:, 2] * 28 + 16
    goal_agent_y = agent_goal_2d[:, 3] * 28 + 16

    agent_start_2d = start_state[:enemy_start_idx].reshape(N_AGENTS, AGENT_FEATURE_DIM)
    start_agent_x = agent_start_2d[:, 2] * 28 + 16
    start_agent_y = agent_start_2d[:, 3] * 28 + 16

    enemy_end_idx = enemy_start_idx + N_ENEMYS * ENEMY_FEATURE_DIM
    enemy_goal_2d = goal_state[enemy_start_idx:enemy_end_idx].reshape(N_ENEMYS, ENEMY_FEATURE_DIM)
    goal_enemy_x = enemy_goal_2d[:, 1] * 28 + 16
    goal_enemy_y = enemy_goal_2d[:, 2] * 28 + 16

    enemy_start_2d = start_state[enemy_start_idx:enemy_end_idx].reshape(N_ENEMYS, ENEMY_FEATURE_DIM)
    start_enemy_x = enemy_start_2d[:, 1] * 28 + 16
    start_enemy_y = enemy_start_2d[:, 2] * 28 + 16

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.set_title(f'Preview: {file_name} - Episode {last_ep} (Start -> Goal)', fontsize=14)
    
    # 绘制带具体颜色的起始点、目标点和箭头
    for agent_idx in range(N_AGENTS):
        # 画起点 (加 alpha=0.3 使其变淡)
        ax.scatter(start_agent_x[agent_idx], start_agent_y[agent_idx],
                   color=AGENT_COLORS[agent_idx], alpha=0.3, s=150, marker='o')
        
        # 画终点 (实心，全饱和度)
        ax.scatter(goal_agent_x[agent_idx], goal_agent_y[agent_idx],
                   color=AGENT_COLORS[agent_idx], s=150, marker='o', 
                   label=f'Agent {agent_idx}' if agent_idx < 3 else "") # 限制图例数量防止杂乱
                   
        # 画箭头指示轨迹
        ax.annotate('', xy=(goal_agent_x[agent_idx], goal_agent_y[agent_idx]),
                    xytext=(start_agent_x[agent_idx], start_agent_y[agent_idx]),
                    arrowprops=dict(arrowstyle='->', color=AGENT_COLORS[agent_idx], alpha=0.5, lw=1.5))
                   
    # 画敌人起点和终点
    ax.scatter(start_enemy_x, start_enemy_y, color=ENEMY_COLOR, alpha=0.3, s=200, marker='*')
    ax.scatter(goal_enemy_x, goal_enemy_y, color=ENEMY_COLOR, s=200, marker='*', label='Enemy')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Relative X')
    ax.set_ylabel('Relative Y')
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    if 'DISPLAY' in os.environ or plt.get_backend() != 'agg':
        plt.show()
    else:
        plt.close(fig)
        print("\n无图形界面环境，跳过预览显示。")