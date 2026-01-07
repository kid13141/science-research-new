import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# ===== 全局配置（便于修改和维护）=====
LOG_DIR = '/home/songshoucheng/GUF_2025/logs_goal_state'
SAVE_PNG_DIR = '/home/songshoucheng/GUF_2025/png_state'
DPI = 300
FIG_SIZE_SINGLE = (8, 8)
N_AGENTS = 5
AGENT_FEATURE_DIM = 7
ENEMY_FEATURE_DIM = 5
AGENT_COLORS = ['red', 'blue', 'green', 'orange', 'purple']
ENEMY_COLOR = 'black'

# 确保图片保存目录存在
os.makedirs(SAVE_PNG_DIR, exist_ok=True)

# 加载指定名称的pkl文件（增加异常处理）
def load_pkl(file_name):
    """加载指定名称的pkl文件，包含异常处理"""
    file_path = os.path.join(LOG_DIR, f'{file_name}.pkl')
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PKL文件不存在：{file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError as e:
        print(f"错误：{e}")
        raise SystemExit(1)
    except pickle.UnpicklingError:
        print(f"错误：{file_path} 不是合法的PKL文件，无法反序列化")
        raise SystemExit(1)
    except Exception as e:
        print(f"加载{file_path}失败：{str(e)}")
        raise SystemExit(1)

# 读取所有pkl文件
print("开始加载PKL文件...")
dis = load_pkl('dis')
goals = load_pkl('goals')
reward = load_pkl('reward')
states = load_pkl('states')
print(states[0][0])
print("所有pkl文件已成功加载到内存！")

# 推断 episode 数量
if hasattr(states, 'shape'):
    n_episodes = states.shape[0]
else:
    n_episodes = len(states)

# 验证 goals 长度
if len(goals) != n_episodes:
    raise ValueError(f"错误：goals 数量 ({len(goals)}) 与 states 的 episode 数量 ({n_episodes}) 不一致！")

print(f"检测到 {n_episodes} 个 episodes。")

# 计算 enemy 起始索引（假设所有状态结构一致）
enemy_start_idx = N_AGENTS * AGENT_FEATURE_DIM

# ========== 每个 episode 一张图：统一 marker，init 虚线边框，goal 实线边框 ==========
print("\n========== 开始生成：Agent 圆圈（init 虚线 / goal 实线），Enemy 五角星 ==========")

# 颜色定义
AGENT_INIT_FACE = 'lightblue'
AGENT_INIT_EDGE = 'steelblue'
AGENT_GOAL_FACE = 'blue'
AGENT_GOAL_EDGE = 'darkblue'

ENEMY_INIT_COLOR = 'lightcoral'
ENEMY_GOAL_COLOR = 'red'

for ep_idx in range(n_episodes):
    # 获取并展平状态
    state_t0 = np.array(states[ep_idx][0]).flatten()
    goal_state = np.array(goals[ep_idx][0]).flatten()

    # 提取坐标
    init_agent_x = state_t0[2::AGENT_FEATURE_DIM][:-1]*28+16
    init_agent_y = state_t0[3::AGENT_FEATURE_DIM][:-1]*28+16
    goal_agent_x = goal_state[2::AGENT_FEATURE_DIM][:-1]*28+16
    goal_agent_y = goal_state[3::AGENT_FEATURE_DIM][:-1]*28+16

    init_enemy_x = float(state_t0[enemy_start_idx + 1])*28+16
    init_enemy_y = float(state_t0[enemy_start_idx + 2])*28+16
    goal_enemy_x = float(goal_state[enemy_start_idx + 1])*28+16
    goal_enemy_y = float(goal_state[enemy_start_idx + 2])*28+16

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.set_xlim(0, 32)      
    ax.set_ylim(0, 32)     
    ax.set_title(f'Episode {ep_idx}: Initial vs Goal Positions', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)

    # === Agent: 使用 plot 绘制带边框样式的圆圈 ===
    # Agent init: 浅填充 + 虚线边框
    ax.plot(
        init_agent_x, init_agent_y,
        marker='o', markersize=8, linestyle='None',
        markerfacecolor=AGENT_INIT_FACE,
        markeredgecolor=AGENT_INIT_EDGE,
        markeredgewidth=1.5,
        label='Agent init'
    )
    # Agent goal: 深填充 + 实线边框
    ax.plot(
        goal_agent_x, goal_agent_y,
        marker='o', markersize=8, linestyle='None',
        markerfacecolor=AGENT_GOAL_FACE,
        markeredgecolor=AGENT_GOAL_EDGE,
        markeredgewidth=1.5,
        label='Agent goal'
    )

    # === Enemy: 五角星（'*'），init 浅色，goal 深色 ===
    ax.plot(
        init_enemy_x, init_enemy_y,
        marker='*', markersize=12, linestyle='None',
        color=ENEMY_INIT_COLOR,
        markeredgecolor='lightcoral',
        markeredgewidth=0.8,
        label='Enemy init'
    )
    ax.plot(
        goal_enemy_x, goal_enemy_y,
        marker='*', markersize=12, linestyle='None',
        color=ENEMY_GOAL_COLOR,
        markeredgecolor='red',
        markeredgewidth=0.8,
        label='Enemy goal'
    )

    # 图例（简洁四项）
    ax.legend(loc='upper right', fontsize=9)

    # 保存
    png_name = f'episode_{ep_idx:04d}.png'
    png_path = os.path.join(SAVE_PNG_DIR, png_name)
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

print(f"\n✅ 成功生成并保存 {n_episodes} 张简化版图像到: {SAVE_PNG_DIR}")

# （可选）如果环境支持，显示最后一张图用于快速检查
if n_episodes > 0:
    last_ep = n_episodes - 1
    if hasattr(states, 'shape') and len(states.shape) == 3:
        state_t0 = states[last_ep, 0, :]
    else:
        state_t0 = np.array(states[last_ep][0])
    goal_state = np.array(goals[last_ep])

    init_agent_x = state_t0[2::AGENT_FEATURE_DIM]
    init_agent_y = state_t0[3::AGENT_FEATURE_DIM]
    init_enemy_x = state_t0[enemy_start_idx + 1]
    init_enemy_y = state_t0[enemy_start_idx + 2]

    goal_agent_x = goal_state[2::AGENT_FEATURE_DIM]
    goal_agent_y = goal_state[3::AGENT_FEATURE_DIM]
    goal_enemy_x = goal_state[enemy_start_idx + 1]
    goal_enemy_y = goal_state[enemy_start_idx + 2]

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.set_title(f'Preview: Episode {last_ep} (Initial vs Goal)', fontsize=14)
    for agent_idx in range(N_AGENTS):
        ax.scatter(init_agent_x[agent_idx], init_agent_y[agent_idx],
                   color=AGENT_COLORS[agent_idx], s=120, marker='o')
        ax.scatter(goal_agent_x[agent_idx], goal_agent_y[agent_idx],
                   color=AGENT_COLORS[agent_idx], s=120, marker='x')
        ax.annotate('', xy=(goal_agent_x[agent_idx], goal_agent_y[agent_idx]),
                    xytext=(init_agent_x[agent_idx], init_agent_y[agent_idx]),
                    arrowprops=dict(arrowstyle='->', color=AGENT_COLORS[agent_idx], alpha=0.6))
    ax.scatter(init_enemy_x, init_enemy_y, color=ENEMY_COLOR, s=150, marker='D')
    ax.scatter(goal_enemy_x, goal_enemy_y, color=ENEMY_COLOR, s=150, marker='*')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Relative X')
    ax.set_ylabel('Relative Y')
    
    if 'DISPLAY' in os.environ or plt.get_backend() != 'agg':
        plt.show()
    else:
        plt.close(fig)
        print("无图形界面环境，跳过预览显示")




