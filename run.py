import datetime
import os
import pprint
import time
import threading
import torch as th
import numpy as np
import pickle
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.lines as mlines

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from modules.TrajBuffer import TrajectoryBuffer
import logging 


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")


    alg_name = args.name
    if str(args.env).startswith('sc2'):
        unique_token = "{}_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), alg_name, args.env, args.env_args['map_name'])
    else:
        unique_token = "{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), alg_name, args.env)
    
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    states = []
    goals = []
    distances = []
    rewards = []
    for _ in range(args.test_nepisode):
        state, goal, dis, reward = runner.run(test_mode=True, diff_return = 1.0, v_thed = 1.0)
        states.append(state)
        goals.append(goal)
        distances.append(dis)
        rewards.append(reward)

    with open('logs_goal_state/states.pkl', 'wb') as f:
        pickle.dump(states, f)
    with open('logs_goal_state/goals.pkl', 'wb') as f:
        pickle.dump(goals, f)
    with open('logs_goal_state/dis.pkl', 'wb') as f:
        pickle.dump(distances, f)
    with open('logs_goal_state/reward.pkl', 'wb') as f:
        pickle.dump(rewards, f)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_enemy = env_info["n_enemy"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.max_length = env_info["episode_limit"]

    # Default/Base scheme
    if "multi" in args.runner:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "goal":{"vshape": env_info["state_shape"]},
            "next_state": {"vshape": env_info["state_shape"]},
            "return": {"vshape": (1,)},
            "factor_reward": {"vshape": (1,), "group": "agents"},
            "cur_return": {"vshape": (1,)},
            "multi_goals": {"vshape": (args.horizon, env_info["state_shape"])},
            "death": {"vshape": (1,)},
            "hilp_vals": {"vshape": (1,), "group": "agents"},
            "lock_states":{"vshape": (1,), "group": "agents"},
        }
    else:
        scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "goal":{"vshape": env_info["state_shape"]},
        "next_state": {"vshape": env_info["state_shape"]},
        "return": {"vshape": (1,)},
        "factor_reward": {"vshape": (1,), "group": "agents"},
        "cur_return": {"vshape": (1,)},
        "death": {"vshape": (1,)},
        "total_reward": {"vshape": (1,)},
        "hilp_vals": {"vshape": (1,), "group": "agents"},
        "lock_states":{"vshape": (1,), "group": "agents"},
        }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # good_buffer = TrajectoryBuffer(N=64, H=args.max_length+1, device="cuda")

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    runner.buffer = buffer

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        timesteps.sort() # 确保步数从小到大排序

        if getattr(args, "vis_state", False):
            if len(timesteps) == 0:
                logger.console_logger.info("未找到任何可用的检查点进行可视化！")
                return
            
            # 设置开始可视化的起始步数 
            start_vis_step = 100311
            valid_timesteps = [t for t in timesteps if t >= start_vis_step]
            if len(valid_timesteps) == 0:
                logger.console_logger.info(f"未找到大于等于 {start_vis_step} 步的检查点进行可视化！")
                return
            

            num_to_vis = min(4, len(valid_timesteps)) # 假设你想看 4 个阶段
            indices = np.linspace(0, len(valid_timesteps) - 1, num_to_vis, dtype=int)
            selected_steps = [valid_timesteps[i] for i in indices]
            
            # 构造输入给 visualize_trajectory_subplots_pdf 的字典
            checkpoint_paths = {}
            for step in selected_steps:
                label = f"{step//1000}k" # 将 10000 转换为 '10k' 方便展示
                checkpoint_paths[label] = os.path.join(args.checkpoint_path, str(step))
            
            logger.console_logger.info(f"开始执行 t-SNE 演变可视化，选取的检查点：{checkpoint_paths}")
            
            # 这里的 runner 必须设置 args.test_nepisode = 1，因为我们的函数目前只画一条轨迹
            original_test_nepisode = args.test_nepisode
            args.test_nepisode = 1 
            
            # 调用你的画图函数
            save_pdf_name = f"tsne_state_{args.env}_{args.name}.pdf"
            visualize_trajectory_subplots_pdf(runner, mac, checkpoint_paths, args, save_path=save_pdf_name)
            
            args.test_nepisode = original_test_nepisode # 恢复原设置
            return  # 画完图直接退出程序

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))

        if not args.pre_train_diff: # If we are not pre-training the diffusion model, only load the diff model normally
            learner.load_diff_model(model_path)
        else:    
            learner.load_models(model_path)
            runner.t_env = timestep_to_load
        
        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    last_hilp_T = -1e5
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    diff_return = 0.0
    v_thed = 1.0
    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        # episode_batch = runner.run(test_mode=False, diff_return = diff_return, v_thed = abs(v_thed))
        episode_batch = runner.run(test_mode=False, diff_return = diff_return)
        buffer.insert_episode_batch(episode_batch)
        episode_return = episode_batch["cur_return"][0,-1].item()
        # new_batch = return_batch(episode_batch)
        # good_buffer.add_trajectory(new_batch, episode_return)

        if buffer.can_sample(args.batch_size):
            if args.use_hilp and runner.t_env - last_hilp_T >= 1e4:
                for hilp_train_num in range(10):
                    episode_sample = buffer.sample(args.batch_size)
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    learner.train_hilp(episode_sample, runner.t_env)
                last_hilp_T = runner.t_env
            else:
                episode_sample = buffer.sample(args.batch_size)
                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                # diff_return, v_thed = learner.train(episode_sample, runner.t_env, episode)
                diff_return = learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config

def return_batch(batch):
    new_batch = {}
    keys = {"state", "obs", "actions", "avail_actions", "reward", "terminated", "actions_onehot", "cur_return"}
    for key in keys:
        new_batch[key] = batch[key]
    return new_batch

def truncate_trajectory(batch, T):
    """
    对轨迹进行截断，确保轨迹长度为T
    :param trajectory: 轨迹字典
    :param T: 截断长度
    :return: 截断后的轨迹
    """
    for key in batch:
        batch[key] = batch[key][:,:T]
        if key in {"terminated"}:
            batch[key][:,-2] = 1
    return batch

def visualize_trajectory_subplots_pdf(runner, mac, checkpoint_paths, args, save_path="tsne_evolution.pdf", step_interval=7, num_trajs=3):
    """
    预加载不同训练步数的模型，在每个阶段跑 `num_trajs` 条测试轨迹。
    每条轨迹及其相关元素（起点、终点、目标、虚线）使用同一种颜色。
    同一个子图中的不同轨迹使用不同颜色，跨子图颜色可复用。
    """
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    all_states = []
    single_goals = []  
    traj_lengths = [] # 二维列表，记录 [stage_1_lengths, stage_2_lengths, ...]
    valid_labels = []
    
    # 强制开启 evaluate 模式
    args.evaluate = True 

    print(f"开始收集不同训练步数下的轨迹与目标数据 (每个阶段 {num_trajs} 条)...")
    
    for label, path in checkpoint_paths.items():
        if not os.path.exists(path):
            print(f"警告: 路径不存在 {path}，跳过该模型。")
            continue
            
        # 1. 加载模型权重
        mac.load_models(path)
        if hasattr(mac, 'load_diff_model'):
            mac.load_diff_model(path)
            
        # 2. 收集 num_trajs 条测试轨迹
        stage_lengths = []
        for _ in range(num_trajs):
            states, goals, dis, exp_rewards = runner.run(test_mode=True)
            all_states.append(states)
            single_goals.append(goals[1:2]) # 取该轨迹唯一的初始目标
            stage_lengths.append(states.shape[0])
            
        traj_lengths.append(stage_lengths)
        valid_labels.append(label)
        print(f"成功收集模型 [{label}] 的 {num_trajs} 条轨迹。")

    n_stages = len(valid_labels)
    if n_stages == 0:
        print("没有成功收集到任何轨迹数据！")
        return

    # 3. 拼接 states 和唯一的 goals 进行全局 t-SNE
    combined_states = np.concatenate(all_states, axis=0)
    combined_goals = np.concatenate(single_goals, axis=0) 
    combined_all = np.concatenate([combined_states, combined_goals], axis=0)
    
    print(f"开始进行全局 t-SNE 降维计算，总数据量: {combined_all.shape}...")
    tsne = TSNE(n_components=2, perplexity=min(30, combined_all.shape[0]-1), random_state=42)
    all_2d = tsne.fit_transform(combined_all)
    print("t-SNE 降维完成！")

    # 4. 获取降维后的全局坐标极值（用于对齐所有子图）
    x_min, x_max = all_2d[:, 0].min(), all_2d[:, 0].max()
    y_min, y_max = all_2d[:, 1].min(), all_2d[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    # 5. 拆分 states_2d 和唯一的 goals_2d
    num_states = combined_states.shape[0]
    states_2d = all_2d[:num_states]
    goals_2d = all_2d[num_states:] 

    # 6. 开始绘制子图
    fig, axes = plt.subplots(1, n_stages, figsize=(6 * n_stages, 6))
    if n_stages == 1:
        axes = [axes] 
        
    cmap = plt.get_cmap('tab10') # 使用 tab10 获取离散的高对比度颜色
    start_idx = 0
    
    for i in range(n_stages):
        ax = axes[i]
        label = valid_labels[i]
        
        # 遍历当前子图中的每一条轨迹
        for j in range(num_trajs):
            length = traj_lengths[i][j]
            end_idx = start_idx + length
            color = cmap(j % 10) # 同一个子图内的不同轨迹取不同颜色 (0, 1, 2)
            
            # 提取当前轨迹的完整 2D state 和对应的 2D goal
            traj_s_2d = states_2d[start_idx:end_idx]
            goal_2d = goals_2d[i * num_trajs + j] 
            
            # 按 step_interval 间隔采样
            traj_s_2d_sub = traj_s_2d[::step_interval]
            sub_length = len(traj_s_2d_sub)
            
            # 1. 绘制状态轨迹连线 (与该轨迹绑定颜色)
            ax.plot(traj_s_2d_sub[:, 0], traj_s_2d_sub[:, 1], color=color, linewidth=2, alpha=0.5, zorder=2)
            
            # 2. 绘制状态散点 (渐变透明度)
            alphas = np.linspace(0.2, 1.0, sub_length)
            for k in range(sub_length):
                ax.scatter(traj_s_2d_sub[k, 0], traj_s_2d_sub[k, 1], color=color, alpha=alphas[k], s=20, zorder=3)
            
            # 3. 绘制起点和终点 (强制使用该轨迹的颜色，用形状区分)
            ax.scatter(traj_s_2d[0, 0], traj_s_2d[0, 1], facecolor=color, s=150, marker='*', edgecolors='black', zorder=5)
            ax.scatter(traj_s_2d[-1, 0], traj_s_2d[-1, 1], facecolor=color, s=100, marker='X', edgecolors='black', zorder=5)

            # 4. 绘制唯一的 Goal (同色大三角形)
            ax.scatter(goal_2d[0], goal_2d[1], facecolor=color, edgecolor='black', marker='^', s=100, alpha=0.9, zorder=6)
            
            # 5. 绘制连接 起点 与 Goal 的意图虚线
            # ax.plot([traj_s_2d[0, 0], goal_2d[0]], [traj_s_2d[0, 1], goal_2d[1]], 
                    # color=color, linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
            
            start_idx = end_idx

        # 设置子图属性
        ax.set_title(f'Training Stage: {label}', fontsize=14, fontweight='bold')
        # ax.set_xlabel('t-SNE Dim 1', fontsize=12)
        # if i == 0:
        #     ax.set_ylabel('t-SNE Dim 2', fontsize=12)
        
        # 统一坐标轴
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 在最后一个子图上添加自定义图例 (仅解释形状含义)
        if i == n_stages - 1:
            star = mlines.Line2D([], [], color='gray', marker='*', linestyle='None', markersize=12, markeredgecolor='black', label='Start State')
            cross = mlines.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=10, markeredgecolor='black', label='End State')
            triangle = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=12, markeredgecolor='black', label='Diffusion Goal')
            dashed_line = mlines.Line2D([], [], color='gray', linestyle='--', label='Exploration Intent')
            
            ax.legend(handles=[star, cross, triangle, dashed_line], loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

    plt.suptitle('Evolution of Agent Multiple Trajectories vs Diffusion Goals', fontsize=18, y=1.05)
    plt.tight_layout()
    
    # 保存为 PDF
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n✅ 可视化完成！已成功保存为高质量 PDF 文件: {save_path}")
    plt.show()