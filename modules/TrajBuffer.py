import torch
import numpy as np

class TrajectoryBuffer:
    def __init__(self, N, H, device='cuda'):
        """
        初始化轨迹缓冲区
        :param N: 最大存储容量
        :param H: 每条轨迹的最大步长
        :param device: 默认存储设备，'cuda' 或 'cpu'
        """
        self.N = N  # 最大存储容量
        self.H = H  # 每条轨迹的最大步长
        self.device = device  # 存储设备
        self.buffer = []  # 存储轨迹的列表
        self.scores = []  # 存储轨迹的最终得分，用于判断替换
        self.idx = 0  # 当前存储的轨迹数量

    def _pad_trajectory(self, trajectory):
        """
        对轨迹进行补齐，确保轨迹长度为H
        :param trajectory: 轨迹字典
        :return: 补齐后的轨迹
        """
        for key in trajectory:
            if isinstance(trajectory[key], torch.Tensor):
                T = trajectory[key].size(0)
                if T < self.H:
                    # 补齐到H长度，填充0或其他适当值
                    padding_size = self.H - T
                    if key not in {"terminated"}:
                        pad_tensor = torch.zeros(padding_size, *trajectory[key].shape[1:], device=self.device)
                    else:
                        pad_tensor = torch.ones(padding_size, *trajectory[key].shape[1:], device=self.device)
                    trajectory[key] = torch.cat([trajectory[key], pad_tensor], dim=0)
            elif isinstance(trajectory[key], np.ndarray):
                T = trajectory[key].shape[0]
                if T < self.H:
                    padding_size = self.H - T
                    if key not in {"terminated"}:
                        pad_array = np.zeros((padding_size, *trajectory[key].shape[1:]), dtype=trajectory[key].dtype)
                    else:
                        pad_array = np.ones((padding_size, *trajectory[key].shape[1:]), dtype=trajectory[key].dtype)
                    trajectory[key] = torch.tensor(np.concatenate([trajectory[key], pad_array], axis=0), device=self.device)
        
        return trajectory

    def _generate_filled_field(self, trajectory):
        """
        自动生成 `filled` 字段，标记轨迹中哪些是填充数据，哪些是真实数据
        :param trajectory: 轨迹字典
        :return: 添加了 `filled` 字段的轨迹
        """
        T = trajectory['state'].size(0)  # 当前轨迹的长度
        filled = torch.zeros(T, dtype=torch.bool, device=self.device)  # 默认所有时间步为真实数据（True）

        # 将 `filled` 标记为 False，表示填充部分
        if T < self.H:
            filled[T:] = True  # 填充部分为 False

        # 将 `filled` 字段添加到轨迹字典
        trajectory['filled'] = filled.unsqueeze(-1)
        return trajectory

    def add_trajectory(self, trajectory, final_return):
        """
        添加一条轨迹，若未达到存储容量，则直接添加；若已满，则判断是否替换
        :param trajectory: 轨迹字典，包含numpy数据
        :param final_return: 当前轨迹的最终收益
        """
        # 将输入的numpy数据转换为torch.Tensor，并移动到指定设备
        trajectory = {key: torch.tensor(value, device=self.device).squeeze(0) if isinstance(value, np.ndarray) else value.squeeze(0)
                      for key, value in trajectory.items()}
        
        # 自动生成 `filled` 字段
        trajectory = self._generate_filled_field(trajectory)

        # 补齐轨迹，确保每个轨迹长度为H
        trajectory = self._pad_trajectory(trajectory)

        if self.idx < self.N:
            # 如果容量未满，直接添加
            self.buffer.append(trajectory)
            self.scores.append(final_return)
            self.idx += 1
        else:
            # 如果容量已满，替换最低得分的轨迹
            min_idx = torch.argmin(torch.tensor(self.scores, device=self.device))
            if final_return > self.scores[min_idx]:
                self.buffer[min_idx] = trajectory
                self.scores[min_idx] = final_return

    def sample_trajectories(self, n):
        """
        随机采样n条轨迹，按字典形式返回批量轨迹
        :param n: 需要采样的轨迹数量
        :return: 字典形式的批量轨迹，所有数据为tensor格式
        """
        # 随机从buffer中采样n条轨迹
        indices = torch.randint(0, self.idx, (n,), device=self.device)
        batch = {key: [] for key in self.buffer[0].keys()}

        for i in indices:
            for key in batch:
                batch[key].append(self.buffer[i][key])

        # 将采样结果转为Tensor，并确保它们在CUDA上
        for key in batch:
            batch[key] = torch.stack(batch[key]).to(self.device)

        return batch