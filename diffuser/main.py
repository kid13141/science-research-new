import json
from typing import Any, Dict, List, Optional, Tuple, Union,Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import torch.nn as nn
import torch.nn.functional as F
import importlib
from src.diffusion import GaussianDiffusion
from src.constructor import Backward_Net,Reward_Net
from os.path import dirname, abspath
import components
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

TensorBatch = List[torch.Tensor]
test =True

def compute_max_min(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    states = states.reshape(-1,states.shape[-1])
    print('states_shape=',states.shape)
    max = np.max(states,axis=0)
    min = np.min(states,axis=0)
    return max, min

def normalize_states(states: np.ndarray, max: np.ndarray, min: np.ndarray):
    states=torch.tensor((states - min) / (max-min))
    states=torch.where(torch.isnan(states), torch.full_like(states, 0), states)
    return states
        

def non_normalized(states:np.ndarray,max:np.ndarray,min:np.ndarray):
    states_new=[]
    for s in states:
        b=s*(max-min)+min
        states_new.append(b.tolist())
    return np.array(states_new)

def sample(states: torch.tensor,batch_size: int,terminal_locs,valid_indices) -> TensorBatch:

    indices = np.random.choice(valid_indices, size=batch_size)
    batch_states = states[indices]

    goal_indx = sample_goals(indices,terminal_locs)
    batch_goals = states[goal_indx]
    return batch_states,batch_goals

def load_and_concatenate_states(buffer_dir):
    all_states = []
    all_terminal_pos = []
    episodes_num = 0
    # 遍历buffer文件夹下的所有子文件夹
    for subdir, _, files in os.walk(buffer_dir):
        for file in files:
            if file == 'replay_buffer.npy':
                # 构建文件路径
                file_path = os.path.join(subdir, file)
                # 加载npy文件
                buffer_data = np.load(file_path, allow_pickle=True).item()
                #获取episodes数量
                episodes_num += buffer_data.episodes_in_buffer
                # 提取state数据并添加到列表中
                states = buffer_data.data.transition_data['state']
                # 提取terminal数据添加到列表中
                terminal_pos = buffer_data.data.transition_data['terminated']
                state_dim = states.shape[-1]
                action_dim = buffer_data.data.transition_data['actions'].shape[-1]
                # print(buffer_data.data.transition_data['actions'][0])

                all_states.append(states)
                all_terminal_pos.append(terminal_pos)

    concatenated_states = np.concatenate(all_states, axis=0)#(26000, 201, 40)
    concatenated_terminal_pos = np.concatenate(all_terminal_pos, axis=0)
    max_seq_length = concatenated_states.shape[1]

    return concatenated_states,concatenated_terminal_pos,state_dim,action_dim,episodes_num,max_seq_length

#index是采样出来的样本的索引数组
def sample_goals(indx,terminal_locs, p_trajgoal=0.625, p_currgoal=0,geom_sample=1,discount=0.99):

    batch_size = len(indx)

    # Goals from the same trajectory and Goals as terminal，返回的是index中每一个样本对应的终止状态索引
    b=np.searchsorted(terminal_locs, indx)
    # print(self.terminal_locs)
    # print(b)
    
    final_state_indx = terminal_locs[b]

    # Random goals 
    # to do 需要改成在同一episode下随机采样goal_indx
    # goal_indx = np.random.randint(self.dataset.size, size=batch_size)
    goal_indx = [np.random.randint(i,j+1) for i,j in zip(indx,final_state_indx)]

    distance = np.random.rand(batch_size)

    # 找到middle_goal_indx,并且小于final_state_indx
    if  geom_sample:
        us = np.random.rand(batch_size)
        middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(discount)).astype(int), final_state_indx)
    else:
        middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

    #以一定的概率分布，选择middle_goal_indx还是random_goal_indx
    goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)

    # Goals at the current state
    goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
    return goal_indx


def read_data(data):
    time_steps = []
    for block in data:
        points = []
        n = 0
        alloc = [int(x) for x in block.split('\n')[0].strip()[1:-1].replace(" ", "").split(',')]
        for line in block.split('\n')[1:]:
            # print(line)
            a = line.strip()[1:-1].replace(" ", "").split(',')
            # print(a)
            # print([int(a[0])] + [float(x) for x in a[1:]])
            # b = [int(a[0])] + [float(x) for x in a[1:]]
            b = [float(x) if i > 0 else int(x) for i, x in enumerate(a)]
            b.append(10)
            points.append(b)
            if b[0] == 0:
                b[-1] = alloc[n]
                # print(n)
                n += 1
        time_steps.append(points)
    return time_steps


if __name__ == '__main__':

# 获取当前时间并格式化为文件名
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join(dirname(abspath(__file__)),"buffer_data","test")
    concatennated_states,concatenated_terminal_pos,state_dim,action_dim,episodes_num,max_seq_length = load_and_concatenate_states(path)

    # buffer_data=np.load(path,allow_pickle=True).item()
    # print(buffer_data.data.transition_data.keys()) #dict_keys(['state', 'obs', 'actions', 'avail_actions', 'reward', 'terminated', 'actions_onehot', 'filled'])
    # print(buffer_data.data.transition_data['state'].shape) #torch.Size([1000, 201, 40])
    
    # #-----train--------

    states_max, states_min = compute_max_min(concatennated_states, eps=1e-3)
    goal_states_max=np.concatenate((states_max,states_max))
    # print(states_max)
    # print(states_min)
    # print(concatennated_states[0][0])
    all_states= normalize_states(concatennated_states, states_max, states_min)
    # print(all_states[0][0])
    # buffer_data.data.transition_data['state'] = normalize_states(concatennated_states, states_max, states_min)

    # action_dim =buffer_data.data.transition_data['actions'].shape[-1]
    # print("action_dim=",action_dim)
    # state_dim =buffer_data.data.transition_data['state'].shape[-1]
    # print("state_dim=",state_dim)

    #train diffusions
    back_generator_network = Backward_Net(observation_dim = state_dim*2, action_dim = action_dim, input_dim = state_dim*2+16, out_dim = state_dim*2 , t_dim = 16, hidden_dim=256)
    diffusion_generator = GaussianDiffusion(back_generator_network, state_dim, action_dim, state_dim*2, goal_states_max, lambda_guide=1,n_timesteps=20).to("cuda")
    g_optimizer = torch.optim.Adam(diffusion_generator.parameters(), lr=3e-4)

    # # reward_network = Reward_Net(observation_dim = state_dim, action_dim = action_dim)
    # # r_optimizer = torch.optim.Adam(reward_network.parameters(), lr=3e-4)
    # # reward_network.to(config.device)
    batch_size = 16
    train_step = 100000

    state = all_states.reshape(-1,all_states.shape[-1])
    terminal_pos = concatenated_terminal_pos.reshape(-1,concatenated_terminal_pos.shape[-1])
    terminal_pos = terminal_pos.flatten()
    terminal_locs, = np.nonzero(terminal_pos > 0) 

    valid_indices=[]
    for episode in range(episodes_num):
        end_index = terminal_locs[episode] 
        if (end_index+1) %  max_seq_length != 0:
            valid_indices.extend([episode * max_seq_length + step for step in range((end_index+1)% max_seq_length -1)])
        else:
            valid_indices.extend([episode *  max_seq_length + step for step in range(max_seq_length-1)])

    if(test==False):
        # 创建“mode”文件夹
        model_dir = os.path.join(os.getcwd(), "mode")
        os.makedirs(model_dir, exist_ok=True)

        for t in range(train_step):
            # batch= buffer_data.sample(batch_size)
            states_batch,goals_batch = sample(state,batch_size,terminal_locs,valid_indices)
            #将state与goal拼接
            states_goals_batch = np.concatenate((states_batch,goals_batch),axis=1)
            # states_batch = torch.tensor([s.cpu().detach().numpy() for s in states_batch]).cuda()
            states_goals_batch = torch.tensor(states_goals_batch).cuda()
            states_batch= torch.tensor(states_batch).cuda()
            # print(states_batch.shape)
            g_loss = diffusion_generator.loss(x = states_goals_batch, cond = None, is_cond = False)
            if(t % 1000 ==0):
                print('train_step {} loss is = {}'.format(t,g_loss))
            if(t % 10 ==0):
                # 格式化为文件名
                model_filename = f"diffuser-{current_time}.pt"
                model_path = os.path.join(model_dir, model_filename)
                # 保存模型
                torch.save(diffusion_generator.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            g_loss.backward()
            g_optimizer.step()
            g_optimizer.zero_grad()
    else:
        #测试
        model_dir = os.path.join(os.getcwd())
        model_path = os.path.join(model_dir,"mode", "diffuser-2024-08-07-19-00-52.pt")
        norm_path = os.path.join(model_dir,"mode", "norm.json")
        diffusion_generator.load_state_dict(state_dict=torch.load(model_path))
        cond_state_batch,_ = sample(state,batch_size,terminal_locs,valid_indices)
        cond_state_batch = torch.tensor(cond_state_batch).cuda()
        goal_states=diffusion_generator.forward(cond=cond_state_batch, is_cond=True)#[20,40]
        next_states=non_normalized(goal_states[:,40:].to('cpu').numpy(),states_max,states_min)
        start_state=non_normalized(cond_state_batch.to('cpu').numpy(),states_max,states_min)
        # start_state = cond_state_batch.to('cpu').numpy()
        np.savetxt('start_data.csv', start_state, delimiter=',')
        np.savetxt('end_data.csv', next_states, delimiter=',')
        dic={}
        dic['states_max']=states_max.tolist()
        dic['states_min']=states_min.tolist()
        with open(norm_path,'w') as file:
            json.dump(dic,file)
        print(next_states.shape)


