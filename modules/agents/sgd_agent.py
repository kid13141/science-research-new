import torch
import torch.nn as nn
import torch.nn.functional as F

class SGDAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SGDAgent, self).__init__()
        self.args = args
        
        # 1. 共享感知主干
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # 2. 双头
        self.q_nav_head = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.q_act_head = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # 3. 门控参数
        self.gating_k = getattr(args, "gating_k", 5.0)
        self.gating_delta = getattr(args, "gating_delta", 0.5)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    # def forward(self, inputs, hidden_state, hilp_val, latch_state):
    #     # inputs: [Batch*Agents, Input_Dim] (PyMARL standard flattened input)
    #     # hilp_val: [Batch*Agents, 1]
    #     # latch_state: [Batch*Agents, 1] 记录上一时刻的锁定状态
        
    #     # --- A. Shared Backbone ---
    #     x = F.relu(self.fc1(inputs))
    #     h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
    #     h_out = self.rnn(x, h_in)
        
    #     # --- B. Dual Heads ---
    #     q_nav = self.q_nav_head(h_out)
    #     q_act = self.q_act_head(h_out)
        
    #     # --- C. Hard Gating & Phase Latching ---
    #     # 确保 hilp_val 维度对齐
    #     if hilp_val.shape[0] != inputs.shape[0]:
    #         hilp_val = hilp_val.reshape(-1, 1)
            
    #     # 1. 计算硬门控系数 hard_alpha (绝对的 0.0 或 1.0)
    #     # 现在的 hilp_val 是距离 (越小越近)。当距离小于等于 delta 时，触发战斗 (1.0)
    #     hard_alpha = (hilp_val <= self.gating_delta).float()

    #     # 2. 相位锁定逻辑 (Phase Latching)
    #     # 更新锁状态：只要当前判定该进入战斗 (hard_alpha=1)，或者历史已经是 1
    #     # 锁就会永久闭合，保持 1.0 状态不掉落
    #     new_latch_state = torch.max(latch_state, hard_alpha).detach() 

    #     # --- D. Hard Switching (硬切换) ---
    #     # 因为 new_latch_state 只有 0 或 1 两种值，这里实际上就是非黑即白的切换
    #     # 0 的时候 100% 用 q_nav，1 的时候 100% 用 q_act
    #     q_final = (1 - new_latch_state) * q_nav + new_latch_state * q_act
        
    #     # 注意：这里返回了 new_latch_state
    #     return q_final, h_out, new_latch_state


    # 用于 Learner 获取分头数据
    # def get_dual_q(self, inputs, hidden_state, hilp_val, latch_state, n_agents, bs):
    #     # 1. 前向传播提取特征
    #     x = F.relu(self.fc1(inputs))
    #     h_out = self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim))
        
    #     # 2. 计算双头 Q 值
    #     q_act = self.q_act_head(h_out)
    #     q_nav = self.q_nav_head(h_out.detach())
        
    #     # 3. 计算硬门控系数 hard_alpha (绝对的 0.0 或 1.0)
    #     hilp_val_reshaped = hilp_val.reshape(-1, 1)
    #     hard_alpha = (hilp_val_reshaped <= self.gating_delta).float()
        
    #     # 4. 相位锁定逻辑 (Phase Latching)
    #     # 只要触发一次 1.0，立刻永久锁定为 1.0
    #     new_latch_state = torch.max(latch_state, hard_alpha).detach() # 必须 detach 断开梯度
        
    #     # 5. 执行硬切换
    #     q_final = (1 - new_latch_state) * q_nav + new_latch_state * q_act
        
    #     # 6. 返回结果 (注意：现在返回的 alpha 也就是 hard_alpha，它只有 0 或 1)
    #     return q_final, q_nav, q_act, hard_alpha, h_out.view(bs, n_agents, -1), new_latch_state

    def forward(self, inputs, hidden_state):
        # --- A. Shared Backbone ---
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        
        # --- B. Dual Heads ---
        q_nav = self.q_nav_head(h_out)
        q_act = self.q_act_head(h_out)
    
        return q_nav, q_act,h_out