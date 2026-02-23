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


    def forward(self, inputs, hidden_state, hilp_val, latch_state):
        # inputs: [Batch*Agents, Input_Dim] (PyMARL standard flattened input)
        # hilp_val: [Batch*Agents, 1]
        # latch_state: [Batch*Agents, 1] 记录上一时刻的锁定状态
        
        # --- A. Shared Backbone ---
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        
        # --- B. Dual Heads ---
        q_nav = self.q_nav_head(h_out)
        q_act = self.q_act_head(h_out)
        
        # --- C. Soft Gating & Phase Latching ---
        # 确保 hilp_val 维度对齐
        if hilp_val.shape[0] != inputs.shape[0]:
            hilp_val = hilp_val.reshape(-1, 1)
            
        # 1. 计算原始门控系数 alpha
        raw_gate = self.gating_k * (hilp_val - self.gating_delta)
        alpha = torch.sigmoid(raw_gate) # (Batch*Agents, 1)

        # 2. 相位锁定逻辑 (Phase Latching)
        # 触发器：如果当前 alpha 大于 0.9，产生锁定信号
        lock_signal = (alpha > 0.9).float()
        
        # 更新锁状态：保持 1.0 状态不掉落
        new_latch_state = torch.max(latch_state, lock_signal).detach() 
        
        # 确定实际使用的 alpha
        locked_alpha = torch.max(alpha, new_latch_state)

        # --- D. Dynamic Blending ---
        # 使用 locked_alpha 融合两个头的 Q 值
        q_final = (1 - locked_alpha) * q_nav + locked_alpha * q_act
        
        # 注意：这里返回了 new_latch_state
        return q_final, h_out, new_latch_state

    # 用于 Learner 获取分头数据
    def get_dual_q(self, inputs, hidden_state, hilp_val, latch_state, n_agents, bs):
        # 1. 前向传播提取特征
        x = F.relu(self.fc1(inputs))
        h_out = self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim))
        
        # 2. 计算双头 Q 值
        q_nav = self.q_nav_head(h_out)
        q_act = self.q_act_head(h_out)
        
        # 3. 计算原始门控系数 alpha
        raw_gate = self.gating_k * (hilp_val.reshape(-1, 1) - self.gating_delta)
        alpha = torch.sigmoid(raw_gate)
        
        # 4. 相位锁定逻辑 (Phase Latching)
        # 触发器：如果当前 alpha 大于 0.9，则产生 1.0 的锁定信号，否则为 0.0
        lock_signal = (alpha > 0.9).float()
        
        # 更新锁状态：只要历史状态或当前信号有任意一个是 1.0，锁就会永久闭合 (保持 1.0)
        new_latch_state = torch.max(latch_state, lock_signal).detach() # 必须 detach 断开梯度
        
        # 决定当前实际使用的 alpha: 
        # 如果未锁定 (new_latch_state=0)，取原 alpha 进行平滑过渡；
        # 如果已锁定 (new_latch_state=1)，取 1.0 (完全由 q_act 主导)。
        locked_alpha = torch.max(alpha, new_latch_state)
        
        # 5. 执行决策融合
        q_final = (1 - locked_alpha) * q_nav + locked_alpha * q_act
        
        # 6. 返回结果 (注意最后增加返回了 new_latch_state)
        return q_final, q_nav, q_act, alpha, h_out.view(bs, n_agents, -1), new_latch_state

