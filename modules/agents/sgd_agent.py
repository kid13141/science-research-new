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

    # 关键修改：显式接收 hilp_val
    def forward(self, inputs, hidden_state, hilp_val):
        # inputs: [Batch*Agents, Input_Dim] (PyMARL standard flattened input)
        # hilp_val: [Batch*Agents, 1]
        
        # --- A. Shared Backbone ---
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        
        # --- B. Dual Heads ---
        q_nav = self.q_nav_head(h_out)
        q_act = self.q_act_head(h_out)
        
        # --- C. Soft Gating ---
        # 确保 hilp_val 维度对齐
        if hilp_val.shape[0] != inputs.shape[0]:
            hilp_val = hilp_val.reshape(-1, 1)
            
        raw_gate = self.gating_k * (hilp_val - self.gating_delta)
        alpha = torch.sigmoid(raw_gate) # (Batch*Agents, 1)

        # --- D. Dynamic Blending ---
        q_final = (1 - alpha) * q_nav + alpha * q_act
        
        return q_final, h_out

    # 用于 Learner 获取分头数据
    def get_dual_q(self, inputs, hidden_state, hilp_val):
        # ... (前向传播逻辑同上) ...
        x = F.relu(self.fc1(inputs))
        h_out = self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim))
        
        q_nav = self.q_nav_head(h_out)
        q_act = self.q_act_head(h_out)
        
        raw_gate = self.gating_k * (hilp_val.reshape(-1, 1) - self.gating_delta)
        alpha = torch.sigmoid(raw_gate)
        
        q_final = (1 - alpha) * q_nav + alpha * q_act
        
        return q_final, q_nav, q_act, alpha, h_out