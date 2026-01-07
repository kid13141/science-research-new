import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy_Head(nn.Module):
    def __init__(self, args):
        super(Policy_Head, self).__init__()
        self.policy = nn.Linear(args.rnn_hidden_dim*2, args.n_actions)

    def forward(self, x):
        x = self.policy(x)
        output = nn.Softmax(dim=1)(x)
        return x

class GUF_Exp_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(GUF_Exp_Agent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim*2)
        self.fc2 = [Policy_Head(args).cuda() for i in range(self.args.n_agents)]

    def init_hidden(self):
        # make hidden states on same device as model
        # return self.fc1.weight.new(1, self.args.rnn_hidden_dim * 2).zero_()
        return

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.float()))
        q = []
        for i in range(self.args.n_agents):
            q.append(self.fc2[i](x))
        q = torch.stack(q, dim=1)
        return q
