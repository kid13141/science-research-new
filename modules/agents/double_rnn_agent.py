import torch.nn as nn
import torch.nn.functional as F


class Double_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Double_RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn1 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc2 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn2 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.head1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.head2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_(), self.fc2.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state_1, hidden_state_2):
        x1 = F.relu(self.fc1(inputs))
        h_in_1 = hidden_state_1.reshape(-1, self.args.rnn_hidden_dim)
        h1 = self.rnn1(x1, h_in_1)

        x2 = F.relu(self.fc2(inputs))
        h_in_2 = hidden_state_2.reshape(-1, self.args.rnn_hidden_dim)
        h2 = self.rnn2(x2, h_in_2)

        q1 = self.head1(h1)
        q2 = self.head2(h2)
        return q1, q2, h1, h2
