import torch.nn as nn
import torch.nn.functional as F
import torch as th

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        '''
        self.fc1_2 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_2 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_3 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_3 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_4 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_4 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_4 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_5 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_5 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_5 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_6 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_6 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_6 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_7 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_7 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_7 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        '''
        '''
        self.fc1_8 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_8 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_8 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_9 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_9 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_9 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_10 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_10 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_10 = nn.Linear(args.rnn_hidden_dim, args.n_actions
        '''

    def init_hidden(self):
        x1 = self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x2 = self.fc1_2.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x3 = self.fc1_3.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x4 =self.fc1_4.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x5 =self.fc1_5.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x6 =self.fc1_6.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x7 = self.fc1_7.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x1
        # make hidden states on same device as model
        #return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #return x1,x2,x3,x4,x5,x6,x7

    def init_hidden2(self):
        x2 = self.fc1_2.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x2

    def init_hidden3(self):
        x3 = self.fc1_3.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x3

    def init_hidden4(self):
        x4 = self.fc1_4.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x4

    def init_hidden5(self):
        x5 = self.fc1_5.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x5

    def init_hidden6(self):
        x6 = self.fc1_6.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x6

    def init_hidden7(self):
        x7 = self.fc1_7.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x7


    def forward(self, inputs, hidden_state):#,hid2,hid3,hid4,hid5,hid6,hid7):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        '''
        x_2 = F.relu(self.fc1_2(inputs))
        h_in_2 = hid2.reshape(-1, self.args.rnn_hidden_dim)
        h_2 = self.rnn_2(x_2, h_in_2)
        q_2 = self.fc2_2(h_2)

        x_3 = F.relu(self.fc1_3(inputs))
        h_in_3 = hid3.reshape(-1, self.args.rnn_hidden_dim)
        h_3 = self.rnn_3(x_3, h_in_3)
        q_3 = self.fc2_3(h_3)

        x_4 = F.relu(self.fc1_4(inputs))
        h_in_4 = hid4.reshape(-1, self.args.rnn_hidden_dim)
        h_4 = self.rnn_4(x_4, h_in_4)
        q_4 = self.fc2_4(h_4)

        x_5 = F.relu(self.fc1_5(inputs))
        h_in_5 = hid5.reshape(-1, self.args.rnn_hidden_dim)
        h_5 = self.rnn_5(x_5, h_in_5)
        q_5 = self.fc2_5(h_5)

        x_6 = F.relu(self.fc1_6(inputs))
        h_in_6 = hid6.reshape(-1, self.args.rnn_hidden_dim)
        h_6 = self.rnn_6(x_6, h_in_6)
        q_6 = self.fc2_6(h_6)

        x_7 = F.relu(self.fc1_7(inputs))
        h_in_7 = hid7.reshape(-1, self.args.rnn_hidden_dim)
        h_7 = self.rnn_7(x_7, h_in_7)
        q_7 = self.fc2_7(h_7)
        '''
        '''
        x8 = F.relu(self.fc1_8(inputs))
        h_in_8 = hid8.reshape(-1, self.args.rnn_hidden_dim)
        h_8 = self.rnn(x_8, h_in_8)
        q_8 = self.fc8(h_8)

        x9 = F.relu(self.fc1_9(inputs))
        h_in_9 = hid9.reshape(-1, self.args.rnn_hidden_dim)
        h_9 = self.rnn(x_9, h_in_9)
        q_9 = self.fc9(h_9)

        x10 = F.relu(self.fc1_10(inputs))
        h_in_10 = hid10.reshape(-1, self.args.rnn_hidden_dim)
        h_10 = self.rnn(x_10, h_in_10)
        q_10 = self.fc10(h_10)
        '''
        return q,h
        return (q+q_2+q_3+q_4+q_5+q_6+q_7)/th.tensor(7) ,q,q_2,q_3,q_4,q_5,q_6,q_7, h,h_2,h_3,h_4,h_5,h_6,h_7
