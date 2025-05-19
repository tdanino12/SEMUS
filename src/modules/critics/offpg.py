import torch as th
import torch.nn as nn
import torch.nn.functional as F

class PGCriticNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PGCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_v = nn.Linear(256, 1)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        a = self.fc3(x)
        q = a + v
        return q


class OffPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(OffPGCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"
        

        '''
        self.network1 = PGCriticNetwork(input_shape, self.n_actions)
        self.network2 = PGCriticNetwork(input_shape, self.n_actions)
        self.network3 = PGCriticNetwork(input_shape, self.n_actions)
        self.network4 = PGCriticNetwork(input_shape, self.n_actions)
        self.network5 = PGCriticNetwork(input_shape, self.n_actions)
        self.network6 = PGCriticNetwork(input_shape, self.n_actions)
        self.network7 = PGCriticNetwork(input_shape, self.n_actions)
        self.network8 = PGCriticNetwork(input_shape, self.n_actions)
        self.network9 = PGCriticNetwork(input_shape, self.n_actions)
        self.network10 = PGCriticNetwork(input_shape, self.n_actions)    
        '''
        # Create a ModuleList to hold the critic ensemble
        self.ensemble = nn.ModuleList([
            PGCriticNetwork(input_shape, self.n_actions) for _ in range(10)
        ])



        #self.network11 = PGCriticNetwork(input_shape, self.n_actions)
        #self.network12 = PGCriticNetwork(input_shape, self.n_actions)
        #self.network13 = PGCriticNetwork(input_shape, self.n_actions)
        #self.network14 = PGCriticNetwork(input_shape, self.n_actions)


    def forward(self, inputs):
        '''
        q = self.network1(inputs)
        q2 = self.network2(inputs)
        q3 = self.network3(inputs)
        q4 = self.network4(inputs)
        q5 = self.network5(inputs)
        q6 = self.network6(inputs)
        q7 = self.network7(inputs)
        q8 = self.network8(inputs)
        q9 = self.network9(inputs)
        q10 = self.network10(inputs)  


        #q11 = self.network7(inputs)
        #q12 = self.network8(inputs)
        #q13 = self.network9(inputs)
        #q14 = self.network10(inputs)
        

        #return (q+q2+q3+q4+q5+q6+q7+q8+q9+q10+q11+q12+q13+q14)/th.tensor(14),q,q2,q3,q4,q5,q6,q7,q8,q9,q10
        return (q+q2+q3+q4+q5+q6+q7+q8+q9+q10)/th.tensor(10),q,q2,q3,q4,q5,q6,q7,q8,q9,q10
        '''
        # Run all networks in parallel and stack their outputs
        q_values = [net(inputs) for net in self.ensemble]  # list of tensors [batch, n_actions]
        q_stack = th.stack(q_values, dim=0)                # [num_nets, batch, n_actions]

        # Mean Q across ensemble
        q_mean = q_stack.mean(dim=0)

        return (q_mean,) + tuple(q_values)  # returns (mean, q1, q2, ..., q10)

    def _build_inputs(self, batch, bs, max_t):
        inputs = []
        # state, obs, action
        inputs.append(batch["state"][:].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        inputs.append(batch["obs"][:])
        #actions = batch["actions_onehot"][:].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        #agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        #agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        #inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))
        # last actions
        #if self.args.obs_last_action:
        #    last_action = []
        #    last_action.append(actions[:, 0:1].squeeze(2))
        #    last_action.append(actions[:, :-1].squeeze(2))
        #    last_action = th.cat([x for x in last_action], dim = 1)
        #    inputs.append(last_action)
        #agent id
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs



    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        # actions and last actions
        #input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        #if self.args.obs_last_action:
        #    input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # agent id
        input_shape += self.n_agents
        return input_shape
