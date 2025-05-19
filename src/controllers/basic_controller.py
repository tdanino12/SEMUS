from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np
import math

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self.h2,self.h3,self.h4,self.h5,self.h6,self.h7 = None,None,None,None,None,None
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False,learner=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode,learner=learner,execute=True)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, agent_outputs


    def forward(self, ep_batch, t, test_mode=False,training=False,learner=None,execute=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        if(self.args.soft_modul):
            if(execute):
                # code for parallel runnner
                x = [float(i) for i in range(self.args.n_agents)]
                if self.args.use_cuda:
                    all_x = th.tensor([x] * ep_batch.batch_size).to("cuda")
                else:
                    all_x = th.tensor([x] * ep_batch.batch_size)
                all_x = all_x.view(agent_inputs.shape[0],1)
                agent_outs = self.pf(agent_inputs,idx=all_x)

                # code for episode runner
                #agent_outs = self.pf(agent_inputs,idx=th.tensor([float(i) for i in range(self.args.n_agents)]).view(self.args.n_agents,1))
            else:
                # code for parallel runnner
                x = [float(i) for i in range(self.args.n_agents)]
                if self.args.use_cuda:
                    all_x = th.tensor([x] * ep_batch.batch_size).to("cuda")
                else:
                    all_x = th.tensor([x] * ep_batch.batch_size)
                all_x = all_x.view(agent_inputs.shape[0],1)
                agent_outs = self.pf(agent_inputs,idx=all_x)
        else:
            agent_outs,self.hidden_states= self.agent(agent_inputs, self.hidden_states)

    
        if(learner!=None and execute==True):
            #t_alpha = min(5.5,4+t/600000)
            inputs = learner.critic._build_inputs(ep_batch,ep_batch.batch_size,ep_batch.max_seq_length)
            q_temp,t_q1,t_q2,t_q3,t_q4,t_q5,t_q6,t_q7,t_q8,t_q9,t_q10 = learner.critic.forward(inputs[:,t:t+1])
            q = q_temp.clone().detach()
            all_tensors = [t_q1.clone().detach(),t_q2.clone().detach(),t_q3.clone().detach(),t_q4.clone().detach(),t_q5.clone().detach(),
                           t_q6.clone().detach(),t_q7.clone().detach(),t_q8.clone().detach(),t_q9.clone().detach(),t_q10.clone().detach()]
            mone = th.zeros_like(q_temp)
            mechane = th.zeros_like(q_temp)
            for a in all_tensors:
                mone+=th.pow(a-q,4)
                mechane+=th.pow(a-q,2)
                #mone+=np.power(a-q,4)
                #mechane+=np.power(a-q,2)

            mechane = th.pow(mechane,2)
            moment = ((th.tensor(10))*mone)/mechane       
            m =moment.view(agent_outs.shape[0],agent_outs.shape[1])
            m2 = th.mean(m,dim=-1)
            m2 = m2-th.tensor(3)
            m2 = m2.unsqueeze(dim=-1)
            m2 = m2.expand(agent_outs.shape[0],agent_outs.shape[1])
            m[m2<0]=th.tensor(0.0)
            m = m*th.tensor(0.0001)
            #m = m*th.tensor(0.001)
            m = th.clamp(m,-0.1,0.1)
            
            agent_outs = agent_outs+m

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e11
                #if(learner!=None and execute==True):
                #    m[reshaped_avail_actions == 0] = -1e11
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            #if(learner!=None and execute==True):
            #    agent_outs = th.tensor(0.9999)*agent_outs+th.tensor(0.0001)*m
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if(not self.args.soft_modul):
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        a=1
    def parameters(self):
        if(self.args.soft_modul):
            return self.pf.parameters()
        else:     
            return self.agent.parameters()

    def load_state(self, other_mac):
        if(self.args.soft_modul):
            self.pf.load_state_dict(other_mac.pf.state_dict())
        else:
            self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        if(self.args.soft_modul):
            self.pf.cuda()
        else:    
            self.agent.cuda()

    def save_models(self, path):
        if(self.args.soft_modul):
            th.save(self.pf.state_dict(), "{}/agent.th".format(path))
        else:
            th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        if(self.args.soft_modul):
            self.pf.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        else:    
            self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
