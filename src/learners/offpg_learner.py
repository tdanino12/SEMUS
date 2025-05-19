import copy
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.offpg import OffPGCritic
import torch as th
from utils.offpg_utils import build_target_q
from utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop
from modules.mixers.qmix import QMixer
import numpy as np

class OffPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.bound = 2
        
        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = OffPGCritic(scheme, args)
        self.mixer = QMixer(args)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.mixer_params = list(self.mixer.parameters())
        self.params = self.agent_params + self.critic_params
        self.c_params = self.critic_params + self.mixer_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.mixer_optimiser = RMSprop(params=self.mixer_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.numpy_list = np.zeros(650000)
        self.index = 0
        self.first_ind = 0

    def train_on(self, batch: EpisodeBatch, t_env: int, log):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        states = batch["state"][:, :-1]

        #build q
        inputs = self.critic._build_inputs(batch, bs, max_t)
        q_vals,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10 = self.critic.forward(inputs)
        q_vals = q_vals.view(bs,max_t,self.n_agents,-1)
        q_vals = q_vals.detach()[:, :-1]
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0


        # Calculated baseline
        q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3)
        pi = mac_out.view(-1, self.n_actions)
        baseline = th.sum(mac_out * q_vals, dim=-1).view(-1).detach()

        
        # Calculate policy grad with mask
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)
        coe = self.mixer.k(states).view(-1)
        #entropy = -(pi*log_pi_taken)
                    
        advantages = (q_taken.view(-1) - baseline).detach()

        coma_loss = - ((coe * (advantages)* log_pi_taken ) * mask).sum() / mask.sum()
        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        #compute parameters sum for debugging
        p_sum = 0.
        for p in self.agent_params:
            p_sum += p.data.abs().sum().item() / 100.0


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(log["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean", "q_max_mean", "q_min_mean", "q_max_var", "q_min_var"]:
                self.logger.log_stat(key, sum(log[key])/ts_logged, t_env)
            self.logger.log_stat("q_max_first", log["q_max_first"], t_env)
            self.logger.log_stat("q_min_first", log["q_min_first"], t_env)
            #self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env
      



    def train(self, batch: EpisodeBatch, t_env: int, log):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        states = batch["state"][:, :-1]

        #build q
        inputs = self.critic._build_inputs(batch, bs, max_t)
        q_vals,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10 = self.critic.forward(inputs)
        
        q_vals = q_vals.view(bs,max_t,self.n_agents,-1)
        q_vals = q_vals.detach()[:, :-1]


        #combined_tensor = th.cat((q1,q2,q3,q4,q5,q6,q7,q8,q9,q10), dim=-1)
        #variance = th.var(combined_tensor,dim=-1)
        #total_var = th.sum(variance,dim=-1)
        #total_var = total_var.unsqueeze(dim=-1)
        #total_var = total_var.clone().detach()
        #total_var = total_var[:,:-1]


        mac_out = []
        '''
        a1_out = []
        a2_out = []
        a3_out = []
        a4_out = []
        a5_out = []
        a6_out = []
        a7_out = []
        '''
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            '''
            agent_outs,a1,a2,a3,a4,a5,a6,a7 = self.mac.forward(batch, t=t,training=True)
            a1_out.append(a1)
            a2_out.append(a2)
            a3_out.append(a3)
            a4_out.append(a4)
            a5_out.append(a5)
            a6_out.append(a6)
            a7_out.append(a7)
            '''
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        '''
        a1_out = th.stack(a1_out, dim=1)  # Concat over time
        a2_out = th.stack(a2_out, dim=1)  # Concat over time
        a3_out = th.stack(a3_out, dim=1)  # Concat over time
        a4_out = th.stack(a4_out, dim=1)  # Concat over time
        a5_out = th.stack(a5_out, dim=1)  # Concat over time
        a6_out = th.stack(a6_out, dim=1)
        a7_out = th.stack(a7_out, dim=1)
        '''
        #mac_out = th.stack(mac_out, dim=1)  # Concat over time
        '''
        q_list = [a1_out,a2_out,a3_out,a4_out,a5_out,a6_out,a7_out]
        mean = (a1_out+a2_out+a3_out+a4_out+a5_out+a6_out+a7_out)/th.tensor(7)
        
        a1_out = a1_out.view(mac_out.shape[0],mac_out.shape[1],mac_out.shape[2],mac_out.shape[3])
        a2_out = a2_out.view(mac_out.shape[0],mac_out.shape[1],mac_out.shape[2],mac_out.shape[3])
        a3_out = a3_out.view(mac_out.shape[0],mac_out.shape[1],mac_out.shape[2],mac_out.shape[3])
        a4_out = a4_out.view(mac_out.shape[0],mac_out.shape[1],mac_out.shape[2],mac_out.shape[3])
        a5_out = a5_out.view(mac_out.shape[0],mac_out.shape[1],mac_out.shape[2],mac_out.shape[3])
        a6_out = a6_out.view(mac_out.shape[0],mac_out.shape[1],mac_out.shape[2],mac_out.shape[3])
        a7_out = a7_out.view(mac_out.shape[0],mac_out.shape[1],mac_out.shape[2],mac_out.shape[3])
        mwan = mean.view(mac_out.shape[0],mac_out.shape[1],mac_out.shape[2],mac_out.shape[3])

        a1_out[avail_actions == 0] = th.tensor(float('-inf'))
        a2_out[avail_actions == 0] = th.tensor(float('-inf'))
        a3_out[avail_actions == 0] = th.tensor(float('-inf'))
        a4_out[avail_actions == 0] = th.tensor(float('-inf'))
        a5_out[avail_actions == 0] = th.tensor(float('-inf'))
        a7_out[avail_actions == 0] = th.tensor(float('-inf'))

        total_div=None
        count=0
        mean = mean.clone().detach()
        
        #mean = mean*mask
        #mean[mean==0]=th.tensor(float('-inf'))
        mean = th.softmax(mean,dim=-1)
        mean[th.isnan(mean)] = 0

        for i in  q_list:
            #ii = i*mask
            #ii[ii==0]=th.tensor(float('-inf'))
            ii = th.softmax(i,dim=-1)
            d = ii*mean
            d = d.clone().detach()
            s_t = th.log(th.sum(th.sqrt(d),dim=-1))
            s_t[th.isnan(s_t)] = 0
            s_t.requires_grad=True
            if(count==0):
                total_div = s_t.clone()
                count=1
            else:
                 total_div = total_div +s_t.clone()


        total_div = total_div.sum()/mask.sum()

        total_div = th.clamp(total_div*th.tensor(-0.0001),-0.1,0.1)
        total_div = total_div/th.tensor(self.n_agents)
        '''







        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0





        # Calculated baseline
        #####q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3)
        pi = mac_out.view(-1, self.n_actions)
        #print(q_vals.shape)
        baseline = th.sum(mac_out * q_vals, dim=-1).view(-1)
        #x = th.gather(q_vals, dim=3, index=actions).squeeze(3)
        
        
        coma_loss = -(self.mixer.forward(baseline.view(actions.shape[0],actions.shape[1],actions.shape[2]), states,False,None)* mask).sum() / mask.sum()
        
        # Calculate policy grad with mask
        #pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        #pi_taken[mask == 0] = 1.0
        #log_pi_taken = th.log(pi_taken)
        #coe = self.mixer.k(states).view(-1)
        #entropy = -(pi*log_pi_taken)
                    
        #advantages = (q_taken.view(-1) - baseline).detach()

        #coma_loss = - ((coe * (advantages)* log_pi_taken ) * mask).sum() / mask.sum()
        #coma_loss = - ((coe * (advantages)* log_pi_taken+total_div ) * mask).sum() / mask.sum()
        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        #compute parameters sum for debugging
        p_sum = 0.
        for p in self.agent_params:
            p_sum += p.data.abs().sum().item() / 100.0

        '''
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(log["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean", "q_max_mean", "q_min_mean", "q_max_var", "q_min_var"]:
                self.logger.log_stat(key, sum(log[key])/ts_logged, t_env)
            self.logger.log_stat("q_max_first", log["q_max_first"], t_env)
            self.logger.log_stat("q_min_first", log["q_min_first"], t_env)
            #self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env
        '''
    def train_critic(self, on_batch, best_batch=None, log=None):
        bs = on_batch.batch_size
        max_t = on_batch.max_seq_length
        rewards = on_batch["reward"][:, :-1]
        actions = on_batch["actions"][:, :]
        terminated = on_batch["terminated"][:, :-1].float()
        mask = on_batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = on_batch["avail_actions"][:]
        states = on_batch["state"]




        
        # original:
        #build_target_q
        target_inputs = self.target_critic._build_inputs(on_batch, bs, max_t)
    
        target_q_vals,t_q1,t_q2,t_q3,t_q4,t_q5,t_q6,t_q7,t_q8,t_q9,t_q10 = self.target_critic.forward(target_inputs)
        target_q_vals = target_q_vals.detach()
        




        '''
        target_list = []
        q_t_list = [t_q1,t_q2,t_q3,t_q4,t_q5,t_q6,t_q7,t_q8,t_q9,t_q10]
        for i in q_t_list:
            t_q = i.detach()
            t_q = th.gather(t_q, dim=3, index=actions).squeeze(3)
            t_q = self.target_mixer.forward(t_q, states)
            t_q = t_q.detach().numpy()
            target_list.append(t_q)
        m = np.minimum.reduce(target_list)
        targets_taken = th.tensor(m)
        '''
        #combined_tensor = th.cat((t_q1,t_q2,t_q3,t_q4,t_q5,t_q6,t_q7,t_q8,t_q9,t_q10), dim=-1)
        '''
        all_tensors = [t_q1.clone().detach(),t_q2.clone().detach(),t_q3.clone().detach(),t_q4.clone().detach(),
                       t_q5.clone().detach(),t_q6.clone().detach(),t_q7.clone().detach(),t_q8.clone().detach(),
                       t_q9.clone().detach(),t_q10.clone().detach()]
        mone = th.zeros_like(target_q_vals)
        mechane = th.zeros_like(target_q_vals)
        for a in all_tensors:
            mone+=th.pow(a-target_q_vals,4)
            mechane+=th.pow(a-target_q_vals,2)

        #mone = th.sum(mone,dim=-1)
        #mechane = th.sum(mechane,dim=-1)
        mechane = th.pow(mechane,2)
        moment = ((th.tensor(10))*mone)/mechane - th.tensor(6)
        moment[moment<0]=th.tensor(0.0)
        moment = th.mean(moment,dim=-1)
        moment = th.tensor(1)/(th.tensor(1)+ th.exp(0.1*moment))
        moment = moment+th.tensor(0.5)
        moment = th.clamp(moment,0,1)
        moment = moment.clone().detach()
        '''

        '''
        moment = th.mean(moment,dim=-1)
        moment = moment.unsqueeze(dim=-1)
        moment = moment.clone().detach()
        '''

        #moment = moment[:,:-1]
        

        #variance = th.var(combined_tensor,dim=-1)
        #total_var = th.sum(variance,dim=-1)
        #total_var = total_var.unsqueeze(dim=-1)
        #total_var = total_var.clone().detach()
        #total_var = total_var[:,:-1]
        
        
        '''
        # Flatten the tensor
        flattened_tensor = total_var.view(-1)
        # Convert to NumPy array
        numpy_array = flattened_tensor.numpy()

        x = min(len(self.numpy_list)-1,self.index+len(numpy_array)))
        self.numpy_list[self.index:x]=numpy_array[:(x-self.index)]
        self.index= (self.index+len(numpy_array))%249999
        qu = np,quantile(self.numpy_list,0.9)
        qu = th.tensor(qu)
        total_var[total_var<qu]=0
        '''
        
        
        # original:
        targets_taken = self.target_mixer(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), states, False,None)
            
        #targets_taken = targets_taken#*moment
        

        target_q = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda).detach()
        #target_q = target_q- moment#th.tensor(0.99)*th.tensor(0.001)*th.clamp(total_var,0,self.bound)
        inputs = self.critic._build_inputs(on_batch, bs, max_t)
        

        #mac_out = []
        #self.mac.init_hidden(bs)
        #for i in range(max_t):
        #    agent_outs = self.mac.forward(on_batch, t=i)
        #    mac_out.append(agent_outs)
        #mac_out = th.stack(mac_out, dim=1).detach()
        # Mask out unavailable actions, renormalise (as in action selection)
        #mac_out[avail_actions == 0] = 0
        #mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        #mac_out[avail_actions == 0] = 0
        



        
        # Calculate 1-step Q-Learning targets
        targets = target_q #rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        q_vals,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10  = self.critic.forward(inputs)
        
        q_vals = th.gather(q_vals, 3, index=actions).squeeze(3)
        q_vals = self.mixer.forward(q_vals, states,False,None)
        # Td-error
        q_vals = q_vals[:, :-1, :]
        td_error = (q_vals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        critic_loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        self.mixer_optimiser.zero_grad()
        critic_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.c_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.mixer_optimiser.step()
        self.critic_training_steps += 1
        




        if best_batch is not None:
            best_target_q, best_inputs, best_mask, best_actions, best_mac_out, best_moment= self.train_critic_best(best_batch)
            #best_target_q2, best_inputs2, best_mask2, best_actions2, best_mac_out2, best_moment2= self.train_critic_best(on_batch)

            log["best_reward"] = th.mean(best_batch["reward"][:, :-1].squeeze(2).sum(-1), dim=0)
            '''
            target_q = th.cat((best_target_q, best_target_q2), dim=0)
            inputs = th.cat((best_inputs, best_inputs2), dim=0)
            mask = th.cat((best_mask, best_mask2), dim=0)
            actions = th.cat((best_actions, best_actions2), dim=0)
            states = th.cat((best_batch["state"],states), dim=0)
            #mac_out = th.cat((mac_out,best_mac_out), dim=0)
            #moment = th.cat((moment,best_moment), dim=0)
            '''

            target_q = best_target_q#th.cat((target_q, best_target_q), dim=0)
            inputs = best_inputs #th.cat((inputs, best_inputs), dim=0)
            mask = best_mask #th.cat((mask, best_mask), dim=0)
            actions = best_actions #th.cat((actions, best_actions), dim=0)
            states = best_batch["state"]#th.cat((states, best_batch["state"]), dim=0)
            #mac_out = best_mac_out #th.cat((mac_out,best_mac_out), dim=0)
            #moment = th.cat((moment,best_moment), dim=0)
            

        #train critic
        #mac_out = mac_out.detach()
        for t in range(max_t - 1):
            mask_t = mask[:, t:t+1]
            if mask_t.sum() < 0.5:
                continue
            #k = self.mixer.k(states[:, t:t+1]).unsqueeze(3)
            #b = self.mixer.b(states[:, t:t+1])
            q_vals,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10  = self.critic.forward(inputs[:, t:t+1])
            
            
            ###############
            #combined_tensor = th.cat((q1,q2,q3,q4,q5,q6,q7,q8,q9,q10), dim=-1)
            # Calculate the variance
            
            #variance = th.var(combined_tensor,dim=-1)
            
            #variance = variance*mask_t
            #total_var = th.sum(variance,dim=(-1,-2))
            #total_var = total_var.unsqueeze(dim=-1)
            #total_var = total_var.clone().detach()
               
            mean = (q1+q2+q3+q4+q5+q6+q7+q8+q9+q10)/th.tensor(10)
            #combined_tensor = th.stack((q1,q2,q3,q4,q5,q6,q7,q8,q9,q10), dim=-1)
            #product = th.prod(combined_tensor,dim=-1)
            #product = th.pow(product, 1/10)
        
            q_list = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]
            '''
            total_div=None
            mean = mean.clone().detach()
            mean = mean*mask_t
            mean = th.softmax(mean-dim=-1)
            for count,i in enumerate(q_list):
            
                ii = th.softmax(i,dim=-1)*men*mask_t
                #ii[ii==0]=1
                ii = th.sum(ii,dim=-1)
                x_i = th.log(ii)*mask_t
                x_i[th.isnan(x_i)] = 0
                #x_mean = (i/mean).clone().detach()
                #x_mean[th.isnan(x_mean)] = 0
                
                
                if(count==0):
                    total_div = x_i.clone()
                else:
                    total_div = total_div +x_i
             
            total_div = total_div/th.tensor(100*self.n_agents)
            total_div = total_div.sum()/mask_t.sum()
            total_div = th.clamp(total_div*th.tensor(-0.00001),-0.1,0.1)
            '''
            '''
            total_div=None
            count=0
            for i in q_list:
                ii = i*mask_t
                ii[ii==0]=-100000000
                ii = th.softmax(ii,dim=-1)
                for j in q_list: 
                    jj = j*mask_t
                    jj[jj==0]=-10000000
                    jj = th.softmax(jj,dim=-1)
                    i_j = th.abs(ii-jj)
                    if(count==0):
                        total_div = i_j.clone()
                        count=1
                    else:
                        total_div = total_div +i_j
            #mean=mean*mask_t
            #mean[mean==0]=-1000000
            '''
            
            total_div=None
            count=0
            mean = mean.clone().detach()
            mean = mean*mask_t
            mean[mean==0]=th.tensor(float('-inf'))
            mean = th.softmax(mean,dim=-1)
            mean[th.isnan(mean)] = 0            
            
            for i in  q_list:
                ii = i*mask_t
                ii[ii==0]=th.tensor(float('-inf'))
                ii = th.softmax(ii,dim=-1)
                d = ii*mean
                d = d.clone().detach()
                s_t = th.log(th.sum(th.sqrt(d),dim=-1))
                s_t[th.isnan(s_t)] = 0
                s_t.requires_grad=True
                if(count==0):
                    total_div = s_t.clone()
                    count=1
                else:
                    total_div = total_div +s_t.clone()
                    

            total_div = total_div.sum()/mask_t.sum()
            
            total_div = th.clamp(total_div*th.tensor(0.0001),-0.1,0.1)
            #total_div = th.clamp(total_div*th.tensor(0.001),-0.1,0.1)

            total_div = total_div/th.tensor(self.n_agents)
             

            q_ori = q_vals
            
            q_vals = th.gather(q_vals, 3, index=actions[:, t:t+1]).squeeze(3)
            q_vals = self.mixer.forward(q_vals, states[:, t:t+1],False,None)
            
            target_q_t = target_q[:, t:t+1].detach()
        
            q_err = (q_vals - target_q_t) * mask_t
         
            '''
            # Flatten the tensor
            flattened_tensor = moment[:,t:t+1].view(-1)
            # Convert to NumPy array
            numpy_array = flattened_tensor.numpy()

            x = min(len(self.numpy_list)-1,self.index+len(numpy_array))
            self.numpy_list[self.index:x]=numpy_array[:(x-self.index)]
            
            if((self.index+len(numpy_array))>649998):
                self.first_ind = 649999
            else:
                self.first_ind = self.index+len(numpy_array)
            self.index= (self.index+len(numpy_array))%649999
            '''
            #qu = np.quantile(self.numpy_list[:self.first_ind],0.95)
            #qu = np.max(self.numpy_list[:self.first_ind])
            #qu = np.mean(self.numpy_list[:self.first_ind])
            #qu = th.tensor(qu)
            
            #total_var[total_var<qu]=th.tensor(0.0)
            #total_var1 = total_var-th.tensor(5)*(qu-total_var)
            #total_var[total_var1<0]=th.tensor(0.0)

            #non_zero_count = th.sum(total_var != 0)
            #print("Number of non-zero elements:", non_zero_count.item())
            '''
            moments_new = moment[:,t:t+1]-th.tensor(qu)
            moments_new[moments_new>0]=th.tensor(0.0)
            sig = th.tensor(1)/(th.tensor(1)+ th.exp(moments_new))
             

            sig = sig+th.tensor(0.5)
            sig = sig.clone().detach()
            
            q_err = q_err.squeeze()*sig.squeeze()
            '''
            if(th.isnan(total_div)):
                 critic_loss = (q_err ** 2).sum() / mask_t.sum()
            else:       
                critic_loss = (q_err ** 2).sum() / mask_t.sum()
                critic_loss+=total_div
             

            ''' 
            if(t==(max_t-2)):
                #################
                regularization_norm = []
                # Get the norm of parameters for each network
                net = [self.critic.network1,self.critic.network2,self.critic.network3,self.critic.network4,self.critic.network5,
                        self.critic.network6,self.critic.network7,self.critic.network8,self.critic.network9,self.critic.network10]
                for network in net:
                    params = th.nn.utils.parameters_to_vector(network.parameters())
                    params_norm = th.norm(params,p=2)
                    regularization_norm.append(params_norm)

                # Stack the tensors along a new dimension (default is dim=0)
                stacked_tensor = th.stack(regularization_norm)

                # Calculate the product of all elements in the stacked tensor
                product = th.prod(stacked_tensor)
                product = th.pow(product, 1/10)

                total_norm = th.tensor(0.0)
                for i in regularization_norm:
                    total_norm += th.pow(th.log(i)- th.log(product),2)
                total_norm = total_norm/th.tensor(10)
                #total_norm = total_norm.clone().detach()
                ################
                critic_loss+=th.tensor(-0.001)*total_norm
            '''
        
            #Here introduce the loss for Qi
            #v_vals = th.sum(q_ori * mac_out[:, t:t+1], dim=3, keepdim=True)
            #ad_vals = q_ori - v_vals
            #goal = th.sum(k * v_vals, dim=2, keepdim=True) + k * ad_vals
            #goal_err = (goal - q_ori) * mask_t
            #goal_loss = 0.1 * (goal_err ** 2).sum() / mask_t.sum()/ self.args.n_actions
            #critic_loss += goal_loss
            self.critic_optimiser.zero_grad()
            self.mixer_optimiser.zero_grad()
            critic_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.c_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.mixer_optimiser.step()
            self.critic_training_steps += 1

            log["critic_loss"].append(critic_loss.item())
            log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            log["td_error_abs"].append((q_err.abs().sum().item() / mask_elems))
            log["target_mean"].append((target_q_t * mask_t).sum().item() / mask_elems)
            log["q_taken_mean"].append((q_vals * mask_t).sum().item() / mask_elems)
            log["q_max_mean"].append((th.mean(q_ori.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)
            log["q_min_mean"].append((th.mean(q_ori.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)
            log["q_max_var"].append((th.var(q_ori.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)
            log["q_min_var"].append((th.var(q_ori.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)

            if (t == 0):
                log["q_max_first"] = (th.mean(q_ori.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems
                log["q_min_first"] = (th.mean(q_ori.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems

        #update target network
        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps



    def train_critic_best(self, batch):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask_t = batch["filled"]
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:]
        states = batch["state"]
        old_outs = batch["outs"][:, :]
        
        # pr for all actions of the episode
        mac_out = []
        self.mac.init_hidden(bs)
        for i in range(max_t):
            agent_outs = self.mac.forward(batch, t=i)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1).detach()
        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0
        critic_mac = th.gather(mac_out, 3, actions).squeeze(3).prod(dim=2, keepdim=True)
    
        old_outs = old_outs.squeeze(3).prod(dim=2, keepdim=True)
        imprtance_sampling = critic_mac/old_outs
        imprtance_sampling = imprtance_sampling.to(dtype=th.float32)
        imprtance_sampling = imprtance_sampling.to('cuda')
        imprtance_sampling = th.where(th.isnan(imprtance_sampling) | (imprtance_sampling == 0), imprtance_sampling.new_tensor(float('inf')), imprtance_sampling)

        #imprtance_sampling = th.where(th.isnan(imprtance_sampling) | (imprtance_sampling == 0), th.tensor(float('inf'), dtype=th.float32).to('cuda'), imprtance_sampling)

        #target_q take
        target_inputs = self.target_critic._build_inputs(batch, bs, max_t)
        target_q_vals,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10 = self.target_critic.forward(target_inputs)
        target_q_vals = target_q_vals.detach()
        #targets_taken = self.target_mixer(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), states,False,None)

         
        all_tensors = [q1.clone().detach(),q2.clone().detach(),q3.clone().detach(),q4.clone().detach(),
                       q5.clone().detach(),q6.clone().detach(),q7.clone().detach(),q8.clone().detach(),
                       q9.clone().detach(),q10.clone().detach()]

        #variance = th.var(all_tensors,dim=-1)
        #variance = variance*mask_t
        #moment = th.tensor(1)/(th.tensor(1)+ th.exp(variance))

        mone = th.zeros_like(target_q_vals)
        mechane = th.zeros_like(target_q_vals)
        for a in all_tensors:
            mone+=th.pow(a-target_q_vals,4)
            mechane+=th.pow(a-target_q_vals,2)

        mechane = th.pow(mechane,2)
        moment = mone/mechane#-th.tensor(3)
        #moment[moment<0]=th.tensor(0.0)
        
        moment = th.mean(moment,dim=-1) #mean(moment,dim=-1)
        
        #normalized = th.tanh(moment/2)      # ~ [-1, 1]

        #moment = 5 * normalized + 6 #10 * normalized   
        
        
        moment = th.tensor(1)/(th.tensor(1)+ th.exp(moment*0.5))
        moment = moment+th.tensor(0.5)
        #moment = moment
        moment = th.clamp(moment,0,1) 
         
        moment = moment.clone().detach()
        
    
        coe = self.target_mixer.k(states).view(-1)
        coe = th.mean(coe,dim=-1)
        coe = coe.detach()
        moment = moment*coe
        moment = moment/coe.sum()
        

        moment = th.mean(moment,dim=-1) # th.prod(moment,dim=-1)
        moment = moment.unsqueeze(dim=-1)
        moment = moment.clone().detach()
        '''
        coe = self.target_mixer.k(states).view(-1)
        coe = th.mean(coe,dim=-1)
        coe = coe.detach()
        moment = moment*coe
        '''
        moment = moment[:,:-1]
        #moment = th.mean(moment,dim=-1)        
         
        moment = th.min(moment,imprtance_sampling[:,:-1])
        #moment = th.min(th.tensor(1),imprtance_sampling[:,:-1])
        #moment = th.min(imprtance_sampling[:, :-1], imprtance_sampling.new_tensor(1.0))
        
        targets_taken = self.target_mixer(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), states,False,None)
        #sig = th.tensor(1)/(th.tensor(1)+ th.exp(0.01*moment))
             
        #sig = sig+th.tensor(0.5)
        #sig = sig.clone().detach()

        #expected q
        exp_q = self.build_exp_q(target_q_vals, mac_out, states).detach()
        # td-error
        targets_taken[:, -1] = targets_taken[:, -1] #* (1 - th.sum(terminated, dim=1))
        targets_taken = targets_taken.detach()     
        targets_taken = targets_taken
        
        
        exp_q[:, -1] = exp_q[:, -1] * (1 - th.sum(terminated, dim=1))
        targets_taken[:, :-1] = targets_taken[:, :-1] * mask
        exp_q[:, :-1] = exp_q[:, :-1] * mask
        
        #td_q = (rewards + moment*(self.args.gamma * exp_q[:, 1:]) - targets_taken[:, :-1]) * mask
        td_q = (rewards + self.args.gamma * exp_q[:, 1:] - targets_taken[:, :-1]) * mask
        
        #print(critic_mac.shape,moment.shape)
          
        new_weight = moment # th.tensor(0.5)*critic_mac + th.tensor(0.5)*moment # th.ones_like(critic_mac)
        
        #new_weight = th.cumprod(new_weight, dim=-1)
        #powers = th.arange(1, new_weight.shape[1] + 1, dtype=th.float).to('cuda').view(1, -1, 1)
        #new_weight = th.pow(new_weight, powers)
             
        #new_weight = th.cumsum(new_weight,dim=-1)
        #new_weight = 1/new_weight
        
        #new_weight = new_weight/new_weight.sum()
        
        #new_weight = new_weight.detach()
        #compute target
        
        target_q =  build_target_q(td_q, targets_taken[:, :-1], new_weight, mask, self.args.gamma, self.args.tb_lambda, self.args.step).detach()
        
        '''
        target_q =  build_target_q(td_q, targets_taken[:, :-1], mask, self.args.gamma, self.args.tb_lambda, self.args.step).detach()
        '''
        #target_q = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda).detach()
        target_q = target_q.detach()
        #combined_tensor = th.cat((q1,q2,q3,q4,q5,q6,q7,q8,q9,q10), dim=-1)

        # Calculate the variance
        #variance = th.var(combined_tensor,dim=-1)
        #all_weights = all_weights.view(variance.shape[0],variance.shape[1],variance.shape[2])
        #####variance = variance*all_weights

        #total_var = th.sum(variance,dim=-1)
        #total_var = total_var.unsqueeze(dim=-1)
        #total_var = total_var.clone().detach()
        #total_var = total_var[:,:-1]

        #target_q = target_q-th.tensor(0.99)*th.tensor(0.001)*th.clamp(total_var,0,self.bound)
        
        '''
        all_tensors = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]
        mone = th.zeros_like(target_q_vals)
        mechane = th.zeros_like(target_q_vals)
        for a in all_tensors:
            mone+=th.pow(a-target_q_vals,4)
            mechane+=th.pow(a-target_q_vals,2)

        #mone = th.sum(mone,dim=-1)
        #mechane = th.sum(mechane,dim=-1)
        mechane = th.pow(mechane,2)
        moment = ((th.tensor(10))*mone)/mechane
        moment = th.mean(moment,dim=-1)
        moment = th.sum(moment,dim=-1)
        moment = moment.unsqueeze(dim=-1)
        moment = moment.clone().detach()
        moment = moment[:,:-1]
        '''
        moment = 0  
        inputs = self.critic._build_inputs(batch, bs, max_t)

        return target_q, inputs, mask, actions, mac_out, moment


    def build_exp_q(self, target_q_vals, mac_out, states):
        target_exp_q_vals = th.sum(target_q_vals * mac_out, dim=3)
        target_exp_q_vals = self.target_mixer.forward(target_exp_q_vals, states,False,None)
        return target_exp_q_vals

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.mixer.cuda()
        self.target_critic.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.mixer_optimiser.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
       # self.target_critic.load_state_dict(self.critic.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimiser.load_state_dict(th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))
