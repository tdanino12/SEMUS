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

    def start_training(self,on_batch, off_episode_sample, episode_sample, log, t_env: int):
        mac_out = []
        self.mac.init_hidden(off_episode_sample.batch_size)
        for t in range(off_episode_sample.max_seq_length):
            agent_outs = self.mac.forward(off_episode_sample, t=t)
            mac_out.append(agent_outs)
        mac_out_critic = th.stack(mac_out, dim=1)  # Concat over time
        mac_out_actor = th.stack(mac_out[:-1], dim=1)  # Concat over time
        self.train_critic(on_batch, best_batch=off_episode_sample, log=log, mac_off = mac_out_critic)
        self.train(off_episode_sample, t_env, log, mac_out_actor)
    
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
      



    def train(self, batch: EpisodeBatch, t_env: int, log, mac_out_off):
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

        '''
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        '''
        mac_out = mac_out_off
        
        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        pi = mac_out.view(-1, self.n_actions)
        baseline = th.sum(mac_out * q_vals, dim=-1).view(-1)    
        coma_loss = -(self.mixer.forward(baseline.view(actions.shape[0],actions.shape[1],actions.shape[2]), states,False,None)* mask).sum() / mask.sum()
        
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
    def train_critic(self, on_batch, best_batch=None, log=None, mac_off = None):
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
        
    
        # original:
        targets_taken = self.target_mixer(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), states, False,None)

        
        target_q = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda).detach()
        inputs = self.critic._build_inputs(on_batch, bs, max_t)
          
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
            best_target_q, best_inputs, best_mask, best_actions, best_mac_out, best_moment= self.train_critic_best(best_batch, mac_off)

            log["best_reward"] = th.mean(best_batch["reward"][:, :-1].squeeze(2).sum(-1), dim=0)
            target_q = best_target_q#th.cat((target_q, best_target_q), dim=0)
            inputs = best_inputs #th.cat((inputs, best_inputs), dim=0)
            mask = best_mask #th.cat((mask, best_mask), dim=0)
            actions = best_actions #th.cat((actions, best_actions), dim=0)
            states = best_batch["state"]#th.cat((states, best_batch["state"]), dim=0)

            
        #train critic
        for t in range(max_t - 1):
            mask_t = mask[:, t:t+1]
            if mask_t.sum() < 0.5:
                continue
            q_vals,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10  = self.critic.forward(inputs[:, t:t+1])     
            mean = (q1+q2+q3+q4+q5+q6+q7+q8+q9+q10)/th.tensor(10)
        
            q_list = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]

            
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
         
   
            if(th.isnan(total_div)):
                 critic_loss = (q_err ** 2).sum() / mask_t.sum()
            else:       
                critic_loss = (q_err ** 2).sum() / mask_t.sum()
                critic_loss+=total_div
            
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



    def train_critic_best(self, batch, mac_out_off):
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

        '''
        # pr for all actions of the episode
        mac_out = []
        self.mac.init_hidden(bs)
        for i in range(max_t):
            agent_outs = self.mac.forward(batch, t=i)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1).detach()
        '''
        mac_out = mac_out_off.detach()
        
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

         
        all_tensors = [q1.clone().detach(),q2.clone().detach(),q3.clone().detach(),q4.clone().detach(),
                       q5.clone().detach(),q6.clone().detach(),q7.clone().detach(),q8.clone().detach(),
                       q9.clone().detach(),q10.clone().detach()]

        mone = th.zeros_like(target_q_vals)
        mechane = th.zeros_like(target_q_vals)
        for a in all_tensors:
            mone+=th.pow(a-target_q_vals,4)
            mechane+=th.pow(a-target_q_vals,2)

        mechane = th.pow(mechane,2)
        moment = mone/mechane

        
        moment = th.mean(moment,dim=-1) #mean(moment,dim=-1)       
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
        moment = moment[:,:-1]
          
         
        moment = th.min(moment,imprtance_sampling[:,:-1])
        #moment = th.min(th.tensor(1),imprtance_sampling[:,:-1])
        #moment = th.min(imprtance_sampling[:, :-1], imprtance_sampling.new_tensor(1.0))
        
        targets_taken = self.target_mixer(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), states,False,None)


        #expected q
        exp_q = self.build_exp_q(target_q_vals, mac_out, states).detach()
        # td-error
        targets_taken[:, -1] = targets_taken[:, -1] #* (1 - th.sum(terminated, dim=1))
        targets_taken = targets_taken.detach()     
        targets_taken = targets_taken
        
        
        exp_q[:, -1] = exp_q[:, -1] * (1 - th.sum(terminated, dim=1))
        targets_taken[:, :-1] = targets_taken[:, :-1] * mask
        exp_q[:, :-1] = exp_q[:, :-1] * mask
        
        td_q = (rewards + self.args.gamma * exp_q[:, 1:] - targets_taken[:, :-1]) * mask
        
          
        new_weight = moment 
        
        target_q =  build_target_q(td_q, targets_taken[:, :-1], new_weight, mask, self.args.gamma, self.args.tb_lambda, self.args.step).detach()
    
        target_q = target_q.detach()
        
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
