import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            #masked_policies = th.clamp(masked_policies, min=0)
            masked_policies = masked_policies.float().cpu()
            picked_actions = Categorical(masked_policies).sample().long()
            picked_actions = picked_actions.to(agent_inputs.device)

            # random_numbers = th.rand_like(agent_inputs[:, :, 0])
            # pick_random = (random_numbers < self.epsilon).long()
            # random_actions = Categorical(avail_actions.float()).sample().long()
            # picked_actions = pick_random * random_actions + (1 - pick_random) * picked_actions

        if not (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
            print("########################################################")
            print("recursion")
            print("########################################################")
            return self.select_action_recursion(agent_inputs, avail_actions, t_env, test_mode)

        return picked_actions

    def select_action_recursion(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        
        avail_actions_cpu = avail_actions.float().cpu()
        random_actions = Categorical(avail_actions_cpu.float()).sample().long()
        random_actions = random_actions.to(agent_inputs.device)
        
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        if not (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
            print((th.gather(avail_actions, dim=2, index=random_actions.unsqueeze(2)) <= 0.99).squeeze())
            print((th.gather(avail_actions, dim=2, index=masked_q_values.max(dim=2)[1].unsqueeze(2)) <= 0.99).squeeze())
            print((th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) <= 0.99).squeeze())

            print('Action Selection Error')
            # raise Exception
            #return self.select_action(agent_inputs, avail_actions, t_env, test_mode)
            try:
                return self.select_action_recursion(agent_inputs, avail_actions, t_env, test_mode)
            except RecursionError as e:
                return self.select_action_recursion(agent_inputs, avail_actions, t_env, test_mode)
        return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        if not (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
            print((th.gather(avail_actions, dim=2, index=random_actions.unsqueeze(2)) <= 0.99).squeeze())
            print((th.gather(avail_actions, dim=2, index=masked_q_values.max(dim=2)[1].unsqueeze(2)) <= 0.99).squeeze())
            print((th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) <= 0.99).squeeze())

            print('Action Selection Error')
            # raise Exception
            return self.select_action(agent_inputs, avail_actions, t_env, test_mode)

        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
