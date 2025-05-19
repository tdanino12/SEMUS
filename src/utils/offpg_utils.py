import torch as th

def build_target_q(td_q, target_q, mac, mask, gamma, td_lambda, n):
    aug = th.zeros_like(td_q[:, :1])

    #Tree diagram
    #mac = mac[:, :-1]
    tree_q_vals = th.zeros_like(td_q)
    coeff = 1.0
    t1 = td_q[:]
    for i in range(10):#range(n):
        tree_q_vals += t1 * coeff

        out = mac.unfold(1, i, 1).prod(dim=-1) #sum(dim=-1)
        out = out[:, :mac.shape[1]]
        mac_copy = mac.clone()
        mac_copy[:, :out.size(1), :] = out
         
        #running_weight = 1/(mac_copy+th.tensor(1))
        t1 = th.cat(((t1* mac_copy)[:, 1:], aug), dim=1)
        #t1 = th.cat(((t1*th.tensor(1))[:, 1:], aug), dim=1)
        coeff *= gamma * td_lambda
        #coeff *= gamma

    return target_q + tree_q_vals

'''
def build_target_q(td_q, target_q, mask, gamma, td_lambda, n):
    aug = th.zeros_like(td_q[:, :1])

    #Tree diagram
    #mac = mac[:, :-1]
    tree_q_vals = th.zeros_like(td_q)
    coeff = 1.0
    t1 = td_q[:]
    for _ in range(n):
        tree_q_vals += t1 * coeff
        t1 = th.cat(((t1)[:, 1:], aug), dim=1)
        #t1 = th.cat(((t1)[:, 1:], aug), dim=1)
        coeff *= gamma * td_lambda
        #coeff *= gamma

    return target_q + tree_q_vals


def build_target_q(td_q, target_q, mac, mask, gamma, td_lambda, n):
    aug = th.zeros_like(td_q[:, :1])

    #Tree diagram
    mac = mac[:, :-1]
    tree_q_vals = th.zeros_like(td_q)
    coeff = 1.0
    t1 = td_q[:]
    for _ in range(n):
        tree_q_vals += t1 * coeff
        t1 = th.cat(((t1 * mac)[:, 1:], aug), dim=1)
        #t1 = th.cat(((t1)[:, 1:], aug), dim=1)        
        coeff *= gamma * td_lambda
        #coeff *= gamma 

    return target_q + tree_q_vals
'''









