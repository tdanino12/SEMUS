import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def calc_next_shape(input_shape, conv_info):
    """
    take input shape per-layer conv-info as input
    """
    out_channels, kernel_size, stride, padding = conv_info
    c, h, w = input_shape
    # for padding, dilation, kernel_size, stride in conv_info:
    h = int((h + 2*padding[0] - ( kernel_size[0] - 1 ) - 1 ) / stride[0] + 1)
    w = int((w + 2*padding[1] - ( kernel_size[1] - 1 ) - 1 ) / stride[1] + 1)
    return (out_channels, h, w )

def _fanin_init(tensor, alpha = 0):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = np.sqrt( 1. / ( (1 + alpha * alpha ) * fan_in) )
    return tensor.data.uniform_(-bound, bound)

def _uniform_init(tensor, param=3e-3):
    return tensor.data.uniform_(-param, param)

def _constant_bias_init(tensor, constant = 0.1):
    tensor.data.fill_( constant )

def _normal_init(tensor, mean=0, std =1e-3):
    return tensor.data.normal_(mean,std)

def layer_init(layer, weight_init = _fanin_init, bias_init = _constant_bias_init ):
    weight_init(layer.weight)
    if hasattr(layer, 'bias'):
        bias_init(layer.bias)

def basic_init(layer):
    layer_init(layer, weight_init = _fanin_init, bias_init = _constant_bias_init)

def uniform_init(layer):
    layer_init(layer, weight_init = _uniform_init, bias_init = _uniform_init )

def normal_init(layer):
    layer_init(layer, weight_init = _normal_init, bias_init = _normal_init)
    
def _orthogonal_init(tensor, gain = np.sqrt(2)):
    nn.init.orthogonal_(tensor, gain = gain)

def orthogonal_init(layer, scale = np.sqrt(2), constant = 0 ):
    layer_init(
        layer,
        weight_init= lambda x:_orthogonal_init(x, gain=scale),
        bias_init=lambda x: _constant_bias_init(x, 0))


def null_activation(x):
    return x

def sample_gumbel(logits):
    eps = 1e-20
    U = logits.clone()
    U.uniform_(0, 1)
    return -torch.log( -torch.log( U + eps))

class MLPBase(nn.Module):
    def __init__(self, input_shape, hidden_shapes, activation_func=F.relu, init_func = basic_init, last_activation_func = None ):
        super().__init__()
        
        self.activation_func = activation_func
        self.fcs = []
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        input_shape = np.prod(input_shape)

        self.output_shape = input_shape
        for i, next_shape in enumerate( hidden_shapes ):
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("fc{}".format(i), fc)

            input_shape = next_shape
            self.output_shape = next_shape

    def forward(self, x):

        out = x
        for fc in self.fcs[:-1]:
            out = fc(out)
            out = self.activation_func(out)
        out = self.fcs[-1](out)
        out = self.last_activation_func(out)
        return out


class DepthRouteModule(nn.Module):
    def __init__(
            self,
            module_input_shape,
            module_num,
            module_hidden,
            gate_input_shape,
            gate_hiddens,
            top_k = None,   # soft if None
            rescale_prob = True,
            route_as_sample = False,
            use_resnet = True,
            resrouting = False,
            explore_sample = False,
            temperature_sample = False,
            module_hidden_init_func = basic_init,
            activation_func = F.relu,
            ):
        super().__init__()

        self.module_num = module_num

        self.top_k = self.module_num if top_k is None else top_k
        self.rescale_prob = rescale_prob

        self.route_as_sample = route_as_sample
        self.use_resnet = use_resnet
        self.resrouting = resrouting
        self.explore_sample = explore_sample
        self.temperature_sample = temperature_sample

        self.activation_func = activation_func

        self.module_fcs = nn.ModuleList([])
        module_fc1 = nn.Linear(module_input_shape, module_hidden)
        module_hidden_init_func(module_fc1)
        self.module_fcs.append(module_fc1)

        for i in range(module_num-1):
            module_fc = nn.Linear(module_hidden, module_hidden)
            module_hidden_init_func(module_fc)
            self.module_fcs.append(module_fc)
        
        self.gates = nn.ModuleList([])
        gate_fc_input_shape = gate_input_shape
        for gate_hidden in gate_hiddens:
            gate_fc = nn.Linear(gate_fc_input_shape, gate_hidden)
            module_hidden_init_func(gate_fc)
            self.gates.append(gate_fc)
            gate_fc_input_shape = gate_hidden
        gate_output_shape = module_num * (module_num+1) // 2
        gate_fc_last = nn.Linear(gate_fc_input_shape, gate_output_shape)
        module_hidden_init_func(gate_fc_last)
        self.gates.append(gate_fc_last)

    def gate_forward(self, gate_input, idx):
        gate_out = gate_input
        for gate_fc in self.gates[:-1]:
            gate_out = gate_fc(gate_out)
            gate_out = self.activation_func(gate_out)
        gate_out = self.gates[-1](gate_out)
        
        gate_shape = [i for i in range(1, self.module_num+1)]
        gate_logits_list = torch.split(gate_out, gate_shape, dim=-1)

        return gate_logits_list
    
    def forward(self, module_input, gate_input, idx=None, gate_sample=None, explore=True, gumbel_temperature=1):
        gate_logits_list = self.gate_forward(gate_input, idx)
        gates_list, gates_onehot_list = [], []
        softmax_gates_list = []
        greater_avg_list = []
        if gate_sample is not None:
            sample_gates_onehot = gate_sample
            gate_shape = [i for i in range(1, self.module_num+1)]
            sample_gates_onehot_list = torch.split(sample_gates_onehot, gate_shape, dim=-1)
        else:
            sample_gates_onehot_list = [None for _ in range(1, self.module_num+1)]

        for up_module_num, gate_logits, sample_gates_onehot in \
            zip(range(1, self.module_num+1) ,gate_logits_list, sample_gates_onehot_list):
        
            if gate_sample is not None and self.route_as_sample:
                _, top_k_gate_indices = sample_gates_onehot.topk(min(self.top_k, up_module_num), dim=-1)
                top_k_gate_logits = torch.gather(gate_logits, dim=-1, index=top_k_gate_indices)
            else:
                if self.explore_sample and explore:
                    if self.temperature_sample:
                        temperature = gumbel_temperature[idx.long()].unsqueeze(-1)
                        sample_gate_logits = gate_logits / temperature
                        gate_probs = F.softmax(sample_gate_logits, dim=-1)
                    else:
                        gate_probs = F.softmax(gate_logits, dim=-1)

                    gate_logits_dim = gate_logits.shape
                    top_k_gate_indices = torch.multinomial(gate_probs.reshape([-1, gate_logits_dim[-1]]), min(self.top_k, up_module_num))
                    top_k_gate_indices = top_k_gate_indices.reshape(list(gate_logits_dim[:-1]) + [-1])
                    top_k_gate_logits = torch.gather(gate_logits, dim=-1, index=top_k_gate_indices)
                else:
                    top_k_gate_logits, top_k_gate_indices = gate_logits.topk(min(self.top_k, up_module_num), dim=-1)

            if self.rescale_prob:
                top_k_gates = F.softmax(top_k_gate_logits, dim=-1)
            else:
                soft_gates = F.softmax(gate_logits, dim=-1)
                top_k_gates = torch.gather(soft_gates, dim=-1, index=top_k_gate_indices)
                
            zeros = torch.zeros_like(gate_logits, requires_grad=True)
            gates = zeros.scatter(dim=-1, index=top_k_gate_indices, src=top_k_gates)

            top_k_gate_onehot = F.one_hot(top_k_gate_indices, up_module_num).float()
            gates_onehot = top_k_gate_onehot.sum(dim=-2)

            softmax_gates = F.softmax(gate_logits, dim=-1)
            greater_avg = (softmax_gates > (1 / (up_module_num+1))).float()

            gates_list.append(gates)
            gates_onehot_list.append(gates_onehot)
            softmax_gates_list.append(softmax_gates)
            greater_avg_list.append(greater_avg)
        

        out = self.activation_func(self.module_fcs[0](module_input))
        out = out.unsqueeze(-2)
        out_detach = out.detach()
        for module_fc, gate, greater_avg in zip(self.module_fcs[1:], gates_list[:-1], greater_avg_list[:-1]):
            if self.resrouting and self.training:
                greater_avg = greater_avg.unsqueeze(-1)
                out_now = greater_avg * out + (1-greater_avg) * out_detach 
                module_fc_input = out_now * gate.unsqueeze(-1)
                module_fc_input = torch.sum(module_fc_input, dim=-2, keepdim=False)
                module_fc_out = self.activation_func(module_fc(module_fc_input))
                module_fc_out_detach = module_fc_out.detach()
                if self.use_resnet:
                    module_fc_input_src = out * gate.unsqueeze(-1)
                    module_fc_input_src = torch.sum(module_fc_input_src, dim=-2, keepdim=False)
                    module_fc_out = module_fc_out + module_fc_input_src
                    module_fc_out_detach = module_fc_out_detach + module_fc_input_src
            else:
                module_fc_input = out * gate.unsqueeze(-1)
                module_fc_input = torch.sum(module_fc_input, dim=-2, keepdim=False)
                module_fc_out = self.activation_func(module_fc(module_fc_input))
                module_fc_out_detach = module_fc_out.detach()
                if self.use_resnet:
                    module_fc_out = module_fc_out + module_fc_input
                    module_fc_out_detach = module_fc_out_detach + module_fc_input

            out = torch.cat([out, module_fc_out.unsqueeze(-2)], dim=-2)
            out_detach = torch.cat([out_detach, module_fc_out_detach.unsqueeze(-2)], dim=-2)
        
        if self.resrouting and self.training:
            greater_avg_last = greater_avg_list[-1].unsqueeze(-1)
            out_now = greater_avg_last * out + (1-greater_avg_last) * out_detach
        else:
            out_now = out
            
        last_gate = gates_list[-1]
        last_out = out_now * last_gate.unsqueeze(-1)
        last_out = torch.sum(last_out, dim=-2, keepdim=False)

        return last_out, torch.cat(gates_list, dim=-1), torch.cat(gates_onehot_list, dim=-1), torch.cat(softmax_gates_list, dim=-1)

class DepthRouteNet(nn.Module):
    def __init__(
            self,
            task_num,
            output_shape,
            input_shape,
            hidden_shapes,
            em_hidden_shapes,
            module_hidden,
            module_num,
            gate_hiddens,
            top_k = None,
            rescale_prob = True,
            route_as_sample = False,
            use_resnet = False,
            resrouting = False,
            cond_ob = True,
            explore_sample = False,
            temperature_sample = False,
            module_hidden_init_func = basic_init,
            last_init_func = uniform_init,
            activation_func = F.relu,
            **kwargs):

        super().__init__()

        self.task_num = task_num

        self.base_input_shape = input_shape
        self.em_input_shape = 1

        self.base = MLPBase( 
                        last_activation_func = null_activation,
                        input_shape = self.base_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,
                        **kwargs)

        self.em_base = MLPBase(
                        last_activation_func = null_activation,
                        input_shape = self.em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,
                        **kwargs)

        self.activation_func = activation_func

        self.depth_route_net = DepthRouteModule(
                                module_input_shape = self.base.output_shape,
                                module_num = module_num,
                                module_hidden = module_hidden,
                                gate_input_shape = self.em_base.output_shape,
                                gate_hiddens = gate_hiddens,
                                top_k = top_k,
                                rescale_prob = rescale_prob,
                                route_as_sample = route_as_sample,
                                use_resnet = use_resnet,
                                resrouting = resrouting,
                                explore_sample = explore_sample,
                                temperature_sample = temperature_sample,
                                module_hidden_init_func = module_hidden_init_func,
                                activation_func = activation_func,       
                            )
        
        self.last = nn.Linear(module_hidden, output_shape)
        last_init_func(self.last)

        self.cond_ob = cond_ob

        self.gumbel_temperature = nn.Parameter(torch.ones(task_num))

    def forward(self, x, idx=None, gate_sample=None, explore=True, return_gate=False):
        explore=False
        base_x = x
        em_x = idx
        out = self.base(base_x)
        embedding = self.em_base(em_x)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        out, gates, gates_onehot, gates_softmax = self.depth_route_net(out, embedding, idx, gate_sample, explore, self.gumbel_temperature)

        out = self.last(out)

        if return_gate:
            return out, gates, gates_onehot, gates_softmax

        return out
    
    def update_gumbel_temperature(self, log_alpha):
        gumbel_temperature = F.softmax(-log_alpha.clone().detach(), dim=-1) * self.task_num
        with torch.no_grad():
            self.gumbel_temperature.copy_(gumbel_temperature)
    
    def copy(self):
        return copy.deepcopy(self)
