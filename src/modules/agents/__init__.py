REGISTRY = {}

from .rnn_agent import RNNAgent
from .grnn_agent import GRNNAgent
from .depth_route_net import DepthRouteNet
REGISTRY["rnn"] = RNNAgent
REGISTRY["grnn"] = GRNNAgent
REGISTRY["soft_new"] = DepthRouteNet
