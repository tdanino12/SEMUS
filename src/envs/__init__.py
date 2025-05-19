from functools import partial
import sys
import os
from .gymma import GymmaWrapper
from .multiagentenv import MultiAgentEnv
from .lbforaging import ForagingEnv
from .starcraft import StarCraft2Env
#from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt
from .matrix_game.nstep_matrix_game import NStepMatrixGame
#from .pacmen_env.gym_foo.envs.pac_men import CustomEnv
from .particle import Particle
from .stag_hunt import StagHunt
try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except:
    gfootball = False

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)
def env_fn_foraging(env, **kwargs) -> MultiAgentEnv:
    del kwargs['state_last_action']
    return env(**kwargs)

def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    del kwargs['state_last_action']
    return GymmaWrapper(**kwargs)
def env_fn2(env, **kwargs) -> MultiAgentEnv:
    del kwargs['state_last_action']
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
#REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["foraging"] =  partial(env_fn_foraging, env=ForagingEnv)
REGISTRY["gymma"] = gymma_fn
#REGISTRY["pac_men"] = partial(env_fn2, env=CustomEnv)
REGISTRY["particle"] = partial(env_fn, env=Particle)
#REGISTRY["nstep_matrix"] = partial(env_fn, env=NStepMatrixGame)
if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
