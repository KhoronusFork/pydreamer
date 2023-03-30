# Ignore annoying warnings from imported envs
import warnings
warnings.filterwarnings("ignore", ".*Box bound precision lowered by casting")  # gym

try:
  import gymnasium as gym
  use_gymnasium = True
except:
  import gym
  use_gymnasium = False
import numpy as np

from .wrappers import *


def create_env(env_id: str, no_terminal: bool, env_time_limit: int, env_action_repeat: int, worker_id: int):

    print('create_env:{}'.format(env_id))

    if env_id.startswith('MiniGrid-'):
        from .minigrid import MiniGrid
        env = MiniGrid(env_id)

    elif env_id.startswith('Atari-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=env_action_repeat)

    elif env_id.startswith('AtariGray-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=env_action_repeat, grayscale=True)

    elif env_id.startswith('MiniWorld-'):
        import miniworld
        #import gym_miniworld.wrappers as wrap
        #print('env_id:{}'.format(env_id))
        #env = gym.make('MiniWorld-Hallway-v0', view='agent', render_mode="human")
        #env = gym.make('MiniWorld-Hallway-v0')#env_id)

        # changed the observation size
        env = gym.make(env_id, render_mode = 'rgb_array', obs_height = 64, obs_width = 64)
        env = DictWrapper(env)
        #env = wrap.DictWrapper(env)
        #env = wrap.MapWrapper(env)
        #env = wrap.AgentPosWrapper(env)
        if env_id.startswith('MiniWorld-ScavengerHunt'):
            env = wrap.GoalPosWrapper(env)
            env = wrap.GoalVisibleWrapper(env)
            env = wrap.GoalVisAgeWrapper(env)

    elif env_id.startswith('DmLab-'):
        from .dmlab import DmLab
        env = DmLab(env_id.split('-', maxsplit=1)[1].lower(), num_action_repeats=env_action_repeat)
        env = DictWrapper(env)
    
    elif env_id.startswith('DMM-'):
        from .dmm import DMMEnv
        env = DMMEnv(env_id.split('-', maxsplit=1)[1].lower(), num_action_repeats=env_action_repeat, worker_id=worker_id)
        env = DictWrapper(env)

    elif env_id.startswith('MineRL'):
        from .minerl import MineRL
        constr = lambda: MineRL(env_id, action_repeat=env_action_repeat)
        env = RestartOnExceptionWrapper(constr)

    elif env_id.startswith('DMC-'):
        from .dmc import DMC
        env = DMC(env_id.split('-', maxsplit=1)[1].lower(), action_repeat=env_action_repeat)
    
    elif env_id.startswith('Embodied-'):
        from .embodied import EmbodiedEnv
        task = env_id.split('-', maxsplit=1)[1].lower()
        env = EmbodiedEnv(task, action_repeat=env_action_repeat, time_limit=env_time_limit)
        env_time_limit = 0  # This is handled by embodied.Env

    else:
        if use_gymnasium:
            env = gym.make(env_id, render_mode = 'rgb_array', max_episode_steps=env_time_limit) # newer version
        else:
            env = gym.make(env_id)
            env.reset()
            env.render(mode = "rgb_array") #older version
        env = DictWrapper(env)

    if hasattr(env.action_space, 'n'):
        env = OneHotActionWrapper(env)
    if env_time_limit > 0:
        env = TimeLimitWrapper(env, env_time_limit)
    env = ActionRewardResetWrapper(env, no_terminal)
    env = CollectWrapper(env)
    return env
