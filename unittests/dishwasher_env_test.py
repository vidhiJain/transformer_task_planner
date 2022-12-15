"""Create Gym Env and get action from expert policy
"""
from temporal_task_planner.policy.expert_policy import (
    ExpertInteractPickOnlyWithPreferencePolicy,
    ExpertPickPlacePolicy,
)
from temporal_task_planner.envs.dishwasher_env import (
    FullVisibleEnv,
    InteractivePartialVisibleEnv,
)
from temporal_task_planner.constants.gen_sess_config.lookup import *

env = FullVisibleEnv()
policy = ExpertPickPlacePolicy(env)
obs = env.reset()
policy.reset(env)

action = policy.get_action(obs)
import ipdb

ipdb.set_trace()
obs, reward, done, info = env.step(action)
