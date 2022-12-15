"""Run an entire rollout of eXpert policy
Store in a json file (to later view as a video)
"""
from numpy import place
from src.data_structures.action import Action
from src.data_structures.instance import PlacementInstance, RigidInstance
from src.envs.utils.dishwasher_env_utils import DishwasherEnvUtils
from src.policy.expert_policy import ExpertPolicy
from src.envs.dishwasher_env import FullVisibleEnv
from constants.gen_sess_config.lookup import *

savepath = "tmp.json"
env_utils = DishwasherEnvUtils()
env = FullVisibleEnv(env_utils)
obs = env.reset()
expert_policy = ExpertPolicy(env, place_pref=True)
for i in range(len(expert_policy.action_script)):
    act = expert_policy.get_action()
    print(f"\nAction {i}: {act}")
    obs, rew, done, info = env.step(act)
session_dict = env.env_utils.get_session_dict()
env.env_utils.save_session(session_dict, savepath)
print(f"Saved expert policy at : {savepath}")
