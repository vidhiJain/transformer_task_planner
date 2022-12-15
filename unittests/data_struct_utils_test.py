"""Construct State from current info from simulator"""

from constants.gen_sess_config import *
from src.envs.utils.kitchen_simulator import KitchenSimulator, make_configuration
from src.utils.data_structure_utils import *

cfg = make_configuration(dataset_filepath, scene_filepath, height, width, cam_info)
sim = KitchenSimulator(cfg)

keyframe = sim.create_session_keyframe()
feasible_pick_ids = sim.get_instances_from_obs()["feasiblePick"]
feasible_pick = [instance_dict[str(idx)] for idx in feasible_pick_ids]

state = construct_rigid_instances_from_keyframe(
    keyframe=keyframe, feasible_pick=feasible_pick
)
print(state)
print("-----\n\nConstructed state for pick only!\n\n-----")

# state = construct_rigid_instances_from_keyframe(keyframe=keyframe, feasible_pick=feasible_pick, pick_only=False)
# print("Constructed state for pick place!")
