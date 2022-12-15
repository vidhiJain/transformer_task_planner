"""Init Dishwasher articulated object from Habitat sim
and print its attributes
"""
from src.envs.utils.dishwasher_object_wrapper import Dishwasher
from src.envs.utils.kitchen_simulator import KitchenSimulator, make_configuration
from constants.gen_sess_config import *

cfg = make_configuration(dataset_filepath, scene_filepath, height, width, cam_info)
sim = KitchenSimulator(cfg)
dw = Dishwasher(sim.get_dishwasher_object())

print(dw.link_vocab)
print(dw.get_open_close_range())
print(dw.populate_all_part_poses())
