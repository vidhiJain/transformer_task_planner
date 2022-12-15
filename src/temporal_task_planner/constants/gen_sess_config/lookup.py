import os
import json
from pathlib import Path
from vocab import Vocab
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

"""Env Constants"""
num_immovable_rigid_obj = 4
top_rack_height = 0.4

d3_colors_rgb = np.load(os.path.join(root, "resources/d3_colors_rgb.npy"))

with open(os.path.join(root, "resources/cat_seg_px_counts.json"), "r") as f:
    catseg_fullview_thresh = json.load(f)

with open(os.path.join(root, "resources/cumulative_bb.json"), "r") as f:
    bounding_box_dict = cumulative_bb_dict = json.load(f)

with open(os.path.join(root, "resources/countertop_settle_height_dict.json"), "r") as f:
    settle_height_dict = json.load(f)

with open(os.path.join(root, "resources/up_orients.json"), "r") as f:
    discretize_orientation_data = json.load(f)
up_orient = discretize_orientation_data["up_orient"]

with open(os.path.join(root, "resources/placement_poses_dict.json"), "r") as f:
    original_potential_placements = json.load(f)


"""Vocabs"""
with open(os.path.join(root, "resources/category_vocab.json"), "r") as f:
    category_dict = json.load(f)

category_vocab = Vocab()
category_vocab.word2index(list(category_dict.keys()), train=True)

with open(os.path.join(root, "resources/instance_vocab.json"), "r") as f:
    instance_dict = json.load(f)

special_tokens = [
    "<pad>",
    "<unk>",
    "<reset>",
    "<act>",
    "<act:pick>",
    "<act:place>",
    "<goal>",
]
instance_vocab = Vocab()
instance_vocab.word2index(list(instance_dict.values()), train=True)

special_instance_dict = special_tokens + list(instance_dict.values())
special_instance_vocab = Vocab()
special_instance_vocab.word2index(special_instance_dict, train=True)


"""File paths for habitat sim
To load lighthouse and replicaCAD assets
"""
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
lighthouse_path = "third_party/kitchen_arrange/data/lighthouse_kitchen_dataset/"
dataset_filepath = Path(
    pkg_root, lighthouse_path, "lighthouse_kitchen.scene_dataset_config.json"
).as_posix()
scene_filepath = Path(
    pkg_root, lighthouse_path, "scenes/max_objects_10.scene_instance.json"
).as_posix()
physics_filepath = Path(
    pkg_root, lighthouse_path, "scenes/max_objects_10.physics_keyframe.json"
).as_posix()


"""Semantic Coordinates Lookup 
To get the center of a semantic location 
"""
default_rot = [1.0, 0.0, 0.0, 0.0]
semantic_coordinates = {
    "sink": [-1.65, 0.96, 0.45],
    "floor": [-1.59, 0.0, 0.946],
    "dishwasher_countertop": [-1.05, 0.94, 0.4],
}
spatial_2d_rack_coordinates = {
    "back-right": {"pos": [-0.9, 0.52, 0.85]},
    "back-left": {"pos": [-0.9, 0.52, 1.15]},
    "front-left": {"pos": [-1.2, 0.52, 1.15]},
    "front-right": {"pos": [-1.2, 0.52, 0.85]},
    "front-center": {"pos": [-1.05, 0.64, 1.15]},
    "back-center": {"pos": [-1.05, 0.64, 0.85]},
    "None": None,
}


"""Lighthouse Object Lists
List of rigid and articulated objects for 
random scene initialization 
"""
already_placed = [
    "ktc_cabinets1_:0000",
    "ktc_cabinets2_open_:0000",
    "ktc_cabinets3_:0000",
    "ktc_dishwasher_:0000_joint_0",
    "ktc_dishwasher_:0000_joint_1",
    "ktc_dishwasher_:0000_joint_2",
    "ktc_dishwasher_:0000_joint_3",
    "kitchen_oven_:0000_joint_0",
    "kitchen_oven_:0000_joint_1",
]
already_placed_ids = [instance_vocab.word2index(name) for name in already_placed]

default_category_order = np.array(
    [
        "frl_apartment_plate_01_",
        "frl_apartment_plate_01_small_",
        "frl_apartment_cup_02_",
        "frl_apartment_bowl_07_small_",
        "frl_apartment_bowl_03_",
        "frl_apartment_kitchen_utensil_06_",
        "ktc_clutter_tray_",
    ]
)

"""Dishwasher part poses: gives center of mass 
in open and close state of the top/bottom racks and door.

Keyframes don't give the pose of the racks and door, 
only a number indicating their joint position. 
But we will (eventually) train with images and their instance segmentation 
where the center of mass will be calculated from the point cloud centers.
"""
dishwasher_part_poses = {
    "bottom": {
        "close": [-1.059, 0.3, 0.35, 1.0, 0.0, 0.0, 0.0],
        "open": [-1.059, 0.3, 0.95, 1.0, 0.0, 0.0, 0.0],
    },
    "door": {
        "close": [-1.059, 0.43, 0.6499999999999999, 1.0, 0.0, 0.0, 0.0],
        "open": [-1.059, 0.01, 0.95, 1.0, 0.0, 0.0, 0.0],
    },
    "top": {
        "close": [-1.059, 0.56, 0.35, 1.0, 0.0, 0.0, 0.0],
        "open": [-1.059, 0.56, 0.95, 1.0, 0.0, 0.0, 0.0],
    },
}

link_vocab = {"bottom": 0, "door": 1, "top": 2}
inv_link_vocab = {val: key for key, val in link_vocab.items()}

open_close_ranges = {
    0: {"close": 0, "open": 0.6},
    1: {"close": 0, "open": -1.5},
    2: {"close": 0, "open": 0.6},
}

link_id_names = {
    "bottom": "ktc_dishwasher_:0000_joint_1",
    "door": "ktc_dishwasher_:0000_joint_2",
    "top": "ktc_dishwasher_:0000_joint_3",
}
link_names_id = {val: key for key, val in link_id_names.items()}
place_name_idx = {"bottom": 0, "top": 1, "sink": 2}
