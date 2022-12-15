from typing import Dict, List
import magnum as mn
import numpy as np
import copy
import json
import pandas as pd
from habitat_sim.utils.common import quat_rotate_vector, quat_from_magnum
from temporal_task_planner.constants.gen_sess_config.lookup import *

def load_session_json(session_path):
    try: 
        with open(session_path, "r") as f:
            data = json.load(f)
    except ValueError as e:
        print(f"Not JSON!!! session_path : {session_path}")
        return None
    return data


def get_list_of_poses(pose_json_path):
    pp_dict = load_session_json(pose_json_path)
    pose_list = []
    for category, poses in pp_dict.items():
        for pose in poses:
            pose_list.append(pose["pos"] + pose["rot"])
    return pose_list


def make_quaternion(float_array):
    return mn.Quaternion(
        mn.Vector3(float_array[1], float_array[2], float_array[3]), float_array[0]
    )


def get_quat_coeffs_from_magnum(magnum_quat):
    return [
        magnum_quat.scalar,
        magnum_quat.vector[0],
        magnum_quat.vector[1],
        magnum_quat.vector[2],
    ]


def get_vector_coeffs_from_magnum(magnum_vec):
    return [magnum_vec[0], magnum_vec[1], magnum_vec[2]]


def flatten(t):
    return [item for sublist in t for item in sublist]


def random_position(min_x, max_x, min_y, max_y, min_z, max_z):
    val = np.random.random_sample((3,))
    possible_position = [
        (max_x - min_x) * val[0] + min_x,
        (max_y - min_y) * val[1] + min_y,
        (max_z - min_z) * val[2] + min_z,
    ]
    return possible_position


def get_discrete_orientation(rigid_obj_rotation):
    # if type(rigid_obj_rotation)!= list:
    #     quat_from_magnum(rigid_obj_rotation)
    new_up = quat_rotate_vector(
        quat_from_magnum(make_quaternion(rigid_obj_rotation)), np.array([0.0, 1.0, 0.0])
    )
    max_sim_score = -1.0
    max_sim_id = None
    for i in range(len(up_orient)):
        name = up_orient[i]["name"]
        ori_list = up_orient[i]["new_up"]
        for up_vec in ori_list:
            sim_score = np.dot(np.array(up_vec), new_up)
            print("\t", name, up_vec, sim_score)
            if max_sim_score <= sim_score:
                max_sim_score = sim_score
                max_sim_id = i
    return max_sim_id


def get_select_instances(seg_map, threshold):
    """
    Heuristic (utility) function for selecting visible/feasible pick instances
    based on pixel counts in segmentation maps
    Args:
        seg_map : np.array HxW
        threshold : int
    Returns:
        selected_instance_ids : List[int]
    """
    ids, counts = np.unique(seg_map, return_counts=True)
    selected_instance_ids = []
    for i in range(len(ids)):
        name = instance_vocab.index2word(ids[i]).split(":")[0]
        if counts[i] > threshold * catseg_fullview_thresh[name]:
            selected_instance_ids.append(int(ids[i]))
    return selected_instance_ids


def get_potential_placements(
    original_potential_placements, is_top_rack=True, offset=0.26
):
    potential_placements = copy.deepcopy(original_potential_placements)
    if not is_top_rack:
        for k, v in original_potential_placements.items():
            for i, p in enumerate(v):
                potential_placements[k][i]["pos"][1] = (
                    p["pos"][1] - offset
                )  # 0.25 # - 0.1455 #9 #(top_rack_height - offset)
    return potential_placements


def closestk_pose(
    available_poses: List[Dict[str, List[float]]],
    last_placed_pose: Dict[str, List[float]],
    k: int = 3,
) -> Dict[str, List[float]]:
    if last_placed_pose is None:
        return np.random.choice(available_poses, 1)[0]
    if available_poses is None:
        return None
    distances = []
    for pose in available_poses:
        dist = (pose["pos"][0] - last_placed_pose["pos"][0]) ** 2 + (
            pose["pos"][2] - last_placed_pose["pos"][2]
        ) ** 2
        distances.append(dist)
    combined = list(zip(available_poses, distances))
    sorted_combo = sorted(combined, key=lambda x: x[1], reverse=True)
    if len(sorted_combo) > k:
        return sorted_combo[-k][0]
    return sorted_combo[0][0]


def load_placement_data(jsonfile):
    with open(jsonfile, "r") as f:
        parsed_json = json.load(f)
    parsed_df = pd.read_json(parsed_json, orient="split")
    return parsed_df, parsed_json


def get_init_utensil_actions_from_session(data):
    init_objects = []
    for useraction in data['session']['userActions']:
        if useraction['actionType'] == 'init':
            init_objects.append({
                'instance': useraction['rigidObj'],
                'pose': {
                    'pos': useraction['initPose'][:3],
                    'rot': useraction['initPose'][3:],
                }
            })
    return init_objects


sink_poses = [{"pos": semantic_coordinates["sink"], "rot": [1.0, 0.0, 0.0, 0.0]}]    
bottom_rack_poses = get_potential_placements(original_potential_placements, is_top_rack=False)
top_rack_poses = get_potential_placements(original_potential_placements, is_top_rack=True)

