from typing import Dict, List, Tuple
from numpy import place
from temporal_task_planner.data_structures.action import Action
from temporal_task_planner.data_structures.instance import (
    ActInstance,
    PlacementInstance,
    RigidInstance,
)
from temporal_task_planner.data_structures.state import State
from temporal_task_planner.constants.gen_sess_config.lookup import (
    bounding_box_dict,
    category_vocab,
    special_instance_vocab,
    dishwasher_part_poses,
    open_close_ranges,
    inv_link_vocab,
    semantic_coordinates,
    link_vocab,
    original_potential_placements,
    link_names_id,
)
from temporal_task_planner.utils.gen_sess_utils import sink_poses, bottom_rack_poses, top_rack_poses
"""
Utility Functions for constructing instances for State 
(i.e., rigid, place and act) from keyframe
"""


def get_dishwasher_part_pose(idx: int, jp: float, reverse: bool = False) -> List[float]:
    """Dishwasher parts like rack and doors movement described in terms of how
    center of mass of the pixels move (instead of arbitrary joint positions)
    """
    if abs(jp - open_close_ranges[idx]["open"]) < abs(jp - open_close_ranges[idx]["close"]):
        status = "open" if not reverse else "close"
    else:
        status = "close" if not reverse else "open"
    part_name = inv_link_vocab[idx]
    return dishwasher_part_poses[part_name][status]


def get_dishwasher_state(pose: List[float], epsilon=1e-4) -> str:
    # HACK to check the state of dishwasher object
    # if z coordinate is 0.95, the part is open!
    if abs(pose[2] - 0.95) < epsilon:
        return "open"
    return "close"


def construct_rigid_instance(
    name: str, pose: List[float], category_name: str, relative_timestep: int = 0
) -> RigidInstance:
    """Constructs a single rigid instance"""
    return RigidInstance(
        timestep=relative_timestep,
        category=bounding_box_dict[category_name],
        pose=pose,
        action_masks=False,
        is_real=True,
        category_token_name=category_name,  # record utensil type
        category_token=category_vocab.word2index(category_name),
        instance_token=special_instance_vocab.word2index(name),
        category_name=category_name,
        instance_name=name,
    )


def construct_placement_instance(
    picked_instance: RigidInstance,
    pose: List,
) -> PlacementInstance:
    """Constructs a single place instance"""
    return PlacementInstance(
        timestep=picked_instance.timestep,
        category=picked_instance.category,
        pose=pose,
        action_masks=False,
        is_real=False,
        category_token_name="<goal>",
        category_token=category_vocab.word2index("<goal>"),
        instance_token=special_instance_vocab.word2index("<goal>"),
        category_name=picked_instance.category_name,
    )


def process_dishwasher_for_rigid_instances(
    keyframe: Dict, relative_timestep: int, epsilon: float = 1e-4
) -> List[RigidInstance]:
    """Processes dishwasher racks and door for rigid instances
    which are feasible for interaction
    Feasibility Constraints :
        - door is feasible, if both racks are closed
        - racks are feasible, if door is open
    """
    rigid_instances = []
    art_obj_name = "ktc_dishwasher_:0000"
    art_obj_pose = keyframe["articulatedObjects"][art_obj_name]
    current_door_joint_value = art_obj_pose["joints"][1]
    # record if both racks are closed
    flag_both_racks_closed = True
    for rack in ["bottom", "top"]:
        idx = link_vocab[rack]
        jp = art_obj_pose["joints"][idx]
        if abs(jp - open_close_ranges[link_vocab[rack]]["open"]) < epsilon:
            flag_both_racks_closed = False
    if flag_both_racks_closed:
        # add door as feasible rigid instance
        idx = link_vocab["door"]
        jp = art_obj_pose["joints"][idx]
        jp_name = art_obj_name + "_joint_" + str(idx + 1)
        pose = get_dishwasher_part_pose(idx, jp)
        rigid_instance = construct_rigid_instance(
            jp_name, pose, jp_name, relative_timestep
        )
        rigid_instances.append(rigid_instance)
    # If dishwasher door is open, racks should be feasible
    if (
        abs(current_door_joint_value - open_close_ranges[link_vocab["door"]]["open"])
        < epsilon
    ):
        for rack in ["bottom", "top"]:
            idx = link_vocab[rack]
            jp = art_obj_pose["joints"][idx]
            jp_name = art_obj_name + "_joint_" + str(idx + 1)
            pose = get_dishwasher_part_pose(idx, jp)
            rigid_instance = construct_rigid_instance(
                jp_name, pose, jp_name, relative_timestep
            )
            rigid_instances.append(rigid_instance)
    return rigid_instances


def process_utensils_for_rigid_instances(
    keyframe: Dict, relative_timestep: int, feasible_pick: List[str]
) -> List[RigidInstance]:
    """Processes utensils as rigid instances"""
    rigid_instances = []
    for name, pose in keyframe["rigidObjects"].items():
        if name in feasible_pick:
            rigid_instance = construct_rigid_instance(
                name,
                pose["pos"] + pose["rot"],
                name.split(":")[0],
                relative_timestep,
            )
            rigid_instances.append(rigid_instance)
    return rigid_instances


def construct_rigid_instances_from_keyframe(
    keyframe: Dict,
    feasible_pick: List[str],
    relative_timestep: int = 0,
) -> List[RigidInstance]:
    """Combines dishwasher parts and utensils as rigid instances for state"""
    rigid_instances = []
    dishwasher_instances = process_dishwasher_for_rigid_instances(
        keyframe, relative_timestep
    )
    utensil_instances = process_utensils_for_rigid_instances(
        keyframe, relative_timestep, feasible_pick
    )
    for rigid_instance in dishwasher_instances + utensil_instances:
        rigid_instances.append(rigid_instance)

    return rigid_instances


def process_dishwasher_for_place_instances(
    keyframe: Dict, track: str, relative_timestep: int = 0
) -> List[PlacementInstance]:
    """Chooses dishwasher parts and utensils as place instances for state"""
    placed_instances = []
    for art_obj_name, art_obj_pose in keyframe["articulatedObjects"].items():
        if art_obj_name == "ktc_dishwasher_:0000":
            for idx, jp in enumerate(art_obj_pose["joints"]):
                jp_name = f"{art_obj_name}_joint_{idx+1}"
                if track == jp_name:
                    # Construct the picked instance
                    current_pose = get_dishwasher_part_pose(idx, jp)
                    picked_instance = construct_rigid_instance(
                        jp_name, current_pose, jp_name, relative_timestep=0
                    )
                    # Add the target placement instance for it
                    target_pose = get_dishwasher_part_pose(idx, jp, reverse=True)
                    target_place_instance = construct_placement_instance(
                        picked_instance, target_pose
                    )
                    placed_instances.append(target_place_instance)
    return placed_instances


def process_utensils_for_place_instances(
    keyframe: Dict,
    track: str,
    target_pose: List[float],
    feasible_pick: List[str],
    feasible_place: Dict,
    max_place_poses: int = 100,
    relative_timestep: int = 0,
) -> List[PlacementInstance]:
    place_instances = []
    for rigid_obj_name, rigid_obj_pose in keyframe["rigidObjects"].items():
        if rigid_obj_name in feasible_pick:
            if track == rigid_obj_name:
                # Construct picked instance
                picked_instance = construct_rigid_instance(
                    rigid_obj_name,
                    rigid_obj_pose,
                    rigid_obj_name.split(":")[0],
                    relative_timestep,
                )
                place_instances = get_potential_place_instances_per_category(
                    picked_instance, original_potential_placements
                )
                # # Add the target placement instance for it
                # target_place_instance = construct_placement_instance(
                #     picked_instance, target_pose
                # )
                # place_instances.append(target_place_instance)
                # # Add the other feasible placement instances for it, if exists
                # if picked_instance.category_name in feasible_place.keys():
                #     for pose in feasible_place[picked_instance.category_name][
                #         :max_place_poses
                #     ]:
                #         feasible_place_instance = construct_placement_instance(
                #             picked_instance,
                #             pose["pos"] + pose["rot"],
                #         )
                #         place_instances.append(feasible_place_instance)
    return place_instances


def construct_place_instances_from_keyframe(
    keyframe: Dict,
    track: str,
    target_pose: List[float],
    feasible_pick: List[str],
    feasible_place: Dict,
    max_place_poses: int = 100,
    relative_timestep: int = 0,
) -> List[PlacementInstance]:
    if track.startswith("ktc_dishwasher_:0000"):
        place_instances = process_dishwasher_for_place_instances(
            keyframe, track, relative_timestep
        )
    else:
        place_instances = process_utensils_for_place_instances(
            keyframe,
            track,
            target_pose,
            feasible_pick,
            feasible_place,
            max_place_poses,
            relative_timestep,
        )
    return place_instances


def get_potential_place_instances_per_category(
    picked_instance: RigidInstance,
    potential_placements: Dict[int, Dict[str, List[float]]],
    num_per_rack: int = -1,
):
    """Place instances list for all bottom and top rack poses for the picked_instance's category
    Note:
    - Enlisted poses could be infeasible due to already placed objects there.
    - TODO: Limit the number by randomly sampling yet ensuring the pose by expert is included in the list.
    """
    place_instances = []
    for place_pose in sink_poses + bottom_rack_poses[picked_instance.category_name] + top_rack_poses[picked_instance.category_name]:
        place_instances.append(
            construct_placement_instance(
                picked_instance, place_pose["pos"] + place_pose["rot"]
            )
        )
    return place_instances


def construct_act_instance(
    is_action_available: bool,
    is_action_to_be_predicted: bool,
    relative_timestep: int = 0,
) -> ActInstance:
    """act_tag indicates whether the target is set or not"""
    act_instance = ActInstance(
        timestep=relative_timestep,
        category=bounding_box_dict["act"],
        pose=[0.0] * 7,
        action_masks=is_action_to_be_predicted,
        is_real=False,
        category_token_name="<act>",
        category_token=category_vocab.word2index("<act>"),
        instance_token=special_instance_vocab.word2index("<act>"),
        instance_name="<act>",
        is_action_available=is_action_available,
    )
    return act_instance


def construct_action(useraction: Dict) -> Action:
    # moving the dishwasher racks/door
    if useraction["articulatedObj"] != "":
        pick_handle = (
            f"{useraction['articulatedObj']}_joint_{useraction['articulatedLink']}"
        )
        pick_category = pick_handle
    # pick place utensils
    else:
        pick_handle = useraction["rigidObj"]
        pick_category = useraction["rigidObj"].split(":")[0]
    pick_instance = construct_rigid_instance(
        pick_handle,
        useraction["initPose"],
        pick_category,
    )
    place_instance = construct_placement_instance(
        pick_instance,
        useraction["endPose"],
    )
    return Action(pick_instance=pick_instance, place_instance=place_instance)


def get_picked_instance(name, pose):
    if name.startswith("ktc_dishwasher_:0000"):
        picked_instance = construct_rigid_instance(
            name,
            pose,
            category_name=name,  # .split(":")[0],
        )
    else:
        # utensil
        picked_instance = construct_rigid_instance(
            name,
            pose,
            category_name=name.split(":")[0],
        )
    return picked_instance


def get_place_instances(picked_instance, potential_placements):
    if picked_instance.category_name.startswith("ktc_dishwasher_:0000"):
        # dishwasher part just has the toggled place pose as instance
        status = get_dishwasher_state(picked_instance.pose)
        toggled_status = "close" if status == "open" else "open"
        target_pose = dishwasher_part_poses[
            link_names_id[picked_instance.instance_name]
        ][toggled_status]
        place_instances = [
            construct_placement_instance(picked_instance, target_pose)
        ]
    else:
        # utensil
        place_instances = get_potential_place_instances_per_category(
            picked_instance, potential_placements
        )
    return place_instances

def get_actions_from_session(data: Dict) -> List[Action]:
    """goes over all useractions in session json and calls construct_action"""
    user_actions = data["session"]["userActions"]
    actions = []
    for user_action in user_actions:
        actions.append(construct_action(user_action))
    return actions


if __name__ == "__main__":
    pick_instance = construct_rigid_instance(
        name="ktc_dishwasher_:0000_joint_2",
        pose=dishwasher_part_poses["door"]["close"],
        category_name="ktc_dishwasher_:0000_joint_2",
        relative_timestep=0,
    )
    place_instance = construct_placement_instance(
        pick_instance,
        pose=dishwasher_part_poses["door"]["open"],
    )
    print("Tested rigid_instance and place_instance construction")
