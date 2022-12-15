import copy
from glob import glob
from typing import Dict, List, Tuple
from pathlib import Path
from temporal_task_planner.data_structures.action import Action
from temporal_task_planner.data_structures.state import State
from temporal_task_planner.data_structures.temporal_context import TemporalContext
from temporal_task_planner.utils.data_structure_utils import (
    construct_act_instance,
    construct_action,
    construct_place_instances_from_keyframe,
    construct_rigid_instances_from_keyframe,
)
from temporal_task_planner.utils.extract_from_session import *
from temporal_task_planner.utils.gen_sess_utils import load_session_json
import torch

from temporal_task_planner.data_structures.instance import Instance


def get_session_path(session_pathname: str, num_sessions_limit: int = 10000) -> List:
    session_list = glob(session_pathname + "/sess_*.json")[:num_sessions_limit]
    session_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][5:]))
    return session_list


def get_attribute_batch(instances: List[Instance], attribute: str):
    return torch.tensor([getattr(instance, attribute) for instance in instances])


def get_state_action_from_useraction(
    keyframe: Dict,
    useraction: Dict,
    relative_timestep: int = 0,
    is_action_available: bool = True,
) -> Tuple[State, Action]:
    """Creates the state and action from starting keyframe of a useraction"""
    # assign from useraction
    track = get_instance_to_track(useraction)
    target_instance_poses = get_target_instance_poses(useraction)
    feasible_pick = get_feasible_picks(useraction)
    feasible_place = get_feasible_place(useraction)

    state = State(pick_only=False)
    state.rigid_instances = construct_rigid_instances_from_keyframe(
        keyframe, feasible_pick, relative_timestep
    )

    state.place_instances = construct_place_instances_from_keyframe(
        keyframe,
        track,
        target_instance_poses["endPose"],
        feasible_pick,
        feasible_place,
        relative_timestep,
    )
    state.act_instances = construct_act_instance(
        is_action_available=is_action_available,
        is_action_to_be_predicted=is_action_available,
        relative_timestep=relative_timestep,
    )
    action = None
    if is_action_available:
        action = construct_action(useraction)
    return state, action

def get_session_paths_split(dataset_path):
    session_list = glob(Path(dataset_path, 'sess_*.json').as_posix())
    train_sessions = session_list[: int(len(session_list) * 0.8)]
    val_sessions = session_list[
        len(train_sessions) : len(train_sessions) + int(len(session_list) * 0.1)
    ]
    test_sessions = session_list[
        len(train_sessions)
        + len(val_sessions) : len(train_sessions)
        + len(val_sessions)
        + int(len(session_list) * 0.1)
    ]
    session_split = {"train": train_sessions, "val": val_sessions, "test": test_sessions}
    return session_split