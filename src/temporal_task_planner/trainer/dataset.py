from typing import Dict, List, Tuple
from glob import glob
import copy
from dataclasses import dataclass, asdict, field
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from temporal_task_planner.constants.gen_sess_config.lookup import *
from temporal_task_planner.data_structures.action import Action
from temporal_task_planner.data_structures.state import State
from temporal_task_planner.data_structures.temporal_context import TemporalContext
from temporal_task_planner.utils.gen_sess_utils import load_session_json
from temporal_task_planner.utils.extract_from_session import *
from temporal_task_planner.utils.data_structure_utils import (
    construct_act_instance,
    construct_action,
    construct_place_instances_from_keyframe,
    construct_rigid_instances_from_keyframe,
)
from temporal_task_planner.utils.datasetpytorch_utils import get_session_path


class DishwasherArrangeDataset(Dataset):
    """Descending order dataset of plates (token index: self.num_instances-1)
    Predict at the ACTION MASKS
    after ACT token (token index: self.num_instances)
    """

    def __init__(
        self,
        session_paths: List[str],
        context_history: int,
        pick_only: bool = True,
        num_sessions_limit: int = -1,
        max_place_poses: int = 100,
    ) -> None:
        self.session_paths = session_paths
        self.context_history = context_history
        self.pick_only = pick_only
        self.num_sessions_limit = num_sessions_limit
        self.max_place_poses = max_place_poses

        self.num_sessions = len(self.session_paths)

        # clip and normalize pos info
        self.min_pos = -3
        self.max_pos = 3
        self.norm_pos = {
            "mean": [-1.8851218, 0.646792, 0.58907884],
            "std": [0.62312627, 0.34161338, 0.46390563],
        }
        self.start_userid = {}
        self.sess_userid_list = self.get_sess_userid_list()
        self.is_action_available = False

    def get_state_action_from_useraction(
        self,
        keyframe: Dict,
        useraction: Dict,
        relative_timestep: int = 0,
    ) -> Tuple[State, Action]:
        """Creates the state and action from starting keyframe of a useraction"""
        # assign from useraction
        track = get_instance_to_track(useraction)
        target_instance_poses = get_target_instance_poses(useraction)
        feasible_pick = get_feasible_picks(useraction)
        feasible_place = get_feasible_place(useraction)
        state = State(pick_only=self.pick_only)
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
            is_action_available=self.is_action_available,
            is_action_to_be_predicted=self.is_action_available,
            relative_timestep=relative_timestep,
        )
        action = None
        if self.is_action_available:
            action = construct_action(useraction)
        return state, action

    def get_sess_userid_list(self) -> List[Tuple[int, int]]:
        """
        Processing start useraction ids for the sessions
        excluding
            - settling and clearing actions
            - init action
        """
        sess_userid_list = []
        for session_id in range(len(self.session_paths)):
            data = load_session_json(self.session_paths[session_id])
            if data is None:
                continue

            startflag = False
            for userid, useraction in enumerate(data["session"]["userActions"]):
                if not startflag:
                    # ignoring settling and clearing actions
                    if useraction["articulatedObj"] == "ktc_dishwasher_:0000":
                        startflag = True
                        self.start_userid[session_id] = userid
                # ignoring init actions
                if startflag and (useraction["actionType"] != "init"):
                    sess_userid_list.append((session_id, userid))
        return sess_userid_list

    def set_action_available(self, *args) -> None:
        self.is_action_available = True

    def process_session_keyframes(
        self, session_id: int, userid: int
    ) -> Tuple[Dict, Dict]:
        """
        Args:
            session_id, userid : which are valid for processing inputs
        Returns:
            inputs: Dict of Instance attributes (timestep, category, pose, etc.)
                with values describing each instance
            outputs: Dict of pick_track_ids (a.k.a 'act'), init_pose and end_pose
                corresponding to each action_mask True
        """
        data = load_session_json(self.session_paths[session_id])
        if data is None:
            return
        original_context_budget = (
            min(userid - self.start_userid[session_id], self.context_history) + 1
        )
        context_budget = copy.deepcopy(original_context_budget)
        useraction_counter = 0
        temporal_context = TemporalContext()
        while (context_budget) and (useraction_counter <= userid):
            useraction = data["session"]["userActions"][userid - useraction_counter]
            useraction_counter += 1
            if useraction["actionType"] == "init":
                continue
            keyframe = data["session"]["keyframes"][useraction["startFrame"]]
            self.set_action_available(context_budget, original_context_budget)
            state, action = self.get_state_action_from_useraction(
                keyframe, useraction, context_budget
            )
            temporal_context.states.insert(0, state)
            if self.is_action_available:
                temporal_context.actions.insert(0, action)
            context_budget -= 1
        assert len(temporal_context.states), " check if temporal context has states?! "
        inputs = temporal_context.process_states()
        targets = temporal_context.process_actions()
        return inputs, targets

    def transform_position(
        self,
        position: List[float],
        transform_pos_limits: bool = True,
        transform_pos_norm: bool = True,
    ) -> List[float]:
        """
        Transformation for 3D position
         - limit the extreme coordinates
         - zero mean distribution for learning
        TODO: test the function whether it is bijective?!
        """
        if transform_pos_limits:
            for i in range(3):
                if position[i] < self.min_limit:
                    position = self.min_limit
                elif position[i] > self.max_limit:
                    position = self.max_limit
            if transform_pos_norm:
                position[i] = (position[i] - self.norm_pos["mean"]) / self.norm_pos[
                    "std"
                ]
        return position

    def __len__(self) -> int:
        return len(self.sess_userid_list)

    def __getitem__(self, index: int) -> Tuple[Dict, Dict]:
        session_id, userid = self.sess_userid_list[index]
        inputs, targets = self.process_session_keyframes(session_id, userid)
        inputs_dict = {key: torch.tensor(inputs[key]) for key in inputs.keys()}
        targets_dict = {key: torch.tensor(targets[key]) for key in targets.keys()}
        return inputs_dict, targets_dict  # , info


class DishwasherArrangeStateActionDataset(DishwasherArrangeDataset):
    def __init__(self, dataset=None, *args, **kwargs) -> None:
        super(DishwasherArrangeStateActionDataset, self).__init__(*args, **kwargs)

    def process_session_keyframes(
        self, session_id: int, userid: int
    ) -> Tuple[State, Action]:
        data = load_session_json(self.session_paths[session_id])
        useraction = data["session"]["userActions"][userid]
        keyframe = data["session"]["keyframes"][useraction["startFrame"]]
        self.set_action_available()
        state, action = self.get_state_action_from_useraction(
            keyframe,
            useraction,
        )
        return state, action

    def __getitem__(self, index: int) -> Tuple[State, Action]:
        session_id, userid = self.sess_userid_list[index]
        return process_session_keyframes(session_id, userid)


class DishwasherArrangeSavedSession(DishwasherArrangeDataset):
    """
    Process one session without repetitions for symbolic accuracy
    First k steps where k < context_window will have shorter inputs
    and 1 target
    later sequences will have inputs of size of the context window
    and the final target.
    """

    def __init__(self, dataset=None, *args, **kwargs) -> None:
        super(DishwasherArrangeSavedSession, self).__init__(*args, **kwargs)

    def get_session_path(self, session_pathname: str) -> List:
        return [session_pathname]

    def set_action_available(
        self, curr_context_budget: int, orig_context_budget: int
    ) -> None:
        if curr_context_budget == orig_context_budget:
            self.is_action_available = True
        else:
            self.is_action_available = False


class DishwasherArrangeAllPlaceDataset(DishwasherArrangeDataset):
    """In every getitem, include place instances for sink, top rack and bottom rack
    as seq of category, pose.
    """

    def __init__(
        self,
        session_paths: List[str],
        context_history: int,
        pick_only: bool = True,
        num_sessions_limit: int = -1,
        max_place_poses: int = 100,
    ) -> None:
        super().__init__(
            session_paths,
            context_history,
            pick_only,
            num_sessions_limit,
            max_place_poses,
        )

    def get_all_place_instances(self):
        pass


def pad_fn(batch) -> Tuple[Dict, Dict]:
    inputs, targets = zip(*batch)
    # measure the padding sequence length with object correspondence and target sequence lengths
    inputs_pad_len = [len(x["timestep"]) for x in inputs]
    src_key_padding_mask = [
        torch.zeros(input_len).bool() for input_len in inputs_pad_len
    ]
    input_tuple = [inputs[i].values() for i in range(len(inputs))]
    timestep, bb, pose, action_masks, is_real, category_token, instance_token = zip(
        *input_tuple
    )
    inputs_padded = {
        "timestep": pad_sequence(timestep, batch_first=True, padding_value=0),
        "category": pad_sequence(bb, batch_first=True, padding_value=0),
        "pose": pad_sequence(pose, batch_first=True, padding_value=0),
        "action_masks": pad_sequence(
            action_masks, batch_first=True, padding_value=False
        ),
        "is_real": pad_sequence(is_real, batch_first=True, padding_value=False),
        "category_token": pad_sequence(
            category_token, batch_first=True, padding_value=0
        ),
        "instance_token": pad_sequence(
            instance_token, batch_first=True, padding_value=0
        ),
        "src_key_padding_mask": pad_sequence(
            src_key_padding_mask, batch_first=True, padding_value=True
        ),
    }
    target_tuple = [targets[i].values() for i in range(len(targets))]
    action_instance, init_pose, end_pose = zip(*target_tuple)
    targets_collated = {
        "action_instance": list(action_instance),
        "init_pose": list(init_pose),
        "end_pose": list(end_pose),
    }
    return inputs_padded, targets_collated


if __name__ == "__main__":
    import os

    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    """
    every target is of context window length or that of the userid 
        - to utilize parallelized training in Transformers 
    """
    dataset = DishwasherArrangeDataset(
        session_paths=os.path.join(pkg_root, "artifacts_sample/full_visibility/"),
        context_history=5,
        num_sessions_limit=5,
        pick_only=True,
    )
    # check if the self.sess_userid_list[index] for index from 0 to len
    # produces a valid session and user id
    for index in [0, 5, 7]:
        print(index)
        inputs, targets = dataset.__getitem__(index)
        print(inputs, targets)
        print("inputs timestep : ", inputs["timestep"])
        print("targets act : ", targets["act"])
        assert (
            len(targets["act"]) == min(index, dataset.context_history) + 1
        ), "number of targets == current and previous states till context window"
        assert inputs["timestep"][-1] == len(
            targets["act"]
        ), "every state should have a target pick instance"
        assert inputs["action_masks"].sum() == len(
            targets["act"]
        ), "number of act instance tokens should match number of targets"

    """
    every target is of size 1; apt padding for inputs
    """
    sess_dataset = DishwasherArrangeSavedSession(
        session_paths=dataset.session_paths[0],
        context_history=5,
        pick_only=False,
    )
    for index in [0, 5, 7]:
        print(index)
        print(inputs, targets)
        inputs, targets = sess_dataset.__getitem__(index)
        print("inputs timestep : ", inputs["timestep"])
        print("targets act : ", targets["act"])
        assert len(targets["act"]) == 1, "Single ACT token for eval needed!"
        assert (
            inputs["timestep"][-1] == inputs["timestep"][targets["act"]]
        ), "Target pick instance timestep should be the most recent timestep"
        assert inputs["action_masks"].sum() == len(
            targets["act"]
        ), "number of act instance tokens should match number of targets"
    assert len(targets["act"]) == 1
    assert inputs["action_masks"].sum() == len(targets["act"])
    """
    # Padding fn test!
    """
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=pad_fn,
    )
    inp, tgt = next(iter(loader))
    print(inp, tgt)
    print("done")
