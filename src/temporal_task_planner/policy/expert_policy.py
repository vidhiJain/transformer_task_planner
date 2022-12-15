from dataclasses import asdict
from typing import Any, Dict, List
import gym
import copy
from temporal_task_planner.constants.gen_sess_config.lookup import (
    link_id_names,
    dishwasher_part_poses,
    place_name_idx,
    semantic_coordinates,
)
from temporal_task_planner.constants.gen_sess_config.area_extents import safe_area
from temporal_task_planner.data_structures.instance import RigidInstance
from temporal_task_planner.data_structures.preference import Preference
from temporal_task_planner.data_structures.state import State
from temporal_task_planner.data_structures.action import Action
from temporal_task_planner.envs.dishwasher_env_utils import DishwasherEnvUtils
from temporal_task_planner.policy.abstract_policy import Policy
from temporal_task_planner.utils.data_structure_utils import (
    construct_placement_instance,
    construct_rigid_instance,
)


class ExpertPickOnlyPolicy(Policy):
    """
    Picks according to category order (does not depend on dict yet)
    Placement according to the preference pose proximity and place close k
    """

    def __init__(self, env: gym.Env) -> None:
        super(ExpertPickOnlyPolicy, self).__init__()
        self.env = env
        # self.reset(env)

    def reset(self, env: gym.Env = None, **kwargs) -> None:
        # min items to construct RigidInstance, PlacementInstance
        self.counter = 0
        self.env = env
        self.dummy_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.init_rack_preference()
        self.expert_seq = self.populate_expert_seq()

    def init_rack_preference(self) -> None:
        """Utensils are segregated into the mutually exclusive categories
        for top and bottom rack.
        Note: category order for placement is not defined.
        """
        self.rack_orders = {
            "top": self.env.env_utils.top_utensils,
            "bottom": self.env.env_utils.bottom_utensils,
        }
        self.load_top_rack_first = self.env.env_utils.preference.load_top_rack_first
        self.first_rack = "top" if self.load_top_rack_first else "bottom"
        self.second_rack = "bottom" if self.load_top_rack_first else "top"
        return

    def populate_expert_seq(self) -> List[Dict]:
        """Pre-generated (offline) sequence of actions that the expert
        takes based on the init state of the dishwasher for execution
        on the number of times `get_action` is called"""
        expert_seq = [
            self.dishwasher_action_util("door", "open"),
            self.dishwasher_action_util(self.first_rack, "open"),
        ]
        is_top_rack = True if self.first_rack == "top" else False
        expert_seq += self.utensil_action_util(
            self.rack_orders[self.first_rack], is_top_rack
        )
        expert_seq += [
            self.dishwasher_action_util(self.first_rack, "close"),
            self.dishwasher_action_util(self.second_rack, "open"),
        ]
        is_top_rack = True if self.second_rack == "top" else False
        expert_seq += self.utensil_action_util(
            self.rack_orders[self.second_rack], is_top_rack
        )
        expert_seq += [
            self.dishwasher_action_util(self.second_rack, "close"),
            self.dishwasher_action_util("door", "close"),
        ]
        return expert_seq

    def dishwasher_action_util(self, link_name: str, target_state: str) -> Dict:
        """Action info for dishwasher toggling for the action script"""
        current_state = "open" if target_state == "close" else "close"
        return {
            "name": link_id_names[link_name],
            "init_pose": dishwasher_part_poses[link_name][current_state],
            "end_pose": dishwasher_part_poses[link_name][target_state],
        }

    def utensil_action_util(
        self, utensil_list: List, is_top_rack: bool = False
    ) -> List:
        """Action info for utensil pick-place for the action script"""
        expert_seq = []
        for utensil in utensil_list:
            category = utensil.split(":")[0]
            expert_seq.append(
                dict(
                    name=utensil,
                    init_pose=self.env.env_utils.sim.getPose(utensil),
                    end_pose=self.dummy_pose,
                    category_name=category,
                )
            )
        return expert_seq

    def get_action_util(self, action_essentials: Dict) -> Action:
        """Constructs the RigidInstance, PlacementInstance for Action class"""
        if "category_name" in action_essentials.keys():
            category_name = action_essentials["category_name"]
        else:
            category_name = action_essentials["name"]
        rigid_instance = construct_rigid_instance(
            name=action_essentials["name"],
            pose=action_essentials["init_pose"],
            category_name=category_name,
        )
        place_instance = construct_placement_instance(
            picked_instance=rigid_instance, pose=action_essentials["end_pose"]
        )
        action = Action(rigid_instance, place_instance)
        return action

    def get_action(self, *args) -> Action:
        """Prepares the Action to taken by the env.step fn
        For pick only, Action has a dummy place_instance
        """
        action_essentials = self.expert_seq[self.counter]
        action = self.get_action_util(action_essentials)
        self.counter += 1
        return action


class ExpertPickOnlyPreferencePolicy(ExpertPickOnlyPolicy):
    def __init__(self, env) -> None:
        super().__init__(env)

    def init_rack_preference(self) -> None:
        """Utensils are segregated into the mutually exclusive categories
        for top and bottom rack.
        Note: category order for placement is not defined.
        """
        self.rack_orders = {
            "top": self.env.env_utils.sort_utensils_by_category(
                self.env.env_utils.top_utensils,
                self.env.env_utils.preference.category_order_top_rack,
            ),
            "bottom": self.env.env_utils.sort_utensils_by_category(
                self.env.env_utils.bottom_utensils,
                self.env.env_utils.preference.category_order_bottom_rack,
            ),
        }
        self.load_top_rack_first = self.env.env_utils.preference.load_top_rack_first
        self.first_rack = "top" if self.load_top_rack_first else "bottom"
        self.second_rack = "bottom" if self.load_top_rack_first else "top"
        return


class ExpertInteractPickOnlyPreferencePolicy(ExpertPickOnlyPreferencePolicy):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.preference_dict = asdict(self.env.env_utils.preference)

    def reset(self, env, **kwargs) -> None:
        super().reset(env, **kwargs)
        self.interim_init_counter = 0

    def get_yet_to_be_placed_utensils(self, obs: State) -> None:
        all_visible_instances = [item.instance_name for item in obs.rigid_instances]
        yet_to_be_placed = []
        all_placed = []
        for val in self.env.env_utils.track_placed.values():
            all_placed += val
        for name in all_visible_instances:
            if name not in all_placed and not name.startswith("ktc_dishwasher_:0000"):
                yet_to_be_placed.append(name)
        return yet_to_be_placed

    def get_rack_for_category_preference(self, given_category: str) -> str:
        for rack in [self.first_rack, self.second_rack]:
            if given_category in getattr(
                self.env.env_utils.preference, f"category_order_{rack}_rack"
            ):
                return rack
        return ""

    def get_utensil_to_place(
        self, rigid_instances: List[RigidInstance], preference: Preference
    ) -> str:
        category_order = (
            preference.category_order_top_rack + preference.category_order_bottom_rack
            if preference.load_top_rack_first
            else preference.category_order_bottom_rack
            + preference.category_order_top_rack
        )
        for category in category_order:
            for rigid_instance in rigid_instances:
                if rigid_instance.category_name == category:
                    return rigid_instance.instance_name
        return ""

    def get_action(self, obs: State) -> Action:
        """Prepares the Action to taken by the env.step fn
        For pick only, Action has a dummy place_instance
        self.interim_init_counter
        """
        if self.counter < len(self.expert_seq):
            action_essentials = self.expert_seq[self.counter]
            action = self.get_action_util(action_essentials)
            self.counter += 1
            return action
        action = None
        potential_placements = self.env.env_utils.place_pose_mgr.potential_placements
        if self.interim_init_counter < len(self.env.remaining_utensils):
            utensils_in_safe_area = []
            for rigid_instance in obs.rigid_instances:
                if safe_area.contains(rigid_instance.get_place_position()):
                    utensils_in_safe_area.append(rigid_instance)
            utensil_to_place = self.get_utensil_to_place(
                utensils_in_safe_area, self.env.env_utils.preference
            )
            if utensil_to_place == "":
                if self.env.env_utils.dishwasher.is_open(self.first_rack):
                    return self.get_action_util(
                        self.dishwasher_action_util(self.first_rack, "close")
                    )
                if self.env.env_utils.dishwasher.is_open(self.second_rack):
                    return self.get_action_util(
                        self.dishwasher_action_util(self.second_rack, "close")
                    )
            # BY TIME ORDER : self.env.remaining_utensils[self.interim_init_counter]
            given_category = utensil_to_place.split(":")[0]
            # rack selection
            rack = self.get_rack_for_category_preference(given_category)
            other_rack = (
                self.first_rack if rack == self.second_rack else self.second_rack
            )
            is_top_rack = True if rack == "top" else False
            # check place pose mgr whether dishwasher interaction is needed?
            if given_category in potential_placements[place_name_idx[rack]].keys():
                # open the door, if it is closed
                if not self.env.env_utils.dishwasher.is_open("door"):
                    return self.get_action_util(
                        self.dishwasher_action_util("door", "open")
                    )
                # close the other rack if it is open
                elif self.env.env_utils.dishwasher.is_open(other_rack):
                    return self.get_action_util(
                        self.dishwasher_action_util(other_rack, "close")
                    )
                # open the desired rack for placing if it is closed
                elif not self.env.env_utils.dishwasher.is_open(rack):
                    return self.get_action_util(
                        self.dishwasher_action_util(rack, "open")
                    )
            else:
                # check if the other rack is open! close it?!
                if self.env.env_utils.dishwasher.is_open(other_rack):
                    return self.get_action_util(
                        self.dishwasher_action_util(other_rack, "close")
                    )
            # pick the object (for placing in the rack or the sink)
            self.interim_init_counter += 1
            return self.get_action_util(
                self.utensil_action_util([utensil_to_place], is_top_rack)[0]
            )
        else:
            # all utensils are placed
            # close the open racks!
            if self.env.env_utils.dishwasher.is_open("bottom"):
                return self.get_action_util(
                    self.dishwasher_action_util("bottom", "close")
                )
            elif self.env.env_utils.dishwasher.is_open("top"):
                return self.get_action_util(self.dishwasher_action_util("top", "close"))
        return self.get_action_util(self.dishwasher_action_util("door", "close"))


class ExpertPickPlacePreferencePolicy(ExpertPickOnlyPreferencePolicy):
    def __init__(self, env: gym.Env) -> None:
        super(ExpertPickPlacePreferencePolicy, self).__init__(env)

    def reset(self, env: gym.Env = None, **kwargs) -> None:
        self.counter = 0
        self.env = env
        self.init_rack_preference()
        self.expert_place_pose_mgr = copy.deepcopy(self.env.env_utils.place_pose_mgr)
        self.expert_seq = self.populate_expert_seq()

    def utensil_action_util(
        self, utensil_list: List, is_top_rack: bool = False
    ) -> List:
        """Action info for utensil pick-place for the action script"""
        expert_seq = []
        for utensil in utensil_list:
            category = utensil.split(":")[0]
            place_pose = self.expert_place_pose_mgr.get_place_pose(
                category, is_top_rack
            )
            if place_pose is None:
                # drop in sink
                place_pose = {
                    "pos": semantic_coordinates["sink"],
                    "rot": [1.0, 0.0, 0.0, 0.0],
                }
            potential_placements = self.expert_place_pose_mgr.remove_occupied_places(
                chosen_name=utensil, place_pose=place_pose, is_top_rack=is_top_rack
            )
            self.expert_place_pose_mgr.set_potential_placements(
                potential_placements, is_top_rack
            )
            expert_seq.append(
                dict(
                    name=utensil,
                    init_pose=self.env.env_utils.sim.getPose(utensil),
                    end_pose=place_pose["pos"] + place_pose["rot"],
                    category_name=category,
                )
            )
        return expert_seq


class ExpertInteractPickPlacePreferencePolicy(
    ExpertPickPlacePreferencePolicy, ExpertInteractPickOnlyPreferencePolicy
):
    def __init__(self, env: gym.Env) -> None:
        super(ExpertInteractPickPlacePreferencePolicy, self).__init__(env)

    def reset(self, env: gym.Env, **kwargs) -> None:
        super().reset(env, **kwargs)
        self.interim_init_counter = 0


if __name__ == "__main__":
    from temporal_task_planner.data_structures.action import Action
    from temporal_task_planner.envs.dishwasher_env_utils import DishwasherEnvUtils
    from temporal_task_planner.envs.dishwasher_env import (
        FullVisibleEnv,
        InteractivePartialVisibleEnv,
    )
    from temporal_task_planner.constants.gen_sess_config.lookup import *
    from temporal_task_planner.rollout import session_rollout

    savepath = "tmp.json"
    env = InteractivePartialVisibleEnv(max_steps=50)
    obs = env.reset()
    expert_policy = ExpertInteractPickOnlyPreferencePolicy(env)
    session_rollout(env, expert_policy, savepath)
    print(f"Saved expert policy at : {savepath}")
