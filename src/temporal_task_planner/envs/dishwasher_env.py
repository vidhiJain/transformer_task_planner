import random
import json
from typing import Any, Dict, List, Tuple
import gym
import copy
from dataclasses import asdict
from temporal_task_planner.constants.gen_sess_config.area_extents import (
    safe_area,
    zero_area,
)
from temporal_task_planner.constants.gen_sess_config.lookup import instance_dict
from temporal_task_planner.data_structures.action import Action
from temporal_task_planner.data_structures.state import State
from temporal_task_planner.data_structures.temporal_context import TemporalContext
from temporal_task_planner.envs.dishwasher_env_utils import DishwasherEnvUtils
from temporal_task_planner.utils.data_structure_utils import (
    construct_act_instance,
    construct_rigid_instances_from_keyframe,
)
from temporal_task_planner.utils.gen_sess_utils import random_position


class DishwasherLoadingTaskEnv(gym.Env):
    """
    Dishwasher loading Gym API Wrapper
    """

    def __init__(
        self,
        reference_session_path: str = None,
        max_objects_in_rack: int = 8,
        max_instantiated_utensils: int = 5,
        num_animate_frames: int = 10,
        init_task_visibility: float = 1,
        max_place_poses: int = 5,
        max_steps: int = None,
        pick_only: bool = True,
    ) -> None:
        super().__init__()
        self.env_utils = DishwasherEnvUtils(
            max_objects_in_rack=max_objects_in_rack,
            init_task_visibility=init_task_visibility,
        )
        if reference_session_path is None or reference_session_path == "None":
            self.ref_sess = None
        else:
            with open(reference_session_path, "r") as f:
                self.ref_sess = json.load(f)
                self.init_object_list = get_init_utensil_actions_from_session(self.ref_sess)
        self.max_objects_in_rack = max_objects_in_rack
        self.max_instantiated_utensils = max_instantiated_utensils
        self.num_animate_frames = num_animate_frames
        self.init_task_visibility = init_task_visibility
        self.max_place_poses = max_place_poses
        self.max_steps = (
            2 * (max_objects_in_rack + 3) if max_steps is None else max_steps
        )
        self.pick_only = pick_only
        self.history = TemporalContext()
        self.num_objects_at_init_per_rack = int(
            self.init_task_visibility * self.max_objects_in_rack
        )

    def reset(self, ref_sess: str = None) -> State:
        """Constructs a new sim object or Re-inits based on a session json!
        1. Take the first useraction startFrame (i.e. first keyframe)
        TODO: check useraction after setting
        2. load the rigid objects listed as per the info
        Returns: (Init) State
        """
        self.step_counter = 0
        self.utensils_picked_step_counter = 0
        if ref_sess is None:
            self.env_utils.init_scene(self.max_instantiated_utensils)
        else:
            with open(ref_sess, "r") as f:
                session_dict = json.load(f)
            rigid_obj_dict = session_dict["session"]["keyframes"][0]["rigidObjects"]
            # placed_utensils = session_dict["session"]["userActions"][-1]["placedUtensils"]
            self.env_utils.init_scene(self.max_instantiated_utensils, rigid_obj_dict)
        self.env_utils.init_session_record()
        state = self.get_state()
        return state

    def step(self, action: Action) -> Tuple[Any, int, bool, Dict]:
        """
        Args: Action as
            - RigidInstance, PlacementInstance
            - Pick handle name, Place pos and rot
        Returns:  (State, reward, done, info)
        """
        self.step_counter += 1
        self.execute_action(action)
        state = self.get_state()
        reward = self.get_reward()
        done = self.check_stopping_condition()
        info = self.get_reference_session_state()
        self.history.states.append(state)
        return state, reward, done, info

    def render(
        self,
        mode="rgb",
        cam_uuid="color_kitchen_sensor",
    ) -> None:
        """Utility function of sim to view the change in env
        - env_utils.sim to visualize the current frame
        - to show or save as rgb image of the `current frame`
        # TODO: get action taken as a `text` mode
        """
        self.env_utils.sim.plot_sensor_observations(cam_uuid)
        return

    def execute_action(self, action: Action) -> None:
        """Handles session recording of useraction
        as the action is applied to sim
        # Dishwasher parts are handled to toggle only
        # Utensils pick only / pick-place
        """
        pick_handle = action.get_pick_handle()
        if pick_handle.startswith("ktc_dishwasher_:0000_joint_"):
            articulated_link = int(pick_handle[-1])
            self.env_utils.interact_with_dishwasher(
                articulated_link, num_steps=self.num_animate_frames
            )
        else:
            if self.pick_only:
                self.env_utils.place_in_rack_else_sink(pick_handle)
            else:
                place_pose = action.get_place_pose()
                self.env_utils.set_utensil_at_pose(pick_handle, place_pose)
        return

    def get_state(self) -> State:
        """convert the keyframe to State : dataset.py update fn"""
        dummy_timestamp = 0
        keyframe = self.env_utils.sim.create_session_keyframe()
        feasible_pick_ids = self.env_utils.sim.get_instances_from_obs()["feasiblePick"]
        feasible_pick = [instance_dict[str(idx)] for idx in feasible_pick_ids]
        rigid_instances = construct_rigid_instances_from_keyframe(
            keyframe,
            feasible_pick=feasible_pick,
            relative_timestep=dummy_timestamp,
        )
        act_instances = construct_act_instance(
            is_action_available=False,
            is_action_to_be_predicted=True,
            relative_timestep=dummy_timestamp,
        )
        state = State(
            rigid_instances=rigid_instances,
            act_instances=act_instances,
            pick_only=self.pick_only,
        )
        return state

    def get_reward(self) -> int:
        # TODO: get_reward(History: TemporalContext)
        """Reward can be based on:
        1. Reference session: how much the policy
        follows the PREFERENCE
        2. physics: how much the policy learned
        about FEASIBILITY
        """
        # raise NotImplementedError
        return 0  # dummy value for testing

    def check_stopping_condition(self) -> bool:
        """check stopping condition and set done
        NOTE : Tested with expert policy
        TODO: test with learned policies
        """
        # is max num steps exceeded? Hard end
        if self.step_counter >= self.max_steps:
            return True
        # are objects removed from the counter? either in racks or sink
        for utensil in (
            self.env_utils.all_bottom_utensils + self.env_utils.all_top_utensils
        ):
            pose = self.env_utils.sim.getPose(utensil)
            # check if within the specific initialization areas -> done=False
            if safe_area.contains(pose[:3]): #  or zero_area.contains(pose[:3]):
                self.utensils_picked_step_counter = 0
                return False
        if self.utensils_picked_step_counter < 2: # and self.step_counter > 3:
            self.utensils_picked_step_counter += 1
            if not self.env_utils.dishwasher.is_open("door"):
                return True
            return False
        return True

    def get_reference_session_state(self) -> Dict:
        # TODO: set actionType from reference session in info
        if self.ref_sess is None:
            return {}
        raise NotImplementedError


class FullVisibleEnv(DishwasherLoadingTaskEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class InteractivePartialVisibleEnv(DishwasherLoadingTaskEnv):
    """
    Objects appear after every n steps of the env
    For 1 instance introduced per step,
     the policy always has to take a pick/place action
    TODO:
    In case no new object introduced,
     optimal action could be NO-OP instance
    """

    def __init__(
        self, init_task_visibility: float = 0.4, interact_frequency: int = 1, **kwargs
    ) -> None:
        super().__init__(init_task_visibility=init_task_visibility, **kwargs)
        self.interact_start_step_threshold = self.num_objects_at_init_per_rack * 2 + 6
        self.interact_frequency = interact_frequency

    def reset(self, ref_sess: str = None) -> State:
        obs = super().reset(ref_sess)
        self.remaining_utensils = copy.deepcopy(
            self.env_utils.all_bottom_utensils[
                self.num_objects_at_init_per_rack : self.max_objects_in_rack
            ]
        ) + copy.deepcopy(
            self.env_utils.all_top_utensils[
                self.num_objects_at_init_per_rack : self.max_objects_in_rack
            ]
        )
        random.shuffle(self.remaining_utensils)
        self.interim_init_counter = 0
        return obs

    def get_state(self) -> State:
        """
        Args: Action as
            - RigidInstance, PlacementInstance
            - Pick handle name, Place pos and rot
        Returns:  (State, reward, done, info)
         Step counter is updated in super()
        """
        if (self.step_counter + 1 >= self.interact_start_step_threshold) and (
            self.step_counter % self.interact_frequency == 0
        ):
            if self.interim_init_counter < len(self.remaining_utensils):
                if self.ref_sess is None: 
                    sampled_instance = self.remaining_utensils[self.interim_init_counter]
                    self.interim_init_counter += 1
                    pose = {
                        "pos": random_position(**asdict(safe_area)),
                        "rot": [1.0, 0.0, 0.0, 0.0],
                    }
                    self.env_utils.init_object(sampled_instance, pose)
                else:
                    # load the object as per init list
                    ref_data = self.init_object_list[self.interim_init_counter]
                    self.interim_init_counter += 1
                    self.env_utils.init_object(**ref_data)
        return super().get_state()


if __name__ == "__main__":
    env = FullVisibleEnv(pick_only=False)
    obs = env.reset()
    print("\n\n\n\nSuccessfully reset the env!!!\n\n\n\n")
