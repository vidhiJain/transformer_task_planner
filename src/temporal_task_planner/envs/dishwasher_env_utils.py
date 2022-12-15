from dataclasses import asdict
from typing import Dict, List, Tuple
import copy
import json
import random
import magnum as mn
import numpy as np
from pathlib import Path
from temporal_task_planner.utils.data_structure_utils import get_dishwasher_part_pose

from temporal_task_planner.utils.gen_sess_utils import make_quaternion, random_position
from temporal_task_planner.data_structures.preference import Preference
from temporal_task_planner.data_structures.useraction import UserAction
from temporal_task_planner.utils.dishwasher_object_wrapper import Dishwasher
from temporal_task_planner.utils.kitchen_simulator import (
    KitchenSimulator,
    make_configuration,
)
from temporal_task_planner.utils.session_recorder import SessionRecorder
from temporal_task_planner.utils.place_pose_mgr import (
    CounterPlacePoseManager,
    DishwasherPlacePoseManager,
)
from temporal_task_planner.constants.gen_sess_config.camera_params import *
from temporal_task_planner.constants.gen_sess_config.area_extents import (
    zero_area,
    safe_area,
)
from temporal_task_planner.constants.gen_sess_config.lookup import (
    instance_vocab,
    original_potential_placements,
    already_placed_ids,
    default_category_order,
    top_rack_height,
    semantic_coordinates,
    spatial_2d_rack_coordinates,
    dataset_filepath,
    scene_filepath,
    physics_filepath,
    num_immovable_rigid_obj,
    settle_height_dict,
    bounding_box_dict,
)
from temporal_task_planner.utils.data_structure_utils import get_dishwasher_part_pose


def get_utensils_by_preference(
    category_order: List[str], rigid_obj_dict: Dict, epsilon = 1e-6
) -> List[str]:
    selected_utensils = []
    for utensil, pose in rigid_obj_dict.items():
        category = utensil.split(":")[0]
        if np.sum((np.array(pose['pos']) - np.array([0.0,0.0,0.0]))**2) > epsilon :
            if category in category_order:
                selected_utensils.append(utensil)
    return selected_utensils


class DishwasherEnvUtils(SessionRecorder):
    """
    Functions for semantic actions on scene
    Note: keyframe is recorded before the action taken
    The resulting change is recorded in the end pose and the subsquent frames
    ..update_record
    ..sim.setPose
    ..useraction composed
    """

    def __init__(
        self,
        pkg_root: str = ".",
        sim: KitchenSimulator = None,
        preference: Preference = None,
        max_objects_in_rack: int = 8,
        init_task_visibility: float = 1,
    ):
        self.pkg_root = pkg_root
        sim = sim if sim is not None else self.default_sim()
        preference = preference if preference is not None else self.default_pref()
        super(DishwasherEnvUtils, self).__init__(sim, preference)
        self.init_task_visibility = init_task_visibility
        self.max_objects_in_rack = max_objects_in_rack
        self.num_objects_at_init = int(
            self.init_task_visibility * self.max_objects_in_rack
        )
        self.init_session_record()

    def init_session_record(
        self,
        max_instantiated_utensils: int = 10,
        rigid_obj_dict: Dict = None,
        all_utensils_ordered: List[str] = None,
    ) -> None:
        """Inits all the essential attributes for env utils"""
        super().init_session_record()
        self.track_placed = dict(top_rack=[], bottom_rack=[], sink=[])
        self.dishwasher = Dishwasher(self.sim.get_dishwasher_object())
        self.place_pose_mgr = DishwasherPlacePoseManager()
        self.place_pose_mgr.init_params(asdict(self.preference))
        return

    def init_scene(
        self,
        max_instantiated_utensils: int = 10,
        rigid_obj_dict: Dict = None,
    ) -> None:
        """Initialize the scene with different combinations of
        bottom and top rack objects

        Construct all_utensils_ordered List from useraction rigidObjects

        For Full Visible case:
            init partial visibility = 1
            max_objects_in_rack = 20
                (much larger than what would actually fit in)
        """
        sim = self.sim
        sim.load_scene_art_obj(self.articulated_objects)
        # init partial visibility
        # get object ids per category TODO: assign them once in json and load?!
        if rigid_obj_dict is None:
            bottom_rack_objects = []
            top_rack_objects = []
            for i in range(max_instantiated_utensils):
                for category in self.preference.category_order_bottom_rack:
                    bottom_rack_objects.append("{}:{:04d}".format(category, i))
                for category in self.preference.category_order_top_rack:
                    top_rack_objects.append("{}:{:04d}".format(category, i))

            # shuffle the lists and select by partial visibility
            random.shuffle(bottom_rack_objects)
            random.shuffle(top_rack_objects)

            sim.load_scene_rigid_obj_from_names(
                bottom_rack_objects[: self.num_objects_at_init]
                + top_rack_objects[: self.num_objects_at_init],
                position_extents=safe_area,
            )
            # init remaining objects to origin (out of view)
            sim.load_scene_rigid_obj_from_names(
                bottom_rack_objects[self.num_objects_at_init :]
                + top_rack_objects[self.num_objects_at_init :],
                position_extents=zero_area,
            )
        else:
            sim.load_scene_rigid_obj_dict(rigid_obj_dict)
            bottom_rack_objects = get_utensils_by_preference(
                self.preference.category_order_bottom_rack, rigid_obj_dict
            )
            top_rack_objects = get_utensils_by_preference(
                self.preference.category_order_top_rack, rigid_obj_dict
            )
        # TODO: sort by category priority and distance from pick close to
        # DOUBLE CHECK
        self.bottom_utensils = copy.deepcopy(
            bottom_rack_objects[: self.num_objects_at_init]
        )
        self.all_bottom_utensils = copy.deepcopy(
            bottom_rack_objects[: self.max_objects_in_rack]
        )
        self.top_utensils = copy.deepcopy(top_rack_objects[: self.num_objects_at_init])
        self.all_top_utensils = copy.deepcopy(
            top_rack_objects[: self.max_objects_in_rack]
        )
        return

    def default_sim(self) -> KitchenSimulator:
        """Setup
        1. Sim
        2. Default agent
        3. Cameras
        4. Cabinet, counters, etc. init
        """
        # 1. Sim
        cfg = make_configuration(
            dataset_filepath, scene_filepath, height, width, cam_info
        )
        sim = KitchenSimulator(cfg)  # habitat_sim.Simulator(cfg)
        # 2. Default agent
        sim.init_default_agent()
        # 3. Cameras
        for cam_type in ["color", "depth", "semantic"]:
            sim.assign_camera_params(f"{cam_type}_kitchen_sensor", **kitchen_cam)
            sim.assign_camera_params(f"{cam_type}_rack_sensor", **rack_cam)
        # 4. Cabinet, counters, etc. init
        with open(physics_filepath, "r") as f:
            scene_config = json.load(f)
        possible_utensils = scene_config["keyframe"]["rigidObjects"]
        self.articulated_objects = scene_config["keyframe"]["articulatedObjects"]
        # Keep cabinets i.e. Remove first 4 rigid objects that part of the scene
        self.scene_rigid_obj = possible_utensils[:num_immovable_rigid_obj]
        sim.load_scene_rigid_obj(self.scene_rigid_obj)
        return sim

    def default_pref(self) -> Dict:
        preference = Preference()
        preference.process_category_order()
        return preference  # asdict(preference)

    def sort_utensils_by_category(
        self, utensil_name_list: List[str], ordering_by_pref: List[str]
    ) -> List:
        sorted_list = []
        for cat in ordering_by_pref:
            for utensil in utensil_name_list:
                utensil_category = utensil.split(":")[0]
                if utensil_category == cat:
                    sorted_list.append(utensil)
        return sorted_list

    def update_record(self, **kwargs) -> Dict:
        """Utility for updating dishwasher loading session record
        1. registers current frame num as prev frame num
        2. feasibility dict
            2a. computes the visible, feasiblePick from current obs
            2b. get feasiblePlace lists
        3. updates the keyframes as per kwargs
        4. update placed utensils
        ---
        Args:
            kwargs: Dict ; (same as update_keyframes)
        Returns:
            useraction_params: Dict (partially filled userAction)
        """
        # 1. registers current frame num as prev frame num
        self.prev_frame_num = copy.deepcopy(self.frame_num)
        # 2. computes the visible, feasiblePick from current obs
        feasibility_dict = self.get_feasibility_dict()
        # 3. updates the keyframes as per kwargs
        self.update_keyframes(**kwargs)
        useraction_params = feasibility_dict
        # 4. update placed utensils
        useraction_params.update({"placedUtensils": copy.deepcopy(self.track_placed)})
        return useraction_params

    def get_feasible_place_poses(self, is_top_rack, is_bottom_rack):
        if is_top_rack:
            feasiblePlace = self.place_pose_mgr.potential_placements[1]
        elif is_bottom_rack:
            feasiblePlace = self.place_pose_mgr.potential_placements[0]
        else:
            feasiblePlace = self.place_pose_mgr.potential_placements[2]
        return feasiblePlace

    def get_feasibility_dict(self):
        feasibility_dict = self.sim.get_instances_from_obs()
        is_top_rack = feasibility_dict["is_top_rack"]
        is_bottom_rack = feasibility_dict["is_bottom_rack"]
        feasibility_dict.update(
            {
                "feasiblePlace": self.get_feasible_place_poses(
                    is_top_rack, is_bottom_rack
                )
            }
        )
        return feasibility_dict

    def get_non_placeable_object(self):
        sampled_objects = []
        category_list = copy.deepcopy(default_category_order)
        np.random.shuffle(category_list)
        for category in category_list:
            if (
                category in self.preference.category_order_bottom_rack
                and category
                not in list(self.place_pose_mgr.potential_placements[0].keys())
            ) or (
                category in self.preference.category_order_top_rack
                and category
                not in list(self.place_pose_mgr.potential_placements[1].keys())
            ):
                i = 0
                found = False
                while not found and i < 10:
                    instance = "{}:{:04d}".format(category, i)
                    if (
                        instance
                        not in self.track_placed["top_rack"]
                        + self.track_placed["bottom_rack"]
                        + self.track_placed["sink"]
                    ):
                        sampled_objects = [instance]
                    i += 1
        return sampled_objects

    """Primary interaction functions"""

    def settling_action(
        self, step_time=500, settle_time=1, enable_physics=True
    ) -> UserAction:
        params = self.update_record(
            step_time=step_time, settle_time=settle_time, enable_physics=enable_physics
        )
        params.update(
            {
                "isSettlingAction": True,
                "rigidObj": "",
                "articulatedObj": "",
                "articulatedLink": -1,  # body
                "startFrame": self.prev_frame_num,
                "endFrame": self.frame_num,
                "initPose": "",
                "endPose": "",
                "actionType": "init",
            }
        )
        # Update useraction
        useraction = UserAction(**params)
        self.record["userActions"].append(useraction.get())
        return useraction

    def interact_with_dishwasher(self, link_id: int, num_steps: int = 10) -> UserAction:
        lower, upper = self.dishwasher.get_joint_position_limits(link_id - 1)
        # TODO: fix the duplication of joint pos list
        init_joint_positions = self.dishwasher.get_joint_positions()
        joint_positions = self.dishwasher.get_joint_positions()
        (
            desired_joint_positions,
            interpolated_joint_positions,
        ) = self.dishwasher.toggle_joint_by_link_id(link_id, num_steps=num_steps)
        self.prev_frame_num = copy.deepcopy(self.frame_num)
        # feasibility_dict = self.get_feasibility_dict()
        params = self.update_record(step_time=1, settle_time=1, enable_physics=False)
        for interim_joint_position in interpolated_joint_positions:
            prev_jp = joint_positions[link_id - 1]
            new_jp = interim_joint_position
            joint_positions[link_id - 1] = interim_joint_position
            self.dishwasher.set_joint_positions(joint_positions)

            # Move the objects in top/bottom rack when it is moved!
            if link_id == 3:  # top rack
                for placed_object in self.track_placed["top_rack"]:
                    # change the z coordinate according to the joint position
                    rigid_obj = self.sim.rigid_obj_mgr.get_object_by_handle(
                        placed_object
                    )
                    rigid_obj.translation = mn.Vector3(
                        [
                            rigid_obj.translation[0],
                            rigid_obj.translation[1],
                            rigid_obj.translation[2] + (new_jp - prev_jp),
                        ]
                    )
            elif link_id == 1:  # bottom rack
                for placed_object in self.track_placed["bottom_rack"]:
                    # change the z coordinate according to the joint position
                    rigid_obj = self.sim.rigid_obj_mgr.get_object_by_handle(
                        placed_object
                    )
                    rigid_obj.translation = mn.Vector3(
                        [
                            rigid_obj.translation[0],
                            rigid_obj.translation[1],
                            rigid_obj.translation[2] + (new_jp - prev_jp),
                        ]
                    )
            # simulate and populate record
            # frame_num, keyframe_list, user_actions_list, potential_placement_list = update_session_entries(sim, prev_frame_num, keyframe_list, user_actions_list, potential_placement_list)
            self.update_keyframes(step_time=1, settle_time=1, enable_physics=False)
        params.update(
            {
                "isSettlingAction": False,
                "rigidObj": "",
                "articulatedObj": "ktc_dishwasher_:0000",
                "articulatedLink": link_id,
                "startFrame": self.prev_frame_num,
                "endFrame": self.frame_num,
                "initPose": get_dishwasher_part_pose(
                    link_id - 1, desired_joint_positions[link_id - 1], reverse=True
                ),
                "endPose": get_dishwasher_part_pose(
                    link_id - 1, desired_joint_positions[link_id - 1]
                ),
                "actionType": "toggle-" + str(link_id - 1),
            }
        )
        useraction = UserAction(**params)
        self.record["userActions"].append(useraction.get())
        return useraction

    def clear_dishwasher(
        self, safe_area, dishwasher_extents, utensil_names
    ) -> UserAction:
        for utensil in utensil_names:
            utensil_obj = self.rigid_obj_mgr.get_object_by_handle(utensil)
            if (
                (
                    utensil_obj.translation[0] > dishwasher_extents[0][0]
                    and utensil_obj.translation[0] < dishwasher_extents[1][0]
                )
                and (
                    utensil_obj.translation[1] > dishwasher_extents[0][1]
                    and utensil_obj.translation[1] < dishwasher_extents[1][1]
                )
                and (
                    utensil_obj.translation[2] > dishwasher_extents[0][2]
                    and utensil_obj.translation[2] < dishwasher_extents[1][2]
                )
            ):
                init_pose = self.sim.getPose(utensil)
                params = self.update_record(
                    step_time=1, settle_time=1, enable_physics=False
                )
                utensil_obj.translation = mn.Vector3d(
                    random_position(
                        safe_area[0][0],
                        safe_area[1][0],
                        safe_area[0][1],
                        safe_area[1][1],
                        safe_area[0][2],
                        safe_area[1][2],
                    )
                )
                utensil_obj.rotation = make_quaternion([1.0, 0.0, 0.0, 0.0])
                end_pose = self.sim.getPose(utensil)
                params.update(
                    {
                        "isSettlingAction": False,
                        "rigidObj": utensil,
                        "articulatedObj": "",
                        "articulatedLink": -1,  # body
                        "startFrame": self.prev_frame_num,
                        "endFrame": self.frame_num,
                        "initPose": init_pose,
                        "endPose": end_pose,
                        "actionType": "clear",
                    }
                )
                useraction = UserAction(**params)
                self.record["userActions"].append(useraction.get())
        return

    def set_utensil_at_pose(
        self, pick_handle: str, place_pose: Dict[str, List[float]], epsilon: float = 1e-6  # List[float]
    ) -> UserAction:
        params = self.update_record(
            step_time=0.016, settle_time=30, enable_physics=False
        )
        is_top_rack = self.dishwasher.is_open("top")
        is_bottom_rack = self.dishwasher.is_open("bottom")
        init_pose = self.sim.getPose(pick_handle)

        if np.sum((np.array(place_pose["pos"]) - np.array(semantic_coordinates["sink"]))**2) < epsilon:
            self.track_placed["sink"].append(pick_handle)
            self.sim.setPose(pick_handle, place_pose)
        elif not is_top_rack and not is_bottom_rack:
            print('WARNING: placing when racks are closed!!!')
        elif is_top_rack:
            self.track_placed["top_rack"].append(pick_handle)
            self.sim.setPose(pick_handle, place_pose)
        elif is_bottom_rack:
            self.track_placed["bottom_rack"].append(pick_handle)
            self.sim.setPose(pick_handle, place_pose)
        else:
            # place pose is for placement in rack but the top/bottom racks are not open.
            # DEFAULT : put in sink!
            place_pose["pos"] = semantic_coordinates["sink"]
            self.sim.setPose(pick_handle, place_pose)
        potential_placements = self.place_pose_mgr.remove_occupied_places(
            chosen_name=pick_handle, place_pose=place_pose, is_top_rack=is_top_rack
        )
        self.place_pose_mgr.set_potential_placements(potential_placements, is_top_rack)
        params.update(
            {
                "isSettlingAction": False,
                "rigidObj": pick_handle,
                "articulatedObj": "",
                "articulatedLink": -1,
                "startFrame": self.prev_frame_num,
                "endFrame": self.frame_num,
                "initPose": init_pose,
                "endPose": place_pose["pos"] + place_pose["rot"],
                "actionType": "pickplace-" + str(is_top_rack),
            }
        )
        useraction = UserAction(**params)
        self.record["userActions"].append(useraction.get())
        return useraction

    def place_one_utensil_in_rack(
        self,
        picked_instance: str,
    ) -> UserAction:
        useraction = None
        is_top_rack = self.dishwasher.is_open("top")
        is_bottom_rack = self.dishwasher.is_open("bottom")
        if (not is_top_rack) and (not is_bottom_rack):
            # print("No rack is open! Cannot place in dishwasher.")
            return useraction
        picked_category = picked_instance.split(":")[0]
        place_pose = self.place_pose_mgr.get_place_pose(picked_category, is_top_rack)
        if place_pose is None:
            # print(
            #     f"could not find placement for {picked_instance} in is_top_rack={is_top_rack}"
            # )
            return useraction
        useraction = self.set_utensil_at_pose(picked_instance, place_pose)
        return useraction

    def place_utensils_in_rack(
        self,
        category_order,
        pick_close_to,
        place_close_to,
        place_close_k=3,
        is_top_rack=True,
        utensil_id=0,
        count_error=0,
        max_error=3,
        max_num_objects=1,
    ) -> List[UserAction]:
        useractions = []
        while utensil_id < max_num_objects:
            # import ipdb; ipdb.set_trace()
            self.dishwasher.set_joint_velocities([0.0, 0.0, 0.0])
            if is_top_rack:
                self.dishwasher.open("top")  # set_joint_positions([0.0,-1.5,0.6])
            else:
                self.dishwasher.open("bottom")  # set_joint_positions([0.6,-1.5,0.0])

            potential_placements = self.place_pose_mgr.potential_placements[
                int(is_top_rack)
            ]
            # A: Recording the state and feasible placements before placement of chosen object category
            params = self.update_record(
                step_time=0.016, settle_time=1, enable_physics=False
            )

            if not len(params["feasiblePick"]):
                print("No feasible object to pick")
                return useractions

            if not len(potential_placements):
                print("No more possible placements. Breaking...")
                return useractions  # record, potential_placements

            placed_objects = (
                self.track_placed["top_rack"]
                if is_top_rack
                else self.track_placed["bottom_rack"]
            )
            placed_utensils_ids = [
                instance_vocab.word2index(objname) for objname in placed_objects
            ]

            if count_error > 0:
                # place at random position if there have been placement errors
                place_close_to = None

            # A: sample category and pose
            pick_place = self.place_pose_mgr.get_pick_place_targets(
                self.sim,
                category_order,
                pick_close_to,
                place_close_to,
                place_close_k,
                # TODO: Refactor the name for visible and feasible!
                params["feasiblePick"],
                potential_placements,  # record['potential_placement_list'][-1],
                already_placed_ids + placed_utensils_ids,
                is_top_rack,
            )

            if pick_place is None:
                print("could not find a feasible pick and placement... ")
                return useractions  # record, potential_placements

            pick_handle, place_pose = pick_place
            if place_pose is None:
                print(f"Picked {pick_handle} but no place pose...")
                # TODO: visualize and check if there is any pick place pose.
                return useractions
            # utensil_name = '{0}:{1:04n}'.format(utensil_category, count_remain[utensil_category]-1)
            print(utensil_id, pick_handle, place_pose)
            # self.record['placed_objects'].append(pick_handle)
            init_pose = self.sim.getPose(pick_handle)
            if is_top_rack:
                self.track_placed["top_rack"].append(pick_handle)
            else:
                self.track_placed["bottom_rack"].append(pick_handle)

            params.update(
                {
                    "isSettlingAction": False,
                    "rigidObj": pick_handle,
                    "articulatedObj": "",
                    "articulatedLink": -1,
                    "startFrame": self.prev_frame_num,
                    "endFrame": self.frame_num,
                    "initPose": init_pose,
                    "endPose": place_pose["pos"] + place_pose["rot"],
                    "actionType": "pickplace-" + str(is_top_rack),
                }
            )
            useraction = UserAction(**params)
            useractions.append(useraction)
            self.record["userActions"].append(useraction.get())
            utensil_category = pick_handle.split(":")[0]

            # B: set pose
            rigid_obj = self.sim.setPose(pick_handle, place_pose)
            # already_placed_ids.append(instance_vocab.word2index(pick_handle))

            # C: remove infeasible poses
            potential_placements = self.place_pose_mgr.remove_occupied_places(
                pick_handle, potential_placements
            )
            # Assign potential placements back to the dict
            self.place_pose_mgr.set_potential_placements(
                potential_placements, is_top_rack
            )
            print([len(poses) for key, poses in potential_placements.items()])
            print("placed_objects", placed_objects, "is_top_rack", is_top_rack)
            error = self.place_pose_mgr.check_placement(placed_objects, is_top_rack)
            print("error: ", error)
            if error:
                if error in [1, 4]:
                    count_error = 0
                count_error += 1
                last_user_action = self.record["userActions"][-count_error]
                print("count_error: ", count_error)
                print(
                    "last_user_action['startFrame']: ", last_user_action["startFrame"]
                )
                # print("record['keyframe_list']: ", len(record['keyframe_list']))
                # assert last_user_action["startFrame"] > len(self.record["keyframes"]), \
                # f'last_user_action["startFrame"] ={last_user_action["startFrame"]} is less than len(self.record["keyframes"]) = {len(self.record["keyframes"])}'

                self.sim.reset_to_frame(
                    self.record["keyframes"][last_user_action["startFrame"]]
                )
                self.record["keyframes"] = self.record["keyframes"][
                    : last_user_action["startFrame"]
                ]
                self.record["userActions"] = self.record["userActions"][:-count_error]
                # self.record['placed_objects'] = self.record['placed_objects'][:-count_error]
                if is_top_rack:
                    self.track_placed["top_rack"] = self.track_placed["top_rack"][
                        :-count_error
                    ]
                else:
                    self.track_placed["bottom_rack"] = self.track_placed["bottom_rack"][
                        :-count_error
                    ]
                self.frame_num = last_user_action["startFrame"]
                potential_placements = original_potential_placements
                for pick_handle in placed_objects:
                    if (
                        self.sim.get_position_by_handle(pick_handle)[1]
                        > top_rack_height
                    ):
                        potential_placements = (
                            self.place_pose_mgr.remove_occupied_places(
                                pick_handle, potential_placements
                            )
                        )

                if count_error > max_error:
                    print("Too many errors")
                    utensil_id = max_num_objects
                    break
            else:
                utensil_id += 1
                place_close_to = place_pose
                count_error = 0
        # TODO: '>>> Updating record for last item... '

        return useractions  # record, potential_placements

    def init_object(
        self,
        instance: str,
        pose: Dict[str, List] = {
            "pos": np.mean(safe_area.aslist(), axis=0).tolist(),
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
    ) -> UserAction:
        init_pose = self.sim.getPose(instance)
        self.dishwasher.set_joint_velocities([0.0, 0.0, 0.0])
        params = self.update_record(step_time=0.16, settle_time=1, enable_physics=False)
        rigid_obj = self.sim.setPose(instance, pose)
        settled_pose = self.sim.getPose(instance)
        params.update(
            {
                "isSettlingAction": False,
                "rigidObj": instance,
                "articulatedObj": "",
                "articulatedLink": -1,
                "startFrame": self.prev_frame_num,
                "endFrame": self.frame_num,
                "initPose": init_pose,
                "endPose": settled_pose,
                "actionType": "init",
            }
        )
        useraction = UserAction(**params)
        self.record["userActions"].append(useraction.get())
        return useraction

    def remove_object(
        self,
        instance: str,
        pose: Dict[str, List] = {"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
    ) -> UserAction:
        init_pose = self.sim.getPose(instance)
        self.dishwasher.set_joint_velocities([0.0, 0.0, 0.0])
        params = self.update_record(step_time=0.16, settle_time=1, enable_physics=False)
        rigid_obj = self.sim.setPose(instance, pose)
        settled_pose = self.sim.getPose(instance)
        params.update(
            {
                "isSettlingAction": False,
                "rigidObj": instance,
                "articulatedObj": "",
                "articulatedLink": -1,
                "startFrame": self.prev_frame_num,
                "endFrame": self.frame_num,
                "initPose": init_pose,
                "endPose": settled_pose,
                "actionType": "remove",
            }
        )
        useraction = UserAction(**params)
        self.record["userActions"].append(useraction.get())
        return useraction

    def drop_object(
        self,
        instance: str,
        pose: Dict[str, List] = {
            "pos": np.array(semantic_coordinates["sink"])
            + np.array([np.random.normal(0, 0.1), 0, np.random.normal(0, 0.005)]),
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
    ) -> UserAction:
        init_pose = self.sim.getPose(instance)
        self.track_placed["sink"].append(instance)
        self.dishwasher.set_joint_velocities([0.0, 0.0, 0.0])
        self.prev_frame_num = copy.deepcopy(self.frame_num)
        # recording the initial pose in the keyframe before dropping
        params = self.update_record(
            step_time=0.16, settle_time=200, enable_physics=True
        )
        # push dishwasher racks back to close!?
        # find pose of the dropped object in record['keyframe_list'][-1]
        self.sim.setPose(instance, pose)
        dropped_pose = self.sim.getPose(instance)
        params.update(
            {
                "isSettlingAction": False,
                "rigidObj": instance,
                "articulatedObj": "",
                "articulatedLink": -1,
                "startFrame": self.prev_frame_num,
                "endFrame": self.frame_num,
                "initPose": init_pose,
                "endPose": dropped_pose,
                "actionType": "drop",
            }
        )
        useraction = UserAction(**params)
        self.record["userActions"].append(useraction.get())
        return useraction

    """Composite interaction functions
    - utility wrapper around primary functions
    """

    def open_place_close_rack(
        self,
        category_order,
        num_objects,
        is_top_rack=True,
        articulated_link=3,
    ) -> List[UserAction]:
        print(
            "Open rack"
        )  # , 'is_top_rack', is_top_rack, 'articulated_link', articulated_link)
        useraction_open = self.interact_with_dishwasher(
            articulated_link=articulated_link
        )
        # TODO: check correct rack is open after interaction
        rack = "top" if is_top_rack else "bottom"
        self.dishwasher.open(rack)

        print("Place utensils in category priority order")
        useractions_place = self.place_utensils_in_rack(
            category_order=np.array(category_order),
            pick_close_to=mn.Vector3(
                semantic_coordinates[self.preference.pick_close_to]
            ),
            place_close_to=spatial_2d_rack_coordinates[self.preference.place_close_to],
            is_top_rack=is_top_rack,
            max_num_objects=num_objects,
        )

        print("Close rack")
        useraction_close = self.interact_with_dishwasher(
            articulated_link=articulated_link
        )
        self.dishwasher.close("top" if is_top_rack else "bottom")
        return [useraction_open, useractions_place, useraction_close]

    def open_dishwasher(self) -> UserAction:
        if not self.dishwasher.is_open("door"):
            print("open dishwasher door:  joint_position=[0.0,-1.5,0.0], link_id=2")
            # Slowly open the door
            useraction = self.interact_with_dishwasher(link_id=2, num_steps=10)
            self.dishwasher.open("door")
        return useraction

    def close_dishwasher(self) -> UserAction:
        if self.dishwasher.is_open("door"):
            print("close dishwasher door:  joint_position=[0.0,0.0,0.0], link_id=2")
            # Slowly close the door
            useraction = self.interact_with_dishwasher(link_id=2, num_steps=10)
            self.dishwasher.close("door")
        return useraction

    def place_in_rack_else_sink(self, picked_instance: str) -> UserAction:
        useraction = self.place_one_utensil_in_rack(picked_instance)
        if useraction is None:
            useraction = self.drop_object(picked_instance)
        return useraction

    def count_by_category_preference(
        self, placed_utensils: List[str], preference_category: List[str]
    ) -> int:
        count = 0
        for utensil in placed_utensils:
            if utensil.split(":")[0] in preference_category:
                count += 1
        return count
