import copy
from dataclasses import asdict
from typing import Any, Dict, List
import numpy as np
import os
import numpy as np
import magnum as mn
import matplotlib.pyplot as plt
from pathlib import Path

import habitat_sim
from habitat_sim.physics import ManagedBulletRigidObject, ManagedBulletArticulatedObject
from temporal_task_planner.utils.gen_sess_utils import (
    already_placed_ids,
    instance_vocab,
    make_quaternion,
    get_quat_coeffs_from_magnum,
    random_position,
    get_select_instances,
)
from temporal_task_planner.constants.gen_sess_config.area_extents import (
    AreaExtent,
    init_area,
)


def make_configuration(
    dataset: str, scene: str, height: float, width: float, cam_info_dict: Dict
) -> habitat_sim.Configuration:
    """Utility function for sensor setup
    Args:
        dataset: path, scene: path, height: int, width: int,
        cam_info_dict: dict('uuid', 'type', 'subtype')
    Returns:
        habitat_sim.Simulator object
    """
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene
    backend_cfg.scene_dataset_config_file = dataset
    assert os.path.exists(backend_cfg.scene_id)
    backend_cfg.enable_physics = True
    backend_cfg.scene_light_setup = ""
    backend_cfg.override_scene_light_defaults = True

    sensor_specs = []
    for cam_info in cam_info_dict:
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = cam_info["uuid"]
        sensor_spec.sensor_type = cam_info["type"]
        sensor_spec.sensor_subtype = cam_info["subtype"]
        sensor_spec.resolution = [height, width]
        sensor_specs.append(sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs  # [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


class KitchenSimulator(habitat_sim.Simulator):
    """Wrapper around the habitat sim with some utility functions"""

    def __init__(self, *args):
        super().__init__(*args)
        self.rigid_obj_mgr = self.get_rigid_object_manager()
        self.art_obj_mgr = self.get_articulated_object_manager()

    def init_default_agent(
        self, translation: List = [0.0, 0.0, 0.0], rotation: List = [1.0, 0.0, 0.0, 0.0]
    ) -> None:
        agent_state = habitat_sim.AgentState()
        self.initialize_agent(0, agent_state)
        agent_node = self.get_agent(0).body.object
        agent_node.translation = translation
        agent_node.rotation = make_quaternion(rotation)
        return

    def assign_camera_params(
        self, camera_uuid: str, translation: List, rotation: List, zoom: int
    ) -> None:
        camera = self._sensors[camera_uuid]._sensor_object
        camera.reset_zoom()
        camera.zoom(zoom)
        cam_obj = self._sensors[camera_uuid]._sensor_object.object
        cam_obj.translation = mn.Vector3(translation)
        cam_obj.rotation = make_quaternion(rotation)
        return

    def simulate(
        self, dt: float = 1.0 / 60.0, sim_steps: int = 60, get_frames: bool = True
    ) -> List:
        # simulate dt seconds at 60Hz to the nearest fixed timestep
        observations = []
        for steps in range(sim_steps):
            self.step_physics(dt)
            if get_frames:
                observations.append(self.get_sensor_observations())
        return observations

    def plot_sensor_observations(self, cam_uuid: str = "color_kitchen_sensor") -> None:
        obs = self.get_sensor_observations()
        # print(self._sensors[cam_uuid]._sensor_object.object.translation)
        plt.imshow(obs[cam_uuid])
        plt.savefig("current_frame")

    def get_instances_from_obs(
        self,
        seg_uuid: str = "semantic_kitchen_sensor",
        visibility_count_thresh: float = 0.0,
        pick_accesible_count_thresh: float = 0.0,
    ) -> Dict:
        """
        update each key of the record. assumes semantic segmentation sensor
        """
        obs = self.get_sensor_observations()
        seg_map = obs[seg_uuid]
        visible = get_select_instances(seg_map, threshold=visibility_count_thresh)
        feasiblePick = get_select_instances(
            seg_map, threshold=pick_accesible_count_thresh
        )
        for ids in already_placed_ids:
            if ids in feasiblePick:
                feasiblePick.remove(ids)

        is_bottom_rack = bool(
            (seg_map == instance_vocab.word2index("ktc_dishwasher_:0000_joint_1")).any()
            == True
        )
        is_top_rack = bool(
            (seg_map == instance_vocab.word2index("ktc_dishwasher_:0000_joint_3")).any()
            == True
        )

        return {
            "visible": visible,
            "feasiblePick": feasiblePick,
            "is_top_rack": is_top_rack,
            "is_bottom_rack": is_bottom_rack
            # "feasiblePlace": feasiblePlace,
        }

    def get_position_by_handle(self, handle_name: str) -> mn.Vector3d:
        handles = self.rigid_obj_mgr.get_object_handles(handle_name)
        assert len(handles)
        rigid_obj = self.rigid_obj_mgr.get_object_by_handle(handles[0])
        return rigid_obj.translation

    def getPose(self, pick_handle: str) -> List[float]:
        rigid_obj = self.rigid_obj_mgr.get_object_by_handle(pick_handle)
        init_pose = [
            rigid_obj.translation[0],
            rigid_obj.translation[1],
            rigid_obj.translation[2],
            rigid_obj.rotation.scalar,
            rigid_obj.rotation.vector[0],
            rigid_obj.rotation.vector[1],
            rigid_obj.rotation.vector[2],
        ]
        return init_pose

    def setPose(
        self, utensil_name: str, pose: Dict[str, List[float]], dropheight: int = 0.0
    ) -> ManagedBulletRigidObject:
        rigid_obj = self.rigid_obj_mgr.get_object_by_handle(utensil_name)
        rigid_obj.translation = mn.Vector3(
            [pose["pos"][0], pose["pos"][1] + dropheight, pose["pos"][2]]
        )
        rigid_obj.rotation = make_quaternion(pose["rot"])
        rigid_obj.semantic_id = instance_vocab.word2index(utensil_name)
        rigid_obj.awake = True
        return rigid_obj

    def set_pos_rot(
        self, object_name: str, position: List, rotation: List = [1.0, 0.0, 0.0, 0.0]
    ) -> ManagedBulletRigidObject:
        _obj = self.rigid_obj_mgr.get_object_by_handle(object_name)
        _obj.translation = mn.Vector3(position)
        _obj.rotation = make_quaternion(rotation)
        _obj.semantic_id = instance_vocab.word2index(object_name)
        return _obj

    def set_articulated_object(
        self,
        object_name: str,
        position: List,
        rotation: List = [1.0, 0.0, 0.0, 0.0],
        joint_positions: List = [0.0, 0.0, 0.0],
    ) -> ManagedBulletArticulatedObject:
        _obj = self.art_obj_mgr.get_object_by_handle(object_name)
        _obj.translation = mn.Vector3(position)
        _obj.rotation = make_quaternion(rotation)
        _obj.joint_positions = joint_positions
        for i in _obj.get_link_ids():
            visual_scene_nodes = _obj.get_link_visual_nodes(i)
            for visual_scene_node in visual_scene_nodes:
                visual_scene_node.semantic_id = instance_vocab.word2index(
                    f"{object_name}_joint_{i}"
                )
        return _obj

    def load_scene_rigid_obj(
        self, rigid_obj_infodicts: List[Dict], set_random_position: bool = False
    ) -> None:
        for infodict in rigid_obj_infodicts:
            if set_random_position:
                infodict["pos"] = random_position(**asdict(init_area))
            obj = self.set_pos_rot(infodict["name"], infodict["pos"], infodict["rot"])
        return

    def load_scene_rigid_obj_from_names(
        self, rigid_obj_name_list: List[str], position_extents: AreaExtent = init_area
    ) -> None:
        for obj_name in rigid_obj_name_list:
            obj_name = self.set_pos_rot(
                obj_name, random_position(**asdict(position_extents))
            )
        return

    def load_scene_art_obj(self, art_obj_infodicts: List[Dict]) -> None:
        for obj in art_obj_infodicts:
            obj = self.set_articulated_object(
                obj["name"], obj["pos"], obj["rot"], obj["joints"]
            )
        return

    def load_scene_rigid_obj_dict(
        self, rigid_obj_dict: Dict[str, Dict], set_random_position: bool = False
    ) -> None:
        for (name, pose_dict) in rigid_obj_dict.items():
            if set_random_position:
                pose_dict["pos"] = random_position(**asdict(init_area))
            obj = self.set_pos_rot(name, pose_dict["pos"], pose_dict["rot"])
        return

    def load_scene_art_obj_dict(self, art_obj_dict) -> None:
        for (name, pose_dict) in art_obj_dict.items():
            obj = self.set_articulated_object(
                name, pose_dict["pos"], pose_dict["rot"], pose_dict["joints"]
            )
        return

    def reset_to_frame(self, keyframe: Dict) -> None:
        print("Resetting to previous keyframe...")
        self.load_scene_rigid_obj(keyframe["rigidObjects"])
        self.load_scene_art_obj(keyframe["articulatedObjects"])
        return

    def create_session_keyframe(self) -> Dict:
        keyframe = {"rigidObjects": {}, "articulatedObjects": {}}
        for obj_name in self.rigid_obj_mgr.get_object_handles():
            obj = self.rigid_obj_mgr.get_object_by_handle(obj_name)
            obj_details = {
                obj_name: {
                    # 'name': obj_name,
                    "pos": np.asarray(obj.translation).tolist(),
                    "rot": get_quat_coeffs_from_magnum(obj.rotation),
                }
            }
            keyframe["rigidObjects"].update(obj_details)
        for obj_name in self.art_obj_mgr.get_object_handles():
            obj = self.art_obj_mgr.get_object_by_handle(obj_name)
            obj_details = {
                obj_name: {
                    # 'name': obj_name,
                    "pos": np.asarray(obj.translation).tolist(),
                    "rot": get_quat_coeffs_from_magnum(obj.rotation),
                    "joints": np.asarray(obj.joint_positions).tolist(),
                }
            }
            keyframe["articulatedObjects"].update(obj_details)
        return keyframe

    def get_dishwasher_object(self) -> ManagedBulletArticulatedObject:
        """
        Get dishwasher object wrapped in custom class Dishwasher for
        more utility functions
        """
        handles = self.art_obj_mgr.get_object_handles("ktc_dishwasher_:0000")
        assert len(handles)
        dishwasher_obj = self.art_obj_mgr.get_object_by_handle(handles[0])
        return dishwasher_obj
