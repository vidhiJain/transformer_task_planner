from typing import Dict, List, Tuple
from dataclasses import dataclass
import magnum as mn
import numpy as np
from temporal_task_planner.constants.gen_sess_config.lookup import *  # instance_vocab, already_placed_ids
from temporal_task_planner.constants.gen_sess_config.area_extents import *
from temporal_task_planner.utils.intersection_detection import *
from temporal_task_planner.utils.kitchen_simulator import KitchenSimulator
from temporal_task_planner.utils.gen_sess_utils import *  # make_quaternion, get_quat_coeffs_from_magnum, get_select_instances, get_potential_placements
from temporal_task_planner.utils.eval_placement import (
    is_facing_upwards,
    is_object_fallen_off,
    is_outside_rack,
    is_rack_moved,
    is_settled,
)

"""TODO: fix inheritance in CounterPlacePoseMgr & DishwasherPlacePoseMgr"""


class CounterPlacePoseManager:
    """Non-overlapping random placement of rigid instances
     in xz plane (counter/dishwasher rack)
    Args: (TODO: add to constants/lookup)
    - settle height dict
    - area extent
    - bounding_box_dict
    """

    def __init__(
        self,
        settle_height_dict: Dict[str, float],
        area_extent: AreaExtent,
        bounding_box_dict: Dict[str, List[float]],
        num_per_category: int = 100,
    ) -> None:
        self.settle_height_dict = settle_height_dict
        self.area_extent = area_extent
        self.bounding_box_dict = bounding_box_dict
        self.category_list = default_category_order.tolist()
        self.num_per_category = num_per_category
        self.potential_placements = self.init_potential_placements()
        self.place_close_to = {
            "pos": semantic_coordinates["dishwasher_countertop"],
            "rot": default_rot,
        }
        self.place_close_k = 3

    def init_potential_placements(self):
        """TODO: fix init potential placements??!"""
        potential_placements = {}
        for category in self.category_list:
            sampled_positions = self.get_extent(category).sample_3d_position(
                num=self.num_per_category
            )
            potential_placements[category] = []
            for i in range(self.num_per_category):
                potential_placements[category].append(
                    {
                        "pos": sampled_positions[i].tolist(),
                        "rot": default_rot,
                    }
                )
        return potential_placements

    def get_extent(self, category):
        """calculate the extent in the area based on the category bb"""
        return AreaExtent(
            min_x=self.area_extent.min_x + self.bounding_box_dict[category][0] / 2,
            max_x=self.area_extent.max_x - self.bounding_box_dict[category][0] / 2,
            min_y=self.settle_height_dict[category],
            max_y=self.settle_height_dict[category],
            min_z=self.area_extent.min_z + self.bounding_box_dict[category][2] / 2,
            max_z=self.area_extent.max_z - self.bounding_box_dict[category][2] / 2,
        )

    def get_place_pose(self, utensil_name: str, potential_placements=None):
        category = utensil_name.split(":")[0]

        if potential_placements is None:
            potential_placements = self.potential_placements
        if self.place_close_to is None:
            idx = np.random.choice(len(potential_placements[category]))
            self.place_close_to = potential_placements[category][idx]
        place_pose = None
        if category in list(potential_placements.keys()):
            place_pose = closestk_pose(
                potential_placements[category],
                self.place_close_to,
                k=self.place_close_k,
            )
            # update last place pose
            self.place_close_to = place_pose
        return place_pose

    def remove_occupied_places(
        self,
        chosen_name: str,
        place_pose: Dict[str, List[float]],
        potential_placements: Dict = None,
    ) -> Dict:
        """
        Assume that the chosen name has been placed in the rack
        """
        if potential_placements is None:
            potential_placements = self.potential_placements
        # rigid_obj = self.rigid_obj_mgr.get_object_by_handle(chosen_name)
        cumulative_bb_max = cumulative_bb_dict[chosen_name.split(":")[0]]
        rect1 = get_polygon_corners_in_xz(
            cumulative_bb_max,
            mn.Vector3d(place_pose["pos"]),
            make_quaternion(place_pose["rot"]),
        )
        new_potential_placements = {}
        for objkey, poses in potential_placements.items():
            new_poses = []
            for pose in poses:
                cumulative_bb_max = mn.Vector3(self.bounding_box_dict[objkey])
                rect2 = get_polygon_corners_in_xz(
                    cumulative_bb_max, pose["pos"], make_quaternion(pose["rot"])
                )
                if not is_intersecting(rect1, rect2):
                    new_poses.append(pose)
            if len(new_poses):
                new_potential_placements.update({objkey: new_poses})
        self.potential_placements = new_potential_placements
        return new_potential_placements

    def set_potential_placements(self, potential_placements: Dict) -> None:
        self.potential_placements = potential_placements
        return


class DishwasherPlacePoseManager:
    """
    [utility] functions for potential placement locations
    """

    def __init__(self) -> None:
        self.potential_placements = self.init_potential_placements()
        self.place_close_to = None
        self.place_close_k = 3

    def init_params(self, preference: Dict) -> None:
        self.place_close_to = {
            # bottom
            0: spatial_2d_rack_coordinates[preference["place_close_to"]],
            # top
            1: spatial_2d_rack_coordinates[preference["place_close_to"]],
        }
        self.place_close_k = preference["place_dist_k"]
        return

    def init_potential_placements(self) -> Dict:
        bottom_rack_placements = get_potential_placements(
            original_potential_placements, is_top_rack=False
        )
        top_rack_placements = get_potential_placements(
            original_potential_placements, is_top_rack=True
        )
        return {
            0: bottom_rack_placements,  # 'bottom'
            1: top_rack_placements,  # 'top'
            2: dict(),  # 'both racks are closed'
        }

    def set_potential_placements(
        self, potential_placements: Dict, is_top_rack: bool
    ) -> None:
        self.potential_placements[int(is_top_rack)] = potential_placements
        return

    def get_place_pose(
        self,
        category: str,
        is_top_rack: bool = None,
        potential_placements: Dict = None,
        place_close_to: Dict = None,
        place_close_k: int = None,
    ) -> Dict[str, List[float]]:
        assert is_top_rack is not None
        assert potential_placements is None
        if potential_placements is None:
            potential_placements = self.potential_placements[int(is_top_rack)]
            place_close_to = self.place_close_to[int(is_top_rack)]
        if place_close_k is None:
            place_close_k = self.place_close_k
        # random spatial placement?!
        assert place_close_to is not None, "place_close_to is None"
        if place_close_to is None:
            place_close_to = np.random.choice(potential_placements[category])
        place_pose = None
        if category in list(potential_placements.keys()):
            place_pose = closestk_pose(
                potential_placements[category], place_close_to, k=place_close_k
            )
            # update last place pose
            self.place_close_to[int(is_top_rack)] = place_pose
        return place_pose

    def get_pick_place_targets(
        self,
        sim: KitchenSimulator,
        category_order: List[str],
        pick_close_to: List[float],
        place_close_to: Dict[str, List],
        place_close_k: int,
        feasible_pick_ids: List[int],
        potential_placements: Dict[str, Dict],
        already_placed_ids: List[int],
        is_top_rack: bool,
    ) -> Tuple[int, Dict]:
        """
        If a utensil is almost completely visible and
            has potential placement poses in the rack,
        only then the pick instance name and
            placement pose is returned, else None
        """
        for ids in already_placed_ids:
            if ids in feasible_pick_ids:
                feasible_pick_ids.remove(ids)

        feasible_id_handles = np.array(
            [instance_vocab.index2word(idx) for idx in feasible_pick_ids]
        )
        feasible_categories = np.array(
            [handle_name.split(":")[0] for handle_name in feasible_id_handles]
        )
        feasible_distances = np.array(
            [
                (pick_close_to - sim.get_position_by_handle(handle_name)).length()
                for handle_name in feasible_id_handles
            ]
        )
        pair = (None, None)
        for cat in category_order:
            if cat in feasible_categories:
                indices = np.where(feasible_categories == cat)  # , True, False)
                pick_id = feasible_id_handles[indices][
                    np.argmin(feasible_distances[indices])
                ]
                place_pose = self.get_place_pose(
                    cat,
                    is_top_rack,
                    potential_placements=potential_placements,
                    place_close_to=place_close_to,
                    place_close_k=place_close_k,
                )
                pair = (pick_id, place_pose)
                if place_pose is not None:
                    return pair
        return pair

    def remove_occupied_places(
        self,
        chosen_name: str,
        place_pose: Dict[str, List[float]],
        potential_placements: Dict = None,
        is_top_rack: bool = True,
    ) -> Dict:
        """
        Assume that the chosen name has been placed in the rack
        """
        if potential_placements is None:
            potential_placements = self.potential_placements[int(is_top_rack)]
        # rigid_obj = self.rigid_obj_mgr.get_object_by_handle(chosen_name)
        cumulative_bb_max = cumulative_bb_dict[chosen_name.split(":")[0]]
        rect1 = get_polygon_corners_in_xz(
            cumulative_bb_max,
            mn.Vector3d(place_pose["pos"]),
            make_quaternion(place_pose["rot"]),
        )
        new_potential_placements = {}
        for objkey, poses in potential_placements.items():
            new_poses = []
            for pose in poses:
                cumulative_bb_max = mn.Vector3(cumulative_bb_dict[objkey])
                rect2 = get_polygon_corners_in_xz(
                    cumulative_bb_max, pose["pos"], make_quaternion(pose["rot"])
                )
                if not is_intersecting(rect1, rect2):
                    new_poses.append(pose)
            if len(new_poses):
                new_potential_placements.update({objkey: new_poses})
        return new_potential_placements

    def check_placement(self, placed_objects: List, is_top_rack: bool) -> int:
        if is_rack_moved(self.dishwasher.get_sim_object(), is_top_rack=is_top_rack):
            return 1
        for placed_name in placed_objects:
            rigid_obj = self.rigid_obj_mgr.get_object_by_handle(placed_name)
            if is_facing_upwards(rigid_obj.rotation):
                return 2

            if is_outside_rack(rigid_obj, get_rack_extents(thresh=0.2)):
                return 3
            # TODO: separate placed_object list needed for arbitrary ordering in loading top and bottom racks
            if is_object_fallen_off(rigid_obj, is_top_rack):
                return 4
            if not is_settled(rigid_obj):
                return 5
        return 0
