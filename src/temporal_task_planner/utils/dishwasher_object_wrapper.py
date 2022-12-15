import copy
from os import link
import numpy as np
from habitat_sim.physics import ManagedBulletArticulatedObject
from temporal_task_planner.constants.gen_sess_config.camera_params import rack_cam
from temporal_task_planner.constants.gen_sess_config.lookup import (
    dishwasher_part_poses,
    link_vocab,
    open_close_ranges,
)


class Dishwasher:
    """Wrapper for habitat_sim.physics.ManagedBulletArticulatedObject
    with utility functions for bottom rack, door, and top rack interactions
    """

    # TODO: change to habitat_sim.physics.ManagedBulletRigidObject for is-a relationship
    def __init__(self, obj: ManagedBulletArticulatedObject):
        self.obj = obj
        for key in obj.__dir__():
            if not key.startswith("__"):
                val = obj.__getattribute__(key)
                self.__setattr__(key, val)
        self.link_vocab = link_vocab
        self.ranges = open_close_ranges
        self.dishwasher_part_poses = dishwasher_part_poses

    def open(self, link_name):
        """Opens the link of the dishwasher.
        - 'bottom' rack,
        - dishwasher main 'door',
        - 'top' rack
        Note: Link names are in self.link_vocab
        """
        link_id = self.link_vocab[link_name]
        self.joint_positions[link_id] = self.ranges[link_id]["open"]

    def close(self, link_name):
        # closing the rack
        link_id = self.link_vocab[link_name]
        self.joint_positions[link_id] = self.ranges[link_id]["close"]

    def is_open(self, link_name, threshold=0.1):
        """Check if the link is open"""
        link_id = self.link_vocab[link_name]
        if (
            abs(self.joint_positions[link_id] - self.ranges[link_id]["close"])
            < threshold
        ):
            return False
        return True

    def get_open_close_range(self):
        """open_close_ranges
        # map from dishwasher index to tuple(value when closed, value when open)
        """
        return {
            0: {"close": 0, "open": 0.6},
            1: {"close": 0, "open": -1.5},
            2: {"close": 0, "open": 0.6},
        }

    def get_position_of_top_rack(self):
        # TODO: dishwasher_rearrange_2/dataset.py logic
        pass

    def get_position_of_bottom_rack(self):
        # TODO: dishwasher_rearrange_2/dataset.py logic
        pass

    def get_joint_positions(self):
        return copy.deepcopy(self.joint_positions)

    def get_joint_position_limits(self, link_id):
        lower_limits, upper_limits = self.joint_position_limits
        return lower_limits[link_id], upper_limits[link_id]

    def set_joint_positions(self, joint_positions):
        self.joint_positions = joint_positions
        self.obj.joint_positions = self.joint_positions
        return

    def set_joint_velocities(self, joint_velocities=[0.0, 0.0, 0.0]):
        self.joint_velocities = joint_velocities
        self.obj.joint_velocities = self.joint_velocities
        return

    def toggle_joint_by_link_id(self, articulated_link, num_steps=10):
        link_id = articulated_link - 1
        curpos = self.joint_positions[link_id]
        lower, upper = self.get_joint_position_limits(link_id)
        if (curpos - lower) < (upper - curpos):
            toggled_joint = upper
            interim_joint_values = np.linspace(lower, upper, num_steps)
        else:
            toggled_joint = lower
            interim_joint_values = np.linspace(upper, lower, num_steps)
        desired_joint_positions = self.get_joint_positions()
        desired_joint_positions[link_id] = toggled_joint
        return desired_joint_positions, interim_joint_values

    def get_toggled_value(self, link_id, orig_jp, thresh=0.1):
        if link_id in [0, 2]:
            if abs(0.6 - orig_jp) < thresh:
                jp = 0.0
            else:  # if abs(jp - 0.) < 0.1:
                jp = 0.6
        else:
            if abs(-1.5 - orig_jp) < thresh:
                jp = 0.0
            else:
                jp = -1.5
        return jp

    def get_rack_pose(self, link_id, jp, reverse=False):
        position = copy.deepcopy(rack_cam["translation"])
        # orig_jp = self.joint_positions[link_id]
        jp = self.get_toggled_value(link_id, jp) if reverse else jp
        if link_id == 0:  # bottom rack
            position[1] = 0.3
            position[2] = position[2] - 0.6 + jp
        elif link_id == 2:
            position[1] = 0.56
            position[2] = position[2] - 0.6 + jp
        elif link_id == 1:
            if abs(jp - -1.5) < 0.1:
                position[1] = 0.01
                position[2] = position[2]
            else:  # jp == 0
                position[1] = 0.43
                position[2] = position[2] - 0.3
        pose = position + [1.0, 0.0, 0.0, 0.0]  # [jp]*4
        return pose

    def populate_all_part_poses(self):
        rack_poses = {}
        for name, link_id in self.link_vocab.items():
            rack_poses[name] = {}
            for status, jp in self.ranges[link_id].items():
                rack_poses[name][status] = self.get_rack_pose(link_id, jp)
        return rack_poses
