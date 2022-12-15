"""
Run the eval functions after the rigid objects' is settled 
i.e. linear velocity and angular velocity is zero

ensure rigid_obj.contact_test() is false
- when the object is released from a drop height > object y bb
    it will fall under gravity and settle with time

"""

import numpy as np
from habitat_sim.utils.common import quat_rotate_vector, quat_from_magnum


def is_settled(rigid_obj):
    if (
        rigid_obj.linear_velocity[0] != 0
        or rigid_obj.linear_velocity[1] != 0
        or rigid_obj.linear_velocity[2] != 0
    ):
        if (
            rigid_obj.angular_velocity[0] != 0
            or rigid_obj.angular_velocity[1] != 0
            or rigid_obj.angular_velocity[2] != 0
        ):
            return False
    return True


def is_facing_upwards(rigid_obj_rotation, thresh=0.5):
    """
    rigid_obj_rotation : mn Quat
    # rigid_obj : habitat-sim object
    """
    up_quats = [
        # [-0.420489, 0.873727, -0.10949, -0.218637],
        [0.0, 0.0, 0.0, 1.0]
    ]
    orig_up = np.array([0.0, 1.0, 0.0])
    new_up = quat_rotate_vector(quat_from_magnum(rigid_obj_rotation), orig_up)

    # if math.cos(angle_between_quats(quat_from_magnum(rigid_obj_rotation), quat_from_coeffs([0.,0.,0.,1.]))) > 0:
    if np.dot(new_up, orig_up) > thresh:
        return True
    return False


def is_rack_moved(
    dishwasher, value=0.6, thresh=0.05, is_top_rack=True
):  # [0.0, -1.5, 0.6]
    """
    dishwasher : habitat-sim object
    """
    index = 0
    if is_top_rack:
        index = 2

    if (
        dishwasher.joint_positions[index] > value + thresh
        or dishwasher.joint_positions[index] < value - thresh
    ):
        return True
    return False


def is_object_fallen_off(rigid_obj, is_top_rack):
    """
    rigid_obj: object to be evaluated
    """
    if not is_top_rack:
        dishwasher_rack_height = 0.15
    else:
        dishwasher_rack_height = 0.4
    y = rigid_obj.translation[1]
    # simulate if velocity : linear or angular is not zero
    if y < dishwasher_rack_height:
        return True
    return False


def is_outside_rack(rigid_obj, rack_extents):
    """
    rigid_obj: habitat-sim obj,
    rack_extents: dict of min and max positions
    """

    chosen_bb_body = rigid_obj.root_scene_node.cumulative_bb.max
    p = rigid_obj.translation
    Rq = quat_rotate_vector(quat_from_magnum(rigid_obj.rotation), chosen_bb_body)
    chosen_bb_world_min = p - Rq
    chosen_bb_world_max = p + Rq

    for i in range(3):
        if (chosen_bb_world_min[i] < rack_extents["min"][i]) or (
            chosen_bb_world_max[i] > rack_extents["max"][i]
        ):
            return True
    return False
