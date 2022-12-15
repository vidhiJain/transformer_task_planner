#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import magnum as mn
import habitat_sim
from habitat_sim.utils import viz_utils as vut
from habitat_sim import SensorType, SensorSubType
from pathlib import Path

from temporal_task_planner.constants.gen_sess_config.lookup import (
    already_placed,
    dataset_filepath,
    scene_filepath,
)
from temporal_task_planner.utils.gen_sess_utils import make_quaternion
from temporal_task_planner.utils.kitchen_simulator import (
    KitchenSimulator,
    make_configuration,
)

step_frame = 1


def init_sim():
    cfg = make_configuration(
        dataset_filepath,
        scene_filepath,
        height=544,
        width=720,
        cam_info_dict=[
            {
                "uuid": "color_sensor",
                "type": SensorType.COLOR,
                "subtype": SensorSubType.PINHOLE,
            }
        ],
    )

    sim = KitchenSimulator(cfg)
    sim.init_default_agent()
    front_view_cam = {
        "translation": [-1.059, 2.5, 1.799],
        "rotation": [0.9423165, -0.534723, 0.0, 0.0],
        "zoom": 1,
    }
    sim.assign_camera_params("color_sensor", **front_view_cam)
    return sim


def session_viewer(sim, args):
    show_video = args.show_video
    make_video = args.make_video

    if make_video and not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    with open(args.session, "r") as f:
        json_root = json.load(f)
        assert "session" in json_root
        session = json_root["session"]

    keyframes = session["keyframes"]
    assert keyframes

    observations = []

    for user_action_index, user_action in enumerate(session["userActions"]):

        print(
            "rendering "
            + str(user_action_index + 1)
            + "/"
            + str(len(session["userActions"]))
            + " actions"
        )

        for keyframe_index in range(
            user_action["startFrame"], user_action["endFrame"], step_frame
        ):
            keyframe = keyframes[keyframe_index]
            rigid_obj_list = {}  # keyframe["rigidObjects"]

            if args.no_clutter:
                for name in keyframe["rigidObjects"]:
                    if name in session["placedUtensils"] + already_placed:
                        rigid_obj_list.update({name: keyframe["rigidObjects"][name]})
            else:
                rigid_obj_list = keyframe["rigidObjects"]

            sim.load_scene_rigid_obj_dict(rigid_obj_list)
            sim.load_scene_art_obj_dict(keyframe["articulatedObjects"])

            obs = sim.get_sensor_observations()
            observations.append(obs)

            if keyframe_index == user_action["endFrame"]:
                # repeat the last frame in the action to produce a pause in the video
                for _ in range(100):
                    observations.append(obs)

    if make_video:
        vut.make_video(
            observations,
            "color_sensor",
            "color",
            args.output_folder
            + "/"
            + os.path.splitext(os.path.basename(args.session))[0],
            open_vid=show_video,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show-video", dest="show_video", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.add_argument("--session")
    parser.add_argument("--output-folder")
    parser.add_argument("--no-clutter", dest="no_clutter", action="store_true")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    curdir = "./"
    for subdir in args.output_folder.split("/"):
        curdir = Path(curdir, subdir)
        if not os.path.exists(curdir):
            os.mkdir(curdir)
    sim = init_sim()
    session_viewer(sim, args)
