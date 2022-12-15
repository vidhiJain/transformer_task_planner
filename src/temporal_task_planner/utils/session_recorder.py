import json
from typing import Dict, List
from dataclasses import asdict, dataclass
from temporal_task_planner.data_structures.preference import Preference
from temporal_task_planner.utils.kitchen_simulator import KitchenSimulator
from temporal_task_planner.constants.gen_sess_config.lookup import *


@dataclass
class SessionRecorder:
    """
    [utility] functions for update record dict
    """

    sim: KitchenSimulator
    preference: Preference

    def init_session_record(self):
        self.record = dict(keyframes=[], userActions=[])
        self.frame_num = 0  # to be updated

    def update_keyframes(self, step_time=0.016, settle_time=200, enable_physics=False):
        for i in range(settle_time):
            if enable_physics:
                self.sim.step_world(dt=step_time)  # 0.016
            keyframe = self.sim.create_session_keyframe()
            self.record["keyframes"].append(keyframe)
            self.frame_num += 1
        return keyframe

    def get_session_dict(self, cam_extrinsics=None):
        session_dict = {
            "session": {
                "dataset": dataset_filepath,
                "scene": scene_filepath,
                "preference": asdict(self.preference),
                "defaultCamera": cam_extrinsics,
                "userActions": self.record["userActions"],
                "keyframes": self.record["keyframes"],
                "placedUtensils": self.track_placed["top_rack"]
                + self.track_placed["bottom_rack"],
            }
        }
        return session_dict

    def save_session(self, session_dict, savepath):
        print(f"Saving session at {savepath}")

        if savepath is not None:
            with open(savepath, "w") as f:
                json.dump(session_dict, f)  # , indent=4)
        return session_dict
