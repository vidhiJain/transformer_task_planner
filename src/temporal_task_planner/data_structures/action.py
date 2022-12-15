from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from temporal_task_planner.data_structures.instance import (
    PlacementInstance,
    RigidInstance,
)


@dataclass
class Action:
    pick_instance: RigidInstance
    place_instance: PlacementInstance

    pick_track_id: int = None
    place_track_id: int = None
    init_pose: List[int] = None
    end_pose: List[int] = None

    def get_pick_track_id(self, to_match, instances, offset):
        for i, _instance in enumerate(instances):
            if _instance == to_match:
                return i + offset 

    def get_place_track_id(self, to_match, instances, offset):
        dist_matrix = np.array([inst.pose for inst in instances])[:, :3]
        err = np.linalg.norm(dist_matrix - np.array(to_match.pose[:3]), ord=2, axis=1) 
        return np.argmin(err) + offset

    def process(self, state, offset=0):
        self.init_pose = self.pick_instance.pose
        self.end_pose = self.place_instance.pose
        self.pick_track_id = self.get_pick_track_id(self.pick_instance, state.rigid_instances, offset)
        self.place_track_id = self.get_place_track_id(self.place_instance, state.place_instances, len(state.rigid_instances) + 1 + offset)
        return 

    def get_pick_handle(self) -> str:
        return self.pick_instance.instance_name

    def get_place_position(self) -> List[float]:
        if self.place_instance is None:
            print("Place instance not found")
        return self.place_instance.pose[:3]

    def get_place_orientation(self) -> List[float]:
        return self.place_instance.pose[3:]

    def get_place_pose(self) -> Dict[str, List[float]]:
        return {"pos": self.get_place_position(), "rot": self.get_place_orientation()}

    def __repr__(self) -> str:
        return f"Action to move {self.pick_instance} to {self.place_instance}"
