from typing import Dict, List
from dataclasses import dataclass


@dataclass
class UserAction:
    visible: List
    feasiblePick: List
    feasiblePlace: List
    placedUtensils: List
    isSettlingAction: bool = False
    actionType: str = None
    rigidObj: str = ""
    articulatedObj: str = ""
    articulatedLink: int = -1
    startFrame: int = 0
    endFrame: int = 0
    initPose: List = None
    endPose: List = None
    is_top_rack: bool = False
    is_bottom_rack: bool = False

    def assert_checks(self):
        assert self.rigidObj == "" or self.articulatedObj == "" or self.isSettlingAction
        assert self.startFrame < self.endFrame

    def get(self):
        return self.__dict__
