from dataclasses import dataclass
from typing import List


@dataclass
class Instance:
    timestep: int
    category: List[float]
    pose: List[float]

    action_masks: bool
    is_real: bool
    category_token: int
    instance_token: int


@dataclass
class RigidInstance(Instance):
    """Real Objects
    For each instance, set:
        is_real: bool = True
    """

    instance_name: str
    category_name: str
    category_token_name: str

    def __repr__(self) -> str:
        return f"rigid body: {self.instance_name}"

    def __eq__(self, __o: object) -> bool:
        if self.is_real and (self.instance_token == __o.instance_token):
            return True
        return False

    def get_place_position(self) -> List[float]:
        return self.pose[:3]


@dataclass
class PlacementInstance(Instance):
    """Placement poses
    For each instance, set:
        is_real: bool = False
    """

    category_name: str
    category_token_name: str

    def __repr__(self) -> str:
        return f"place pose: {self.pose}"

    def __eq__(self, __o: object, epsilon: float = 1e-5) -> bool:
        assert len(__o.pose) == 7, f"self.pose: {self.pose}, other pose: {__o.pose}"
        err = sum((self.pose[i] - __o.pose[i])**2 for i in range(3))
        if err < epsilon:
            return True
        return False


@dataclass
class ActInstance(Instance):
    """Dummy instance to trigger action
    For each instance, set:
        is_real: bool = False
        is_action_available: bool ; is target in Action available for this prediction
            True : during training
            False: during rollout
            -   Used in the TemporalContext process_states function
        is_action_to_be_predicted: bool ; to get output latent from the model at this instance
    """

    instance_name: str
    category_token_name: str
    is_action_available: bool

    def __repr__(self) -> str:
        return f"act token"
