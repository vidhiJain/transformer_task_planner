from dataclasses import asdict, dataclass, field
from typing import Dict, List
from temporal_task_planner.data_structures.instance import (
    Instance,
    RigidInstance,
    PlacementInstance,
    ActInstance,
)


@dataclass
class State:
    """
    Objects and feasible poses at current timestep
    ACT token is appended to the end
    """

    rigid_instances: List[RigidInstance] = field(default_factory=list)
    place_instances: List[PlacementInstance] = field(default_factory=list)
    act_instances: ActInstance = None

    pick_only: bool = None

    def get_all_instances(self) -> List[Instance]:
        if self.pick_only:
            return self.rigid_instances + [self.act_instances]
        return (
            self.rigid_instances
            + [self.act_instances]
            + self.place_instances
            + [self.act_instances]
        )

    def apply_timestep(self, t) -> None:
        for instance in self.get_all_instances():
            instance.timestep = t
        return

    def process(self) -> Dict[str, List]:
        """
        Process as dict_of_lists
        """
        instances = self.get_all_instances()
        data = [asdict(instance) for instance in instances]
        return {key: [x[key] for x in data] for key in Instance.__annotations__.keys()}
