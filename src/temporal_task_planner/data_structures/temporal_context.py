from dataclasses import dataclass, field
from typing import Any, List
import copy
from temporal_task_planner.data_structures.action import Action
from temporal_task_planner.data_structures.instance import Instance
from temporal_task_planner.data_structures.state import State


@dataclass
class TemporalContext:
    """Combine several states based on prev context window
    to the input
    """

    # context_history: int
    states: List[State] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)

    pick_only: bool = None  # TODO: set from states

    def init_sub_copy(self, start_id: int, end_id: int) -> Any:
        """Selective Copy Constructor to create another TemporalContext object
        with limited set of state and action lists
        """
        return copy.deepcopy(TemporalContext(
            states=self.states[start_id:end_id], actions=self.actions[start_id:end_id]
        ))

    def process_pick_only(self):
        if len(self.states):
            return self.states[0].pick_only
        return True

    def process_states(self):
        """
        Process as dict_of_lists of input attributes
        Additional: If action is available, process those actions
            for current state and with previous instances' offset
        """
        self.pick_only = self.process_pick_only()
        data = []
        offset = 0
        action_counter = 0
        for i, state in enumerate(self.states):
            state.apply_timestep(i + 1)
            data.append(state.process())
            if state.act_instances.is_action_available:
                self.actions[action_counter].process(state, offset)
                action_counter += 1
            offset += len(data[-1]["timestep"])

        combined_data_dict = {
            key: [x[key] for x in data] for key in Instance.__annotations__.keys()
        }
        flatten_combined = {
            key: sum(val, []) for key, val in combined_data_dict.items()
        }
        return flatten_combined

    def process_actions(self):
        target = {"act": [], "init_pose": [], "end_pose": []}
        for action in self.actions:
            if self.pick_only:
                target["act"] += [action.pick_track_id]
            else:
                target["act"] += [action.pick_track_id, action.place_track_id]
            target["init_pose"].append(action.init_pose)
            target["end_pose"].append(action.end_pose)
        return target


if __name__ == "__main__":
    tc1 = TemporalContext()
