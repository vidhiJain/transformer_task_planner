from abc import ABC, abstractmethod
from typing import Any
from temporal_task_planner.data_structures.action import Action
from temporal_task_planner.data_structures.state import State


class Policy(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def reset(self, **kwargs) -> None:
        self.counter = 0
        raise NotImplementedError

    @abstractmethod
    def get_action(self, state: Any) -> Action:
        raise NotImplementedError
