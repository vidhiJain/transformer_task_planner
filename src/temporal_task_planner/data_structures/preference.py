from dataclasses import asdict, dataclass
from typing import Any, List
from temporal_task_planner.constants.gen_sess_config.lookup import (
    default_category_order,
)
from temporal_task_planner.data_structures.instance import RigidInstance


@dataclass
class Preference:
    """Preference structure
    Note: utensil category numbers are indexing as per
        default category order
    """

    load_top_rack_first: bool = False
    pick_close_to: str = "sink"
    place_close_to: str = "front-center"
    place_dist_k: int = 1

    category_order_numbers_top_rack: str = "523"
    category_order_numbers_bottom_rack: str = "0641"
    category_order_top_rack: Any = default_category_order
    category_order_bottom_rack: Any = default_category_order

    def process_category_order(self):
        category_indices = [int(cidx) for cidx in self.category_order_numbers_top_rack]
        self.category_order_top_rack = default_category_order[category_indices].tolist()
        category_indices = [
            int(cidx) for cidx in self.category_order_numbers_bottom_rack
        ]
        self.category_order_bottom_rack = default_category_order[
            category_indices
        ].tolist()
        return

    def get_utensil_to_place(self, rigid_instances: List[RigidInstance]) -> str:
        category_order = (
            self.category_order_top_rack + self.category_order_bottom_rack
            if self.load_top_rack_first
            else self.category_order_bottom_rack + self.category_order_top_rack
        )
        for category in category_order:
            for rigid_instance in rigid_instances:
                if rigid_instance.category_name == category:
                    return rigid_instance.instance_name
        return ""

    def get(self):
        self.process_category_order()
        return self.__dict__


if __name__ == "__main__":
    p = Preference()
    print(p.get())
    print(asdict(p))
