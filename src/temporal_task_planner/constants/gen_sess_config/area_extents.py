from dataclasses import dataclass
from typing import List
import numpy as np

"""Area extents describes the endpoints of 
a 3D bounding box enveloping a semantic area like 
- sink + left side counter (init) area 
- dishwasher, 
- right-side counter ('safe') area,
"""


@dataclass
class AreaExtent:
    min_x: float = -2.7
    max_x: float = -1.5
    min_y: float = 0.98
    max_y: float = 1.2
    min_z: float = 0.3
    max_z: float = 0.65

    def get_center(self):
        return [
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2,
        ]

    def expand_area(self, thresh=0.1):
        """Reduces the min and increases the max
        by the given threshold
        Note: not modify the original extents
        Returns
            AreaExtent
        """
        min_x = self.min_x - thresh
        min_z = self.min_z - thresh
        max_x = self.max_x + thresh
        max_z = self.max_z + thresh
        return AreaExtent(min_x, max_x, self.min_y, self.max_y, min_z, max_z)

    def get_min_max_center(self):
        return {
            "min": [self.min_x, self.min_y, self.min_z],
            "max": [self.max_x, self.max_y, self.max_z],
            "center": self.get_center(),
        }

    def aslist(self):
        return [
            [self.min_x, self.min_y, self.min_z],
            [self.max_x, self.max_y, self.max_z],
        ]

    def contains(self, position: List[int]) -> bool:
        if (
            (self.min_x <= position[0])
            and (self.max_x >= position[0])
            and (self.min_y <= position[1])
            and (self.max_y >= position[1])
            and (self.min_z <= position[2])
            and (self.max_z >= position[2])
        ):
            return True
        return False

    def sample_3d_position(self, num: int = 1) -> np.array:
        val = np.random.random_sample((3, num))
        possible_positions = [
            (self.max_x - self.min_x) * val[0] + self.min_x,
            (self.max_y - self.min_y) * val[1] + self.min_y,
            (self.max_z - self.min_z) * val[2] + self.min_z,
        ]
        return np.array(possible_positions).T


init_area = AreaExtent(
    min_x=-2.7, min_y=0.98, min_z=0.3, max_x=-1.5, max_y=1.2, max_z=0.65
)

dishwasher_extents = AreaExtent(
    min_x=-1.35, min_y=0.0, min_z=-0.1, max_x=-0.70, max_y=0.82, max_z=1.2
)

safe_area = AreaExtent(
    min_x=-1.35, min_y=0.96, min_z=0.2, max_x=-0.73, max_y=0.98, max_z=0.6
)

rack_area = AreaExtent(
    min_x=-1.2, min_y=0.54, min_z=0.8, max_x=-0.9, max_y=0.64, max_z=1.15
)

zero_area = AreaExtent(min_x=0.0, min_y=0.0, min_z=0.0, max_x=0.0, max_y=0.0, max_z=0.0)


def get_rack_extents(thresh=0.1):
    # return {
    #     "min": [-1.2 - thresh, 0.54, 0.8 - thresh],
    #     "max": [-0.9 + thresh, 0.64, 1.15 + thresh],
    #     "center": [-1.05, 0.64, 0.975],
    # }
    expanded_rack = rack_area.expand_area(thresh=thresh)
    return expanded_rack.get_min_max_center()
