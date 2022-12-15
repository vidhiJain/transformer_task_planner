"""
Utility functions to check if utensil bounding boxes intersect in xz plane
Source: <url> from stackoverflow
"""
from typing import List
import magnum as mn


class Vector:
    """
    Handles 2d vectors
    """

    def __init__(self, coords):
        self.x = coords[0]
        self.y = coords[1]

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector([self.x + v.x, self.y + v.y])

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector([self.x - v.x, self.y - v.y])

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x * v.y - self.y * v.x


class Line:
    """
    Constructs a line from two vectors
    # ax + by + c = 0
    """

    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a * p.x + self.b * p.y + self.c

    def intersection(self, other):
        # See e.g. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a * other.b - self.b * other.a
        return Vector(
            [
                (self.b * other.c - self.c * other.b) / w,
                (self.c * other.a - self.a * other.c) / w,
            ]
        )


def transform_pts(quat: mn.Quaternion, vecs: List) -> List:
    """Magnum quaternion rotates each point vector in vecs"""
    return [quat.transform_vector(vec) for vec in vecs]


def project_to_xz(vecs: mn.Vector3) -> List:
    """Projects points in vecs on xz plane"""
    return [[vec.x, vec.z] for vec in vecs]


def select_corners(pts2d: List) -> List[Vector]:
    """Get the min and max for x and z coordinates
    as corners of the projected rectangle on xz plane
    """
    pts2d.sort(key=lambda x: x[0])
    pt_min_x = Vector(pts2d[0])
    pt_max_x = Vector(pts2d[-1])

    pts2d.sort(key=lambda x: x[1])
    pt_min_z = Vector(pts2d[0])
    pt_max_z = Vector(pts2d[-1])
    return [pt_min_x, pt_min_z, pt_max_x, pt_max_z]


def get_8_bb_points(cumulative_bb_max: List) -> List:
    x, y, z = cumulative_bb_max
    return [
        [-x, -y, -z],
        [-x, -y, z],
        [x, -y, -z],
        [x, -y, z],
        [-x, y, -z],
        [-x, y, z],
        [x, y, -z],
        [x, y, z],
    ]


def get_polygon_corners_in_xz(
    cumulative_bb_max: List, translation: mn.Vector3, rotation: mn.Quaternion
) -> List[Vector]:
    """Extracting corners of polygon (i.e. rotated rectangle in this case)
    1. Get 8 pts of the bounding box in object's body frame
    2. Transform each pt by the rotation quaternion
    3. Project the transformed points to xz plane
    4. Select the 4 pts with min and max - x and z values
    5. Translate the pts to the object's center of mass position

    Args:
        cumulative_bb_max: List, of size 8x3
        translation: mn.Vector3,
        rotation: mn.Quaternion

    Returns
        vertices_sf: List[Vector] of size 4
    """
    # 1. Get 8 pts of the bounding box in object's body frame
    vecs_body = get_8_bb_points(cumulative_bb_max)
    # 2. Transform each pt by the rotation quaternion
    quat = rotation
    vecs_bf_ori = transform_pts(quat, vecs_body)
    # 3. Project the transformed points to xz plane
    vecs_bf_ori_2d = project_to_xz(vecs_bf_ori)
    # 4. Select the 4 pts with min and max - x and z values
    vertices_bf = select_corners(vecs_bf_ori_2d)
    # 5. Translate the pts to the object's center of mass position
    vertices_sf = [Vector([translation[0], translation[2]]) + v for v in vertices_bf]
    return vertices_sf


def is_intersecting(rect1, rect2, epsilon=1e-7):
    """Calculates the intersection area between rect1 and rect2
    1. Iterate over the consequective points listed in rect2
    2. Consider a line between these points
    3. For rect1, calculate the sign of each vertex through this line.
    4. For any 2 consecutive vertex in rect1, do they have the same sign?
       If no, there is an intersection
    5. Detect intersection point between edges of rect1 and rect2
       Add the point to the new intersection list

    For Intersection area from intersection points:
        Ref: https://www.mathopenref.com/coordpolygonarea.html

    Args:
        rect1 : List[(Vector, Vector)] of size 4
        rect2 : List[(Vector, Vector)] of size 4
    Returns:
        boolean :
            True, if area of intersection > epsilon
            False, otherwise
    """
    intersection = rect1
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break  # No intersection
        line = Line(p, q)
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection,
            intersection[1:] + intersection[:1],
            line_values,
            line_values[1:] + line_values[:1],
        ):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)
        intersection = new_intersection
    if len(intersection) <= 2:
        return False
    area = 0.5 * sum(
        p.x * q.y - p.y * q.x
        for p, q in zip(intersection, intersection[1:] + intersection[:1])
    )
    # print('area of intersection: ',  area)
    if area < 1e-7:
        return False
    return True


if __name__ == "__main__":
    rect1 = [Vector([i, i + 1]) for i in range(1, 5)]
    rect2 = [Vector([i, i + 1]) for i in range(2, 6)]
    is_intersecting(rect1, rect2)
