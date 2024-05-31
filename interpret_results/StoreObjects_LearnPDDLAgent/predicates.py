import numpy as np
from typing import List, Dict, Set, FrozenSet
from perception_api import (
    get_detected_object_list,
    get_object_xy_position,
    get_object_xy_size,
    get_object_category,
    get_object_water_amount,
    get_gripper_position,
    get_gripper_open_width,
    get_in_gripper_mass,
    get_gripper_y_size,
    get_gripper_max_open_width,
    get_table_x_range,
    get_table_y_height,
)




# Utility functions:
# <utility>
def get_object_xy_bbox(a) -> np.ndarray:
    """
    Get the xyxy bounding box of object a
    :param a: string, name of detected object
    :return: np.ndarray, [x1, y1, x2, y2], where x1 is left, x2 is right, y1 is bottom, y2 is top
    """
    object_a_position = get_object_xy_position(a)
    object_a_size = get_object_xy_size(a)
    return [
        object_a_position[0] - object_a_size[0] / 2,
        object_a_position[1] - object_a_size[1] / 2,
        object_a_position[0] + object_a_size[0] / 2,
        object_a_position[1] + object_a_size[1] / 2,
    ]
# <end-of-utility>

# <utility>
def get_in_gripper_xy_bbox() -> np.ndarray:
    """
    Get the xyxy bounding box of space within the gripper
    :return: np.ndarray, [x1, y1, x2, y2], where x1 is left, x2 is right, y1 is bottom, y2 is top
    """
    gripper_position = get_gripper_position()
    gripper_open_width = get_gripper_open_width()
    gripper_y_size = get_gripper_y_size()
    return [
        gripper_position[0] - gripper_open_width / 2,
        gripper_position[1] - gripper_y_size / 2,
        gripper_position[0] + gripper_open_width / 2,
        gripper_position[1] + gripper_y_size / 2,
    ]
# <end-of-utility>

# Predicates:
# <predicate>
def obj_in_gripper(a) -> bool:
    """
    Description: <<whether object a is held by the gripper>>
    The predicate holds True when the mass in gripper is non-zero, object a is aligned with the opened gripper along x axis, and overlaps with the gripper along y axis
    :param a: string, name of detected object
    :return: bool
    """
    eps = 0.2
    # get in gripper mass
    in_gripper_mass = get_in_gripper_mass()
    # get in_gripper xyxy bbox
    in_gripper_xyxy_bbox = get_in_gripper_xy_bbox()

    # get bbox_xyxy of object a
    object_a_xyxy_bbox = get_object_xy_bbox(a)

    # check whether the mass in gripper is non-zero
    # in_gripper_mass > eps
    if in_gripper_mass > eps:
        # check whether the object a is aligned with the gripper along x axis
        # abs(a.x1 - gripper.x1) < eps and abs(a.x2 - gripper.x2) < eps
        if (
            np.abs(object_a_xyxy_bbox[0] - in_gripper_xyxy_bbox[0]) < eps
            and np.abs(object_a_xyxy_bbox[2] - in_gripper_xyxy_bbox[2]) < eps
        ):
            # check whether the object a overlaps with the gripper along y axis
            # a.y1 < gripper.y2 + eps and a.y2 > gripper.y1 - eps
            if (
                object_a_xyxy_bbox[1] < in_gripper_xyxy_bbox[3] + eps
                and object_a_xyxy_bbox[3] > in_gripper_xyxy_bbox[1] - eps
            ):
                return True
            else:
                return False
        else:
            return False
    else:
        return False
# <end-of-predicate>

# <predicate>
def obj_on_obj(a: str, b: str) -> bool:
    """
    Description: <<check whether object a is on top of object b>>
    The predicate holds True if the bottom of object a is within a small epsilon of the top of object b, and their x-axis projections overlap.
    :param a: string, name of object a
    :param b: string, name of object b
    :return: bool
    """
    eps = 0.2
    # Get the bounding boxes for both objects
    object_a_bbox = get_object_xy_bbox(a)
    object_b_bbox = get_object_xy_bbox(b)

    # Check if the bottom of object a is within epsilon of the top of object b
    if np.abs(object_a_bbox[1] - object_b_bbox[3]) < eps:
        # Check if the x-axis projections of the two objects overlap
        # This is true if one object's right side is further to the right than the other object's left side and vice versa
        if not (object_a_bbox[2] < object_b_bbox[0] or object_a_bbox[0] > object_b_bbox[2]):
            return True
    return False
# <end-of-predicate>

# <predicate>
def graspable(a: str) -> bool:
    """
    Description: <<checks whether object a can be grasped by the gripper, considering only the object's width (x dimension) relative to the gripper's maximum open width>>
    The predicate holds True if the object's width (x dimension) is less than or equal to the gripper's maximum open width.
    :param a: string, name of detected object
    :return: bool
    """
    # Get the gripper's maximum open width
    gripper_max_open_width = get_gripper_max_open_width()
    
    # Get the object's size
    object_a_size = get_object_xy_size(a)
    
    # Check if the object's width is less than or equal to the gripper's maximum open width
    if object_a_size[0] <= gripper_max_open_width:
        return True
    else:
        return False
# <end-of-predicate>

# <predicate>
def obj_on_table(a: str) -> bool:
    """
    Description: <<check whether object a is on top of the table>>
    The predicate holds True if the bottom of object a is within a small epsilon of the table height, and its x-axis projection is within the table's x range.
    :param a: string, name of object a
    :return: bool
    """
    eps = 0.2
    table_height = get_table_y_height()
    table_x_range = get_table_x_range()
    
    # Get the bounding box for object a
    object_a_bbox = get_object_xy_bbox(a)
    
    # Check if the bottom of object a is within epsilon of the table height
    if np.abs(object_a_bbox[1] - table_height) < eps:
        # Check if the x-axis projection of object a is within the table's x range
        if object_a_bbox[0] >= table_x_range[0] and object_a_bbox[2] <= table_x_range[1]:
            return True
    return False
# <end-of-predicate>

# <predicate>
def gripper_empty() -> bool:
    """
    Description: <<check whether the gripper is not holding any object, the predicate holds when the gripper is empty>>
    The predicate holds True if the mass in the gripper is zero or very close to zero.
    :return: bool
    """
    eps = 0.01  # A small epsilon to account for floating-point inaccuracies
    in_gripper_mass = get_in_gripper_mass()
    return in_gripper_mass < eps
# <end-of-predicate>

# <predicate>
def obj_clear(a: str) -> bool:
    """
    Description: <<check whether object a has no other objects on it, the predicate holds true if there are no objects placed on object a>>
    The predicate iterates through all detected objects and checks if any of them is on top of object a.
    :param a: string, name of object a
    :return: bool
    """
    detected_objects = get_detected_object_list()
    for obj in detected_objects:
        if obj != a and obj_on_obj(obj, a):
            return False
    return True
# <end-of-predicate>