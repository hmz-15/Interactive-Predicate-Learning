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

eps = 0.2

# define function: obj_xyxy_box = get_object_xy_bbox(obj)
def get_object_xy_bbox(obj) -> np.ndarray:
    """
    Get the xyxy bounding box of object obj
    :param obj: string, name of detected object
    :return: np.ndarray, [x1, y1, x2, y2], where x1 is left, x2 is right, y1 is bottom, y2 is top
    """
    object_a_position = get_object_xy_position(obj)
    object_a_size = get_object_xy_size(obj)
    return [
        object_a_position[0] - object_a_size[0] / 2,
        object_a_position[1] - object_a_size[1] / 2,
        object_a_position[0] + object_a_size[0] / 2,
        object_a_position[1] + object_a_size[1] / 2,
    ]

# define function: in_gripper_xyxy_bbox = get_in_gripper_xy_bbox()
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

# define function: res = check_obj_in_gripper(obj)
def check_obj_in_gripper(obj) -> bool:
    """
    Description: check whether obj is held by the gripper
    The predicate holds True when the mass in gripper is non-zero, obj is aligned with the opened gripper along x axis, and overlaps with the gripper along y axis
    :param obj: string, name of detected object
    :return: bool
    """
    # get in gripper mass
    in_gripper_mass = get_in_gripper_mass()
    # get in_gripper xyxy bbox
    in_gripper_xyxy_bbox = get_in_gripper_xy_bbox()

    # get bbox_xyxy of obj
    object_a_xyxy_bbox = get_object_xy_bbox(obj)

    # check whether the mass in gripper is non-zero
    # in_gripper_mass > eps
    if in_gripper_mass > eps:
        # check whether the obj is aligned with the gripper along x axis
        # abs(a.x1 - gripper.x1) < eps and abs(a.x2 - gripper.x2) < eps
        if (
            np.abs(object_a_xyxy_bbox[0] - in_gripper_xyxy_bbox[0]) < eps
            and np.abs(object_a_xyxy_bbox[2] - in_gripper_xyxy_bbox[2]) < eps
        ):
            # check whether the obj overlaps with the gripper along y axis
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