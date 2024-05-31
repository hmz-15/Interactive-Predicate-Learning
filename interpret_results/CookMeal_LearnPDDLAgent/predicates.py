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

# <utility>
def is_bbox_on_table(bbox: np.ndarray) -> bool:
    """
    Check if the bounding box of an object is on the table based on its y position and the table's height.
    :param bbox: np.ndarray, [x1, y1, x2, y2], where x1 is left, x2 is right, y1 is bottom, y2 is top
    :return: bool, True if the object is on the table, False otherwise
    """
    table_y_height = get_table_y_height()
    # Check if the bottom of the object is approximately at the table's height
    return np.isclose(bbox[1], table_y_height, atol=0.1)
# <end-of-utility>

# <utility>
def is_appropriate_container(a: str, b: str) -> bool:
    """
    Check if the container type of object a is appropriate for containing object b.
    :param a: string, name of detected object considered as container
    :param b: string, name of detected object considered to be contained
    :return: bool, True if the container is appropriate, False otherwise
    """
    # For simplicity, let's assume only pots can contain food, and cups can contain small items (not food)
    object_a_category = get_object_category(a)
    object_b_category = get_object_category(b)
    if object_a_category == 'pot' and object_b_category == 'food':
        return True
    if object_a_category == 'cup' and object_b_category != 'food':
        # Assuming cups can only contain small non-food items, not implemented here due to lack of size detail
        return False
    return False
# <end-of-utility>

# Predicates:
# <predicate>
def obj_in_gripper(a) -> bool:
    """
    Description: <<whether object a is held by the gripper>>
    The predicate holds True when the mass in gripper is non-zero, and a significant portion of object a's bounding box intersects with the opened gripper's bounding box
    :param a: string, name of detected object
    :return: bool
    """
    # get in gripper mass
    in_gripper_mass = get_in_gripper_mass()
    if in_gripper_mass <= 0:
        return False

    # get in_gripper xyxy bbox
    in_gripper_xyxy_bbox = get_in_gripper_xy_bbox()

    # get bbox_xyxy of object a
    object_a_xyxy_bbox = get_object_xy_bbox(a)

    # check intersection between object a and the gripper along x axis
    x_overlap = min(object_a_xyxy_bbox[2], in_gripper_xyxy_bbox[2]) - max(object_a_xyxy_bbox[0], in_gripper_xyxy_bbox[0])
    significant_x_overlap = x_overlap > 0 and x_overlap >= 0.25 * (object_a_xyxy_bbox[2] - object_a_xyxy_bbox[0])

    # adjust y-axis overlap check to consider the relative size and position of the object
    y_overlap = min(object_a_xyxy_bbox[3], in_gripper_xyxy_bbox[3]) - max(object_a_xyxy_bbox[1], in_gripper_xyxy_bbox[1])
    significant_y_overlap = y_overlap > 0 and y_overlap >= 0.25 * (object_a_xyxy_bbox[3] - object_a_xyxy_bbox[1])

    return significant_x_overlap and significant_y_overlap
# <end-of-predicate>

# <predicate>
def obj_filled_with_water(a) -> bool:
    """
    Description: <<check whether object a is filled with water or not>>
    The predicate holds True if the detected water amount in object a is greater than 0.
    :param a: string, name of detected object
    :return: bool
    """
    water_amount = get_object_water_amount(a)
    return water_amount > 0
# <end-of-predicate>

# <predicate>
def obj_on_table(a) -> bool:
    """
    Description: <<check whether object a is on the table or not>>
    The predicate holds True if the object's bounding box bottom y coordinate is approximately equal to the table height.
    :param a: string, name of detected object
    :return: bool
    """
    object_a_bbox = get_object_xy_bbox(a)
    return is_bbox_on_table(object_a_bbox)
# <end-of-predicate>

# <predicate>
def obj_is_food(a) -> bool:
    """
    Description: <<check whether object a is food, the predicate holds true if the object is a type of food>>
    :param a: string, name of detected object
    :return: bool
    """
    object_category = get_object_category(a)
    return object_category == 'food'
# <end-of-predicate>

# <predicate>
def obj_is_container(a) -> bool:
    """
    Description: <<check whether object a is a container. The predicate holds true if the object is a type of container such as a cup, pot, or basket>>
    :param a: string, name of detected object
    :return: bool
    """
    object_category = get_object_category(a)
    return object_category in ['cup', 'pot', 'basket']
# <end-of-predicate>

# <predicate>
def obj_graspable(a) -> bool:
    """
    Description: <<check whether object a can be grasped by the gripper, the predicate holds true if the object's size or shape allows it to be grasped by the gripper>>
    :param a: string, name of detected object
    :return: bool
    """
    object_a_size = get_object_xy_size(a)
    gripper_max_open_width = get_gripper_max_open_width()
    # Check if the object's width (x dimension) is less than or equal to the gripper's maximum open width
    return object_a_size[0] <= gripper_max_open_width
# <end-of-predicate>

# <predicate>
def gripper_empty() -> bool:
    """
    Description: <<check whether the gripper is not holding any object. The predicate holds true if the gripper is empty.>>
    :return: bool
    """
    # get in gripper mass
    in_gripper_mass = get_in_gripper_mass()
    return in_gripper_mass <= 0
# <end-of-predicate>

# <predicate>
def obj_can_contain(a: str, b: str) -> bool:
    """
    Description: <<check whether object a can contain object b, considering size, capacity, and the appropriateness of the container type. The predicate holds true if object a has sufficient capacity or size to contain object b and is an appropriate type of container.>>
    :param a: string, name of detected object considered as container
    :param b: string, name of detected object considered to be contained
    :return: bool
    """
    # First, ensure that object a is a container
    if not obj_is_container(a):
        return False

    # Ensure that the container is appropriate for the type of object it is supposed to contain
    if not is_appropriate_container(a, b):
        return False

    # Get the size of both objects
    object_a_size = get_object_xy_size(a)
    object_b_size = get_object_xy_size(b)

    # Assuming capacity is directly proportional to the size of the object for simplicity
    # Check if object a is larger than object b in both dimensions (x and y)
    # This is a simplified check and does not account for actual volume or irregular shapes
    return object_a_size[0] > object_b_size[0] and object_a_size[1] > object_b_size[1]
# <end-of-predicate>

# <predicate>
def obj_inside(a: str, b: str) -> bool:
    """
    Description: <<check whether object a is inside object b or not>>
    The predicate holds True if the entire bounding box of object a is within the bounding box of object b.
    :param a: string, name of detected object considered to be inside
    :param b: string, name of detected object considered to contain a
    :return: bool
    """
    # Get the bounding boxes of both objects
    bbox_a = get_object_xy_bbox(a)
    bbox_b = get_object_xy_bbox(b)

    # Check if bbox_a is inside bbox_b
    # This means all coordinates of bbox_a must be within the corresponding coordinates of bbox_b
    inside_x = bbox_a[0] >= bbox_b[0] and bbox_a[2] <= bbox_b[2]
    inside_y = bbox_a[1] >= bbox_b[1] and bbox_a[3] <= bbox_b[3]

    return inside_x and inside_y
# <end-of-predicate>