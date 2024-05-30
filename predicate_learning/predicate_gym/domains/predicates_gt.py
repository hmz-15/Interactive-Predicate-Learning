# the import are only used to demonstrate the perception API
# from perception_api import (
#     get_detected_object_list,
#     get_object_xy_position,
#     get_object_xy_size,
#     get_object_category,
#     get_object_water_amount,
#     get_gripper_position,
#     get_gripper_open_width,
#     get_in_gripper_mass,
#     get_gripper_y_size,
#     get_gripper_max_open_width,
#     get_table_x_range,
#     get_table_y_height,
# )
# import numpy as np

eps = 0.2

############## Helper Functions ##############


def get_object_xyxy_bbox(a):
    object_a_position = get_object_xy_position(a)
    object_a_size = get_object_xy_size(a)
    return [
        object_a_position[0] - object_a_size[0] / 2,
        object_a_position[1] - object_a_size[1] / 2,
        object_a_position[0] + object_a_size[0] / 2,
        object_a_position[1] + object_a_size[1] / 2,
    ]


def get_gripper_xyxy_bbox():
    gripper_position = get_gripper_position()
    gripper_open_width = get_gripper_open_width()
    gripper_y_size = get_gripper_y_size()
    return [
        gripper_position[0] - gripper_open_width / 2,
        gripper_position[1] - gripper_y_size / 2,
        gripper_position[0] + gripper_open_width / 2,
        gripper_position[1] + gripper_y_size / 2,
    ]


def align_x(bbox_xyxy_1, bbox_xyxy_2):
    if abs(bbox_xyxy_1[0] - bbox_xyxy_2[0]) < eps and abs(bbox_xyxy_1[2] - bbox_xyxy_2[2]) < eps:
        return True
    else:
        return False


def overlap_x(bbox_xyxy_1, bbox_xyxy_2):
    if bbox_xyxy_1[0] - bbox_xyxy_2[2] < eps and bbox_xyxy_1[2] - bbox_xyxy_2[0] > -eps:
        return True
    else:
        return False


def overlap_y(bbox_xyxy_1, bbox_xyxy_2):
    if bbox_xyxy_1[1] - bbox_xyxy_2[3] < eps and bbox_xyxy_1[3] - bbox_xyxy_2[1] > -eps:
        return True
    else:
        return False


def inside_x_range(bbox_xyxy_1, x_range):
    if bbox_xyxy_1[0] - x_range[0] > -eps and bbox_xyxy_1[2] - x_range[1] < eps:
        return True
    else:
        return False


def inside_x(bbox_xyxy_1, bbox_xyxy_2):
    return inside_x_range(bbox_xyxy_1, [bbox_xyxy_2[0], bbox_xyxy_2[2]])


def inside_y(bbox_xyxy_1, bbox_xyxy_2):
    if bbox_xyxy_1[1] - bbox_xyxy_2[1] > -eps and bbox_xyxy_1[3] - bbox_xyxy_2[3] < eps:
        return True


def on_surface(bbox_xyxy, surface_y):
    if abs(bbox_xyxy[1] - surface_y) < eps:
        return True
    else:
        return False


############## Predicate Functions ##############
def is_graspable(a):
    # x width < gripper max open width
    object_a_size = get_object_xy_size(a)
    gripper_max_open_width = get_gripper_max_open_width()
    if object_a_size[0] < gripper_max_open_width:
        return True
    else:
        return False


def is_thin(a):
    # height < 1
    object_a_size = get_object_xy_size(a)
    if object_a_size[1] < 1:
        return True
    else:
        return False


def on_table(a):
    table_x_range = get_table_x_range()
    table_y_height = get_table_y_height()
    object_a_bbox_xyxy = get_object_xyxy_bbox(a)
    if inside_x_range(object_a_bbox_xyxy, table_x_range) and on_surface(
        object_a_bbox_xyxy, table_y_height
    ):
        return True
    else:
        return False


def in_gripper(a):
    griper_xyxy_bbox = get_gripper_xyxy_bbox()
    object_a_xyxy_bbox = get_object_xyxy_bbox(a)
    if (
        align_x(griper_xyxy_bbox, object_a_xyxy_bbox)
        and overlap_y(griper_xyxy_bbox, object_a_xyxy_bbox)
        and get_in_gripper_mass() > eps
    ):
        return True
    else:
        return False


def gripper_empty():
    detected_object_list = get_detected_object_list()
    for detected_object in detected_object_list:
        if in_gripper(detected_object):
            return False
    return True


def on(a, b):
    object_a_xyxy_bbox = get_object_xyxy_bbox(a)
    object_b_xyxy_bbox = get_object_xyxy_bbox(b)
    if overlap_x(object_a_xyxy_bbox, object_b_xyxy_bbox) and on_surface(
        object_a_xyxy_bbox, object_b_xyxy_bbox[3]
    ):
        return True
    else:
        return False


def clear_on_top(a):
    detected_object_list = get_detected_object_list()
    for object in detected_object_list:
        if on(object, a):
            return False
    return True


def is_container(a):
    object_a_category = get_object_category(a)
    if object_a_category in ["cup", "pot", "basket"]:
        return True
    else:
        return False


def has_water(a):
    if get_object_water_amount(a) > eps:
        return True
    else:
        return False


def has_no_water(a):
    return not has_water(a)


def is_food(a):
    object_a_category = get_object_category(a)
    if object_a_category == "food":
        return True
    else:
        return False


def inside_container(a, b):
    if a == b:
        return False

    object_a_xyxy_bbox = get_object_xyxy_bbox(a)
    object_b_xyxy_bbox = get_object_xyxy_bbox(b)
    if (
        inside_x(object_a_xyxy_bbox, object_b_xyxy_bbox)
        and inside_y(object_a_xyxy_bbox, object_b_xyxy_bbox)
        and is_container(b)
    ):
        return True
    else:
        return False


def not_in_container(a):
    detected_object_list = get_detected_object_list()
    for object in detected_object_list:
        if object != a and inside_container(a, object):
            return False
    return True


def can_contain_food(a):
    # large enough
    object_a_size = get_object_xy_size(a)
    if object_a_size[0] >= 10:
        return True
    else:
        return False


def is_plate(a):
    object_a_category = get_object_category(a)
    if object_a_category == "plate":
        return True
    else:
        return False


def is_dinningplace(a):
    object_a_category = get_object_category(a)
    if object_a_category in ["placemat", "plate"]:
        return True
    else:
        return False
