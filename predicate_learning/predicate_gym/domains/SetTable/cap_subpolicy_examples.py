import numpy as np
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
from action_api import pick_up, place_on_table, place_first_on_second, push_plate_on_object
from plan_utils import parse_objects
{extra_vars_imports}

# clear all objects above block onto table
obj_stack = ["block"]
while not check_obj_clear(obj_stack[0]):
    objects = parse_objects(f"object on {obj_stack[0]}")
    assert len(objects) == 1, f"Expected 1 object on {obj_stack[0]}, but got {len(objects)}."
    obj_stack = objects + obj_stack
for obj in obj_stack[:-1]:
    assert pick_up(obj), f"Failed to pick up {obj}."
    assert place_on_table(obj), f"Failed to place {obj} on the table."