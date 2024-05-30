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
from plan_utils import parse_objects, execute_subpolicy
{extra_vars_imports}

# Stack block on mat.
execute_subpolicy("clear all objects above block onto table")
execute_subpolicy("clear all objects above mat onto table")
if check_obj_is_graspable("block"):
    assert pick_up("block"), f"Failed to pick up block."
    assert place_first_on_second("block", "mat"), f"Failed to place block on mat."
else:
    assert push_plate_on_object("block", "mat"), "Failed to push block on mat."
# Put block on the table.
execute_subpolicy("clear all objects above bread onto table")
assert pick_up("bread"), "Failed to pick up bread."
assert place_on_table("bread"), "Failed to place bread on the table."