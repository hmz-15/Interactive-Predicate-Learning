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
from action_api import pick_up, place_on_table, place_first_on_second
from plan_utils import parse_objects, execute_subpolicy
{extra_vars_imports}

# Stack tomatocan on shelf.
execute_subpolicy("clear all objects above tomatocan onto table")
execute_subpolicy("clear all objects above shelf onto table")
assert pick_up("tomatocan"), "Failed to pick up tomatocan."
assert place_first_on_second("tomatocan", "shelf"), "Failed to place tomatocan on shelf."
# Put coaster on the table.
execute_subpolicy("clear all objects above coaster onto table")
assert pick_up("coaster"), "Failed to pick up coaster."
assert place_on_table("coaster"), "Failed to place coaster on the table."