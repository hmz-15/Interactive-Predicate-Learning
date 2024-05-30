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
from action_api import pick_up, place_on_table, place_first_in_second, get_water_from_faucet, pour_water_from_first_to_second
from plan_utils import parse_objects, execute_subpolicy
{extra_vars_imports}

# Pour water into bucket on the table.
execute_subpolicy("pour water into bucket")
# Put carrot in pot.
assert not check_obj_in_any_container("carrot"), "Carrot is already in a container."
assert pick_up("carrot"), "Failed to pick up carrot."
assert place_first_in_second("carrot", "pot"), "Failed to place carrot in pot."