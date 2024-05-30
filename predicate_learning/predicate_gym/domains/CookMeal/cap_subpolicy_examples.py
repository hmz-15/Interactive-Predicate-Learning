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
from plan_utils import parse_objects
{extra_vars_imports}

# pour water into bucket
if check_obj_is_graspable("bucket"):
    print("Bucket is graspable, directly get water from faucet.")
    assert pick_up("bucket"), "Failed to pick up bucket."
    assert get_water_from_faucet("bucket"), "Failed to get water from faucet."
    assert place_on_table("bucket"), "Failed to place bucket on the table."
else:
    print("Bucket is not graspable, need to use a cup.")
    all_cups = parse_objects("all cups other than object 'bucket'")
    assert len(all_cups) > 0, "No free cup detected."
    any_cup = all_cups[0]
    assert pick_up(any_cup), f"Failed to pick up {any_cup}."
    assert get_water_from_faucet(any_cup), f"Failed to get water from faucet with {any_cup}."
    assert pour_water_from_first_to_second(any_cup, "bucket"), f"Failed to pour water from {any_cup} to bucket."
    assert place_on_table(any_cup), f"Failed to place {any_cup} on the table."