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
{extra_vars_imports}

# object in gripper
all_objects = get_detected_object_list()
ret_objects = [obj for obj in all_objects if check_obj_in_gripper(obj)]
assert len(ret_objects) <= 1, "More than one object in gripper."
# objects on table
all_objects = get_detected_object_list()
ret_objects = [obj for obj in all_objects if check_obj_on_table(obj)]
# object on block
all_objects = get_detected_object_list()
ret_objects = [obj for obj in all_objects if obj != "block" and check_obj_on_obj(obj, "block")]
