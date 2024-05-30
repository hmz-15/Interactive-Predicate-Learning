import numpy as np
from omegaconf import OmegaConf
import random

from predicate_learning.predicate_gym.base import BaseWorld, WORLD
from predicate_learning.predicate_gym.kitchen_constants import *
from predicate_learning.utils.env_util import sample_on_position_help_func


@WORLD.register()
class SetTable(BaseWorld):
    def _create_single_instance(self, known_objects, goals, instance_id):
        # initialize object positions
        objects_pos = {}

        # first place placemat & plate
        assert "placemat" in known_objects and "plate" in known_objects
        placemat_info = STATIC_POLYGON_OBJS["placemat"]
        plate_info = DYNAMIC_POLYGON_OBJS["plate"]
        placemat_pos = (-5, 0)

        # sample plate position (left or right)
        sign = np.random.choice([-1, 1])
        plate_pos = (sign * (placemat_info["w"] / 2 + plate_info["w"] / 2 + 2) + placemat_pos[0], 0)
        objects_pos["placemat"] = placemat_pos
        objects_pos["plate"] = plate_pos

        all_obj_boxes = {
            "placemat": (
                np.array([placemat_pos[0] - placemat_info["w"] / 2, placemat_pos[1]]),
                np.array(
                    [
                        placemat_pos[0] + placemat_info["w"] / 2,
                        placemat_pos[1] + placemat_info["h"],
                    ]
                ),
            ),
            "plate": (
                np.array([plate_pos[0] - plate_info["w"] / 2, plate_pos[1]]),
                np.array([plate_pos[0] + plate_info["w"] / 2, plate_pos[1] + plate_info["h"]]),
            ),
            "table": (np.array([SINK_POS_X - LEFT_TABLE_WIDTH, -1.0]), np.array([SINK_POS_X, 0.0])),
        }

        # select goal name, then sample items
        goals = OmegaConf.to_container(goals)  # convert to list
        instance_id = instance_id % len(goals)
        goal_name = list(goals.keys())[instance_id]
        goals = goals[goal_name]
        goal_idx = np.random.choice(len(goals))
        goal_lits_raw = goals[goal_idx]

        on_placemat_obj = None  # object on placemat initailly
        on_plate_obj = None
        if "primitive" in goal_name:
            # primitive goal
            goal_lit = goal_lits_raw[0]
            if goal_lit[0] == "on":
                goal_text = f"Stack {goal_lit[1]} on {goal_lit[2]}."
                if goal_lit[2] != "placemat":
                    on_placemat_obj = goal_lit[1]
                if goal_lit[1] == "plate":
                    if random.random() < 0.5:
                        on_placemat_obj = "bread"

            elif goal_lit[0] == "on_table":
                goal_text = f"Put {goal_lit[1]} on the table."
                on_plate_obj = goal_lit[1]

            else:
                raise Exception(f"Goal not allowed {goal_lit}")

        elif "full" in goal_name:
            # full goal
            goal_text_list = [f"{goal_lit[1]} on {goal_lit[2]}" for goal_lit in goal_lits_raw]
            goal_text = f"Set a breakfast table with {', '.join(goal_text_list)}."

            candidates = set(known_objects) - set(["placemat", "plate"])
            on_placemat_obj = np.random.choice(list(candidates), size=1)[0]

        # small objects on table
        for name in known_objects:
            if name in ["placemat", "plate"]:
                continue

            # sample position
            if name in BLOCK_LIKE_OBJS:
                obj_info = BLOCK_LIKE_OBJS[name]
            elif name in CONTAINER_OBJS:
                obj_info = CONTAINER_OBJS[name]
            else:
                raise Exception(f"Object not allowed {name}")

            obj_size = np.array([obj_info["w"], obj_info["h"]])

            if name == on_placemat_obj:
                pos_list = sample_on_position_help_func(
                    obj_size, "placemat", all_obj_boxes, num_sample=1
                )
            elif name == on_plate_obj:
                pos_list = sample_on_position_help_func(
                    obj_size, "plate", all_obj_boxes, num_sample=1
                )
            else:
                pos_list = sample_on_position_help_func(
                    obj_size, "table", all_obj_boxes, num_sample=1
                )
            pos = pos_list[0]

            # save & update all_obj_boxes
            objects_pos[name] = pos
            all_obj_boxes[name] = (
                np.array([pos[0] - obj_size[0] / 2, pos[1]]),
                np.array([pos[0] + obj_size[0] / 2, pos[1] + obj_size[1]]),
            )

        instance_config = {
            "objects": objects_pos,
            "goal_text": goal_text,
            "goal_lits_gt": goal_lits_raw,
        }
        return instance_config
