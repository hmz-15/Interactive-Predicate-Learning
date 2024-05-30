import numpy as np
from omegaconf import OmegaConf

from predicate_learning.predicate_gym.base import BaseWorld, WORLD
from predicate_learning.predicate_gym.kitchen_constants import *
from predicate_learning.utils.env_util import sample_on_position_help_func


@WORLD.register()
class StoreObjects(BaseWorld):
    def _create_single_instance(self, known_objects, goals, instance_id):
        # initialize object positions
        objects_pos = {}

        # first place shelf
        assert "shelf" in known_objects
        shelf_info = BLOCK_LIKE_OBJS["shelf"]
        shelf_pos = (-5, 0)
        objects_pos["shelf"] = shelf_pos

        all_obj_boxes = {
            "shelf": (
                np.array([shelf_pos[0] - shelf_info["w"] / 2, shelf_pos[1]]),
                np.array([shelf_pos[0] + shelf_info["w"] / 2, shelf_pos[1] + shelf_info["h"]]),
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

        initial_on_shelf = None
        initial_on_table = None
        if "primitive" in goal_name:
            # primitive goal
            goal_lit = goal_lits_raw[0]
            if goal_lit[0] == "on":
                goal_text = f"Stack {goal_lit[1]} on {goal_lit[2]}."
                initial_on_table = goal_lit[1]  # initialize it on table
            elif goal_lit[0] == "on_table":
                goal_text = f"Put {goal_lit[1]} on the table."
                initial_on_shelf = goal_lit[1]  # initialize it on shelf
            else:
                raise Exception(f"Goal not allowed {goal_lit}")

        elif "full" in goal_name:
            # full goal
            goal_text_list = [f"{goal_lit[1]} on {goal_lit[2]}" for goal_lit in goal_lits_raw]
            goal_text = f"Store objects on shelf following the order: {', '.join(goal_text_list)}."

            initial_on_table = goal_lits_raw[0][1]  # initialize it on table

        # put object on shelf / table
        if initial_on_shelf is not None:
            obj_info = BLOCK_LIKE_OBJS[initial_on_shelf]
            obj_size = np.array([obj_info["w"], obj_info["h"]])

            # sample on shelf
            pos_list = sample_on_position_help_func(obj_size, "shelf", all_obj_boxes, num_sample=1)
            pos = pos_list[0]

            # save & update all_obj_boxes
            objects_pos[initial_on_shelf] = pos
            all_obj_boxes[initial_on_shelf] = (
                np.array([pos[0] - obj_size[0] / 2, pos[1]]),
                np.array([pos[0] + obj_size[0] / 2, pos[1] + obj_size[1]]),
            )

        on_table_list = (
            ["coaster"] if "coaster" in known_objects and initial_on_shelf != "coaster" else []
        )
        if initial_on_table is not None:
            on_table_list.append(initial_on_table)

        for obj in on_table_list:
            obj_info = BLOCK_LIKE_OBJS[obj]
            obj_size = np.array([obj_info["w"], obj_info["h"]])

            # sample on table
            pos_list = sample_on_position_help_func(obj_size, "table", all_obj_boxes, num_sample=1)
            pos = pos_list[0]

            # save & update all_obj_boxes
            objects_pos[obj] = pos
            all_obj_boxes[obj] = (
                np.array([pos[0] - obj_size[0] / 2, pos[1]]),
                np.array([pos[0] + obj_size[0] / 2, pos[1] + obj_size[1]]),
            )

        # other objects on random receptacle
        available_receptacles = set(all_obj_boxes.keys())
        if initial_on_shelf is not None:
            available_receptacles.remove("shelf")
        for name in known_objects:
            if name in ["shelf", initial_on_shelf, initial_on_table, "coaster"]:
                continue

            # sample receptacle
            receptacle = np.random.choice(list(available_receptacles))
            # sample position
            obj_info = BLOCK_LIKE_OBJS[name]
            obj_size = np.array([obj_info["w"], obj_info["h"]])
            pos_list = sample_on_position_help_func(
                obj_size, receptacle, all_obj_boxes, num_sample=1
            )
            pos = pos_list[0]

            # save & update all_obj_boxes
            objects_pos[name] = pos
            all_obj_boxes[name] = (
                np.array([pos[0] - obj_size[0] / 2, pos[1]]),
                np.array([pos[0] + obj_size[0] / 2, pos[1] + obj_size[1]]),
            )
            available_receptacles.remove(receptacle)
            available_receptacles.add(name)

        instance_config = {
            "objects": objects_pos,
            "goal_text": goal_text,
            "goal_lits_gt": goal_lits_raw,
        }
        return instance_config
