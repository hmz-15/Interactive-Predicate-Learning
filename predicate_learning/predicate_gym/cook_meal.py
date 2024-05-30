import numpy as np
from omegaconf import OmegaConf

from predicate_learning.predicate_gym.base import BaseWorld, WORLD
from predicate_learning.predicate_gym.kitchen_constants import *
from predicate_learning.utils.env_util import sample_on_position_help_func


@WORLD.register()
class CookMeal(BaseWorld):
    def _create_single_instance(self, known_objects, goals, instance_id):
        # initialize object positions
        objects_pos = {}

        # first place pot
        assert "pot" in known_objects
        pot_info = CONTAINER_OBJS["pot"]
        pot_pos = (-5, 0)
        objects_pos["pot"] = pot_pos

        # small objects on table
        all_obj_boxes = {
            "pot": (
                np.array([pot_pos[0] - pot_info["w"] / 2, pot_pos[1]]),
                np.array([pot_pos[0] + pot_info["w"] / 2, pot_pos[1] + pot_info["h"]]),
            ),
            "table": (np.array([SINK_POS_X - LEFT_TABLE_WIDTH, -1.0]), np.array([SINK_POS_X, 0.0])),
        }
        for name in known_objects:
            if name in ["pot"]:
                continue

            # sample position
            if name in BLOCK_LIKE_OBJS:
                obj_info = BLOCK_LIKE_OBJS[name]
            elif name in CONTAINER_OBJS:
                obj_info = CONTAINER_OBJS[name]
            else:
                raise Exception(f"Object not allowed {name}")

            obj_size = np.array([obj_info["w"], obj_info["h"]])
            pos = sample_on_position_help_func(obj_size, "table", all_obj_boxes, num_sample=1)[0]

            # save & update all_obj_boxes
            objects_pos[name] = pos
            all_obj_boxes[name] = (
                np.array([pos[0] - obj_size[0] / 2, pos[1]]),
                np.array([pos[0] + obj_size[0] / 2, pos[1] + obj_size[1]]),
            )

        # select goal name, then sample items
        goals = OmegaConf.to_container(goals)  # convert to list
        instance_id = instance_id % len(goals)
        goal_name = list(goals.keys())[instance_id]
        goals = goals[goal_name]
        goal_idx = np.random.choice(len(goals))
        goal_lits_raw = goals[goal_idx]

        if "primitive" in goal_name:
            # primitive goal
            goal_lit = goal_lits_raw[0]
            if goal_lit[0] == "inside_container":
                goal_text = f"Put {goal_lit[1]} in {goal_lit[2]}."
            elif goal_lit[0] == "has_water" or goal_lit[0] == "on_table":
                goal_text = f"Pour water into {goal_lit[1]} on the table."
            else:
                raise NotImplementedError

        elif "full" in goal_name:
            # full goal
            in_pot_objects = []
            filled_cups = []
            for goal_lit in goal_lits_raw:
                if goal_lit[0] == "has_water":
                    if goal_lit[1] == "pot":
                        continue
                    filled_cups.append(goal_lit[1])
                elif goal_lit[0] == "inside_container":
                    assert goal_lit[2] == "pot"
                    in_pot_objects.append(goal_lit[1])

            goal_text = f"Pour water and put {', '.join(in_pot_objects)} in pot"
            if len(filled_cups) > 0:
                goal_text += f", and pour water into {', '.join(filled_cups)} on the table"
            goal_text += "."

        instance_config = {
            "objects": objects_pos,
            "goal_text": goal_text,
            "goal_lits_gt": goal_lits_raw,
        }
        return instance_config
