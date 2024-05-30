from typing import *
import numpy as np

from predicate_learning.utils.pddl_util import DEFAULT_TYPE


class PerceptionAPIWrapper:
    def __init__(self, world):
        self._world = world
        self._observation = {}

    def update_observation(self, observation=None):
        if observation is not None:
            self._observation = observation
        else:
            self._observation = self._world.get_observation()

    def get_all_apis(self):
        return {
            "get_detected_object_list": self.get_detected_object_list,
            "check_known_object": self.check_known_object,
            "get_object_xy_position": self.get_object_xy_position,
            "get_object_xy_size": self.get_object_xy_size,
            "get_object_category": self.get_object_category,
            "get_object_water_amount": self.get_object_water_amount,
            "get_table_y_height": self.get_table_y_height,
            "get_table_x_range": self.get_table_x_range,
            "get_gripper_position": self.get_gripper_position,
            "get_gripper_y_size": self.get_gripper_y_size,
            "get_gripper_open_width": self.get_gripper_open_width,
            "get_gripper_max_open_width": self.get_gripper_max_open_width,
            "get_in_gripper_mass": self.get_in_gripper_mass,
        }

    def get_detected_object_list(self) -> List[str]:
        return list(self._observation["objects"].keys())

    def check_known_object(self, a) -> bool:
        return a in self._observation["objects"].keys()

    def get_object_xy_position(self, a) -> np.ndarray:
        if not self.check_known_object(a):
            raise KeyError(f"Object {a} doesn't exist in the scene!")

        object_state = self._observation["objects"][a]
        return object_state["position"]

    def get_object_xy_size(self, a) -> np.ndarray:
        if not self.check_known_object(a):
            raise KeyError(f"Object {a} doesn't exist in the scene!")

        object_state = self._observation["objects"][a]
        return object_state["size"]

    def get_object_category(self, a) -> str:
        if not self.check_known_object(a):
            raise KeyError(f"Object {a} doesn't exist in the scene!")

        object_state = self._observation["objects"][a]
        return object_state["category"]

    def get_object_water_amount(self, a) -> float:
        if not self.check_known_object(a):
            raise KeyError(f"Object {a} doesn't exist in the scene!")

        object_state = self._observation["objects"][a]
        if "liquid_amount" not in object_state:
            return 0.0

        return float(object_state["liquid_amount"])

    def get_table_y_height(self) -> float:
        table_state = self._observation["table"]
        return table_state["table_height"]

    def get_table_x_range(self) -> np.ndarray:
        table_state = self._observation["table"]
        return table_state["table_range"]

    def get_gripper_position(self) -> np.ndarray:
        gripper_state = self._observation["gripper"]
        return gripper_state["position"]

    def get_gripper_y_size(self) -> float:
        gripper_state = self._observation["gripper"]
        return gripper_state["height"]

    def get_gripper_open_width(self) -> float:
        # from 0 to 1
        gripper_state = self._observation["gripper"]
        return gripper_state["open_width"]

    def get_gripper_max_open_width(self) -> float:
        gripper_state = self._observation["gripper"]
        return gripper_state["max_open_width"]

    def get_in_gripper_mass(self) -> float:
        gripper_state = self._observation["gripper"]
        return gripper_state["in_gripper_mass"]


class ActionAPIWrapper:
    def __init__(self, world):
        self._world = world

    def get_all_apis(self):
        return {
            "pick_up": self.pick_up,
            "place_on_table": self.place_on_table,
            "place_first_on_second": self.place_first_on_second,
            "place_first_in_second": self.place_first_in_second,
            "push_plate_on_object": self.push_plate_on_object,
            "get_water_from_faucet": self.get_water_from_faucet,
            "pour_water_from_first_to_second": self.pour_water_from_first_to_second,
        }

    def pick_up(self, a) -> bool:
        # execute action
        success = self._world.apply_action(self._world._actions["pick_up"](DEFAULT_TYPE(a)))
        # reset perception api
        self._world.perception_api_wrapper.update_observation()
        return success

    def place_on_table(self, a) -> bool:
        # execute action
        success = self._world.apply_action(self._world._actions["place_on_table"](DEFAULT_TYPE(a)))
        # reset perception api
        self._world.perception_api_wrapper.update_observation()
        return success

    def place_first_on_second(self, a, b) -> bool:
        # execute action
        success = self._world.apply_action(
            self._world._actions["place_first_on_second"](DEFAULT_TYPE(a), DEFAULT_TYPE(b))
        )
        # reset perception api
        self._world.perception_api_wrapper.update_observation()
        return success

    def place_first_in_second(self, a, b) -> bool:
        # execute action
        success = self._world.apply_action(
            self._world._actions["place_first_in_second"](DEFAULT_TYPE(a), DEFAULT_TYPE(b))
        )
        # reset perception api
        self._world.perception_api_wrapper.update_observation()
        return success

    def push_plate_on_object(self, a, b) -> bool:
        # execute action
        success = self._world.apply_action(
            self._world._actions["push_plate_on_object"](DEFAULT_TYPE(a), DEFAULT_TYPE(b))
        )
        # reset perception api
        self._world.perception_api_wrapper.update_observation()
        return success

    def get_water_from_faucet(self, a) -> bool:
        # execute action
        success = self._world.apply_action(
            self._world._actions["get_water_from_faucet"](DEFAULT_TYPE(a))
        )
        # reset perception api
        self._world.perception_api_wrapper.update_observation()
        return success

    def pour_water_from_first_to_second(self, a, b) -> bool:
        # execute action
        success = self._world.apply_action(
            self._world._actions["pour_water_from_first_to_second"](
                DEFAULT_TYPE(a), DEFAULT_TYPE(b)
            )
        )
        # reset perception api
        self._world.perception_api_wrapper.update_observation()
        return success
