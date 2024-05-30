import abc
import itertools
import pygame
import logging
from pathlib import Path
import numpy as np

from predicate_learning.predicate_gym.Kitchen2D import Gripper
from predicate_learning.predicate_gym.kitchen2d_env import Kitchen2DEnv
from predicate_learning.predicate_gym.api_wrapper import PerceptionAPIWrapper, ActionAPIWrapper


from pddlgym.structs import Literal, LiteralConjunction
from pddlgym.parser import PDDLDomainParser

from predicate_learning.utils.registry import Registry
from predicate_learning.utils.pddl_util import (
    PDDLProblem,
    DEFAULT_TYPE,
    lifted_literals,
    select_operator,
)
from predicate_learning.utils.env_util import (
    sample_on_position_help_func,
    get_obj_size,
    get_bounding_box,
    get_obj_pos,
    check_on,
)
from predicate_learning.utils.common_util import EnvFeedback, AgentAction
from predicate_learning.utils.io_util import load_json, dump_json, load_txt
from predicate_learning.utils.llm_util import pddl_literal_to_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

WORLD = Registry("World")


def create_world(world_name, *args, **kwargs):
    return WORLD.get(world_name)(*args, **kwargs)


class BaseWorld(abc.ABC):
    def __init__(self, world_cfg, use_diverse_feedback=False):
        """
        Base world class.
        """
        # environment
        self.env = None
        self.world_cfg = world_cfg

        self.use_diverse_feedback = use_diverse_feedback

        # load domain info
        domains_dir = Path(__file__).resolve().parent / "domains"
        self._all_actions = load_json(domains_dir / "actions.json")
        self._all_predicates = load_json(domains_dir / "predicates.json")
        self._all_domain_desc = load_json(domains_dir / "domain_desc.json")

        # load current pddl domain (ground truth)
        self._world_name = world_cfg.world
        self._pddl_domain = PDDLDomainParser(
            domains_dir / self._world_name / f"{self._world_name}.pddl"
        )

        # actions
        self._actions = {
            action_name: self._pddl_domain.predicates[action_name]
            for action_name in self._pddl_domain.actions
        }

        # entities
        self._entities = {}

        # goal lits & pddl problem
        self._goal_text = None
        self._goal_lits = None
        self._pddl_problem = None

        # mode: train / test
        self.mode = "train"

        # perception / action APIs
        self.perception_api_wrapper = PerceptionAPIWrapper(self)
        self.action_api_wrapper = ActionAPIWrapper(self)

        # ground truth predicate vars
        gt_predicates_str = load_txt(domains_dir / "predicates_gt.py")
        global_scope = self.perception_api_wrapper.get_all_apis()
        exec(gt_predicates_str, global_scope)
        self._gt_predicate_vars = {
            pred_name: global_scope[pred_name]
            for pred_name in self._pddl_domain.predicates.keys()
            if pred_name not in self._actions
        }

    def __str__(self):
        return self._world_name

    @property
    def actions(self):
        return sorted(self._actions.values())

    @property
    def entities(self):
        return sorted(self._entities.values())

    @property
    def goal_text(self):
        return self._goal_text

    @property
    def goal_lits(self):
        return self._goal_lits.literals

    @property
    def train_instances_dict(self):
        return self._train_instances_dict

    @property
    def test_instances_dict(self):
        return self._test_instances_dict

    @property
    def pddl_domain(self):
        return self._pddl_domain

    @property
    def pddl_problem(self):
        return self._pddl_problem

    def reset_env(self, instance_config, render):
        """
        Reset the environment given initial configuration, and goal.
        """
        # destroy env
        if self.env is not None:
            self.destroy_env()

        # clear entities & goal
        self._entities = {}
        self._goal_lits = None
        self._pddl_problem = None

        # reset environment
        self.env = Kitchen2DEnv(render=render)
        for name, pos in instance_config["objects"].items():
            self.env.add_object(name, pos)
        self.gripper = Gripper(self.env, (0, 25), 0)

        # update entities (for pddl stuff)
        for obj_name in self.env.objects:
            self._entities[obj_name] = DEFAULT_TYPE(obj_name)

        # goal lits; only positive goals are allowed
        raw_goal_lits = instance_config["goal_lits_gt"]
        raw_goal_text = instance_config["goal_text"]

        goal_lits_list = []
        predicates = self._pddl_domain.predicates
        for raw_lit in raw_goal_lits:
            lit = predicates[raw_lit[0]](*raw_lit[1:])
            goal_lits_list.append(lit)

        self._goal_lits = LiteralConjunction(goal_lits_list)
        self._goal_text = raw_goal_text

        # run simulation for a few iterations to initialize
        for _ in range(15):
            self.env.step()

        # pddl problem
        initial_state = self.parse_pddl_literals()
        self._pddl_problem = PDDLProblem(
            self.pddl_domain,
            self.entities,
            initial_state=initial_state,
            goal_lits=self._goal_lits,
        )

        # initial state should not satisfy goal
        if self.check_goal(initial_state):
            return False, None

        # prepare initial env feedback
        env_feedback = EnvFeedback(observation=self.get_observation())

        # reset perception api
        self.perception_api_wrapper.update_observation()

        return True, env_feedback

    def step(self, agent_action: AgentAction):
        """
        Apply an action with a gripper.

        Args:
            agent_action: AgentAction object consisting of action literal and query

        Returns:
            env_feedback: EnvFeedback object
        """
        # TODO: remove agent query as it's not necessary
        # parse literals before action
        lits_before, infeasible_lits = self.parse_pddl_literals(return_infeasible_lits=True)

        # apply action
        success = True
        action, query = agent_action.action, agent_action.query
        if action is not None:
            assert (
                action.predicate.name in self._actions.keys()
            ), f"Unknown action {action.predicate.name}"

            # in train mode: detect failure with gt preconditions (avoid execution failures that are hard to recover from)
            if self.mode == "train":
                op, _ = select_operator(lits_before, action, self._pddl_domain.operators)
                if op is None:
                    success = False
                else:
                    # apply action
                    success = self.apply_action(action)
            else:
                # in test mode: apply action directly
                success = self.apply_action(action)
        else:
            logger.info("Action is None!")

        # get observation
        observation = self.get_observation()

        # parse literals after action
        lits_after = self.parse_pddl_literals()

        # check goal
        goal_achieved = self.check_goal(lits_after)

        # generate failure explanation
        failure_explanation = ""
        if query.failure_explain and not success:
            failure_explanation = self.explain_failure(lits_before, infeasible_lits, action)

        # generate non-goal explanation
        non_goal_explain = ""
        if query.non_goal_explain:
            non_goal_explain = self.explain_non_goal(lits_before)

        # prepare env feedback
        env_feedback = EnvFeedback(
            observation=observation,
            success=success,
            goal_achieved=goal_achieved,
            failure_explain=failure_explanation,
            non_goal_explain=non_goal_explain,
        )

        if goal_achieved:
            logger.info("Episode ends!")
            self.destroy_env()

        return env_feedback

    def apply_action(self, action):
        maxspeed = 5.0
        # only allow actions in pddl domain
        assert action.predicate.name in self._actions.keys(), f"Unknown action {action}"

        # pick
        if action.predicate.name == "pick_up":
            # can't pick if has attached object
            if self.gripper.attached:
                return False

            obj_to_pick = self.get_box2d_obj(action.variables[0].name)

            success = self.gripper.grasp(obj_to_pick, 0.5, maxspeed=maxspeed)
            if success:
                self.gripper.find_path(
                    (self.gripper.position[0], self.gripper.position[1] + 4),
                    0,
                    maxspeed=maxspeed,
                )

            return success

        # place
        elif action.predicate.name in [
            "place_on_table",
            "place_first_on_second",
            "place_first_in_second",
        ]:
            # gripper must attached with an obj
            if not self.gripper.attached:
                return False

            # the first var must be the attached obj
            if self.get_box2d_obj(action.variables[0].name) != self.gripper.attached_obj:
                return False

            # can't place on the attached object
            attached_obj = self.gripper.attached_obj
            if len(action.variables) >= 2:
                obj_to_place = self.get_box2d_obj(action.variables[1].name)
                if attached_obj == obj_to_place:
                    return False

            # sample position
            table_box = (
                np.array([self.env.table_info["table_range"][0], -1.0]),
                np.array([self.env.table_info["table_range"][1], 0.0]),
            )
            all_boxes = {name: get_bounding_box(obj) for name, obj in self.env.objects.items()}
            all_boxes["table"] = table_box
            if action.predicate.name == "place_on_table":
                receptacle_name = "table"
            else:
                receptacle_name = action.variables[1].name

            # sample position
            positions = sample_on_position_help_func(
                get_obj_size(attached_obj), receptacle_name, all_boxes
            )
            if positions is None:
                return False

            for pos in positions:
                # adjust position
                if action.predicate.name == "place_first_in_second":
                    pos = (pos[0], pos[1] - 1)

                success = self.gripper.place(pos, 0, maxspeed=maxspeed)
                if success:
                    self.gripper.find_path(
                        (self.gripper.position[0], self.gripper.position[1] + 4),
                        0,
                        maxspeed=maxspeed,
                    )
                    return True
            return False

        # push
        elif action.predicate.name in ["push_first_on_second", "push_plate_on_object"]:
            # can't push if has attached object
            if self.gripper.attached:
                return False

            # can't push one object to itself
            if action.variables[0].name == action.variables[1].name:
                return False

            # get first object
            obj_to_push = self.get_box2d_obj(action.variables[0].name)

            # calculate goal
            goal_obj = self.get_box2d_obj(action.variables[1].name)
            goal_pos = get_obj_pos(goal_obj)
            goal_pos_x = goal_pos[0]

            self.gripper.push(
                obj_to_push, (0.5, 0.0), goal_pos_x, reach_maxspeed=maxspeed, maxspeed=0.6
            )

            dpos = self.gripper.position
            dpos[1] += 3
            self.gripper.apply_lowlevel_control(dpos, 0, maxspeed=maxspeed)

            # check whether the object is pushed onto the other object
            success = check_on(obj_to_push, goal_obj)
            return success

        # get water
        elif action.predicate.name == "get_water_from_faucet":
            # gripper must attached with an obj
            if not self.gripper.attached:
                return False

            # # must be cup
            # attached_obj = self.gripper.attached_obj
            # if attached_obj.userData != "cup":
            #     return False

            # arg must match the attached obj
            attached_obj = self.gripper.attached_obj
            obj_get_water = self.get_box2d_obj(action.variables[0].name)
            if attached_obj != obj_get_water:
                return False

            success = self.gripper.get_liquid_from_faucet(6, maxspeed=maxspeed)
            return success

        # pour into
        elif action.predicate.name == "pour_water_from_first_to_second":
            # gripper must attached with an obj
            if not self.gripper.attached:
                return False

            # must be cup
            attached_obj = self.gripper.attached_obj
            if attached_obj.userData != "cup":
                return False

            # arg must match the attached obj
            obj_pour_from = self.get_box2d_obj(action.variables[0].name)
            if attached_obj != obj_pour_from:
                return False

            # cup must have water
            incupparticles, stopped = self.gripper.compute_post_grasp_mass()
            n_particles = len(incupparticles)
            if n_particles < 5:
                return False

            # the other object must be container
            obj_to_pour = self.get_box2d_obj(action.variables[1].name)
            if obj_to_pour.userData not in ["cup", "pot"]:
                return False

            success = self.gripper.pour(
                obj_to_pour, (3, get_obj_size(obj_to_pour)[1] / 2 + 10), 1.8, maxspeed=maxspeed
            )[0]
            return success

        else:
            raise NotImplementedError(f"Please implement action {action.predicate.name}")

    def get_observation(self):
        """
        Return observation of entity states of different types
        """
        obj_states = self.env.obj_states
        gripper_state = self.gripper.gripper_state
        table_info = self.env.table_info

        observation = {
            "table": table_info,
            "gripper": gripper_state,
            "objects": obj_states,
            "entities": list(obj_states.keys()),
        }

        if self.env.gui_world is not None:
            # global image (H, W, 3)
            image = pygame.surfarray.array3d(self.env.gui_world.screen).swapaxes(0, 1)
            observation["image"] = image

        return observation

    def set_mode(self, is_train):
        self.mode = "train" if is_train else "test"

    def get_box2d_obj(self, name):
        if name not in self.env.objects:
            import pdb

            pdb.set_trace()
        assert name in self.env.objects, f"Unknown object {name}!"
        return self.env.objects[name]

    def create_instances(
        self,
        task_spec,
        num_instances,
        save_to_file=False,
        instance_file=None,
        overwrite=False,
    ):
        """
        Create task instances based on inputs.
        """
        # if already exists, load from file
        if instance_file is not None and instance_file.exists() and not overwrite:
            instances_json = load_json(instance_file)
            logger.info(f"Load from existing file {instance_file}.")

        else:
            # generate initial states & goal lits per task instance
            instances_json = {}
            create_instance_iter = 0
            while create_instance_iter < num_instances:
                # create one instance
                instance_config = self._create_single_instance(
                    **task_spec, instance_id=create_instance_iter
                )
                instances_json[len(instances_json) + 1] = instance_config
                create_instance_iter += 1

            if save_to_file:
                assert (
                    instance_file is not None
                ), "instance_file must be specified when save_to_file is True"

                dump_json(instances_json, instance_file)

        return instances_json

    def check_goal(self, literals=set()):
        # if goal not specified, return False
        if self._goal_lits is None:
            return False

        # if literals not specified, parse them
        if len(literals) == 0:
            literals = self.parse_pddl_literals()

        if isinstance(self._goal_lits, list):
            goal_lits = self._goal_lits
        elif isinstance(self._goal_lits, Literal):
            goal_lits = [self._goal_lits]
        elif isinstance(self._goal_lits, LiteralConjunction):
            goal_lits = self._goal_lits.literals

        goal_achieved = set(goal_lits).issubset(literals)
        return goal_achieved

    def parse_pddl_literals(self, return_infeasible_lits: bool = False):
        """
        Parse current continuous state into a list of ground predicates.
        """
        pddl_lits = set()
        infeasible_lits = set()
        # pddl_lits_text = ""

        predicates = self._pddl_domain.predicates
        entities = self._entities

        # update observation of perception API
        self.perception_api_wrapper.update_observation()

        # parse literals
        for pred_name, predicate in predicates.items():
            # remove action predicates
            if pred_name in self._actions.keys():
                continue

            # parse all literals of this predicate
            arity = predicate.arity

            python_func = self._gt_predicate_vars[predicate.name]
            for combine in itertools.permutations(list(entities.keys()), arity):
                value = python_func(*combine)

                # positive only
                pddl_lit = predicate(*[entities[var] for var in combine])

                if value:
                    pddl_lits.add(pddl_lit)
                elif value is None:
                    infeasible_lits.add(pddl_lit)

        if return_infeasible_lits:
            return pddl_lits, infeasible_lits
        else:
            return pddl_lits

    def explain_failure(self, lits_before, infeasible_lits, action):
        # check precondition violation
        operators = self._pddl_domain.operators

        # collect all operators that match this action, compute ground preconds
        valid_op_preconds = []
        for operator in operators.values():
            precond_lits = operator.preconds.literals
            for lit in precond_lits:
                if lit.predicate == action.predicate:
                    assign = {
                        var_lift: var_ground
                        for var_lift, var_ground in zip(lit.variables, action.variables)
                    }

                    precond_set = set(precond_lits)
                    precond_set.remove(lit)
                    # compute ground preconds
                    ground_preconds = lifted_literals(
                        precond_set, assignment=assign, assign_unknown_vars=False
                    )
                    valid_op_preconds.append(ground_preconds)
                    break

        # get the intersection of these preconditions
        intersec = valid_op_preconds[0]
        for preconds in valid_op_preconds[1:]:
            intersec = intersec.intersection(preconds)

        # check violation
        failure_list = []
        for precond in intersec:
            if precond not in lits_before and precond not in infeasible_lits:
                if self.use_diverse_feedback:
                    # sample one feedback
                    index = np.random.choice(
                        range(
                            len(
                                self._all_predicates[precond.predicate.name]["precondition_explain"]
                            )
                        )
                    )
                else:
                    index = 0

                precond_explain = self._all_predicates[precond.predicate.name][
                    "precondition_explain"
                ][index]
                for i, var in enumerate(precond.variables):
                    precond_explain = precond_explain.replace(f"<arg{i+1}>", var.name)

                failure_list.append(precond_explain)
                # todo: only one
                break

        if len(failure_list) == 0:
            logger.warning(
                f"Weired! No precondition violation found! Lits: {lits_before} Action: {action}"
            )
            return ""

        # finalize failure explanation
        explanation = (
            f"You fail to execute {pddl_literal_to_text(action)} because {', '.join(failure_list)}."
        )
        return explanation

    def explain_non_goal(self, lits_before):
        assert self._goal_lits is not None, "Goal is not specified!"

        non_goal_list = []
        for goal_lit in self._goal_lits.literals:
            if goal_lit not in lits_before:
                if self.use_diverse_feedback:
                    # sample one feedback
                    index = np.random.choice(
                        range(
                            len(self._all_predicates[goal_lit.predicate.name]["non_goal_explain"])
                        )
                    )
                else:
                    index = 0

                non_goal_explain = self._all_predicates[goal_lit.predicate.name][
                    "non_goal_explain"
                ][index]
                for i, var in enumerate(goal_lit.variables):
                    non_goal_explain = non_goal_explain.replace(f"<arg{i+1}>", var.name)

                non_goal_list.append(non_goal_explain)
                # todo: only one
                break

        if len(non_goal_list) == 0:
            logger.warning(
                f"Weired! No goal violation found! Goal: {self._goal_lits} Lits: {lits_before}"
            )
            return ""

        # finalize non goal explanation
        explanation = f"You haven't reach the goal because {', '.join(non_goal_list)}."
        return explanation

    def destroy_env(self):
        if self.env is not None:
            self.env.destroy()
            self.env = None

    @abc.abstractmethod
    def _create_single_instance(self):
        """
        Create single problem instances.
        """
        raise NotImplementedError("Override me!")
