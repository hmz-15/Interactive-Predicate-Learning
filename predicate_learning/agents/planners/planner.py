import random
from pathlib import Path
from termcolor import colored
import numpy as np
import logging
import abc
import copy
import json
from typing import List, Dict, Any, Set, Tuple, Optional

from predicate_learning.utils.io_util import load_txt
from predicate_learning.utils.pddl_util import select_operator, DEFAULT_TYPE
from predicate_learning.utils.common_util import Trajectory
from predicate_learning.utils.registry import Registry

from predicate_learning.utils.llm_util import LLMBase
from predicate_learning.utils.llm_util import (
    create_literal_from_raw,
    pddl_literal_to_text,
    textualize_observation,
)

from pddlgym.spaces import LiteralSpace
from pddlgym.core import State
from pddlgym.parser import Operator, PDDLDomain
from pddlgym.structs import LiteralConjunction, Predicate, Literal

try:
    from pddlgym_planners.fd import FD
    from pddlgym_planners.planner import PlanningFailure, PlanningTimeout
except ModuleNotFoundError:
    raise Exception(
        "To run this demo file, install the "
        + "PDDLGym Planners repository (https://github.com/ronuchit/pddlgym_planners)"
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


PLANNER = Registry("Planner")


def create_planner(planner_name: str, *args, **kwargs):
    return PLANNER.get(planner_name)(*args, **kwargs)


class _BasePlanner(object):
    def __init__(self, action_predicates: List[Predicate], *args, **kwargs):
        self._action_predicates = action_predicates
        self._action_space = LiteralSpace(action_predicates)
        self._pddl_known = False

    def __str__(self):
        return self.__class__.__name__

    def sample_random_action(self, observation: Dict[str, Any], *args, **kwargs):
        entities = [DEFAULT_TYPE(var_name) for var_name in observation["entities"]]
        state = State(None, frozenset(entities), None)
        return self._action_space.sample(state)

    @abc.abstractmethod
    def plan_actions(self, observation, *args, **kwargs):
        raise NotImplementedError("Override me!")


@PLANNER.register()
class RandomPlanner(_BasePlanner):
    def plan_actions(self, observation: Dict[str, Any], *args, **kwargs):
        action = self.sample_random_action(observation)
        return [action]


@PLANNER.register()
class RandomFeasiblePlanner(_BasePlanner):
    def __init__(self, action_predicates: List[Predicate], *args, **kwargs):
        super().__init__(action_predicates)
        self._predicates = None  # Dict[str, Predicate]
        self._operators = None  # Dict[str, Operator]

    def _sample_actions(
        self,
        literals: Set[Literal],
        state: State,
        success_action_execution: Optional[Dict[str, int]] = None,
        feasible_only: bool = True,
        infeasible_weight: float = 0.01,
    ):
        all_ground_actions = self._action_space.all_ground_literals(state)
        all_valid_actions = set()
        all_invalid_actions = set()

        for action in all_ground_actions:
            op, _ = select_operator(
                literals,
                action,
                self._operators,
                inference_mode="csp",
                require_unique_assignment=False,
            )
            if op is not None:
                all_valid_actions.add(action)
            else:
                all_invalid_actions.add(action)

        all_valid_actions = list(all_valid_actions)
        all_invalid_actions = list(all_invalid_actions)
        all_actions = all_valid_actions + all_invalid_actions
        if feasible_only:
            probs = [1.0 for _ in all_valid_actions] + [0.0 for _ in all_invalid_actions]
        else:
            probs = [1.0 for _ in all_valid_actions] + [
                infeasible_weight for _ in all_invalid_actions
            ]
            print(probs)

        probs = np.array(probs)
        # if success_action_execution is not None:
        #     for idx, action in enumerate(all_actions):
        #         probs[idx] *= np.exp(-success_action_execution[action.predicate.name] / 10)

        # normalize
        probs = probs / np.sum(probs) if np.sum(probs) > 0 else probs
        logger.debug(f"{dict(zip(all_actions, probs))}")

        if len(all_valid_actions) > 0 or not feasible_only:
            action = np.random.choice(all_actions, size=1, p=probs)[0]
            # print prob of action
            logger.debug(f"Sampled prob: {probs[all_actions.index(action)]}")
            logger.debug(f"Max prob: {np.max(probs)}")

            return action
        else:
            return None

    def set_pddl_stuff(self, predicates: Dict[str, Predicate], operators: Dict[str, Operator]):
        self._pddl_known = True
        self._predicates = predicates
        self._operators = operators

    def sample_random_action(
        self,
        observation: Dict[str, Any],
        parsed_literals: Set[Literal],
        success_action_execution: Optional[Dict[str, int]] = None,
        feasible_only: bool = True,
        **kwargs,
    ):
        assert self._pddl_known, "PDDL stuff unknown! Run 'set_pddl_stuff' first."
        entities = [DEFAULT_TYPE(var_name) for var_name in observation["entities"]]
        state = State(None, frozenset(entities), None)
        sampled_action = self._sample_actions(
            parsed_literals, state, success_action_execution, feasible_only
        )

        logger.info(colored(f"Literals: {str(parsed_literals)}", "green"))
        if sampled_action is not None:
            action = sampled_action
            logger.info(colored(f"Random feasible action: {str(action)}", "green"))
        else:
            action = _BasePlanner.sample_random_action(self, observation)
            logger.info(colored(f"Random action: {str(action)}", "green"))

        return action

    def plan_actions(
        self, observation: Dict[str, Any], parsed_literals: Set[Literal], *args, **kwargs
    ):
        action = self.sample_random_action(observation, parsed_literals)
        return [action]


@PLANNER.register()
class PDDLPlanner(RandomFeasiblePlanner):
    def __init__(self, action_predicates: List[Predicate], *args, **kwargs):
        super().__init__(action_predicates)
        self._pddl_planner = FD()
        self._pddl_domain = None

    def _plan_actions(self, state: State):
        if len(self._pddl_domain.operators) == 0:
            return None, True

        logger.info(colored("PDDL plan for action!", "green"))
        plan = None
        planning_failed = False
        try:
            plan = self._pddl_planner(self._pddl_domain, state, timeout=100)
        except PlanningFailure as e:
            logger.warning(f"Planning failed with exception: {e}")
            planning_failed = True
        except PlanningTimeout as e:
            planning_failed = True
        return plan, planning_failed

    def set_pddl_stuff(self, predicates: Dict[str, Predicate], operators: Dict[str, Operator]):
        super().set_pddl_stuff(predicates, operators)

        logger.debug(f"Set predicates: {str(list(predicates.keys()))}")
        logger.debug(f"Set operators: {str(list(operators.keys()))}")

        # create pddl domain (merge action predicates)
        predicates.update({action.name: action for action in self._action_predicates})
        self._pddl_domain = PDDLDomain(
            domain_name="domain",
            types={"default": DEFAULT_TYPE},
            type_hierarchy={},
            predicates=predicates,
            operators=operators,
            actions=self._action_predicates,
        )

    def plan_actions(
        self,
        observation: Dict[str, Any],
        parsed_literals: Set[Literal],
        goal_literals: Set[Literal],
        *args,
        **kwargs,
    ):
        assert self._pddl_known, "PDDL stuff unknown! Run 'set_pddl_stuff' first."
        entities = [DEFAULT_TYPE(var_name) for var_name in observation["entities"]]
        pddl_state = State(
            frozenset(parsed_literals),
            frozenset(entities),
            LiteralConjunction(goal_literals),
        )

        logger.info(colored(f"Goal: {str(goal_literals)}", "green"))
        logger.info(colored(f"Literals: {str(parsed_literals)}", "green"))
        plan, planning_failed = self._plan_actions(pddl_state)
        logger.info(colored(f"PDDL plan: {str(plan)}", "green"))
        if planning_failed:
            # import pdb; pdb.set_trace()
            plan = RandomFeasiblePlanner.plan_actions(self, observation, parsed_literals)

        return plan

    def explore_action(
        self,
        observation: Dict[str, Any],
        parsed_literals: Set[Literal],
        success_action_executions: Optional[Dict[str, int]] = None,
    ):
        logger.info(colored(f"Literals: {str(parsed_literals)}", "green"))
        # allow sample random action at a low possibility
        return self.sample_random_action(
            observation, parsed_literals, success_action_executions, feasible_only=False
        )


@PLANNER.register()
class LLMPlanner(RandomFeasiblePlanner, LLMBase):
    def __init__(
        self,
        action_predicates: List[Predicate],
        use_gpt_4: bool,
        domain_desc: str,
        action_desc: str,
        max_examples: int = 5,
        *args,
        **kwargs,
    ):
        RandomFeasiblePlanner.__init__(self, action_predicates=action_predicates)
        LLMBase.__init__(self, use_gpt_4=use_gpt_4)

        # load planning prompt template
        prompt_template_folder = Path(__file__).resolve().parent.parent.parent / "prompts"
        planning_prompt_template = load_txt(
            prompt_template_folder / "llm_planner_prompt_template.txt"
        )

        # load generate explore action prompt template
        explore_action_prompt_template = load_txt(
            prompt_template_folder / "llm_planner_explore_prompt_template.txt"
        )

        # instantiate prompts
        self._planning_prompt = planning_prompt_template.replace(
            "{domain_desc}", domain_desc
        ).replace("{action_desc}", action_desc)
        self._explore_prompt = explore_action_prompt_template.replace(
            "{domain_desc}", domain_desc
        ).replace("{action_desc}", action_desc)

        # maximum number of examples
        self._max_examples = max_examples

        # to be set
        self._planning_examples = []
        self._planning_examples_text = ""
        self._predicates_text = ""
        self._general_knowledge = []

    def _prepare_planning_prompt(
        self,
        observation: Dict[str, Any],
        goal_text: str,
        is_explore: bool,
        parsed_literals: Optional[Set[Literal]] = None,
        goal_literals: Optional[Set[Literal]] = None,
        extra_info: Optional[str] = None,
        completed_actions: Optional[List[str]] = None,
        use_both_text_and_predicates: Optional[bool] = False,
    ):
        planning_prompt = self._planning_prompt if not is_explore else self._explore_prompt

        # predicates
        predicates_text = ""
        if parsed_literals is not None:
            predicates_text = "The available state predicates are: " + str(self._predicates_text)

        planning_prompt = planning_prompt.replace("{predicates}", predicates_text)

        # knowledge
        if len(self._general_knowledge) > 0:
            knowledge_text = "\nThe available knowledge are: \n" + "\n".join(
                self._general_knowledge
            )
        else:
            knowledge_text = ""
        planning_prompt = planning_prompt.replace("{knowledge}", knowledge_text)

        # prepare state lits
        entities = observation["entities"]
        raw_state_text = "The current state is: " + textualize_observation(observation)
        if parsed_literals is not None:
            symbolic_state_text = "The current symbolic literals are: " + ", ".join(
                [f"{pddl_literal_to_text(lit)}" for lit in parsed_literals]
            )
            if use_both_text_and_predicates:
                state_text = raw_state_text + "\n" + symbolic_state_text
            else:
                state_text = symbolic_state_text
        else:
            state_text = raw_state_text

        # prepare goal lits
        # todo: run ablation on goal lits or goal text or translated goal lits
        # if goal_literals is not None:
        #     goal_text = ", ".join([f"{pddl_literal_to_text(lit)}" for lit in goal_literals])
        # else:
        #     goal_text = goal_text

        # past actions
        if completed_actions is not None and len(completed_actions) > 0:
            completed_actions_text = f"{', '.join(completed_actions)}."
        else:
            completed_actions_text = "No completed actions."

        # replace problem
        planning_prompt = planning_prompt.replace(
            "{problem}",
            f"The scene has objects: {str(entities)}\n{state_text}\nThe goal is: {goal_text}\nThe completed actions are: {completed_actions_text}\nHere is some extra info: {extra_info}\n",
        )

        # replace examples
        if not is_explore:
            if self._planning_examples_text != "":
                planning_prompt = planning_prompt.replace(
                    "{examples}", self._planning_examples_text
                )
            elif len(self._planning_examples) > 0:
                examples_text = (
                    "\nHere are some planning examples:\n"
                    + "\n\n".join(self._planning_examples)
                    + "\n"
                )
                planning_prompt = planning_prompt.replace("{examples}", examples_text)
            else:
                planning_prompt = planning_prompt.replace("{examples}", "")

        return planning_prompt

    def set_demonstrations(
        self, trajectories: List[Trajectory], use_both_text_and_predicates: bool = False
    ):
        # select demo with different goal
        demo_trajs = []
        candidate_trajs = []
        goals = set()
        for traj in trajectories:
            if traj.goal_text not in goals:
                demo_trajs.append(traj)
                goals.add(traj.goal_text)
            else:
                candidate_trajs.append(traj)

        demo_trajs += candidate_trajs
        demo_trajs = demo_trajs[: self._max_examples]
        self._planning_examples = []

        for idx, traj in enumerate(demo_trajs):
            entities = traj.entities
            init_state = traj.state_history[0]

            raw_state_text = "The current state is: " + textualize_observation(
                init_state.observation
            )
            if len(init_state.abstract_observation) > 0:
                symbolic_state_text = "The current symbolic literals are: " + ", ".join(
                    [f"{pddl_literal_to_text(lit)}" for lit in init_state.abstract_observation]
                )
                if use_both_text_and_predicates:
                    state_text = raw_state_text + "\n" + symbolic_state_text
                else:
                    state_text = symbolic_state_text
            else:
                state_text = raw_state_text

            goal_text = traj.goal_text
            plan_text = {
                "Plan": [pddl_literal_to_text(action.action) for action in traj.action_history]
            }

            example_text = f"Example {idx+1}:\n"
            example_text += (
                f"The scene has objects: {str(entities)}\n{state_text}\nThe goal is: {goal_text}\n"
            )
            example_text += f"The plan is:\n {plan_text}"

            self._planning_examples.append(example_text)

        logger.info(
            colored(f"{str(self)} set demonstrations: {len(self._planning_examples)}", "green")
        )

    def set_examples(self, planning_examples: str):
        self._planning_examples_text = copy.deepcopy(planning_examples)

    def set_knowledge(self, predicates_text: str, knowledge: Optional[List[str]] = []):
        self._predicates_text = copy.deepcopy(predicates_text)
        self._general_knowledge = copy.deepcopy(knowledge)

    def plan_actions(
        self,
        observation: Dict[str, Any],
        goal_text: str,
        parsed_literals: Optional[Set[Literal]] = None,
        goal_literals: Optional[Set[Literal]] = None,
        completed_actions: Optional[List[str]] = None,
        extra_info: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        use_both_text_and_predicates: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        # prompt llm
        planning_prompt = self._prepare_planning_prompt(
            observation,
            goal_text,
            is_explore=False,
            parsed_literals=parsed_literals,
            goal_literals=goal_literals,
            extra_info=extra_info,
            completed_actions=completed_actions,
            use_both_text_and_predicates=use_both_text_and_predicates,
        )
        llm_output = json.loads(self.prompt_llm(planning_prompt, temperature, force_json=True))

        # parse llm output
        action_predicates = {x.name: x for x in self._action_predicates}
        plan = [
            create_literal_from_raw(raw_action, predicates=action_predicates)
            for raw_action in llm_output["Plan"]
        ]

        return plan

    def explore_action(
        self,
        observation: Dict[str, Any],
        parsed_literals: Set[Literal],
        completed_actions: Optional[List[str]] = None,
    ):
        # prompt llm
        planning_prompt = self._prepare_planning_prompt(
            observation,
            goal_text="",
            is_explore=True,
            parsed_literals=parsed_literals,
            completed_actions=completed_actions,
        )
        llm_output = json.loads(self.prompt_llm(planning_prompt, force_json=True))

        # parse llm output
        action_predicates = {x.name: x for x in self._action_predicates}
        explore_action = create_literal_from_raw(
            llm_output["Exploration action"], predicates=action_predicates
        )

        return explore_action
