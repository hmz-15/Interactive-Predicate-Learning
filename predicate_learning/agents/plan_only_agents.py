from typing import Any, Dict
from pathlib import PosixPath
from termcolor import colored
import logging
import copy

from predicate_learning.agents.base_agent import BaseAgent, AGENT
from predicate_learning.agents.learners.predicate_learner import PredicateLearner
from predicate_learning.utils.common_util import AgentAction, AgentQuery
from predicate_learning.utils.io_util import save_npz, load_npz
from predicate_learning.utils.llm_util import pddl_literal_to_text
from predicate_learning.utils.pddl_util import select_operator

from pddlgym.parser import PDDLDomainParser


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@AGENT.register()
class RandomPlanAgent(BaseAgent):
    """
    An agent that explore with random actions and learn nothing.
    """

    def __init__(self, world, agent_cfg, *args, **kwargs):
        super().__init__(world, agent_cfg)

        # restrict planners
        assert agent_cfg.train_planner == "RandomPlanner"
        assert agent_cfg.test_planner == "RandomPlanner"

    def act(self, observation: Dict[str, Any]):
        # prepare agent query
        query = AgentQuery()

        # plan actions
        if len(self._plan) > 0:
            action = self._plan.pop(0)
        else:
            # if empty plan buffer, replan
            planner = self._train_planner if self._is_train else self._test_planner

            # replan
            actions = planner.plan_actions(observation=observation)
            assert actions is not None, "No feasible action found!"
            self._plan = actions
            action = self._plan.pop(0)

        return AgentAction(action=action, query=query)


@AGENT.register()
class OraclePlanAgent(RandomPlanAgent):
    """
    An agent that plan with ground truth pddl and learn nothing.
    """

    def __init__(self, world, agent_cfg, *args, **kwargs):
        BaseAgent.__init__(self, world, agent_cfg)

        # restrict planners
        assert agent_cfg.train_planner == "PDDLPlanner"
        assert agent_cfg.test_planner == "PDDLPlanner"

        # set pddl
        self._train_planner.set_pddl_stuff(
            world.pddl_domain.predicates, world.pddl_domain.operators
        )
        self._test_planner.set_pddl_stuff(world.pddl_domain.predicates, world.pddl_domain.operators)

        # whether to save demo traj
        self.save_demo_traj = agent_cfg.save_demo_traj
        self.collected_demo_trajs = []

    def act(self, observation: Dict[str, Any]):
        # prepare agent query
        query = AgentQuery()

        # plan actions
        if len(self._plan) > 0:
            action = self._plan.pop(0)
        else:
            # if empty plan buffer, replan
            planner = self._train_planner if self._is_train else self._test_planner

            parsed_lits = self._world.parse_pddl_literals()
            goal_literals = self._goal_lits

            # replan
            actions = planner.plan_actions(
                observation=observation,
                parsed_literals=parsed_lits,
                goal_text=self._goal_text,
                goal_literals=goal_literals,
            )
            assert actions is not None, "No feasible action found!"
            self._plan = actions
            action = self._plan.pop(0)
            # replan
            self._plan = []

        return AgentAction(action=action, query=query)

    def save_agent_knowledge(self, save_dir: PosixPath):
        """
        Save current agent knowledge.
        """
        if len(self.collected_demo_trajs) > 0:
            logger.info(f"Save demo traj {save_dir}.")
            save_npz({"demo_traj": self.collected_demo_trajs}, save_dir / "demo_traj.npz")

    def reset_trail(self, goal_spec: str, init_obs: Dict[str, Any], *args, **kwargs):
        super().reset_trail(goal_spec, init_obs)
        # set goal literals
        self._goal_lits = frozenset(self._world.goal_lits)

    def flag_end_of_trail(self, *args, **kwargs):
        """
        Flag the end of current trail.
        """
        super().flag_end_of_trail(*args, **kwargs)

        # save demo
        traj = self._current_traj
        if traj.goal_achieved:
            demo_traj = copy.deepcopy(traj)
            success_actions = [
                action
                for action, state in zip(traj.action_history, traj.state_history[1:])
                if state.success
            ]
            demo_traj.action_history = success_actions

            self.collected_demo_trajs.append(demo_traj)


@AGENT.register()
class LLMPlanAgent(RandomPlanAgent):
    """
    An agent that plan with LLM and learn nothing.
    """

    def __init__(self, world, agent_cfg, *args, **kwargs):
        BaseAgent.__init__(self, world, agent_cfg)

        # restrict planners
        assert agent_cfg.train_planner == "LLMPlanner"
        assert agent_cfg.test_planner == "LLMPlanner"

        # whether use learned predicates & knowledge
        self.use_learned_predicates = agent_cfg.use_learned_predicates
        if self.use_learned_predicates:
            assert self._load_path is not None and self._load_path.exists(), "Specify load path!"
            self._predicate_learner = PredicateLearner(
                domain_desc=self._domain_desc,
                prompt_dir=agent_cfg.prompt_dir,
                example_predicate_prompt_file=agent_cfg.example_predicate_prompt_file,
                perception_api_wrapper=world.perception_api_wrapper,
                use_gpt_4=agent_cfg.use_gpt_4,
            )
            self.load_agent_knowledge()

            # set pddl
            self._train_planner.set_knowledge(
                self._predicate_learner.get_predicates_text(),  # not include negative predicates
                # self._predicate_learner.general_knowledge, # todo: currently disable this
            )
            self._test_planner.set_knowledge(
                self._predicate_learner.get_predicates_text(),  # not include negative predicates
                # self._predicate_learner.general_knowledge, # todo: currently disable this
            )

            # recompute demo traj initial state
            for traj in self._demo_trajs:
                updated_lits = self._predicate_learner.parse_literals(
                    traj.state_history[0].observation,
                    self.perception_api_wrapper,
                    return_pddl_lits=True,
                    essential_only=True,
                    include_negative_predicates=False,
                )
                traj.state_history[0].abstract_observation = frozenset(updated_lits)

        else:
            self._predicate_learner = None

        # use both text and predicates
        self.use_both_text_and_predicates = agent_cfg.use_both_text_and_predicates

        # set demo trajs
        if len(self._demo_trajs) > 0:
            self._train_planner.set_demonstrations(
                self._demo_trajs, self.use_both_text_and_predicates
            )
            self._test_planner.set_demonstrations(
                self._demo_trajs, self.use_both_text_and_predicates
            )

        if self._planning_examples != "":
            self._train_planner.set_examples(self._planning_examples)
            self._test_planner.set_examples(self._planning_examples)

    def act(self, observation: Dict[str, Any]):
        # prepare agent query
        query = AgentQuery()

        # parse lits for planning (only parse essential, and positive predicates)
        parsed_lits = None
        if self.use_learned_predicates:
            parsed_lits = self._predicate_learner.parse_literals(
                observation,
                self.perception_api_wrapper,
                return_pddl_lits=True,
                essential_only=True,
                include_negative_predicates=False,
            )

        # completed actions with feedback
        completed_actions = []
        for action in self._current_traj.action_history:
            if action.action is not None:
                completed_actions.append(pddl_literal_to_text(action.action))
            else:
                completed_actions.append("None")

        # keep maximum of 5 actions
        completed_actions = completed_actions[-5:]

        # plan actions
        if len(self._plan) > 0:
            action = self._plan.pop(0)
        else:
            # if empty plan buffer, replan
            planner = self._train_planner if self._is_train else self._test_planner

            # replan
            actions = planner.plan_actions(
                observation=observation,
                parsed_literals=parsed_lits,
                goal_text=self._goal_text,
                completed_actions=completed_actions,
                use_both_text_and_predicates=self.use_both_text_and_predicates,
            )
            assert actions is not None, "No feasible action found!"
            self._plan = actions
            if len(self._plan) > 0:
                action = self._plan.pop(0)
            else:
                # agent thinks it achieves the goal
                action = None

        return AgentAction(action=action, query=query)

    @property
    def agent_knowledge(self):
        if self.use_learned_predicates:
            return {
                "learned_predicates": self._predicate_learner.learned_predicates,
                "general_knowledge": self._predicate_learner.general_knowledge,
            }

    @agent_knowledge.setter
    def agent_knowledge(self, loaded_agent_knowledge: Dict[str, Any]):
        if self.use_learned_predicates:
            self._predicate_learner.learned_predicates = loaded_agent_knowledge[
                "learned_predicates"
            ].item()
            self._predicate_learner.general_knowledge = loaded_agent_knowledge[
                "general_knowledge"
            ].tolist()


@AGENT.register()
class LLMPlanPrecondAgent(LLMPlanAgent):
    """
    An agent that plan with LLM and precondition model.
    """

    def __init__(self, world, agent_cfg, *args, **kwargs):
        super().__init__(world, agent_cfg)
        assert agent_cfg.use_learned_predicates, "Must use learned predicates!"

        # load pddl file & set pddl
        self._pddl_domain = PDDLDomainParser(self._load_path / "domain.pddl")

        # max replan iterations
        self._max_replan_iterations = agent_cfg.max_replan_iterations
        self._current_replan_iter = 0

    def act(self, observation: Dict[str, Any]):
        # prepare agent query
        query = AgentQuery()

        # parse lits for planning
        parsed_lits = None
        parsed_lits_all = None
        if self.use_learned_predicates:
            # positive only (for planning)
            parsed_lits = self._predicate_learner.parse_literals(
                observation,
                self.perception_api_wrapper,
                return_pddl_lits=True,
                essential_only=True,
                include_negative_predicates=False,
            )
            # all (for checking preconds)
            parsed_lits_all = self._predicate_learner.parse_literals(
                observation,
                self.perception_api_wrapper,
                return_pddl_lits=True,
                essential_only=True,
                include_negative_predicates=True,
            )

        # completed actions with feedback
        completed_actions = []
        for action in self._current_traj.action_history:
            if action.action is not None:
                completed_actions.append(pddl_literal_to_text(action.action))
            else:
                completed_actions.append("None")

        # keep maximum of 5 actions
        completed_actions = completed_actions[-5:]

        # other cases, iteratively find action until feasible
        feasible = False
        infeasible_actions = []
        while not feasible and self._current_replan_iter <= self._max_replan_iterations:
            if len(self._plan) > 0:
                # check feasibility
                action = self._plan.pop(0)
                logger.info(colored(f"Current lits: {parsed_lits}", "green"))
                if self.check_action_feasibility(action, parsed_lits_all):
                    logger.info(colored(f"Planned action is feasible: {action}!", "green"))
                    feasible = True
                else:
                    logger.info(colored(f"Planned action is infeasible: {action}!", "red"))
                    feasible = False
                    # reset plan buffer
                    self._plan = []

                    infeasible_actions.append(action)
                    self._current_replan_iter += 1
            else:
                # if empty plan buffer, replan
                planner = self._train_planner if self._is_train else self._test_planner
                # prepare extra info
                extra_info = f"The actions below are infeasible in current state: {', '.join(pddl_literal_to_text(action) for action in infeasible_actions)}. Please propose other actions."

                # replan
                actions = planner.plan_actions(
                    observation=observation,
                    parsed_literals=parsed_lits,
                    goal_text=self._goal_text,
                    goal_literals=self._goal_lits,
                    extra_info=extra_info,
                    use_both_text_and_predicates=self.use_both_text_and_predicates,
                )
                assert actions is not None, "No feasible action found!"
                self._plan = actions

                # agent thinks it achieves the goal
                if len(self._plan) == 0:
                    action = None
                    feasible = True

        return AgentAction(action=action, query=query)

    def check_action_feasibility(self, action, current_lits):
        """
        Check whether an action is feasible given the current literals.
        """
        op, _ = select_operator(
            current_lits,
            action,
            self._pddl_domain.operators,
            inference_mode="csp",
            require_unique_assignment=False,
        )
        if op is None:
            return False
        else:
            return True

    def reset_trail(self, goal_spec: str, init_obs: Dict[str, Any], *args, **kwargs):
        super().reset_trail(goal_spec, init_obs)
        # reset replan iter
        self._current_replan_iter = 0
