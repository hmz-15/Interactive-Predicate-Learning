from termcolor import colored
from typing import Any, Dict
import logging
import random
import copy
from pathlib import PosixPath

from predicate_learning.agents.base_agent import AGENT, BaseAgent
from predicate_learning.agents.learn_predicate_agent import LearnPredicateAgent
from predicate_learning.agents.learners.predicate_learner import PredicateLearner

# from predicate_learning.agents.learners.precond_learner import PrecondLearner

from predicate_learning.agents.learners.operator_learner import OperatorLearner

from predicate_learning.utils.common_util import AgentAction, AgentQuery, EnvFeedback
from predicate_learning.utils.io_util import merge_number_dict
from predicate_learning.utils.general_util import Timer

# from predicate_learning.agents.learners.operator_learner import OperatorLearner
from predicate_learning.utils.pddl_util import DEFAULT_TYPE
from pddlgym.parser import PDDLDomain


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
t = Timer(logger=logger, color="green")


@AGENT.register()
class LearnPDDLAgent(LearnPredicateAgent):
    def __init__(self, world, agent_cfg, *args, **kwargs):
        BaseAgent.__init__(self, world, agent_cfg)

        # restrict planners
        assert agent_cfg.train_planner in ["PDDLPlanner"]
        assert agent_cfg.test_planner in ["PDDLPlanner"]

        # learn predicates from language feedback
        self._predicate_learner = PredicateLearner(
            domain_desc=self._domain_desc,
            prompt_dir=agent_cfg.prompt_dir,
            example_predicate_prompt_file=agent_cfg.example_predicate_prompt_file,
            perception_api_wrapper=self.perception_api_wrapper,
            use_gpt_4=agent_cfg.use_gpt_4,
        )

        # learn from failure
        # self._precond_learner = PrecondLearner(self._action_predicates)

        # learn from transitions
        self._operator_learner = OperatorLearner(
            action_predicates=self._action_predicates, predicate_learner=self._predicate_learner
        )

        # successful action executions
        self._success_action_executions = {action.name: 0 for action in self._action_predicates}

        # replan
        self.replan_at_each_step_at_test = agent_cfg.replan_at_each_step_at_test

    def act(self, observation: Dict[str, Any]):
        # prepare agent query
        if self._is_train:
            # initialize
            query = AgentQuery(failure_explain=True)
        else:
            query = AgentQuery()

        # parse lits for planning (only parse essential predicates)
        parsed_lits = self._predicate_learner.parse_literals(
            observation,
            self.perception_api_wrapper,
            return_pddl_lits=True,
            essential_only=True,
            include_negative_predicates=True,
        )

        # at train mode, if agent thinks is goal, prepare non_goal query with None action
        if self._is_train:
            is_goal = self._goal_lits.issubset(set(parsed_lits))
            if is_goal:
                query.non_goal_explain = True
                logger.info(f"Agent thinks it achieves the goal! {parsed_lits}")
                action = None

                return AgentAction(action=action, query=query)

        # at train mode, if random exploration
        if self._is_train:
            if random.random() < self._random_act_prob:
                logger.info(colored("Random exploration!", "green"))
                # set pddl operators (contain preconditions)
                self._train_planner.set_pddl_stuff(
                    self._predicate_learner.pddl_predicates, self._operator_learner.pddl_operators
                )
                # priortize unexplored actions
                action = self._train_planner.explore_action(
                    observation, parsed_lits, self._success_action_executions
                )
                # reset plan buffer
                self._plan = []

                return AgentAction(action=action, query=query)

        # other cases, plan actions
        plan_time = 0.0
        if len(self._plan) > 0:
            action = self._plan.pop(0)
        else:
            # if empty plan buffer, replan
            planner = self._train_planner if self._is_train else self._test_planner

            # todo: handle this
            planner.set_pddl_stuff(
                self._predicate_learner.pddl_predicates, self._operator_learner.pddl_operators
            )

            # planner.set_knowledge(
            #     self._predicate_learner.get_predicates_raw(),  # not include negative predicates
            #     self._predicate_learner.general_knowledge,
            # )  # todo: and here

            # replan
            t.tic("plan actions")
            actions = planner.plan_actions(
                observation=observation,
                parsed_literals=parsed_lits,
                goal_text=self._goal_text,
                goal_literals=self._goal_lits,
            )
            plan_time = t.toc()
            if actions is not None and len(actions) > 0:
                self._plan = actions
                action = self._plan.pop(0)

                # replan at each step
                if self._is_train or self.replan_at_each_step_at_test:
                    self._plan = []
            else:
                action = None

        return AgentAction(action=action, query=query, pddl_plan_time=plan_time)

    def step(
        self, pre_observation: Dict[str, Any], agent_action: AgentAction, env_feedback: EnvFeedback
    ):
        # Call LearnPredicateAgent step
        action_preconds, agent_state = super().step(pre_observation, agent_action, env_feedback)

        # learn preconditions
        # if self._is_train and precond_lits is not None:
        #     self._precond_learner.add_new_preconds(agent_action.action, copy.deepcopy(precond_lits))

        # learn operators
        # if self._is_train and env_feedback.goal_achieved:
        #     # add trajectory to operator learner
        #     self._operator_learner.add_raw_trajectory(self._current_traj)

        #     # learn operators
        #     self._operator_learner.learn_operators()

        # logger.info(self.get_pddl_operators())

        if self._is_train:
            if action_preconds[0] is not None:
                self._operator_learner.add_new_preconds(action_preconds[0], action_preconds[1])
            elif (
                env_feedback.success and not agent_action.query.non_goal_explain
            ):  # if success and the transition is valid
                # both pre_obs and obs need to have all objects detected
                if set(pre_observation["objects"].keys()) == set(
                    pre_observation["entities"]
                ) and set(env_feedback.observation["objects"].keys()) == set(
                    env_feedback.observation["entities"]
                ):
                    precond_lits = self._operator_learner.add_new_transition(
                        obs_before=pre_observation,
                        action=agent_action.action,
                        env_feedback=env_feedback,
                    )

                    # correct predicate functions
                    t.tic("learn predicates from success execution")
                    num_llm_calls = self._predicate_learner.learn_predicates_from_success_execution(
                        pre_observation, precond_lits
                    )
                    learn_pred_time = t.toc()
                    agent_state.learn_pred_time += learn_pred_time
                    agent_state.num_llm_calls += num_llm_calls

                    # learn operators
                    t.tic("learn operators")
                    self._operator_learner.learn_operators()
                    learn_op_time = t.toc()
                    agent_state.learn_op_time = learn_op_time

                    merge_number_dict(
                        self._success_action_executions, {agent_action.action.predicate.name: 1}
                    )

        logger.info(self._operator_learner.pddl_operators)

        # update agent state
        agent_state.learned_operators = self._operator_learner.pddl_operators
        return agent_state

    @property
    def agent_knowledge(self):
        return {
            "learned_predicates": self._predicate_learner.learned_predicates,
            "general_knowledge": self._predicate_learner.general_knowledge,
            "learned_operators": self._operator_learner.learned_operators,
        }

    @agent_knowledge.setter
    def agent_knowledge(self, loaded_agent_knowledge: Dict[str, Any]):
        self._predicate_learner.learned_predicates = loaded_agent_knowledge[
            "learned_predicates"
        ].item()
        self._predicate_learner.general_knowledge = loaded_agent_knowledge[
            "general_knowledge"
        ].tolist()
        self._operator_learner.learned_operators = loaded_agent_knowledge[
            "learned_operators"
        ].item()

    def save_agent_knowledge(self, save_dir: PosixPath):
        """
        Save current agent knowledge.
        """
        LearnPredicateAgent.save_agent_knowledge(self, save_dir)

        # save pddl
        all_predicates = copy.deepcopy(self._predicate_learner.pddl_predicates)
        all_predicates.update({pred.name: pred for pred in self._action_predicates})
        pddl_domain = PDDLDomain(
            domain_name="domain",
            types={"default": DEFAULT_TYPE},
            type_hierarchy={},
            predicates=all_predicates,
            operators=self._operator_learner.pddl_operators,
            actions=self._action_predicates,
        )
        pddl_domain.write(save_dir / "domain.pddl")

    # def get_pddl_operators(self):
    #     # merge pddl operators
    #     traj_operators = self._operator_learner.pddl_operators
    #     precond_operators = self._precond_learner.pddl_operators

    #     # if no traj operators, use precond operators; otherwise merge them
    #     if len(traj_operators) == 0:
    #         traj_operators.update(precond_operators)
    #     else:
    #         for action_name, precond_op in precond_operators.items():
    #             if_op_learned = False
    #             for op_name, traj_op in traj_operators.items():
    #                 # if action name matches, the preconds should be learned
    #                 if action_name in op_name:
    #                     assert check_satisfy(
    #                         precond_op.preconds.literals, traj_op.preconds.literals
    #                     ), f"Preconds not satisfied! "
    #                     if_op_learned = True

    #             # if not learned, add to traj operators
    #             if not if_op_learned:
    #                 traj_operators[action_name] = precond_op

    #     return traj_operators
