from typing import Any, Dict
from termcolor import colored
from pathlib import PosixPath
import logging
import random

from predicate_learning.agents.base_agent import BaseAgent, AGENT
from predicate_learning.utils.common_util import AgentAction, AgentQuery, EnvFeedback, AgentState
from predicate_learning.agents.learners.predicate_learner import PredicateLearner
from predicate_learning.utils.llm_util import pddl_literal_to_text
from predicate_learning.utils.io_util import write_txt
from predicate_learning.utils.general_util import Timer


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
t = Timer(logger=logger, color="green")


@AGENT.register()
class LearnPredicateAgent(BaseAgent):
    """
    An agent that learns predicates from human inputs, while the planning is driven by LLMs.
    """

    def __init__(self, world, agent_cfg, *args, **kwargs):
        super().__init__(world, agent_cfg)

        # restrict planners
        assert agent_cfg.train_planner == "LLMPlanner"
        assert agent_cfg.test_planner == "LLMPlanner"

        # learn predicates
        self._predicate_learner = PredicateLearner(
            domain_desc=self._domain_desc,
            prompt_dir=agent_cfg.prompt_dir,
            example_predicate_prompt_file=agent_cfg.example_predicate_prompt_file,
            perception_api_wrapper=self.perception_api_wrapper,
            use_gpt_4=agent_cfg.use_gpt_4,
        )

    def act(self, observation: Dict[str, Any]):
        # prepare agent query
        if self._is_train:
            # initialize
            query = AgentQuery(failure_explain=True)
        else:
            query = AgentQuery()

        # parse lits for planning (only parse essential, and positive predicates)
        parsed_lits = self._predicate_learner.parse_literals(
            observation,
            self.perception_api_wrapper,
            return_pddl_lits=True,
            essential_only=True,
            include_negative_predicates=False,
        )

        # completed actions with feedback
        completed_actions = []
        for action, feedback in zip(
            self._current_traj.action_history, self._current_traj.state_history[1:]
        ):
            feedback_list = []
            if len(feedback.failure_explain) > 0:
                feedback_list.append(feedback.failure_explain)
            if len(feedback.non_goal_explain) > 0:
                feedback_list.append(feedback.non_goal_explain)

            feedback = f"Feedback: {', '.join(feedback_list)}" if len(feedback_list) > 0 else ""
            # feedback = "Failed" if not feedback.success else "Success"
            action_text = (
                pddl_literal_to_text(action.action) if action.action is not None else "None"
            )
            if feedback != "":
                action_text = f"{action_text} ({feedback})"
            completed_actions.append(action_text)

        # keep maximum of 5 actions
        completed_actions = completed_actions[-5:]

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
                action = self._train_planner.explore_action(
                    observation, parsed_lits, completed_actions
                )
                # reset plan buffer
                self._plan = []

                return AgentAction(action=action, query=query)

        # other cases, plan actions
        if len(self._plan) > 0:
            action = self._plan.pop(0)
        else:
            # if empty plan buffer, replan
            planner = self._train_planner if self._is_train else self._test_planner
            # update knowledge into planner
            planner.set_knowledge(
                self._predicate_learner.get_predicates_text(),  # not include negative predicates
                self._predicate_learner.general_knowledge,
            )

            # set demos
            if not self._is_train:
                for traj in self._demo_trajs:
                    updated_lits = self._predicate_learner.parse_literals(
                        traj.state_history[0].observation,
                        self.perception_api_wrapper,
                        return_pddl_lits=True,
                        essential_only=True,
                        include_negative_predicates=False,
                    )
                    traj.state_history[0].abstract_observation = frozenset(updated_lits)

                if len(self._demo_trajs) > 0:
                    planner.set_demonstrations(self._demo_trajs)

            # replan
            actions = planner.plan_actions(
                observation=observation,
                parsed_literals=parsed_lits,
                goal_text=self._goal_text,
                goal_literals=self._goal_lits,
                completed_actions=completed_actions,
            )
            assert actions is not None, "No feasible action found!"
            self._plan = actions

            if len(self._plan) > 0:
                action = self._plan.pop(0)
            else:
                # agent thinks it achieves the goal
                action = None
                query.non_goal_explain = True

        return AgentAction(action=action, query=query)

    def step(
        self, pre_observation: Dict[str, Any], agent_action: AgentAction, env_feedback: EnvFeedback
    ):
        # call base method
        super().step(pre_observation, agent_action, env_feedback)

        agent_state = AgentState()

        # learn predicates
        action_preconds = (None, None)
        num_llm_calls = 0
        if self._is_train:
            t.tic("learn predicates")
            # if get failure explanations, learn predicates & precondition lits
            if agent_action.query.failure_explain and not env_feedback.success:
                if env_feedback.failure_explain:
                    (
                        action_preconds,
                        num_llm_fail,
                    ) = self._predicate_learner.learn_predicates_from_failure(
                        agent_action, env_feedback
                    )
                    agent_state.failure_explain = env_feedback.failure_explain
                    num_llm_calls += num_llm_fail

            # if get non-goal explanations, learn predicates
            if agent_action.query.non_goal_explain:
                if env_feedback.non_goal_explain:
                    num_llm_non_goal = self._predicate_learner.learn_predicates_from_non_goal(
                        env_feedback
                    )
                    agent_state.non_goal_explain = env_feedback.non_goal_explain
                    num_llm_calls += num_llm_non_goal

            # if goal achieved
            if env_feedback.goal_achieved:
                num_llm_goal = self._predicate_learner.learn_predicates_from_goal_achieved(
                    env_feedback
                )
                agent_state.goal_reached_feedback = True
                num_llm_calls += num_llm_goal

            learn_pred_time = t.toc()

            # update info
            agent_state.num_llm_calls = num_llm_calls
            agent_state.learn_pred_time = learn_pred_time

        # update learned predicates
        agent_state.learned_predicates = self._predicate_learner.pddl_predicates

        return action_preconds, agent_state

    @property
    def agent_knowledge(self):
        return {
            "learned_predicates": self._predicate_learner.learned_predicates,
            "general_knowledge": self._predicate_learner.general_knowledge,
        }

    @agent_knowledge.setter
    def agent_knowledge(self, loaded_agent_knowledge: Dict[str, Any]):
        self._predicate_learner.learned_predicates = loaded_agent_knowledge[
            "learned_predicates"
        ].item()
        self._predicate_learner.general_knowledge = loaded_agent_knowledge[
            "general_knowledge"
        ].tolist()

    def save_agent_knowledge(self, save_dir: PosixPath):
        """
        Save current agent knowledge.
        """
        super().save_agent_knowledge(save_dir)

        # save predicates in a python file
        all_text = self._predicate_learner.predicates_functions
        write_txt(save_dir / "predicates.py", all_text)

    def reset_trail(self, goal_spec: str, init_obs: Dict[str, Any], *args, **kwargs):
        super().reset_trail(goal_spec, init_obs)

        learn_pred_time = 0.0
        parse_goal_time = 0.0
        num_llm_calls = 0
        if self._is_train:
            # learn predicates from goal
            t.tic("learn predicates from goal spec")
            self._goal_lits, num_llm_calls = self._predicate_learner.learn_predicates_from_goal(
                goal_spec, init_obs
            )
            learn_pred_time = t.toc()

        else:
            # parse goal lits
            t.tic("parse goal lits")
            self._goal_lits = self._predicate_learner.parse_goal(init_obs, goal_spec)
            parse_goal_time = t.toc()

        agent_state = AgentState(
            goal_spec=goal_spec,
            learned_predicates=self._predicate_learner.pddl_predicates,
            num_llm_calls=num_llm_calls,
            learn_pred_time=learn_pred_time,
            parse_goal_time=parse_goal_time,
        )
        return agent_state
