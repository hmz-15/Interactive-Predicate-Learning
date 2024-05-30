import abc
from pathlib import Path, PosixPath
from termcolor import colored
from typing import Any, Dict
import logging

from predicate_learning.utils.common_util import AgentAction, EnvFeedback, Trajectory
from predicate_learning.utils.io_util import load_txt, save_npz, load_npz, load_json
from predicate_learning.utils.registry import Registry

from predicate_learning.agents.planners import create_planner
from predicate_learning.utils.llm_util import parse_actions_from_text, prepare_action_desc


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


AGENT = Registry("Agent")


def create_agent(agent_name: str, *args, **kwargs):
    return AGENT.get(agent_name)(*args, **kwargs)


class BaseAgent:
    def __init__(self, world, agent_cfg, *args, **kwargs):
        # load domain & action information
        self.domains_dir = Path(agent_cfg.domain_dir)
        domain_desc = load_json(self.domains_dir / "domain_desc.json")[str(world)]
        self._domain_desc = domain_desc["domain_desc"]

        all_actions = load_json(self.domains_dir / "actions.json")
        self._action_desc = prepare_action_desc(all_actions, domain_desc["actions"])
        self._action_predicates = parse_actions_from_text(self._action_desc)

        logger.info(f"Parsed action predicates: {self._action_predicates}")

        # perception api wrapper
        self.perception_api_wrapper = world.perception_api_wrapper

        # create train & test planner
        self._train_planner = create_planner(
            agent_cfg.train_planner,
            action_predicates=self._action_predicates,
            use_gpt_4=agent_cfg.use_gpt_4 if hasattr(agent_cfg, "use_gpt_4") else False,
            domain_desc=self._domain_desc,
            action_desc=self._action_desc,
        )

        self._test_planner = create_planner(
            agent_cfg.test_planner,
            action_predicates=self._action_predicates,
            use_gpt_4=agent_cfg.use_gpt_4 if hasattr(agent_cfg, "use_gpt_4") else False,
            domain_desc=self._domain_desc,
            action_desc=self._action_desc,
        )

        # load agent state path
        self._load_path = Path(agent_cfg.load_path) if agent_cfg.load_path is not None else None

        # load few-shot demo
        # whether use few-shot demos
        self._use_few_shot_demos = agent_cfg.use_few_shot_demos
        self._demo_trajs = []
        self._planning_examples = ""
        if self._use_few_shot_demos:
            examples_file = self.domains_dir / str(world) / "examples.txt"
            demos_file = self.domains_dir / str(world) / "demo_traj.npz"
            if examples_file.exists():
                self._planning_examples = load_txt(examples_file)
            if demos_file.exists():
                self._demo_trajs = load_npz(self.domains_dir / str(world) / "demo_traj.npz")[
                    "demo_traj"
                ].tolist()

        # to be set: goal
        self._goal_text = None  # str
        self._goal_lits = None  # FrozenSet[Literal]

        # world (only for retrieving demonstrations in training)
        self._world = world

        # to be set: train / test
        self._is_train = False

        # whether trail is started, number of steps in trail
        self._trail_started = False
        self._num_steps = 0

        # random exploration prob
        self._random_act_prob = agent_cfg.random_act_prob

    def __str__(self):
        return self.__class__.__name__

    def set_mode(self, is_train: bool):
        # switch mode only when the previous trail is finished
        assert not self._trail_started, "Cannot switch mode in the middle of a trail!"
        self._is_train = is_train

        logger.info(f"Agent mode set: is_train is {is_train}.")

    def step(
        self, pre_observation: Dict[str, Any], agent_action: AgentAction, env_feedback: EnvFeedback
    ):
        """
        Update agent state based on env feedback.
        """
        assert self._trail_started, "Agent not initialized! Run 'reset_trail' first."

        # save to current traj
        self._current_traj.action_history.append(agent_action)
        self._current_traj.state_history.append(env_feedback)
        self._current_traj.goal_achieved = env_feedback.goal_achieved

        # clear plan buffer to replan if failed
        if not env_feedback.success:
            self._plan = []

        self._num_steps += 1

    @abc.abstractmethod
    def act(self, observation: Dict[str, Any]):
        """
        Predict action.
        """
        raise NotImplementedError("Override me!")

    @property
    def agent_knowledge(self):
        """
        Current agent knowledge.
        """
        pass

    @agent_knowledge.setter
    def agent_knowledge(self, loaded_agent_knowledge: Dict[str, Any]):
        """
        Set agent knowledge.
        """
        pass

    def save_agent_knowledge(self, save_dir: PosixPath):
        """
        Save current agent knowledge.
        """
        if self.agent_knowledge is not None:
            logger.info(f"Save agent knowledge to {save_dir}.")
            save_npz(self.agent_knowledge, save_dir / "agent_knowledge.npz")

    def load_agent_knowledge(self):
        """
        Load saved agent knowledge.
        """
        if self._load_path is not None and self._load_path.exists():
            logger.info(f"Load agent knowledge from {self._load_path}.")
            # load knowledge
            self.agent_knowledge = load_npz(self._load_path / "agent_knowledge.npz")
        else:
            logger.warning(f"Agent knowledge path {self._load_path} does not exist! Skip loading.")

    def reset_trail(self, goal_spec: str, init_obs: Dict[str, Any], *args, **kwargs):
        """
        Reset agent to process new epoch.
        """
        assert self._trail_started is False, "Cannot reset trail in the middle of a trail!"

        # reset goal
        self._goal_text = goal_spec

        # set initialized & num steps
        self._trail_started = True
        self._num_steps = 0

        # reset plan buffer
        self._plan = []

        # reset current traj
        self._current_traj = Trajectory(
            goal_text=goal_spec, entities=init_obs["entities"]
        )
        init_env_feedback = EnvFeedback(observation=init_obs)
        self._current_traj.state_history.append(init_env_feedback)

    def flag_end_of_trail(self, *args, **kwargs):
        """
        Flag the end of current trail.
        """
        self._trail_started = False
        logger.info(colored("End of trail!"))
