import logging
import cv2
import numpy as np
from pathlib import Path
from termcolor import colored
from predicate_learning.agents import create_agent
from predicate_learning.predicate_gym import create_world

from predicate_learning.utils.common_util import Trajectory
from predicate_learning.utils.io_util import dump_json, mkdir, save_npz
from predicate_learning.utils.registry import Registry

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

RUNNER = Registry("Runner")


def create_runner(runner_name, *args, **kwargs):
    return RUNNER.get(runner_name)(*args, **kwargs)


class _BaseRunner:
    def __init__(self, cfg):
        self.cfg = cfg

        self.world = create_world(
            cfg.world.world, world_cfg=cfg.world, use_diverse_feedback=cfg.use_diverse_feedback
        )
        self.agent = create_agent(cfg.agent.agent, world=self.world, agent_cfg=cfg.agent)

        self.max_steps = cfg.max_steps
        self.save_dir = Path(cfg.save_dir)

        self.render = cfg.render

        # stop on failure during execution (only in test mode)
        self.max_failed_steps_on_test = cfg.max_failed_steps_on_test
        self.max_repeated_failed_steps = cfg.max_repeated_failed_steps

        # load agent knowledge
        self.load_agent_knowledge = cfg.load_agent_knowledge

        # world dir
        self.world_dir = (
            Path(__file__).resolve().parent / "predicate_gym" / "domains" / str(cfg.world.world)
        )

        logger.info(
            f"Run {str(self)} with world {str(cfg.world.world)} with agent {str(cfg.agent.agent)}"
        )
        logger.info(f"Save to path: {self.save_dir}")

    def __str__(self):
        return self.__class__.__name__

    def run_single_instance(self, instance_config, mode, dataset_name, instance_id):
        # reset env instance; not valid when the initial goal matches the goal already
        valid, init_env_fedback = self.world.reset_env(instance_config, self.render)
        if not valid:
            return None

        # reset trail for agent
        goal_text = instance_config["goal_text"]
        current_obs = init_env_fedback.observation
        agent_state = self.agent.reset_trail(goal_spec=goal_text, init_obs=current_obs)

        # record history
        traj = Trajectory(
            goal_text=goal_text,
            entities=current_obs["entities"],
            available_actions=self.world._pddl_domain.actions,
        )
        traj.state_history.append(init_env_fedback)
        traj.agent_state_history.append(agent_state)
        imgs = []

        if self.save_to_file:
            assert self.save_dir is not None, "Save dir is not specified!"
            episode_path = self.save_dir / dataset_name / str(instance_id)
            assert not episode_path.exists(), f"The episode path {episode_path} already exists!"
            mkdir(episode_path)

        # run episode
        num_steps = 0
        num_failed_actions = 0
        num_repeated_failed_actions = 0

        # initial state
        if "image" in current_obs:
            image = current_obs.pop("image")
            imgs.append(image)

        for step_idx in range(self.max_steps):
            logger.info(f"-------- Step {step_idx} --------")

            # propose action (action and query)
            agent_action = self.agent.act(current_obs)

            # step world & get env feedback
            env_feedback = self.world.step(agent_action)
            if "image" in env_feedback.observation:
                image = env_feedback.observation.pop("image")
                imgs.append(image)

            # post-process env feedback
            success = env_feedback.success
            goal_achieved = env_feedback.goal_achieved
            num_steps += 1
            num_failed_actions += success == False

            if success:
                num_repeated_failed_actions = 0
            else:
                num_repeated_failed_actions += 1

            if not success:
                logger.info(colored(f"Failed action: {agent_action.action}", "red"))
                logger.info(colored(f"Failed action: {num_failed_actions}", "red"))
                logger.info(
                    colored(f"Repeated failed action: {num_repeated_failed_actions}", "red")
                )
            else:
                logger.info(colored(f"Success action: {agent_action.action}", "green"))

            # update agent
            agent_state = self.agent.step(current_obs, agent_action, env_feedback)

            # record & update
            traj.state_history.append(env_feedback)
            traj.action_history.append(agent_action)
            traj.agent_state_history.append(agent_state)
            current_obs = env_feedback.observation

            # break when goal achieved or max_steps achieved
            if goal_achieved or num_repeated_failed_actions >= self.max_repeated_failed_steps:
                break
            if mode == "test" and num_failed_actions >= self.max_failed_steps_on_test:
                break

        # flag agent for end of trail
        self.agent.flag_end_of_trail()

        # destroy env
        self.world.destroy_env()

        episode_data = {
            "trajectory": traj,
            "num_steps": num_steps,
            "num_failed_actions": num_failed_actions,
            "goal_achieved": goal_achieved,
            "goal_lits_gt": np.asanyarray(instance_config["goal_lits_gt"], dtype=object),
            "goal_text": goal_text,
        }

        logger.info(
            f"step: {num_steps}, failed actions: {num_failed_actions}, goal_achieved: {goal_achieved}"
        )

        # save episode data to file
        if self.save_to_file:
            # save npz
            save_npz(episode_data, episode_path / "trajectory.npz")

            # save images
            for idx in range(len(imgs)):
                # convert into BGR for visualization
                cv2.imwrite(
                    str(episode_path / f"{idx}.png"),
                    cv2.cvtColor(imgs[idx], cv2.COLOR_RGB2BGR),
                )

            # save agent knowledge
            if mode == "train":
                self.agent.save_agent_knowledge(episode_path)

            logger.info(f"Save episode data at: {str(episode_path)}")

        return episode_data

    def run_instances(self, instances_dict, mode):
        eval_json = {}

        # run instances in multiple datasets
        for dataset_name, instances_json in instances_dict.items():
            eval_json[dataset_name] = {}

            goal_list = []
            step_list = []
            failed_action_list = []
            for instance_id, instance_config in instances_json.items():
                logger.info(f"-------------- {dataset_name} / {instance_id} --------------")
                episode_data = self.run_single_instance(
                    instance_config, mode, dataset_name=dataset_name, instance_id=instance_id
                )

                goal_list.append(episode_data["goal_achieved"])
                step_list.append(episode_data["num_steps"])
                failed_action_list.append(episode_data["num_failed_actions"])

            success_rate = np.mean(goal_list)
            avg_steps = np.mean(step_list)
            failed_actions = np.sum(failed_action_list).astype(float)

            logger.info(f"Success rate: {success_rate}")
            logger.info(f"Avg steps: {avg_steps}")
            logger.info(f"All failed actions: {failed_actions}")

            eval_json[dataset_name] = {
                "success_rate": success_rate,
                "avg_steps": avg_steps,
                "all_failed_actions": failed_actions,
            }
        return eval_json


@RUNNER.register()
class SingleModeRunner(_BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)

        # task mode: train / test
        self.mode = cfg.mode if hasattr(cfg, "mode") else "train"
        self.save_to_file = True

        # world-specific instance config
        self.instances_config_dict = cfg.world.instances_list[self.mode]

    def run(self):
        self.run_mode(self.mode, self.instances_config_dict)

    def run_mode(self, mode, instances_config_dict):
        logger.info(colored(f"Running mode: {mode}", "green"))

        # create / load instances
        if self.save_to_file:
            mkdir(self.world_dir / mode)

        instances_dict = {}
        for dataset_name, instances_config in instances_config_dict.items():
            instances_dict[dataset_name] = self.world.create_instances(
                **instances_config,
                instance_file=self.world_dir / mode / f"{dataset_name}.json",
                save_to_file=self.save_to_file,
                overwrite=self.cfg.overwrite_instances,
            )

        # set world & agent mode
        self.world.set_mode(is_train=True if mode == "train" else False)
        self.agent.set_mode(is_train=True if mode == "train" else False)

        # load agent state
        if mode == "test" or self.load_agent_knowledge:
            self.agent.load_agent_knowledge()

        # run instances
        eval_json = self.run_instances(instances_dict, mode)

        # always save eval
        dump_json(eval_json, self.save_dir / f"{mode}_eval.json")

        # save agent state
        if mode == "train":
            self.agent.save_agent_knowledge(self.save_dir)


@RUNNER.register()
class TrainTestRunner(SingleModeRunner):
    """
    Complete pipeline for training and testing.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_to_file = True

        # world-specific instance config
        self.train_instances = cfg.world.instances_list["train"]
        self.test_instances = cfg.world.instances_list["test"]

    def run(self):
        # run train
        self.run_mode("train", self.train_instances)

        # run test
        self.run_mode("test", self.test_instances)


@RUNNER.register()
class CaPSingleModeRunner(SingleModeRunner):
    def run_single_instance(self, instance_config, mode, dataset_name, instance_id):
        # reset env instance; not valid when the initial goal matches the goal already
        valid, init_env_fedback = self.world.reset_env(instance_config, self.render)
        if not valid:
            return None

        # reset trail for agent
        goal_text = instance_config["goal_text"]
        current_obs = init_env_fedback.observation
        self.agent.reset_trail()

        if self.save_to_file:
            assert self.save_dir is not None, "Save dir is not specified!"
            episode_path = self.save_dir / dataset_name / str(instance_id)
            assert not episode_path.exists(), f"The episode path {episode_path} already exists!"
            mkdir(episode_path)

        # run episode
        policy_code, error = self.agent.generate_and_run_policy(goal_text)
        goal_achieved = self.world.check_goal()

        # destroy env
        self.world.destroy_env()

        episode_data = {
            "policy_code": str(policy_code),
            "error": str(error),
            "num_steps": -1,
            "num_failed_actions": -1,
            "goal_achieved": goal_achieved,
            "goal_lits_gt": np.asanyarray(instance_config["goal_lits_gt"], dtype=object),
            "goal_text": goal_text,
        }

        logger.info(f"goal_achieved: {goal_achieved}")

        # save episode data to file
        if self.save_to_file:
            # save npz
            save_npz(episode_data, episode_path / "trajectory.npz")
            logger.info(f"Save episode data at: {str(episode_path)}")

        return episode_data
