import numpy as np
import logging
import traceback
from pathlib import Path
from termcolor import colored
from typing import List, Dict, Set, FrozenSet, Any

from predicate_learning.agents.base_agent import AGENT
from predicate_learning.agents.planners.lmp import LMPFGen, LMP

from predicate_learning.utils.io_util import load_txt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@AGENT.register()
class CaPAgent:
    """
    An agent that plan with LLM and learn nothing.
    """

    def __init__(self, world, agent_cfg, *args, **kwargs):
        # perception & action api wrapper
        self.perception_api_wrapper = world.perception_api_wrapper
        self.action_api_wrapper = world.action_api_wrapper

        # load prompts
        prompt_template_folder = Path(agent_cfg.prompt_dir)
        domain_folder = Path(agent_cfg.domain_dir)
        self._generate_function_prompt = load_txt(
            prompt_template_folder / "cap_generate_function_examples.py"
        )
        self._parse_objects_prompt = load_txt(
            prompt_template_folder / "cap_parse_objects_examples.py"
        )
        self._subpolicy_prompt = load_txt(domain_folder / str(world) / "cap_subpolicy_examples.py")
        self._main_policy_prompt = load_txt(domain_folder / str(world) / "cap_policy_examples.py")

        self._agent_cfg = agent_cfg

        # reset LMPs
        self.reset_lmps()

    def reset_lmps(self):
        # fixed vars
        fixed_vars = {"np": np}
        fixed_vars.update({name: eval(name) for name in ["List", "Dict", "Set", "FrozenSet"]})

        # variable vars
        variable_vars = {}
        variable_vars.update(self.perception_api_wrapper.get_all_apis())
        variable_vars.update(self.action_api_wrapper.get_all_apis())

        # creating the lowest-level function-generating LMP
        lmp_fgen = LMPFGen(
            self._generate_function_prompt,
            fixed_vars,
            variable_vars,
            use_gpt_4=self._agent_cfg.use_gpt_4,
        )

        # creating low-level LMPs
        variable_vars["parse_objects"] = LMP(
            "parse_objects",
            self._parse_objects_prompt,
            lmp_fgen,
            fixed_vars,
            variable_vars,
            maintain_session=False,
            return_val_name="ret_objects",
            use_gpt_4=self._agent_cfg.use_gpt_4,
        )

        variable_vars["execute_subpolicy"] = LMP(
            "execute_subpolicy",
            self._subpolicy_prompt,
            lmp_fgen,
            fixed_vars,
            variable_vars,
            maintain_session=False,
            use_gpt_4=self._agent_cfg.use_gpt_4,
        )

        # main policy LMP
        self._main_lmp = LMP(
            "main_policy",
            self._main_policy_prompt,
            lmp_fgen,
            fixed_vars,
            variable_vars,
            maintain_session=True,
            use_gpt_4=self._agent_cfg.use_gpt_4,
        )

    def __str__(self):
        return self.__class__.__name__

    def generate_and_run_policy(self, goal_spec: str):
        code_str = ""
        error = ""
        full_traceback = ""
        try:
            _, code_str = self._main_lmp(goal_spec, return_srcs=True)
        except Exception as e:
            error = str(e)
            full_traceback = str(traceback.format_exc())
            logger.error(colored(f"Error: {error}", "red"))
            logger.error(colored(f"Trace: {full_traceback}", "red"))
            # import pdb; pdb.set_trace()

        return code_str, (error, full_traceback)

    def reset_trail(self, *args, **kwargs):
        self.reset_lmps()
        # self._main_lmp.clear_exec_hist()

    def save_agent_knowledge(self, *args, **kwargs):
        pass

    def load_agent_knowledge(self, *args, **kwargs):
        pass

    def set_mode(self, *args, **kwargs):
        pass
