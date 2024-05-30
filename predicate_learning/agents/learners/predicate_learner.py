from pathlib import Path
from termcolor import colored
from typing import Any, List, Dict, Tuple, Union, Set, FrozenSet
import itertools
import logging
import re
import traceback
import copy
import json
import numpy as np

from predicate_learning.predicate_gym.api_wrapper import PerceptionAPIWrapper

from predicate_learning.utils.common_util import AgentAction, EnvFeedback
from predicate_learning.utils.io_util import load_txt
from predicate_learning.utils.general_util import Timer

from predicate_learning.utils.llm_util import LLMBase
from predicate_learning.utils.llm_util import (
    create_literal_from_raw,
    parse_util_functions_from_text,
    parse_predicate_functions_from_text,
    textualize_observation,
    create_predicate_from_raw,
    pddl_literal_to_text,
    get_predicate_name_from_raw,
)

from predicate_learning.utils.pddl_util import DEFAULT_TYPE
from pddlgym.structs import Literal, Predicate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: change all json.load to safe loading


class PredicateLearner(LLMBase):
    """
    Invent, ground and correct predicates from human inputs & demonstrations.
    """

    def __init__(
        self,
        domain_desc: str,
        prompt_dir: str,
        example_predicate_prompt_file: str,
        perception_api_wrapper: PerceptionAPIWrapper,
        use_gpt_4: bool = True,
    ):
        super().__init__(use_gpt_4=use_gpt_4)

        # load prompt templates
        prompt_template_folder = Path(prompt_dir)

        # load invent predicates prompts
        self._predicates_from_goal_prompt = load_txt(
            prompt_template_folder / "llm_predicate_from_goal_template.txt"
        ).replace("{domain_desc}", domain_desc)
        # self._predicates_from_failure_prompt = load_txt(
        #     prompt_template_folder / "llm_predicate_from_failure_template.txt"
        # ).replace("{domain_desc}", domain_desc)
        self._predicates_from_failure_prompt = load_txt(
            prompt_template_folder / "llm_predicate_from_failure_template_simple.txt"
        ).replace("{domain_desc}", domain_desc)
        self._predicates_from_non_goal_prompt = load_txt(
            prompt_template_folder / "llm_predicate_from_non_goal_template.txt"
        ).replace("{domain_desc}", domain_desc)

        # load predicate grounding / correction prompts
        self._predicate_grounding_prompt = load_txt(
            prompt_template_folder / "llm_predicate_grounding_template.txt"
        ).replace("{domain_desc}", domain_desc)
        self._predicate_correction_prompt = load_txt(
            prompt_template_folder / "llm_predicate_correction_template.txt"
        ).replace("{domain_desc}", domain_desc)
        # self._predicate_correction_consistency_prompt = load_txt(
        #     prompt_template_folder / "llm_predicate_correction_consistency_template.txt"
        # ).replace("{domain_desc}", domain_desc)
        self._predicate_correction_exec_error_prompt = load_txt(
            prompt_template_folder / "llm_predicate_correction_exec_error_template.txt"
        ).replace("{domain_desc}", domain_desc)

        # load parse goal lits prompt
        self._parse_goal_prompt = load_txt(
            prompt_template_folder / "llm_parse_goal_into_lits_template.txt"
        ).replace("{domain_desc}", domain_desc)

        # perception api wrapper
        self.perception_api_wrapper = perception_api_wrapper

        # primary memory: learned predicates
        self._predicates = LearnedPredicates()

        # load util functions
        example_predicate_prompt_file = Path(example_predicate_prompt_file)
        loaded_python_functions = load_txt(example_predicate_prompt_file)
        util_functions = parse_util_functions_from_text(loaded_python_functions)
        context = loaded_python_functions.split("# Utility functions:")[0]
        self._predicates.add_context_code(context)
        self._predicates.add_util_functions(util_functions)

        # load example predicates
        parsed_predicates, new_functions, dependency = parse_predicate_functions_from_text(
            loaded_python_functions, known_predicate_names=[]
        )
        # todo: seems that we don't need essential any longer
        self._predicates.add_predicates(
            parsed_predicates, new_functions, dependency, essential=True
        )

        # general knowledge
        self._general_knowledge = []

        # current goal lits
        self._current_goal_lits = {}

    def get_predicates_raw(self, include_negative: bool = False) -> List[str]:
        return self._predicates.get_available_predicates_raw(include_negative=include_negative)

    def get_predicates_text(self, include_negative: bool = False) -> str:
        return self._predicates.get_available_predicates_text(include_negative=include_negative)

    def learn_predicates_from_goal(
        self, goal: str, observation: Dict[str, Any]
    ) -> Tuple[FrozenSet[Literal], int]:
        # get entities
        entities = observation["entities"]

        num_llm_calls = 0
        # invent predicates based on goal
        pddl_goal_lits = self._invent_predicates_from_goal(entities, goal)
        num_llm_calls += 1

        # ground new predicates
        if not self._predicates.check_predicates_grounding_func():
            generate_func_llm_calls = self._generate_predicates_grounding_func(observation)
            num_llm_calls += generate_func_llm_calls

        return pddl_goal_lits, num_llm_calls

    def learn_predicates_from_failure(
        self, failed_action: AgentAction, env_feedback: EnvFeedback
    ) -> Tuple[FrozenSet[Literal], int]:
        # get info
        observation = env_feedback.observation
        entities = observation["entities"]
        failure_explain = env_feedback.failure_explain

        num_llm_calls = 0
        # invent predicates, infer literals
        assert len(failure_explain) > 0, "No failure explain!"
        raw_current_lits, action_preconds = self._invent_predicates_from_failure(
            entities,
            failure_explain,
            failed_action,
        )
        num_llm_calls += 1

        # ground new predicates
        if not self._predicates.check_predicates_grounding_func():
            generate_func_llm_calls = self._generate_predicates_grounding_func(observation)
            num_llm_calls += generate_func_llm_calls

        # correct grounding functions
        if len(raw_current_lits) > 0 and set(observation["objects"].keys()) == set(entities):
            correct_func_llm_calls = self._correct_predicates_grounding_func(
                observation, raw_current_lits
            )
            num_llm_calls += correct_func_llm_calls

        return action_preconds, num_llm_calls

    def learn_predicates_from_non_goal(self, env_feedback: EnvFeedback) -> int:
        # get info
        observation = env_feedback.observation
        entities = observation["entities"]
        non_goal_explain = env_feedback.non_goal_explain

        num_llm_calls = 0
        # invent predicates (optional), infer literals
        assert len(non_goal_explain) > 0, "No non-goal explain!"
        raw_current_lits = self._invent_predicates_from_non_goal(entities, non_goal_explain)
        num_llm_calls += 1

        # ground new predicates
        if not self._predicates.check_predicates_grounding_func():
            generate_func_llm_calls = self._generate_predicates_grounding_func(observation)
            num_llm_calls += generate_func_llm_calls

        # correct grounding functions
        # import pdb; pdb.set_trace()
        if len(raw_current_lits) > 0 and set(observation["objects"].keys()) == set(entities):
            corect_func_llm_calls = self._correct_predicates_grounding_func(
                observation, raw_current_lits
            )
            num_llm_calls += corect_func_llm_calls

        return num_llm_calls

    def learn_predicates_from_goal_achieved(self, env_feedback: EnvFeedback) -> int:
        # get info
        observation = env_feedback.observation

        num_llm_calls = 0
        # ground new predicates
        if not self._predicates.check_predicates_grounding_func():
            generate_func_llm_calls = self._generate_predicates_grounding_func(observation)
            num_llm_calls += generate_func_llm_calls

        # correct grounding functions
        if len(self._current_goal_lits) > 0 and set(observation["objects"].keys()) == set(
            observation["entities"]
        ):
            correct_func_llm_calls = self._correct_predicates_grounding_func(
                observation, self._current_goal_lits
            )
            num_llm_calls += correct_func_llm_calls

        return num_llm_calls

    def learn_predicates_from_success_execution(
        self, pre_observation: Dict[str, Any], precond_lits: FrozenSet[Literal]
    ) -> int:
        current_lits = {}
        for lit in precond_lits:
            text_lit, val = self._predicates.find_matched_grounded_literal(lit)
            current_lits[text_lit] = val

        logger.debug("Learn from success execution:")
        logger.debug(str(precond_lits))
        logger.debug(str(current_lits))

        num_llm_calls = 0
        # correct grounding functions
        if len(current_lits) > 0 and set(pre_observation["objects"].keys()) == set(
            pre_observation["entities"]
        ):
            corect_func_llm_calls = self._correct_predicates_grounding_func(
                pre_observation, current_lits
            )
            num_llm_calls += corect_func_llm_calls

        return num_llm_calls

    def parse_literals(
        self,
        observation: Dict[str, Any],
        perception_api_wrapper: PerceptionAPIWrapper,
        return_pddl_lits: bool = False,
        essential_only: bool = False,
        include_negative_predicates: bool = False,
    ) -> Union[Dict[str, bool], Set[Literal]]:
        return self._predicates.parse_literals(
            observation,
            perception_api_wrapper,
            return_pddl_lits=return_pddl_lits,
            essential_only=essential_only,
            include_negative_predicates=include_negative_predicates,
        )

    def parse_goal(self, observation: Dict[str, Any], goal: str) -> FrozenSet[Literal]:
        # get entities
        entities = observation["entities"]

        # instantiate prompts
        parse_goal_prompt = self._parse_goal_prompt.replace("{entities}", str(entities))
        parse_goal_prompt = parse_goal_prompt.replace(
            "{predicates}", self._predicates.get_available_predicates_text(include_negative=True)
        )
        parse_goal_prompt = parse_goal_prompt.replace("{goal_spec}", goal)

        # feed into llm
        llm_output = json.loads(self.prompt_llm(parse_goal_prompt, force_json=True))

        # pddl goal lits
        goal_lits = set()
        for raw_lit in llm_output["Symbolic goal"]:
            # create pddl lit
            lit = create_literal_from_raw(
                raw_lit,
                predicates=self._predicates.get_pddl_predicates(),
                skip_unknown=True,
            )
            if lit is not None:
                goal_lits.add(lit)

        logger.info(f"Goal literals: {llm_output['Symbolic goal']}")
        return frozenset(goal_lits)

    @property
    def pddl_predicates(self) -> Dict[str, Predicate]:
        return self._predicates.get_pddl_predicates()

    @property
    def predicates_functions(self) -> str:
        return self._predicates.get_all_func_text()

    @property
    def learned_predicates(self) -> "LearnedPredicates":
        return self._predicates

    @learned_predicates.setter
    def learned_predicates(self, learned_predicates: "LearnedPredicates") -> None:
        self._predicates = learned_predicates

    @property
    def general_knowledge(self) -> List[str]:
        return self._general_knowledge

    @general_knowledge.setter
    def general_knowledge(self, general_knowledge: List[str]) -> None:
        self._general_knowledge = general_knowledge

    def _invent_predicates_from_goal(self, entities: List[str], goal: str) -> FrozenSet[Literal]:
        logger.info(colored(f"Inventing predicates from goal: {goal}", "yellow"))
        # instantiate prompts
        predicates_from_goal_prompt = self._predicates_from_goal_prompt.replace(
            "{entities}", str(entities)
        )
        predicates_from_goal_prompt = predicates_from_goal_prompt.replace(
            "{predicates}", self._predicates.get_available_predicates_text()
        )
        predicates_from_goal_prompt = predicates_from_goal_prompt.replace("{goal_spec}", goal)

        # feed into llm
        llm_output = json.loads(self.prompt_llm(predicates_from_goal_prompt, force_json=True))

        # save to predicates
        self._predicates.add_predicates(llm_output["Invented predicates"], essential=True)

        # save to current goal lit
        self._current_goal_lits = llm_output["Symbolic goal"]

        # pddl goal lits (positive only)
        goal_lits = set()
        for raw_lit, lit_val in llm_output["Symbolic goal"].items():
            pred_name = get_predicate_name_from_raw(raw_lit)

            # if negative, set corresponding predicate to include_predicate & create new predicate
            if not lit_val:
                raw_lit = "not_" + raw_lit
                self._predicates.set_include_negative(pred_name)

            # create pddl lit (positive only)
            pos_lit = create_literal_from_raw(
                raw_lit, predicates=self._predicates.get_pddl_predicates()
            )
            goal_lits.add(pos_lit)

        logger.info(f"Invented predicates: {llm_output['Invented predicates']}")
        logger.info(f"Goal literals: {goal_lits}")
        return frozenset(goal_lits)

    def _invent_predicates_from_failure(
        self, entities: List[str], failure_explain: str, failed_action: AgentAction
    ) -> Tuple[Dict[str, bool], Tuple[Literal, FrozenSet[Literal]]]:
        failed_pddl_action = failed_action.action
        failed_action_text = pddl_literal_to_text(failed_pddl_action)

        logger.info(
            colored(f"Inventing predicates from failure explanation: {failure_explain}", "yellow")
        )
        # instantiate prompts
        predicates_from_failure_prompt = self._predicates_from_failure_prompt.replace(
            "{entities}", str(entities)
        )
        predicates_from_failure_prompt = predicates_from_failure_prompt.replace(
            "{predicates}", self._predicates.get_available_predicates_text()
        )
        predicates_from_failure_prompt = predicates_from_failure_prompt.replace(
            "{general_knowledge}", str(self._general_knowledge)
        )
        predicates_from_failure_prompt = predicates_from_failure_prompt.replace(
            "{failed_action}", failed_action_text
        )
        predicates_from_failure_prompt = predicates_from_failure_prompt.replace(
            "{failure_explain}", failure_explain
        )

        # feed into llm
        llm_output = json.loads(self.prompt_llm(predicates_from_failure_prompt, force_json=True))

        # save new predicates
        self._predicates.add_predicates(llm_output["2. Invented predicates"], essential=True)

        # save general knowledge
        new_general_knowledge = llm_output["5. New general knowledge"]
        if len(new_general_knowledge) > 0:
            self._general_knowledge.append(new_general_knowledge)

        # convert precondition literals
        precondition_lits = set()
        action_preconditions = llm_output["3. New action preconditions"]
        lifted_action = action_preconditions["action"]
        preconditions = action_preconditions["new preconditions"]

        lifted_action = create_literal_from_raw(
            lifted_action,
            predicates={failed_pddl_action.predicate.name: failed_pddl_action.predicate},
        )
        for raw_lit, lit_val in preconditions.items():
            pred_name = get_predicate_name_from_raw(raw_lit)

            # if negative, set corresponding predicate to include_predicate & create new predicate
            if not lit_val:
                raw_lit = "not_" + raw_lit
                self._predicates.set_include_negative(pred_name)

            # create pddl lit
            pos_lit = create_literal_from_raw(
                raw_lit, predicates=self._predicates.get_pddl_predicates()
            )
            precondition_lits.add(pos_lit)

        logger.info(f"Invented predicates: {llm_output['2. Invented predicates']}")
        logger.info(f"Precondition literals: {precondition_lits}")
        logger.info(f"New general knowledge: {llm_output['5. New general knowledge']}")
        logger.info(f"Current literals: {llm_output['4. Current state literals']}")
        return llm_output["4. Current state literals"], (
            lifted_action,
            frozenset(precondition_lits),
        )

    def _invent_predicates_from_non_goal(
        self, entities: List[str], non_goal_explain: str
    ) -> Dict[str, bool]:
        logger.info(
            colored(f"Inventing predicates from non goal explanation: {non_goal_explain}", "yellow")
        )
        # instantiate prompts
        predicates_from_non_goal_prompt = self._predicates_from_non_goal_prompt.replace(
            "{entities}", str(entities)
        )
        predicates_from_non_goal_prompt = predicates_from_non_goal_prompt.replace(
            "{predicates}", self._predicates.get_available_predicates_text()
        )
        predicates_from_non_goal_prompt = predicates_from_non_goal_prompt.replace(
            "{human_explain}", non_goal_explain
        )

        # feed into llm
        llm_output = json.loads(self.prompt_llm(predicates_from_non_goal_prompt, force_json=True))
        logger.info(f"Current literals: {llm_output['Current state']}")
        return llm_output["Current state"]

    def _generate_predicates_grounding_func(
        self, observation: Dict[str, Any], max_iter: int = 3
    ) -> int:
        """
        Generate python functions to ground predicates.
        Iteratively prompt LLM until all predicates are grounded.
        Note that we feed all predicates to ground in one trail - usually there will not be many predicates to ground each step.
        """
        logger.info(colored(f"Grounding predicates...", "yellow"))

        num_llm_calls = 0
        grounding_iter = 0
        while not self._predicates.check_predicates_grounding_func():
            logger.info(f"Iter {grounding_iter} / {max_iter}:")

            # prepare prompt
            known_functions = self._predicates.get_all_func_text()
            predicate_grounding_prompt = self._predicate_grounding_prompt.replace(
                "{known_functions}", known_functions
            )
            predicate_grounding_prompt = predicate_grounding_prompt.replace(
                "{observation}", textualize_observation(observation)
            )
            predicate_grounding_prompt = predicate_grounding_prompt.replace(
                "{new_predicates}", self._predicates.get_predicates_to_ground_text()
            )

            # prompt llm and parse output
            llm_output = self.prompt_llm(predicate_grounding_prompt)
            parsed_predicates, new_functions, dependency = parse_predicate_functions_from_text(
                llm_output,
                known_predicate_names=self._predicates.get_all_grounded_predicate_names(),
            )
            util_functions = parse_util_functions_from_text(llm_output)
            num_llm_calls += 1

            # save new predicates & grounding functions & dependency & util functions
            self._predicates.add_predicates(parsed_predicates, new_functions, dependency)
            self._predicates.add_util_functions(util_functions)

            # check execution error
            fix_err_llm_calls = self._correct_predicates_grounding_func_exec_error(observation)
            num_llm_calls += fix_err_llm_calls

            grounding_iter += 1
            if grounding_iter >= max_iter:
                raise Exception(f"Fail to ground all functions within {max_iter} iterations!")

        return num_llm_calls

    def _correct_predicates_grounding_func_exec_error(
        self, observation: Dict[str, Any], max_iter: int = 3
    ) -> int:
        logger.info(colored(f"Correcting predicates based on execution error...", "yellow"))

        num_llm_calls = 0
        correct_iter = 0
        while correct_iter < max_iter:
            logger.info(f"Iter {correct_iter} / {max_iter}:")
            error = ""
            full_traceback = ""
            # try parse current literals
            try:
                self.parse_literals(observation, self.perception_api_wrapper)
            except Exception as e:
                error = str(e)
                full_traceback = str(traceback.format_exc())

            if len(error) == 0:
                logger.info(colored("No execution error found!", "green"))
                break

            # instantiate prompt
            known_functions = self._predicates.get_all_func_text()
            predicate_correction_exec_error_prompt = (
                self._predicate_correction_exec_error_prompt.replace(
                    "{known_functions}", known_functions
                )
            )
            predicate_correction_exec_error_prompt = predicate_correction_exec_error_prompt.replace(
                "{observation}", textualize_observation(observation)
            )
            predicate_correction_exec_error_prompt = predicate_correction_exec_error_prompt.replace(
                "{error}", error
            )
            predicate_correction_exec_error_prompt = predicate_correction_exec_error_prompt.replace(
                "{trace}", full_traceback
            )

            # prompt llm and parse output
            llm_output = self.prompt_llm(predicate_correction_exec_error_prompt)
            split = re.split("1.|2.", llm_output)
            reasoning = split[1].strip()
            print(reasoning)

            parsed_predicates, new_functions, dependency = parse_predicate_functions_from_text(
                llm_output,
                known_predicate_names=self._predicates.get_all_grounded_predicate_names(),
            )
            util_functions = parse_util_functions_from_text(llm_output)
            num_llm_calls += 1

            # save new predicates & grounding functions & dependency & util functions
            self._predicates.add_predicates(parsed_predicates, new_functions, dependency)
            self._predicates.add_util_functions(util_functions)
            assert self._predicates.check_predicates_grounding_func(), "Predicate not grounded!"

            correct_iter += 1

        return num_llm_calls

    def _correct_predicates_grounding_func(
        self,
        observation: Dict[str, Any],
        current_lits: Dict[str, bool],
        max_iter: int = 3,
    ) -> int:
        # Note that since modifying one function may affect others, so we iteratively check mismatch and make correction
        logger.info(colored(f"Correcting predicates...", "yellow"))

        num_llm_calls = 0
        correct_iter = 0
        while correct_iter < max_iter:
            logger.info(f"Iter {correct_iter} / {max_iter}:")
            # parse current literals based on the current grounding functions
            parsed_lits = self.parse_literals(observation, self.perception_api_wrapper)

            print([lit for lit in parsed_lits if parsed_lits[lit]])
            # import pdb

            # pdb.set_trace()

            # check mismatch & sort parsed_lits
            corection_list = []
            predicate_list = []
            mismatch_exist = False
            for lit, val in current_lits.items():
                # check mismatch
                if lit not in parsed_lits:
                    logger.warning(f"Predicate {str(lit)} not in parsed literals!")
                    continue

                if parsed_lits[lit] != val:
                    corection_list.append(
                        f"{lit} should be {val}, but the function predicts {parsed_lits[lit]}"
                    )
                    mismatch_exist = True
                    # else:
                    #     corection_list.append(f"{lit} is {val}, which is correct")

                    # record predicate
                    predicate_name = lit.split("(")[0]
                    if predicate_name not in predicate_list:
                        predicate_list.append(predicate_name)

            correction_text = f"{'; '.join(corection_list)}."
            logger.debug(correction_text)

            if not mismatch_exist:
                logger.info(colored("All corrections completed!", "green"))
                break

            # import pdb; pdb.set_trace()
            # instantiate prompt
            known_functions = self._predicates.get_all_relevant_func_text(predicate_list)
            predicate_correction_prompt = self._predicate_correction_prompt.replace(
                "{known_functions}", known_functions
            )
            predicate_correction_prompt = predicate_correction_prompt.replace(
                "{observation}", textualize_observation(observation)
            )
            predicate_correction_prompt = predicate_correction_prompt.replace(
                "{correction}", correction_text
            )

            # prompt llm and parse output
            llm_output = self.prompt_llm(predicate_correction_prompt)
            # ['', '{reasoning}', '']
            split = re.split("1.|2.", llm_output)
            reasoning = split[1].strip()
            print(reasoning)
            parsed_predicates, new_functions, dependency = parse_predicate_functions_from_text(
                llm_output,
                known_predicate_names=self._predicates.get_all_grounded_predicate_names(),
            )
            util_functions = parse_util_functions_from_text(llm_output)
            num_llm_calls += 1

            # import pdb; pdb.set_trace()

            # save new predicates & grounding functions & dependency & util functions
            self._predicates.add_predicates(parsed_predicates, new_functions, dependency)
            self._predicates.add_util_functions(util_functions)
            assert self._predicates.check_predicates_grounding_func(), "Predicate not grounded!"

            # import pdb; pdb.set_trace()

            # check execution error
            fix_err_llm_calls = self._correct_predicates_grounding_func_exec_error(observation)
            num_llm_calls += fix_err_llm_calls

            correct_iter += 1

        return num_llm_calls

    # def _correct_predicates_grounding_func_consistency(
    #     self, observation: Dict[str, Any], max_iter: int = 3
    # ):
    #     logger.info(colored(f"Correcting predicates based on consistency...", "yellow"))

    #     correct_iter = 0
    #     while correct_iter < max_iter:
    #         logger.info(f"Iter {correct_iter} / {max_iter}:")
    #         # parse current literals based on the current grounding functions (positive only)
    #         literals, pddl_literals = self.parse_literals(observation, return_pddl_lits=True)
    #         lits_text = ", ".join([str(lit) for lit in pddl_literals])

    #         print(lits_text)
    #         import pdb

    #         pdb.set_trace()

    #         # instantiate prompt
    #         known_functions = self._predicates.get_all_func_text()
    #         predicate_correction_consistency_prompt = (
    #             self._predicate_correction_consistency_prompt.replace(
    #                 "{known_functions}", known_functions
    #             )
    #         )
    #         predicate_correction_consistency_prompt = (
    #             predicate_correction_consistency_prompt.replace("{parsed_literals}", lits_text)
    #         )

    #         # prompt llm and parse output
    #         llm_output = self.prompt_llm(predicate_correction_consistency_prompt)
    #         # ['', '{yes/no}', '{reasoning}', ...]
    #         split = re.split("1.|2.|3.", llm_output)
    #         is_inconsistent = split[1].strip()
    #         if is_inconsistent == "No":
    #             logger.info(colored("No inconsistency found!", "green"))
    #             break

    #         reasoning = split[2].strip()
    #         print(reasoning)
    #         parsed_predicates, new_functions, dependency = parse_predicate_functions_from_text(
    #             llm_output,
    #             known_predicate_names=self._predicates.get_all_grounded_predicate_names(),
    #         )

    #         logger.info(f"parsed_predicates: {parsed_predicates}")
    #         logger.info(f"new_functions: {new_functions}")
    #         logger.info(f"dependency: {dependency}")

    #         # save new predicates & grounding functions & dependency
    #         self._predicates.add_predicates(parsed_predicates, new_functions, dependency)
    #         assert self._predicates.check_predicates_grounding_func(), "Predicate not grounded!"

    #         import pdb

    #         pdb.set_trace()

    #         correct_iter += 1


class LearnedPredicates:
    def __init__(self):
        # Dict[predicate_raw, explanation]
        self._raw_explain_mapping = {}
        # Dict[predicate_name, LearnedPredicate]
        self._predicates = {}
        # Dict[predicate_raw, predicate_name]
        self._raw_name_mapping = {}
        # Dict[predicate_name, List[predicate_name]]
        self._dependency = {}
        # Set[predicate_name]
        self._essential_predicates = set()

        # Dict[util_name, util_func]
        self._util_functions = {}
        self._context_code = ""

    def add_predicates(
        self,
        predicates_dict: Dict[str, str],
        predicates_func_dict: Dict[str, str] = {},
        predicates_dependency_dict: Dict[str, List[str]] = {},
        essential: bool = False,
    ) -> None:
        for raw, explanation in predicates_dict.items():
            predicate = LearnedPredicate(raw, explanation)

            # if new predicates, create & add LearnedPredicate
            if raw not in self._raw_name_mapping:
                self._predicates[predicate.name] = predicate
                self._raw_name_mapping[raw] = predicate.name
                self._raw_explain_mapping[raw] = explanation

            # essential
            if essential:
                self._essential_predicates.add(predicate.name)

        # set grounding functions (allow overwrite)
        for raw, func in predicates_func_dict.items():
            assert raw in self._raw_name_mapping, f"Predicate {raw} not in predicates!"
            name = self._raw_name_mapping[raw]
            self._predicates[name].func = func

        # set dependency (allow overwrite)
        for name, dependency in predicates_dependency_dict.items():
            assert name in self._predicates, f"Predicate {name} not in predicates!"
            for name_dep in dependency:
                assert name_dep in self._predicates, f"Unknown dependent predicate {name_dep}!"
            self._dependency[name] = dependency

    def add_util_functions(self, util_functions: Dict[str, str]) -> None:
        # allow overwrite
        for name, func in util_functions.items():
            self._util_functions[name] = func

    def add_context_code(self, context: str) -> None:
        # allow overwrite
        self._context_code = context

    def check_predicates_grounding_func(self) -> bool:
        for predicate in self._predicates.values():
            if not predicate.has_grounding:
                return False
        return True

    def parse_literals(
        self,
        observation: Dict[str, Any],
        perception_api_wrapper: PerceptionAPIWrapper,
        return_pddl_lits: bool = False,
        predicate_name: str = None,
        include_relevant_lits: bool = False,
        essential_only: bool = False,
        include_negative_predicates: bool = False,
    ) -> Union[Dict[str, bool], Set[Literal]]:
        """
        Parse literals with predicates.
        Args:
            observation: Dict[str, Any]
            perception_api_wrapper: PerceptionAPIWrapper
            return_pddl_lits: whether to return pddl literals
            predicate_name: if not None, only parse this predicate
            include_relevant_lits: whether to include relevant literals (if predicate_name is not None)
            positive_only: only consider predicates that mark essential
            include_negative_predicates: include invented negative predicates
        Returns:
            literals: Dict[str, bool]
            pddl_literals: List[Literal]
        """
        # entities
        entities = observation["entities"]

        # update observation
        perception_api_wrapper.update_observation(observation)

        # prepare global scope
        global_scope = {"np": np}
        global_scope.update({name: eval(name) for name in ["List", "Dict", "Set", "FrozenSet"]})
        global_scope.update(perception_api_wrapper.get_all_apis())

        # gather grounding functions & execute
        util_funcs = self.get_all_util_func_text()
        if predicate_name is not None:
            predicate_funcs = self.get_predicate_func_text(predicate_name, include_relevant_lits)
        else:
            predicate_funcs = self.get_all_predicate_func_text()

        # execute util functions and predicate functions (local)
        exec(util_funcs, global_scope)
        exec(predicate_funcs, global_scope)

        # predicate to consider
        predicates_to_parse = []
        if predicate_name is not None:
            predicates_to_parse.append(self._predicates[predicate_name])
            if include_relevant_lits:
                for dep_name in self._dependency[predicate_name]:
                    predicates_to_parse.append(self._predicates[dep_name])

        else:
            for pred_name, predicate in self._predicates.items():
                if essential_only and pred_name not in self._essential_predicates:
                    continue
                predicates_to_parse.append(predicate)

        # parse literals
        literals = {}
        if return_pddl_lits:
            pddl_literals = set()
        for predicate in predicates_to_parse:
            # parse all literals of this predicate
            arity = predicate.predicate.arity

            python_func = global_scope[predicate.name]
            for combine in itertools.permutations(entities, arity):
                # only parse detected objects
                all_valid = True
                for ent in combine:
                    if ent not in observation["objects"].keys():
                        all_valid = False
                        break
                if not all_valid:
                    continue

                literal_text = f"{predicate.name}({', '.join(combine)})"
                value = python_func(*combine)
                literals[literal_text] = value

                # only keep positive literals in state
                if return_pddl_lits:
                    if value:
                        pddl_lit = predicate.predicate(*[DEFAULT_TYPE(var) for var in combine])
                        pddl_literals.add(pddl_lit)

                    elif predicate.include_negative and include_negative_predicates:
                        pddl_lit = predicate.negative_predicate(
                            *[DEFAULT_TYPE(var) for var in combine]
                        )
                        pddl_literals.add(pddl_lit)

        if return_pddl_lits:
            return pddl_literals
        else:
            return literals

    def get_dependency(self, predicate_name: str) -> List[str]:
        return self._dependency[predicate_name]

    def get_available_predicates_text(self, include_negative: bool = False) -> str:
        predicates_text_dict = {}
        for raw, pred_name in self._raw_name_mapping.items():
            # add positive predicate
            predicates_text_dict[raw] = self._raw_explain_mapping[raw]
            # add negative predicate when needed
            if include_negative and self._predicates[pred_name].include_negative:
                neg_raw = self._predicates[pred_name].negative_raw
                predicates_text_dict[neg_raw] = f"negative to {raw}"

        return str(predicates_text_dict)

    def get_available_predicates_raw(
        self, include_negative: bool = False, essential_only: bool = False
    ) -> List[str]:
        predicates_raw = []
        for raw, pred_name in self._raw_name_mapping.items():
            if essential_only and pred_name not in self._essential_predicates:
                continue

            # add positive predicate
            predicates_raw.append(raw)
            # add negative predicate when needed
            if include_negative and self._predicates[pred_name].include_negative:
                neg_raw = self._predicates[pred_name].negative_raw
                predicates_raw.append(neg_raw)

        return predicates_raw

    def get_predicates_to_ground_text(self) -> str:
        predicates_to_ground = {}
        for raw, explain in self._raw_explain_mapping.items():
            name = self._raw_name_mapping[raw]
            if not self._predicates[name].has_grounding:
                predicates_to_ground[raw] = explain
        return str(predicates_to_ground)

    def get_predicate_func_text(
        self, predicate_list: List[str], include_relevant_func: bool = False
    ) -> str:
        func_list = [self._predicates[predicate_name].func for predicate_name in predicate_list]
        if include_relevant_func:
            # iteratively add dependent functions
            pred_list = copy.deepcopy(predicate_list)
            checked_pred = set()
            while len(pred_list) > 0:
                pred_to_check = pred_list.pop(0)
                checked_pred.add(pred_to_check)
                for dep_name in self._dependency[pred_to_check]:
                    if dep_name not in checked_pred:
                        pred_list.append(dep_name)
                        func_list.append(self._predicates[dep_name].func)

        return "# Predicates:\n" + "\n\n".join(func_list)

    def get_all_predicate_func_text(self) -> str:
        func_list = []
        for predicate in self._predicates.values():
            if predicate.has_grounding:
                func_list.append(predicate.func)
        return "# Predicates:\n" + "\n\n".join(func_list)

    def get_all_util_func_text(self) -> str:
        return "# Utility functions:\n" + "\n\n".join(self._util_functions.values())

    def get_all_func_text(self) -> str:
        return (
            self._context_code
            + "\n\n"
            + self.get_all_util_func_text()
            + "\n\n"
            + self.get_all_predicate_func_text()
        )

    def get_all_relevant_func_text(self, predicate_list: List[str]) -> str:
        return (
            self._context_code
            + "\n\n"
            + self.get_all_util_func_text()
            + "\n\n"
            + self.get_predicate_func_text(predicate_list, include_relevant_func=True)
        )

    def get_all_grounded_predicate_names(self) -> List[str]:
        return [
            predicate.name for predicate in self._predicates.values() if predicate.has_grounding
        ]

    def get_pddl_predicates(self, include_negative: bool = True) -> Dict[str, Predicate]:
        all_predicates = {}
        for predicate in self._predicates.values():
            all_predicates[predicate.name] = predicate.predicate
            if include_negative:
                all_predicates[predicate.negative_name] = predicate.negative_predicate
        return all_predicates

    def set_include_negative(self, pred_name: str):
        assert pred_name in self._predicates, "Unknown predicate!"
        self._predicates[pred_name].include_negative = True

    def find_matched_grounded_literal(self, pddl_literal: Literal) -> Tuple[str, bool]:
        for learned_predicate in self._predicates.values():
            if pddl_literal.predicate.name == learned_predicate.predicate.name:
                return (pddl_literal_to_text(pddl_literal), True)
            elif (
                learned_predicate.include_negative
                and pddl_literal.predicate.name == learned_predicate.negative_name
            ):
                new_literal = learned_predicate.predicate(*pddl_literal.variables)
                return (pddl_literal_to_text(new_literal), False)


class LearnedPredicate:
    def __init__(self, raw: str, explanation: str, func: str = None):
        # on(x, y)
        self._raw = raw
        # x is on y
        self._explanation = explanation
        self._func = func

        self._predicate = create_predicate_from_raw(raw)
        # on
        self._name = self._predicate.name

        # include its negative predicate
        self._include_negative = False
        self._neg_name = "not_" + self._name
        self._neg_raw = "not_" + raw
        self._neg_predicate = Predicate(
            self._neg_name, self._predicate.arity, self._predicate.var_types
        )

    @property
    def has_grounding(self) -> bool:
        return self._func is not None

    @property
    def func(self) -> str:
        assert self._func is not None, "The grounding function hasn't been set"
        return self._func

    @func.setter
    def func(self, func: str) -> None:
        assert func is not None or len(func) == 0, f"Invalid grounding function!"
        # check arity
        in_parenth = re.split("\(|\)", func)[1]
        if len(in_parenth) == 0:
            parsed_arity = 0
        elif in_parenth.count(",") == 0:
            parsed_arity = 1
        else:
            parsed_arity = in_parenth.count(",") + 1
        assert (
            parsed_arity == self._predicate.arity
        ), f"Wrong arity in grounding function! {self._predicate} with grounding function arity {parsed_arity}"
        self._func = func

    @property
    def raw(self) -> str:
        return self._raw

    @property
    def explanation(self) -> str:
        return self._explanation

    @property
    def predicate(self) -> Predicate:
        return self._predicate

    @property
    def name(self) -> str:
        return self._name

    @property
    def include_negative(self) -> bool:
        return self._include_negative

    @include_negative.setter
    def include_negative(self, include_neg: bool = True) -> bool:
        self._include_negative = include_neg

    @property
    def negative_name(self) -> str:
        return self._neg_name

    @property
    def negative_raw(self) -> str:
        return self._neg_raw

    @property
    def negative_predicate(self) -> Predicate:
        return self._neg_predicate
