from retry import retry
from pathlib import Path
from predicate_learning.utils.io_util import load_json
import openai
import abc
import logging


openai_key_folder = Path(__file__).resolve().parent.parent.parent / "openai_keys"
OPENAI_KEYS = load_json(openai_key_folder / "openai_key.json")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@retry(tries=5, delay=60)
def connect_openai(
    engine,
    messages,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    response_format,
):
    return openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        # stop=stop,
        response_format=response_format,
    )


class GPT_Chat:
    def __init__(
        self,
        engine,
        stop=None,
        max_tokens=4000,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ):
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

        # add key
        openai.api_key = OPENAI_KEYS["key"]
        if "org" in OPENAI_KEYS:
            openai.organization = OPENAI_KEYS["org"]
        if "proxy" in OPENAI_KEYS:
            openai.proxy = {
                "http": "http://" + OPENAI_KEYS["proxy"],
                "https": "https://" + OPENAI_KEYS["proxy"],
            }

    def get_response(
        self,
        prompt,
        messages=None,
        end_when_error=False,
        max_retry=2,
        temperature=0.0,
        force_json=False,
    ):
        conn_success, llm_output = False, ""
        if messages is not None:
            messages = messages
        else:
            messages = [{"role": "user", "content": prompt}]

        if force_json:
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}

        n_retry = 0
        while not conn_success:
            n_retry += 1
            if n_retry >= max_retry:
                break
            try:
                logger.info("[INFO] connecting to the LLM ...")

                response = connect_openai(
                    engine=self.engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.freq_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop,
                    response_format=response_format,
                )
                llm_output = response["choices"][0]["message"]["content"]
                conn_success = True
            except Exception as e:
                logger.info(f"[ERROR] LLM error: {e}")
                if end_when_error:
                    break
        return conn_success, llm_output


class LLMBase(abc.ABC):
    def __init__(self, use_gpt_4: bool, *args, **kwargs):
        engine = "gpt-4-0125-preview" if use_gpt_4 else "gpt-3.5-turbo"
        # gpt-4-1106-preview
        self.llm_gpt = GPT_Chat(engine=engine)

    def prompt_llm(self, prompt: str, temperature: float = 0.0, force_json: bool = False):
        # feed prompt to llm
        logger.debug("\n" + "#" * 50)
        logger.debug(f"Prompt:\n{prompt}")
        messages = [{"role": "user", "content": prompt}]

        conn_success, llm_output = self.llm_gpt.get_response(
            prompt=None,
            messages=messages,
            end_when_error=False,
            temperature=temperature,
            force_json=force_json,
        )
        if not conn_success:
            raise Exception("Fail to connect to the LLM")

        logger.debug("\n" + "#" * 50)
        logger.debug(f"LLM output:\n{llm_output}")

        return llm_output


####################### Convert Formats #####################

from predicate_learning.utils.pddl_util import DEFAULT_TYPE
from pddlgym.structs import Predicate, Literal
from typing import Dict, List

import numpy as np
import re


def create_predicate_from_raw(raw_predicate: str):
    # e.g. "on_table(x, y)"
    split = re.split("\(|\)|,", raw_predicate)
    split = [x.strip(" \"'") for x in split if x != ""]

    predicate_name = split[0]
    arity = len(split) - 1
    return Predicate(predicate_name, arity, var_types=[DEFAULT_TYPE] * arity)


def create_literal_from_raw(
    raw_lit: str,
    predicates: Dict[str, Predicate],
    skip_unknown: bool = False,
):
    # always positive
    split = re.split("\(|\)|,", raw_lit)
    split = [x.strip(" \"'") for x in split if x != ""]
    predicate_name = split[0]

    if skip_unknown and predicate_name not in predicates:
        return None

    assert predicate_name in predicates, f"Unknown predicate in literal {raw_lit}: {predicate_name}"
    assert (
        len(split) - 1 == predicates[predicate_name].arity
    ), f"Wrong number of arguments in literal {raw_lit}: {len(split) - 1} instead of {predicates[predicate_name].arity}"

    predicate = predicates[predicate_name]
    positive_lit = predicate(*[DEFAULT_TYPE(var_name) for var_name in split[1:]])
    return positive_lit


def get_predicate_name_from_raw(raw: str):
    # a raw literal or predicate
    split = re.split("\(", raw)
    return split[0].strip()


def json_to_actions(json_data: Dict[str, List[str]], action_predicates: List[Predicate]):
    # convert to dict for fast access
    action_predicates = {x.name: x for x in action_predicates}

    raw_plan = json_data["Plan"]
    action_plan = []

    for raw_action in raw_plan:
        action = create_literal_from_raw(raw_action, predicates=action_predicates)
        action_plan.append(action)

    return action_plan


# def convert_num_dict_with_decimal(input_dict, decimals=2):
#     output_dict = {}
#     for key, val in input_dict.items():
#         if isinstance(val, Dict):
#             output_dict[key] = convert_num_dict_with_decimal(val)
#         elif isinstance(val, np.ndarray):
#             output_dict[key] = np.round(val, decimals=decimals).tolist()
#         elif isinstance(val, float):
#             output_dict[key] = round(val, ndigits=decimals)
#         else:
#             output_dict[key] = val

#     return output_dict


def textualize_array(array):
    return np.round(array, decimals=2).tolist()


def textualize_observation(observation, with_liquid_state=False):
    if "real_robot" not in observation:
        # add table info
        table_info = observation["table"]
        table_info_text = f"The table has height {table_info['table_height']:.2f}, and ranges from {table_info['table_range'][0]:.2f} to {table_info['table_range'][1]:.2f} along x axis.\n"

        # add gripper state
        gripper_state = observation["gripper"]
        gripper_state_text = f"The gripper is at position {textualize_array(gripper_state['position'])} with open width {gripper_state['open_width']:.2f}, and the mass held by the gripper is {gripper_state['in_gripper_mass']:.2f}. "
        gripper_state_text += f"It has maximum open width {gripper_state['max_open_width']:.2f} and height {gripper_state['height']:.2f}.\n"

        # add objects state
        objects_state = observation["objects"]
        objects_state_text = f"The detected objects are: {', '.join(objects_state.keys())}."
        for obj_name, obj_state in objects_state.items():
            objects_state_text += f"\nObject {obj_name} is at position {textualize_array(obj_state['position'])} with size {textualize_array(obj_state['size'])}. Its category is {obj_state['category']}."
            if with_liquid_state and "liquid_amount" in obj_state:
                water_amount = obj_state["liquid_amount"]
                objects_state_text += f"It has {water_amount} units of water."

    else:
        # add table info
        table_info = observation["table"]
        table_info_text = f"The table has height {table_info['table_height']:.2f}.\n"

        # gripper state
        gripper_state = observation["gripper"]
        gripper_state_text = f"The gripper is at position {textualize_array(gripper_state['ee_position'])} with open width {gripper_state['open_width']:.2f}. "
        gripper_state_text += f"It has maximum open width {gripper_state['max_open_width']:.2f}.\n"

        # add objects state
        objects_state = observation["objects"]
        objects_state_text = f"The detected objects are: {', '.join(objects_state.keys())}."
        for obj_name, obj_state in objects_state.items():
            objects_state_text += f"\nObject {obj_name} is at position {textualize_array(obj_state['position'])} with size {textualize_array(obj_state['size'])}. Its category is {obj_state['category']}."

    obs_text = table_info_text + gripper_state_text + objects_state_text
    return obs_text


def parse_util_functions_from_text(text: str):
    function_starts = [m.start() for m in re.finditer("# <utility>", text)]
    function_ends = [m.end() for m in re.finditer("# <end-of-utility>", text)]
    function_starts = sorted(function_starts)
    function_ends = sorted(function_ends)
    if len(function_starts) != len(function_ends):
        assert len(function_starts) == len(
            function_ends
        ), f"Number of function starts and ends don't match."

    # parse functions
    functions = {}  # Dict[function_name, function]
    for start, end in zip(function_starts, function_ends):
        func = text[start:end]
        # ['', '', '{function_name}', ...]
        split = re.split("# <utility>|def |\(", func)
        function_name = split[2].strip()
        # record
        functions[function_name] = func

    return functions


def parse_predicate_functions_from_text(text: str, known_predicate_names: List[str]):
    function_starts = [m.start() for m in re.finditer("# <predicate>", text)]
    function_ends = [m.end() for m in re.finditer("# <end-of-predicate>", text)]
    function_starts = sorted(function_starts)
    function_ends = sorted(function_ends)
    if len(function_starts) != len(function_ends):
        assert len(function_starts) == len(
            function_ends
        ), f"Number of function starts and ends don't match."

    # parse functions
    functions = {}  # Dict[predicate_raw, function]
    parsed_predicates = {}  # Dict[predicate_raw, explanation]
    raw_to_name_mapping = {}  # Dict[predicate_raw, predicate_name]
    for start, end in zip(function_starts, function_ends):
        func = text[start:end]
        # ['', '', '{predicate}', '{args}', ...]
        split = re.split("# <predicate>|def |\(|\)", func)
        predicate_name = split[2].strip()

        # split args and remove ": str"
        arg_split = re.split(",", split[3].strip())
        arg_split = [x.split(":")[0].strip() for x in arg_split if x != ""]
        predicate_raw = f"{predicate_name}({', '.join(arg_split)})"

        # ['', '{explanation}', '', ...]
        split_2 = re.split("<<|>>", func)
        explanation = split_2[1].strip()

        # record
        functions[predicate_raw] = func
        parsed_predicates[predicate_raw] = explanation
        raw_to_name_mapping[predicate_raw] = predicate_name

    # calculate dependency
    dependency = {}  # Dict[predicate_raw, List[predicate_raw]]
    for predicate_raw, func in functions.items():
        predicate_name = raw_to_name_mapping[predicate_raw]
        dependency[predicate_name] = []
        # check dependency on known predicates
        for known_predicate_name in known_predicate_names:
            if known_predicate_name == predicate_name:
                continue
            if f"{known_predicate_name}(" in func:
                dependency[predicate_name].append(known_predicate_name)
        # check dependency on other predicates
        for other_predicate_raw, other_predicate_name in raw_to_name_mapping.items():
            if other_predicate_raw == predicate_raw:
                continue
            if f"{other_predicate_name}(" in func:
                dependency[predicate_name].append(other_predicate_name)

    return parsed_predicates, functions, dependency


def parse_actions_from_text(text: str):
    raw_action_starts = [m.end() for m in re.finditer("- ", text)]
    raw_action_ends = [m.start() - 1 for m in re.finditer(":", text)][1:]  # remove first :
    raw_action_starts = sorted(raw_action_starts)
    raw_action_ends = sorted(raw_action_ends)

    if len(raw_action_starts) != len(raw_action_ends):
        assert len(raw_action_starts) == len(
            raw_action_ends
        ), f"Number of raw action starts and ends don't match."

    action_predicates = []
    for start, end in zip(raw_action_starts, raw_action_ends):
        raw_action = text[start:end]
        action_predicate = create_predicate_from_raw(raw_action)
        action_predicates.append(action_predicate)

    return action_predicates


def pddl_literal_to_text(literal: Literal):
    lit_text = f"{literal.predicate.name}({', '.join([var.name for var in literal.variables])})"
    if literal.is_negative or literal.is_anti:
        lit_text = f"not {lit_text}"
    return lit_text


def prepare_action_desc(all_action_desc, actions):
    action_desc = "The robot can execute the actions below:"
    for action in actions:
        action_desc += (
            f"\n- {all_action_desc[action]['raw']}: {all_action_desc[action]['description']}"
        )

    return action_desc
