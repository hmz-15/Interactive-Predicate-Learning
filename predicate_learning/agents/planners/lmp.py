import ast
import astunparse
import logging
from termcolor import colored

from predicate_learning.utils.llm_util import LLMBase


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LMP(LLMBase):
    def __init__(
        self,
        name,
        base_prompt,
        lmp_fgen,
        fixed_vars,
        variable_vars,
        maintain_session=False,
        return_val_name=None,
        use_gpt_4=True,
    ):
        super().__init__(use_gpt_4=use_gpt_4)

        self._name = name
        self._base_prompt = base_prompt
        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ""
        self._query_prefix = "# "
        self._query_postfix = (
            "\n# please output code directly, without giving explanation or adding ```python"
        )

        # extra variable_vars for new functions, used to update "import" in prompt
        self._extra_vars = {}

        self._maintain_session = maintain_session
        self._return_val_name = return_val_name

        # update extra_vars with those in lmp_fgen
        self._extra_vars.update(self._lmp_fgen.extra_vars)

    def clear_exec_hist(self):
        self.exec_hist = ""

    def build_prompt(self, query, context=""):
        if len(self._extra_vars) > 0:
            extra_vars_imports_str = f"from extra_utils import {', '.join(self._extra_vars.keys())}"
        else:
            extra_vars_imports_str = ""
        prompt = self._base_prompt.replace("{extra_vars_imports}", extra_vars_imports_str)

        if self._maintain_session:
            prompt += f"\n{self.exec_hist}"

        if context != "":
            prompt += f"\n{context}"

        use_query = f"{self._query_prefix}{query}{self._query_postfix}"
        prompt += f"\n{use_query}"

        return prompt

    def __call__(self, query, context="", return_srcs=False, include_context=False, **kwargs):
        logger.info(colored(f"Call {self._name}", "green"))
        prompt = self.build_prompt(query, context=context)
        code_str = self.prompt_llm(prompt)

        if context != "" and include_context:
            to_exec = f"{context}\n{code_str}"
        else:
            to_exec = code_str

        new_fs, _ = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._extra_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars, self._extra_vars])
        lvars = kwargs

        exec_safe(to_exec, gvars, lvars)
        self.exec_hist += f"\n{to_exec}"

        if self._maintain_session:
            self._variable_vars.update(lvars)

        return_val = lvars[self._return_val_name] if self._return_val_name is not None else None
        logger.info(colored(f"Return {self._return_val_name}: {str(return_val)}", "green"))

        if return_srcs:
            return return_val, code_str
        else:
            return return_val


class LMPFGen(LLMBase):
    def __init__(self, base_prompt, fixed_vars, variable_vars, use_gpt_4=True):
        super().__init__(use_gpt_4=use_gpt_4)
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = base_prompt
        self._query_prefix = "# define function: "
        self._query_postfix = (
            "\n# please output code directly, without giving explanation or adding ```python"
        )

        # extract functions & constants from base_prompt as known
        extra_fs = {}
        parsed_code = ast.parse(self._base_prompt)
        for node in ast.walk(parsed_code):
            if isinstance(node, ast.FunctionDef):
                function_code = astunparse.unparse(node)
                extra_fs[node.name] = function_code

            if isinstance(node, ast.Assign):
                assign_code = astunparse.unparse(node)
                if isinstance(
                    node.value, ast.Constant
                ):  # ast.Str and ast.Num for Python < 3.8 compatibility
                    for target in node.targets:
                        if isinstance(target, ast.Name):  # Ensure the target is a simple name
                            extra_fs[target.id] = assign_code

        extra_fs_str = "\n".join(extra_fs.values())
        lvars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        exec_safe(extra_fs_str, gvars, lvars)
        gvars = merge_dicts([gvars, lvars])
        exec_safe(extra_fs_str, gvars, lvars)  # run twice for functions to be aware of each other
        self._extra_vars = {f_name: lvars[f_name] for f_name in extra_fs.keys()}

    @property
    def extra_vars(self):
        return self._extra_vars

    def create_f_from_sig(self, f_name, f_sig, other_vars=None):
        logger.info(colored(f"Creating function: {f_sig}", "green"))

        use_query = f"{self._query_prefix}{f_sig}{self._query_postfix}"
        prompt = f"{self._base_prompt}\n{use_query}"

        f_src = self.prompt_llm(prompt)

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, self._extra_vars, other_vars])
        lvars = {}

        # import pdb; pdb.set_trace()
        exec_safe(f_src, gvars, lvars)
        f = lvars[f_name]

        self._extra_vars[f_name] = f

        return f, f_src

    def create_new_fs_from_code(self, code_str, other_vars=None):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts(
                [self._fixed_vars, self._variable_vars, self._extra_vars, new_fs, other_vars]
            )
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(f_name, f_sig, new_fs)

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts(
                        [
                            self._fixed_vars,
                            self._variable_vars,
                            self._extra_vars,
                            new_fs,
                            other_vars,
                        ]
                    )
                    lvars = {}
                    # import pdb; pdb.set_trace()
                    exec_safe(f_src, gvars, lvars)

                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        return new_fs, srcs


class FunctionParser(ast.NodeTransformer):
    def __init__(self, fs, f_assigns):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ["import", "__"]
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])
    exec(code_str, custom_gvars, lvars)
