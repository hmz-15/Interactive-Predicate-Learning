import copy
import logging
import itertools
import functools
from pddlgym.parser import PDDLProblemParser, Operator
from pddlgym.core import (
    InvalidAction,
    PDDLEnv,
    _check_struct_for_strips,
    _compute_new_state_from_lifted_effects,
)
from pddlgym.spaces import LiteralSpace, LiteralSetSpace, LiteralActionSpace
from pddlgym.structs import (
    LiteralConjunction,
    LiteralDisjunction,
    Literal,
    ForAll,
    Exists,
    State,
    ground_literal,
    ProbabilisticEffect,
    Type,
)
from pddlgym.inference import find_satisfying_assignments

from predicate_learning.utils.symbol_util import get_object_combinations
from predicate_learning.utils.general_util import Timer

import numpy as np
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
t = Timer(logger=logger, color="green")

# Default type (in PDDL)
DEFAULT_TYPE = Type("default")


class PDDLProblem(PDDLProblemParser):
    def __init__(
        self,
        pddl_domain,
        entities,
        initial_state,
        goal_lits,
    ):
        self.domain_name = pddl_domain.domain_name
        self.types = pddl_domain.types
        self.predicates = pddl_domain.predicates

        self.objects = entities
        self.initial_state = initial_state
        self.goal = goal_lits


class ExtendedPDDLEnv(PDDLEnv):
    def __init__(
        self,
        domain,
        raise_error_on_invalid_action=False,
        dynamic_action_space=False,
    ):
        self._state = None
        self._render = None
        self._raise_error_on_invalid_action = raise_error_on_invalid_action

        # Parse the PDDL files
        self.domain = domain

        # Determine if the domain is STRIPS
        self._domain_is_strips = check_operators_for_strips(self.domain.operators)
        self._inference_mode = "csp" if self._domain_is_strips else "prolog"

        # Initialize action space with problem-independent components
        self.action_predicates = list(self.domain.actions)
        self._action_space = LiteralSpace(
            self.action_predicates,
            type_hierarchy=self.domain.type_hierarchy,
            type_to_parent_types=self.domain.type_to_parent_types,
        )

        # Initialize action space with problem-independent components
        actions = list(self.domain.actions)
        self.action_predicates = [self.domain.predicates[a] for a in actions]
        self._dynamic_action_space = dynamic_action_space
        if dynamic_action_space:
            if self._domain_is_strips:
                self._action_space = LiteralActionSpace(
                    self.domain,
                    self.action_predicates,
                    type_hierarchy=self.domain.type_hierarchy,
                    type_to_parent_types=self.domain.type_to_parent_types,
                )
            else:
                self._action_space = LiteralSpace(
                    self.action_predicates,
                    lit_valid_test=self._action_valid_test,
                    type_hierarchy=self.domain.type_hierarchy,
                    type_to_parent_types=self.domain.type_to_parent_types,
                )

        else:
            self._action_space = LiteralSpace(
                self.action_predicates,
                type_to_parent_types=self.domain.type_to_parent_types,
            )

        # Initialize observation space with problem-independent components
        self._observation_space = LiteralSetSpace(
            set(self.domain.predicates.values()) - set(self.action_predicates),
            type_hierarchy=self.domain.type_hierarchy,
            type_to_parent_types=self.domain.type_to_parent_types,
        )

    def _get_debug_info(self):
        return {}

    def reset(self, problem):
        """
        Set up a new PDDL problem and start a new episode.
        """
        self._problem = problem

        initial_state = State(
            frozenset(self._problem.initial_state),
            frozenset(self._problem.objects),
            self._problem.goal,
        )
        initial_state = self._handle_derived_literals(initial_state)
        self.set_state(initial_state)

        self._goal = self._problem.goal
        debug_info = self._get_debug_info()

        self._action_space.reset_initial_state(initial_state)

        return self.get_state(), debug_info


def check_operators_for_strips(operators):
    """
    Check whether all operators in a domain are STRIPS
    """
    for operator in operators.values():
        if not _check_struct_for_strips(operator.preconds):
            return False
    return True


def compute_operator_probability(
    action,
    lits_before,
    lits_after,
    lits_before_derived,
    operators,
    selected_operator,
    assign,
):
    if selected_operator is None:
        # if no effect, no operator with this action can be applied
        selected_op, _ = select_operator(
            lits_before | lits_before_derived,
            action,
            operators[action.predicate.name],
            inference_mode="csp",
            require_unique_assignment=False,
        )
        if selected_op is not None:
            return 0.0
        else:
            return 1.0

    else:
        predicted_lits_after = apply_effects(
            lits_before,
            selected_operator.effects.literals,
            assign,
        )
        # basic literals only
        if predicted_lits_after == lits_after:
            return 1.0
        else:
            return 0.0


def compute_ground_operators_and_assignments(operators, type_to_entities, action):
    """operators: dict of {action_name: {operator_name: operator}}"""
    all_op_and_assignments = []
    for operator in operators[action.predicate.name].values():
        params = operator.params

        op_action_lit = None
        for precond in operator.preconds.literals:
            if precond.predicate.name == action.predicate.name:
                op_action_lit = precond

        for combination in get_object_combinations(
            type_to_entities,
            var_types=[param.var_type for param in params],
        ):
            assignment = {param: entity for param, entity in zip(params, combination)}

            # only keep those matching the action literal
            if ground_literal(op_action_lit, assignment) == action:
                all_op_and_assignments.append((operator, assignment))

    return all_op_and_assignments


def get_successor_state_literals(
    literals,
    action,
    operators,
    raise_error_on_invalid_action=False,
    inference_mode="csp",
    require_unique_assignment=True,
    get_all_transitions=False,
    return_probs=False,
):
    """
    Compute successor state(s) using operators in the domain

    Parameters
    ----------
    literals : List[Literal]
    action : Literal
    operators : List[Operator]
    require_unique_assignment : bool

    Returns
    -------
    next_state : State
    """
    selected_operator, assignment = select_operator(
        literals,
        action,
        operators,
        inference_mode=inference_mode,
        require_unique_assignment=require_unique_assignment,
    )
    # A ground operator was found; execute the ground effects
    if assignment is not None:
        # Get operator effects
        if isinstance(selected_operator.effects, LiteralConjunction):
            effects = selected_operator.effects.literals
        else:
            assert isinstance(selected_operator.effects, Literal)
            effects = [selected_operator.effects]

        # Need assignment for all parameters!
        if not set(selected_operator.params) == set(assignment.keys()):
            pdb.set_trace()
        assert set(selected_operator.params) == set(
            assignment.keys()
        ), f"Assignment is ambigious, with operator "

        if len(list(assignment.values())) != len(set(assignment.values())):
            pdb.set_trace()

        new_literals = apply_effects(
            literals,
            effects,
            assignment,
            get_all_transitions,
            return_probs=return_probs,
        )
        return new_literals
    # No operator was found
    elif raise_error_on_invalid_action:
        raise InvalidAction(
            f"called get_successor_state with invalid action '{action}' for given state"
        )
    else:
        return None


def select_operator(
    literals,
    action,
    operators,
    inference_mode="csp",
    require_unique_assignment=True,
):
    """
    Helper for successor generation
    """
    if inference_mode == "infer":
        inference_mode = "csp" if _check_struct_for_strips(operators) else "prolog"

    # Possibly multiple operators per action
    possible_operators = set(operators.values())

    # Knowledge base: literals in the state + action taken
    kb = literals | {action}

    selected_operator = None
    assignment = None
    for operator in possible_operators:
        if isinstance(operator.preconds, Literal):
            conds = [operator.preconds]
        else:
            conds = operator.preconds.literals
        # Check whether action is in the preconditions
        action_literal = None
        for lit in conds:
            if lit.predicate == action.predicate:
                action_literal = lit
                break
        if action_literal is None:
            continue
        # For proving, consider action variable first
        action_variables = action_literal.variables
        variable_sort_fn = lambda v: (not v in action_variables, v)
        assignments = find_satisfying_assignments(
            kb,
            conds,
            variable_sort_fn=variable_sort_fn,
            mode=inference_mode,
            allow_redundant_variables=False,
        )
        num_assignments = len(assignments)
        if num_assignments > 0:
            if require_unique_assignment:
                assert num_assignments == 1, "Nondeterministic envs not supported"
            selected_operator = operator
            assignment = assignments[0]

            if len(list(assignment.values())) != len(set(assignment.values())):
                pdb.set_trace()

            break

    # Random assign other vars
    if selected_operator is not None:
        unassigned_vars = set([var for lit in kb for var in lit.variables])
        for var in assignment.values():
            if var in unassigned_vars:
                unassigned_vars.remove(var)

        for lifted_var in selected_operator.params:
            if lifted_var not in assignment:
                for var in unassigned_vars:
                    if var.var_type == lifted_var.var_type:
                        assignment[lifted_var] = var
                        unassigned_vars.remove(var)
                        break

    return selected_operator, assignment


def apply_effects(
    literals,
    lifted_effects,
    assignments,
    get_all_transitions=False,
    return_probs=False,
):
    """
    Update a state given lifted operator effects and
    assignment of variables to objects.

    Parameters
    ----------
    state : State
        The state on which the effects are applied.
    lifted_effects : { Literal }
    assignment : { TypedEntity : TypedEntity }
        Maps variables to objects.
    """
    new_literals = set(literals)
    determinized_lifted_effects = []
    # Handle probabilistic effects.

    # Each element of this list contain
    #   a pair of outcomes from a probabilistic effect
    probabilistic_lifted_effects = []
    for lifted_effect in lifted_effects:
        if isinstance(lifted_effect, ProbabilisticEffect):
            effect_outcomes = lifted_effect.literals
            probas = dict(zip(lifted_effect.literals, lifted_effect.probabilities))
            cur_probabilistic_lifted_effects = []

            if get_all_transitions:
                lifted_effects_list = cur_probabilistic_lifted_effects
            else:
                lifted_effects_list = determinized_lifted_effects
            sampled_effect = lifted_effect.sample()

            # If get_all_transitions == False, create list with sampled state only
            # Otherwise, populate it with possible outcomes
            effects_to_process = [sampled_effect] if not get_all_transitions else effect_outcomes

            for chosen_effect in effects_to_process:
                if isinstance(chosen_effect, LiteralConjunction):
                    for lit in chosen_effect.literals:
                        lifted_effects_list.append(lit)
                        lit.proba = probas[chosen_effect]
                else:
                    lifted_effects_list.append(chosen_effect)
                    chosen_effect.proba = probas[chosen_effect]

            if get_all_transitions:
                probabilistic_lifted_effects.append(cur_probabilistic_lifted_effects)
        else:
            determinized_lifted_effects.append(lifted_effect)

    if not get_all_transitions:
        new_literals = _compute_new_state_from_lifted_effects(
            determinized_lifted_effects, assignments, new_literals
        )
        return new_literals

    # else - get all possible transitions

    # Construct combinations of probabilistic effects
    probabilistic_effects_combinations = list(itertools.product(*probabilistic_lifted_effects))

    new_lits_to_probs = {}
    new_lits_set = {}
    for prob_efs_combination in probabilistic_effects_combinations:
        total_proba = np.prod([lit.proba for lit in prob_efs_combination])
        if total_proba == 0:
            continue
        new_prob_literals = set(literals)
        new_determinized_lifted_effects = determinized_lifted_effects + list(prob_efs_combination)
        new_prob_literals = _compute_new_state_from_lifted_effects(
            new_determinized_lifted_effects, assignments, new_prob_literals
        )
        new_prob_frozen_literals = frozenset(new_prob_literals)
        if new_prob_frozen_literals in new_lits_to_probs:
            # If there are multiple ways of reaching next state,
            #   then these probabilities have to be summed
            new_lits_to_probs[new_prob_frozen_literals] += total_proba
        else:
            new_lits_to_probs[new_prob_frozen_literals] = total_proba
        new_lits_set.add(new_prob_frozen_literals)
    if return_probs:
        return new_lits_to_probs
    # convert list of states to set
    return frozenset(new_lits_set)


# In fact, prolog reasoning is not scalable
def handle_derived_literals(literals, all_basic_literals, predicates, return_derived_only=False):
    # first remove any old derived literals since they're outdated
    to_remove = set()
    for lit in literals:
        if lit.predicate.is_derived:
            to_remove.add(lit)
    literals = literals - to_remove

    # add negative basic literals for checking derived predicates
    literals_with_neg = copy.deepcopy(literals)
    for lit in all_basic_literals:
        if not lit.predicate.is_derived and lit not in literals:
            literals_with_neg = {lit.negative} | literals_with_neg

    derived_lits = frozenset()
    while True:  # loop, because derived predicates can be recursive
        new_derived_literals = set()
        for pred in predicates.values():
            if not pred.is_derived:
                continue

            # get relevant basic predicates to reduce search space
            # output_predicates = set()
            # relevant_literals = set()
            # get_all_related_basic_predicates(pred.body, output_predicates)
            # for lit in literals_with_neg:
            #     if lit.is_negative and lit.predicate.positive in output_predicates:
            #         relevant_literals.add(lit)
            #     elif not lit.is_negative and lit.predicate in output_predicates:
            #         relevant_literals.add(lit)

            assignments = find_satisfying_assignments(
                literals_with_neg,
                pred.body,
                type_to_parent_types={},
                mode="prolog",
                max_assignment_count=99999,
                allow_redundant_variables=False,
            )
            for assignment in assignments:
                objects = [
                    assignment[param_type(param_name)]
                    for param_name, param_type in zip(pred.param_names, pred.var_types)
                ]
                derived_literal = pred(*objects)
                if derived_literal not in literals:
                    new_derived_literals.add(derived_literal)
        if new_derived_literals:
            # update literals for recursive checking
            literals_with_neg = literals_with_neg | new_derived_literals
            # save results
            derived_lits = derived_lits | new_derived_literals
            literals = literals | new_derived_literals
        else:  # terminate
            break

    if return_derived_only:
        return derived_lits
    else:
        return literals


def ground_expression(
    expression,
    assignment,
    type_to_entities,
    output_literals,
):
    assert not isinstance(expression, LiteralDisjunction) and not isinstance(
        expression, Exists
    ), "Not supporting instantiate LiteralDisjunction or Exists"

    # literal
    if isinstance(expression, Literal):
        # derived literal (assume all positive)
        if expression.predicate.is_derived:
            new_assign = copy.deepcopy(assignment)
            # compose assignment: predicate.param_names -> variables -> entities
            for param_name, var, var_type in zip(
                expression.predicate.param_names,
                expression.variables,
                expression.predicate.var_types,
            ):
                assert var in new_assign
                new_assign[var_type(param_name)] = new_assign[var]
                new_assign.pop(var)

            ground_expression(
                expression.predicate.body, new_assign, type_to_entities, output_literals
            )
        else:
            # basic literal
            ground_lit = ground_literal(expression, assignment)
            if len(ground_lit.variables) == len(set(ground_lit.variables)):
                output_literals.add(ground_lit)

    # conjunction
    elif isinstance(expression, LiteralConjunction):
        for lit in expression.literals:
            ground_expression(lit, assignment, type_to_entities, output_literals)
    # forall
    elif isinstance(expression, ForAll):
        assert not expression.is_negative, "Not supporting negative universal quantifier"

        for combination in get_object_combinations(
            type_to_entities,
            var_types=[var.var_type for var in expression.variables],
        ):
            for var, entity in zip(expression.variables, combination):
                assignment[var] = entity

            ground_expression(expression.body, assignment, type_to_entities, output_literals)

        # pop new assignment
        for var in expression.variables:
            assignment.pop(var)
    else:
        raise NotImplementedError


def get_all_related_basic_predicates(expression, output_predicates):
    # literal
    if isinstance(expression, Literal):
        # derived literal (assume all positive)
        if expression.predicate.is_derived:
            get_all_related_basic_predicates(expression.predicate.body, output_predicates)
        else:
            # basic literal
            predicate = expression.predicate
            if expression.predicate.is_negative:
                predicate = expression.predicate.positive
            if predicate not in output_predicates:
                output_predicates.add(predicate)

    # conjunction
    elif isinstance(expression, LiteralConjunction):
        for lit in expression.literals:
            get_all_related_basic_predicates(lit, output_predicates)
    # disjunction
    elif isinstance(expression, LiteralDisjunction):
        for lit in expression.literals:
            get_all_related_basic_predicates(lit, output_predicates)
    # forall
    elif isinstance(expression, ForAll):
        get_all_related_basic_predicates(expression.body, output_predicates)
    # exists
    elif isinstance(expression, Exists):
        get_all_related_basic_predicates(expression.body, output_predicates)
    else:
        raise NotImplementedError(f"{expression}")


def check_satisfy(
    conditions,
    parsed_literals,
    require_eq=False,
    init_assignment=None,
    max_assignment_count=10,
):
    """Return a tuple of (whether the given frozensets lits1 and lits2 can be
    unified, the mapping if the first return value is True).
    Also returns the mapping.
    """
    if require_eq:
        # Terminate quickly if there is a mismatch between lits
        predicate_order1 = [lit.predicate for lit in sorted(conditions)]
        predicate_order2 = [lit.predicate for lit in sorted(parsed_literals)]
        if predicate_order1 != predicate_order2:
            return False, None

    assignments = find_satisfying_assignments(
        parsed_literals,
        conditions,
        allow_redundant_variables=False,
        max_assignment_count=max_assignment_count,
        init_assignments=init_assignment,
    )

    valid_assignments = []
    for assignment in assignments:
        if init_assignment is None or check_assignment(assignment, init_assignment):
            valid_assignments.append(assignment)
        else:
            # redundant assignment in init_assign
            pass

    if len(valid_assignments) > 0:
        return True, valid_assignments
    return False, None


def check_eq_literals(
    new_lits,
    lifted_lits,
    init_assignment=None,
    enforce_free_to_free=False,
    return_all=False,
):
    """Return a tuple of (whether the given frozensets new_effect and clustered_effect can be
    unified, the mapping if the first return value is True).
    Also returns the mapping.
    """
    satisfy, assignments = check_satisfy(
        new_lits,
        lifted_lits,
        require_eq=True,
        init_assignment=init_assignment,
    )

    valid_assignments = []
    if enforce_free_to_free and assignments is not None:
        for assignment in assignments:
            for key, val in assignment.items():
                # ?f_1
                if key not in init_assignment and val[1] != "f":
                    break

            valid_assignments.append(assignment)
        assignments = valid_assignments

    if satisfy and len(assignments) > 0:
        if return_all:
            return True, assignments
        else:
            return True, assignments[0]

    return False, None


# @functools.lru_cache(maxsize=10000)
def check_satisfy_precondition(
    precondition, preimage, max_assignment_count=10, require_fixed_non_fix_vars=False
):
    # return precondition.issubset(preimage)
    # init_assignment = {}
    # for lit in precondition:
    #     for var in lit.variables:
    #         # ?v_1
    #         if var[1] == "v" and var not in init_assignment:
    #             init_assignment[var] = var

    # for lit in preimage:
    #     for var in lit.variables:
    #         if var[0] == "v" and var not in init_assignment:
    #             init_assignment[var] = var

    satisfy, assignments = check_satisfy(
        precondition,
        preimage,
        max_assignment_count=max_assignment_count,
    )

    if satisfy and require_fixed_non_fix_vars:
        valid_assignments = []
        for assignment in assignments:
            valid = True
            for key, val in assignment.items():
                if key[1] == "v" and key != val:
                    valid = False
                    break
            if valid:
                valid_assignments.append(assignment)

        satisfy = len(valid_assignments) > 0
        assignments = valid_assignments

    return satisfy, assignments

    # if satisfy:
    #     for assignments in assignments_list:
    #         valid = True
    #         for var in precondition.constrained_variables:
    #             # It's possible that var not in assignments;
    #             # we need to remove some constrained_variables for precondition
    #             if var in assignments and assignments[var] != var:
    #                 valid = False
    #                 break

    #         if valid:
    #             return True, assignments
    # return False, None


@functools.lru_cache(maxsize=10000)
def check_satisfy_goal(goal_lits, lits):
    for goal_lit in goal_lits:
        if goal_lit.is_negative:
            if goal_lit.negative in lits:
                return False
        else:
            if goal_lit not in lits:
                return False
    return True


def lifted_literal(
    lit,
    assignment,
    assign_unknown_vars=True,
    unknown_var_letter="v",
    inverted_anti=False,
    return_debug=False,
):
    lifted_vars = []
    extra_ass = {}
    for var_type, var in zip(lit.predicate.var_types, lit.variables):
        if var not in assignment:
            if assign_unknown_vars:
                type_prefix = f"?{unknown_var_letter}"
                max_lifted_var_idx = get_max_assignment_val_idx(
                    assignment, var_letter=unknown_var_letter
                )
                assignment[var] = var_type(f"{type_prefix}_{max_lifted_var_idx+1}")
                extra_ass[var] = var_type(f"{type_prefix}_{max_lifted_var_idx+1}")
            else:
                # use original variable
                lifted_vars.append(var)
                continue
        lifted_vars.append(assignment[var])

    if inverted_anti:
        if return_debug:
            return lit.predicate.inverted_anti(*lifted_vars), extra_ass
        return lit.predicate.inverted_anti(*lifted_vars)
    else:
        if return_debug:
            return lit.predicate(*lifted_vars), extra_ass
        return lit.predicate(*lifted_vars)


def lifted_literals(lits, assignment, assign_unknown_vars=True, unknown_var_letter="v"):
    lifted_lit_set = set()
    for lit in lits:
        lifted_lit = lifted_literal(
            lit,
            assignment,
            assign_unknown_vars=assign_unknown_vars,
            unknown_var_letter=unknown_var_letter,
        )
        lifted_lit_set.add(lifted_lit)

    return frozenset(lifted_lit_set)


def get_max_assignment_val_idx(assignment, var_letter="v"):
    # e.g., "?v_10"
    return (
        max([int(val.name[3:]) for val in assignment.values() if val.name[1] == var_letter])
        if len(assignment) > 0
        else -1
    )


def check_assignment(assignment, init_assignment):
    for key, val in assignment.items():
        for init_key, init_val in init_assignment.items():
            if val == init_val and key != init_key:
                return False

    return True


def read_predicates(x):
    if isinstance(x, Literal):
        x_pred = x.predicate
        if x_pred.is_negative:
            x_pred = x_pred.positive
        elif x_pred.is_anti:
            x_pred = x_pred.inverted_anti
        return {x_pred.name: x_pred}
    elif isinstance(x, LiteralConjunction):
        predicates = {}
        for lit in x.literals:
            predicates.update(read_predicates(lit))
        return predicates
    elif isinstance(x, Exists):
        return read_predicates(x.body)
    elif isinstance(x, Operator):
        predicates = {}
        predicates.update(read_predicates(x.preconds))
        predicates.update(read_predicates(x.effects))
        return predicates


def read_entity_types(x):
    if isinstance(x, Literal):
        predicates = {}
        for v in x.variables:
            predicates[v.var_type] = v.var_type
        return predicates
    elif isinstance(x, LiteralConjunction):
        predicates = {}
        for lit in x.literals:
            predicates.update(read_predicates(lit))
        return predicates
    elif isinstance(x, Exists):
        return read_predicates(x.body)
    elif isinstance(x, Operator):
        predicates = {}
        for v in x.params:
            predicates[v.var_type] = v.var_type
        return predicates


# class LiftedLiterals(
#     namedtuple("LiftedLiterals", ["literals", "constrained_variables"])
# ):
#     __slots__ = ()

#     def _check_valid_assignment(self, assignment):
#         for var in assignment:
#             if var in self.constrained_variables:
#                 return False
#         return True

#     @staticmethod
#     def _remove_redundant_constrained_vars(literals, constrained_variables):
#         all_vars = set(v for lit in literals for v in lit.variables)
#         constrained_vars_set = set(constrained_variables)
#         for var in constrained_variables:
#             if var not in all_vars:
#                 constrained_vars_set.remove(var)
#         return frozenset(constrained_vars_set)

#     def with_new_assignment(self, assignment):
#         is_valid = self._check_valid_assignment(assignment)
#         if not is_valid:
#             return False, None

#         literals = set()
#         for lit in self.literals:
#             literals.append(
#                 lifted_literal(lit, assignment, assign_unknown_vars=False)
#             )

#         # update constrained_variables
#         constrained_variables = set(self.constrained_variables)
#         for var in assignment:
#             constrained_variables.remove(var)
#         for var in assignment.values():
#             constrained_variables.add(var)

#         return True, self._replace(
#             literals=frozenset(literals),
#             constrained_variables=frozenset(constrained_variables),
#         )

#     def with_constrained_variables(self, constrained_variables):
#         return self._replace(constrained_variables=frozenset(constrained_variables))

#     def with_cleaned_up_constrained_variables(self):
#         constrained_variables = self._remove_redundant_constrained_vars(
#             self.literals, self.constrained_variables
#         )
#         return self.with_constrained_variables(constrained_variables)
