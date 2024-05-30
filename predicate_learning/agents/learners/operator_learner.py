import logging
import copy
import math
from dataclasses import dataclass
from termcolor import colored
from typing import Any, List, Dict, Tuple, Set, Union, Optional, FrozenSet
from predicate_learning.utils.pddl_util import (
    check_eq_literals,
    check_satisfy_precondition,
    lifted_literal,
    lifted_literals,
)
from pddlgym.structs import LiteralConjunction, Predicate, Literal, NoChange, TypedEntity
from pddlgym.parser import Operator

from predicate_learning.utils.common_util import EnvFeedback
from predicate_learning.utils.io_util import merge_number_dict
from predicate_learning.utils.pddl_util import DEFAULT_TYPE
from predicate_learning.agents.learners.predicate_learner import PredicateLearner


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DELTA = 1e-5


@dataclass
class Transition:
    action: Literal
    obs_before: Dict[str, Any]
    obs_after: Dict[str, Any]


@dataclass
class LiftedTransition:
    lifted_action: Literal
    lifted_preimage: FrozenSet[Literal]
    lifted_effect_count: Dict[FrozenSet[Literal], int]

    def add_effect(self, effect: FrozenSet[Literal]):
        merge_number_dict(self.lifted_effect_count, {effect: 1})


class OperatorLearner:
    """
    Learn symbolic operators (potentially probabilistic)
    """

    def __init__(
        self,
        action_predicates: List[Predicate],
        predicate_learner: Optional[PredicateLearner] = None,
    ):
        self._actions = action_predicates  # List[Predicate]
        self._action_names = [
            action_predicate.name for action_predicate in action_predicates
        ]  # List[str]

        # shared component, will update from time to time
        self._predicate_learner = predicate_learner

        # necessary (common) preconditions from human. Note that this does not need to be reset as it's binded with human input
        self._necessary_preconds = {}  # Dict[Literal, FrozenSet[Literal]]
        for action_predicate in action_predicates:
            lifted_action = action_predicate(
                *[DEFAULT_TYPE(f"?v_{i}") for i in range(action_predicate.arity)]
            )
            self._necessary_preconds[lifted_action] = frozenset()

        self.reset_memory()

    @property
    def pddl_operators(self) -> Dict[str, Operator]:
        pddl_operators = {}

        # if necessary preconds not covered, create operator
        all_op_names = set(self._learned_operators.keys())
        for lifted_action, lifted_preconds in self._necessary_preconds.items():
            action_name = lifted_action.predicate.name
            if action_name in all_op_names:
                continue

            pddl_operators[action_name] = Operator(
                name=action_name,
                params=set(v for lit in lifted_preconds | {lifted_action} for v in lit.variables),
                preconds=LiteralConjunction(list(lifted_preconds | {lifted_action})),
                effects=LiteralConjunction([]),
            )

        # add learned ops
        for learned_operators in self._learned_operators.values():
            for learned_op in learned_operators:
                pddl_op = learned_op._pddl_operator
                if pddl_op is not None:
                    pddl_operators[pddl_op.name] = pddl_op

        return pddl_operators

    @property
    def learned_operators(self) -> Dict[str, Set["LearnedOperator"]]:
        return self._learned_operators

    @learned_operators.setter
    def learned_operators(self, learned_operators: Dict[str, Set["LearnedOperator"]]) -> None:
        self._learned_operators = learned_operators

        # load into necessary preconds
        # self._necessary_preconds = {}
        for operators in self._learned_operators.values():
            # get the intersection of these preconditions
            intersect = frozenset()
            for idx, op in enumerate(operators):
                if idx == 0:
                    intersect = op.precond_set
                    lifted_action = op.lifted_action
                else:
                    intersect = intersect.intersection(op.precond_set)

            # literals in different sessions have different hash
            lifted_action = lifted_action.predicate(
                *[DEFAULT_TYPE(f"?v_{i}") for i in range(lifted_action.predicate.arity)]
            )
            new_intersect = set()
            for lit in intersect:
                new_lit = lit.predicate(*lit.variables)
                new_intersect.add(new_lit)

            self._necessary_preconds[lifted_action] = frozenset(new_intersect)

    def reset_memory(self, keep_transitions: bool = False) -> None:
        # learned operators
        self._learned_operators = {}  # Dict[str, Set[LearnedOperator]]
        # lifted transitions
        self._lifted_transitions = {}  # Dict[str, List[LiftedTransition]]

        # new transitions: transitions will be first saved here and moved to transition pool after learned
        if keep_transitions:
            self._new_transitions = self._transition_pool + self._new_transitions
        else:
            self._new_transitions = []

        # transition pool: store processed transitions
        self._transition_pool = []

    def add_new_transition(
        self,
        obs_before: Dict[str, Any],
        action: Literal,
        env_feedback: EnvFeedback,
    ) -> None:
        """
        Learn the symbolic operator from a transition.
        """
        if action is None or not env_feedback.success:
            return

        # add to new transitions
        transition = Transition(action, obs_before, env_feedback.observation)
        self._new_transitions.append(transition)

        # calculate grounded necessary preconds
        find_match = False
        ground_preconds = frozenset()
        for lifted_action in self._necessary_preconds.keys():
            satisfy, assign = check_eq_literals([lifted_action], [action])
            if satisfy:
                find_match = True

                for lifted_precond in self._necessary_preconds[lifted_action]:
                    ground_precond = lifted_literal(lifted_precond, assign)
                    ground_preconds |= frozenset([ground_precond])
                break

        assert find_match, f"{action} is unknown!"
        return ground_preconds

    def add_new_preconds(
        self,
        action: Literal,
        preconds: FrozenSet[Literal],
    ) -> None:
        """
        Learn the preconditions of an action.
        """
        # add to self._necessary_preconds
        find_match = False
        new_lifted_preconds = frozenset()
        for lifted_action in self._necessary_preconds.keys():
            satisfy, assign = check_eq_literals([action], [lifted_action])
            if satisfy:
                find_match = True

                # add new preconds
                known_vars = set(assign.keys())
                filtered_preconds = frozenset(
                    [lit for lit in preconds if set(lit.variables).issubset(known_vars)]
                )
                new_lifted_preconds = lifted_literals(filtered_preconds, assign)
                self._necessary_preconds[lifted_action] |= new_lifted_preconds

                logger.info(
                    f"New preconds! Action {lifted_action}, Preconds {self._necessary_preconds[lifted_action]}"
                )
                break

        assert find_match, f"{action} is unknown!"

        # add to existing learned operators
        for action_name, operators in self._learned_operators.items():
            if action_name == action.predicate.name:
                for operator in operators:
                    new_preconds = set(
                        operator._pddl_operator.preconds.literals + list(new_lifted_preconds)
                    )
                    operator._pddl_operator.preconds = LiteralConjunction(list(new_preconds))

    def learn_operators(self) -> None:
        """
        Learn operators from transitions.
        """
        # if learned predicates are updated, learn from scratch
        # if self._predicate_learner.predicates_updated_flag:
        self.reset_memory(keep_transitions=True)

        # nothing to learn
        if len(self._new_transitions) == 0:
            return

        # 1. Process the new transitions one by one & initialize learned operators
        while len(self._new_transitions) > 0:
            transition = self._new_transitions.pop()
            # compute, cluster and record lifted transition
            self._compute_lifted_transition(transition)
            # add the transition to pool
            self._transition_pool.append(transition)

        # 2. Initialize and merge operators (always from scratch for now)
        self._learn_operators()

    def _compute_lifted_transition(self, transition: Transition) -> None:
        # 1. Parse observations
        lits_before = self._predicate_learner.parse_literals(
            transition.obs_before,
            self._predicate_learner.perception_api_wrapper,
            return_pddl_lits=True,
            essential_only=True,
            include_negative_predicates=True,
        )
        lits_after = self._predicate_learner.parse_literals(
            transition.obs_after,
            self._predicate_learner.perception_api_wrapper,
            return_pddl_lits=True,
            essential_only=True,
            include_negative_predicates=True,
        )
        lits_before = frozenset(lits_before)
        lits_after = frozenset(lits_after)

        # compute ground effect
        effect = self._compute_effect(lits_before, lits_after)

        action = transition.action
        action_name = action.predicate.name

        # 2. Compute lifted transition; if precondition matches, merge into existing lifted transition
        assignment = {}  # Dict[current_var, lifted_var], will be updated in functions
        # compute lifted action
        lifted_action = lifted_literal(
            action, assignment, assign_unknown_vars=True, unknown_var_letter="f"
        )

        # compare to existing operators w.r.t. preimage
        existing_transitions = (
            self._lifted_transitions[action_name] if action_name in self._lifted_transitions else []
        )
        create_new_flag = True
        for existing_transition in existing_transitions:
            # check if the preimage matches
            is_eq, new_assign = check_eq_literals(
                lits_before,
                existing_transition.lifted_preimage,
                init_assignment=assignment,
            )
            # if match, merge into existing lifted transition
            if is_eq:
                assignment.update(new_assign)
                create_new_flag = False

                # compute & add lifted effect
                lifted_effect = lifted_literals(
                    effect,
                    assignment,
                    assign_unknown_vars=True,
                    unknown_var_letter="f",
                )
                existing_transition.add_effect(lifted_effect)
                break

        # if no match found, create new lifted transition
        if create_new_flag:
            # new lifted preimage
            lifted_preimage = lifted_literals(
                lits_before,
                assignment,
                assign_unknown_vars=True,
                unknown_var_letter="f",
            )
            # new lifted effect
            lifted_effect = lifted_literals(
                effect,
                assignment,
                assign_unknown_vars=True,
                unknown_var_letter="f",
            )
            # new lifted transition
            new_lifted_transition = LiftedTransition(
                lifted_action, lifted_preimage, {lifted_effect: 1}
            )
            # add to existing operators
            if action_name not in self._lifted_transitions:
                self._lifted_transitions[action_name] = []
            self._lifted_transitions[action_name].append(new_lifted_transition)

    def _learn_operators(self, max_iter: int = 10) -> None:
        """
        Initialize and merge operators.
        """
        self._learned_operators = {}

        # 1. Initialize operators from lifted transitions
        for action_name, lifted_transitions in self._lifted_transitions.items():
            if action_name not in self._learned_operators:
                self._learned_operators[action_name] = set()

            for lifted_transition in lifted_transitions:
                new_learned_operator = LearnedOperator(
                    lifted_transition.lifted_action, lifted_transition.lifted_preimage
                )
                new_learned_operator.add_effects(lifted_transition.lifted_effect_count)
                self._learned_operators[action_name].add(new_learned_operator)

        # 2. Merge operators
        for action_name, all_op_per_action in self._learned_operators.items():
            # sort to put operators with less preconditions first (the list is used for the for loop)
            all_op_per_action_list = sorted(
                list(all_op_per_action), key=lambda op: len(op.precond_set)
            )
            for learned_op in all_op_per_action_list:
                # if already merged, skip
                if learned_op not in all_op_per_action:
                    continue

                # change "?f" to "?v" for learned_op (for ease of later assignment)
                learned_op.replace_variables(change_from="f", change_to="v")

                lifted_action = learned_op.lifted_action
                precond_set = learned_op.precond_set

                necessary_preconds = (
                    self._necessary_preconds[lifted_action]
                    if lifted_action in self._necessary_preconds
                    else frozenset()
                )
                merged_effect_counts = copy.deepcopy(learned_op.lifted_effect_counts)

                # merge other operators directly into it
                logger.debug("----------- Direct -------------")
                for learned_op_other in all_op_per_action - set([learned_op]):
                    satisfy, assignments = check_satisfy_precondition(
                        frozenset(precond_set) | {lifted_action},
                        frozenset(learned_op_other.precond_set) | {learned_op_other.lifted_action},
                    )
                    logger.debug(f"Checked set: {learned_op_other.precond_set}")
                    logger.debug(satisfy)
                    if satisfy:
                        logger.debug(assignments[0])
                        logger.debug("Transform!")
                        logger.debug(str(learned_op_other.lifted_effect_counts))
                        new_effect_counts = transform_effect_counts(
                            learned_op_other.lifted_effect_counts, assignments[0], reverse=True
                        )
                        logger.debug(str(new_effect_counts))
                        merge_number_dict(merged_effect_counts, new_effect_counts)

                        # remove
                        all_op_per_action.remove(learned_op_other)

                # remove precondition literals while merge operators
                # unnormalized objective
                best_objective = self._compute_objective(all_op_per_action, normalize=False)
                # sort to prioritize removing literals with more vars not existing in the effect & action
                used_vars = set(
                    [
                        v
                        for effect in learned_op.lifted_effect_counts
                        for lit in effect
                        for v in lit.variables
                    ]
                ) | set(lifted_action.variables)

                precond_list = sorted(
                    list(precond_set),
                    key=lambda lit: len(set(lit.variables) - used_vars) + len(lit.variables) * 0.1,
                    reverse=True,
                )

                logger.debug("---------- Remove --------------")
                for lit in precond_list:
                    logger.debug(str(lit))
                    # do not remove necessary preconditions
                    if lit in necessary_preconds:
                        continue
                    else:
                        precond_set.remove(lit)

                    allow_merge = True
                    candidate_op_merged = set()
                    temp_merged_effect_counts = copy.deepcopy(merged_effect_counts)
                    # remove the current operator's effects
                    current_objective = best_objective + compute_entropy(
                        learned_op.lifted_effect_counts_list, unnormalize=True
                    )
                    logger.debug("------------------------------------------")
                    logger.debug(f"Precond set: {precond_set}")
                    # loop over all unmerged operators
                    for learned_op_other in all_op_per_action - set([learned_op]):
                        satisfy, assignments = check_satisfy_precondition(
                            frozenset(precond_set) | {lifted_action},
                            frozenset(learned_op_other.precond_set)
                            | {learned_op_other.lifted_action},
                        )
                        logger.debug(f"Checked set: {learned_op_other.precond_set}")
                        logger.debug(satisfy)
                        if satisfy:
                            logger.debug(assignments[0])

                            logger.debug("Transform!")
                            logger.debug(str(learned_op_other.lifted_effect_counts))
                            new_effect_counts = transform_effect_counts(
                                learned_op_other.lifted_effect_counts, assignments[0], reverse=True
                            )
                            logger.debug(str(new_effect_counts))
                            merge_number_dict(temp_merged_effect_counts, new_effect_counts)
                            current_objective += compute_entropy(
                                learned_op_other.lifted_effect_counts_list, unnormalize=True
                            )  # remove
                            candidate_op_merged.add(learned_op_other)

                    if allow_merge:
                        # unify effects
                        fixed_vars = set([v for lit in precond_set for v in lit.variables]) | set(
                            lifted_action.variables
                        )
                        logger.debug(f"unify: {temp_merged_effect_counts}")
                        unify_effects(temp_merged_effect_counts, fixed_vars=fixed_vars)
                        logger.debug(f"after unify: {temp_merged_effect_counts}")
                        # assume all effects counts are merged into the current operator
                        current_objective -= compute_entropy(
                            list(temp_merged_effect_counts.values()), unnormalize=True
                        )
                        logger.debug(f"{current_objective}, {best_objective}")
                        if current_objective >= best_objective - DELTA:
                            logger.debug("Merge!")
                            # better objective, accept lit removal & merge operators
                            best_objective = current_objective
                            all_op_per_action -= candidate_op_merged
                            learned_op.lifted_effect_counts = temp_merged_effect_counts
                            # import pdb

                            # pdb.set_trace()

                            continue

                    # back to the previous state
                    precond_set.add(lit)

                # update preconds
                logger.debug(str(precond_set))
                learned_op.precond_set = precond_set

            # Finalize learned operators
            op_idx = 1
            for final_op in all_op_per_action:
                op_name = f"{action_name}_{op_idx}"
                final_op.create_pddl_operator(op_name)

                op_idx += 1

            logger.info(
                colored(
                    f"Learned {len(all_op_per_action)} operators for action {action_name}, with {op_idx-1} valid.",
                    "yellow",
                )
            )
            # import pdb

            # pdb.set_trace()

    @staticmethod
    def _compute_objective(learned_operators: Set["LearnedOperator"], normalize: bool) -> float:
        """
        Compute log likelihood of observing the lifted transitions given the list of learned operators.
        """
        objective = 0.0
        sum_count = 0
        for learned_op in learned_operators:
            objective -= compute_entropy(learned_op.lifted_effect_counts_list, unnormalize=True)
            sum_count += learned_op.num_transitions

        if normalize and sum_count > 0:
            objective /= sum_count
        return objective

    @staticmethod
    def _compute_effect(
        lits_before: FrozenSet[Literal], lits_after: FrozenSet[Literal]
    ) -> FrozenSet[Literal]:
        # all literals are positive
        effect = set()
        for lit_after in lits_after:
            if lit_after not in lits_before:
                effect.add(lit_after)

        for lit_before in lits_before:
            if lit_before not in lits_after:
                effect.add(lit_before.inverted_anti)

        return frozenset(effect)

    # def _compute_lifted_effect(
    #     self,
    #     action_name: str,
    #     lits_before: FrozenSet[Literal],
    #     lits_after: FrozenSet[Literal],
    #     assignment: Dict[str, str],
    # ) -> :
    #     """
    #     Compute lifted effects of an observed transition.
    #     If the effect matches existing lifted effects, return the matched one; otherwise create a new one.
    #     """
    #     effect = self._compute_effect(lits_before, lits_after)

    #     # check if the effect matches existing lifted effects
    #     if action_name in self._lifted_effects:
    #         for effect_id, clustered_effect in enumerate(self._lifted_effects[action_name]):
    #             is_eq, new_assign = check_eq_literals(
    #                 effect,
    #                 clustered_effect,
    #                 init_assignment=assignment,
    #             )
    #             if len(effect) == 0:
    #                 print(is_eq, new_assign)
    #                 import pdb

    #                 pdb.set_trace()

    #             if is_eq:
    #                 assignment.update(new_assign)
    #                 return clustered_effect, effect_id

    #     # if no match found, create new lifted effect
    #     new_effect = lifted_literals(effect, assignment, assign_unknown_vars=True)
    #     if action_name not in self._lifted_effects:
    #         self._lifted_effects[action_name] = []

    #     effect_id = len(self._lifted_effects[action_name])
    #     self._lifted_effects[action_name].append(new_effect)

    #     return new_effect, effect_id

    # def _compute_lifted_preimage(
    #     self,
    #     action_name: str,
    #     lits_before: FrozenSet[Literal],
    #     assignment: Dict[str, str],
    # ):
    #     if action_name in self._lifted_preimages:
    #         for preimage_id, preimage in enumerate(self._lifted_preimages[action_name]):
    #             # Enforce new assignment must be made to free variables
    #             is_eq, new_assign = check_eq_literals(
    #                 lits_before,
    #                 preimage,
    #                 init_assignment=assignment,
    #                 enforce_free_to_free=True,
    #             )

    #             if is_eq:
    #                 assignment.update(new_assign)
    #                 return preimage, preimage_id

    #     # Create new lifted preimage (assign free variables)
    #     new_preimage = lifted_literals(
    #         lits_before,
    #         assignment,
    #         assign_unknown_vars=True,
    #         assign_unknown_as_free_vars=True,
    #     )
    #     if action_name not in self._lifted_preimages:
    #         self._lifted_preimages[action_name] = []

    #     preimage_id = len(self._lifted_preimages[action_name])
    #     self._lifted_preimages[action_name].append(new_preimage)
    #     return new_preimage, preimage_id


class LearnedOperator:
    """
    For each lifted action and each preimage, we initialize a LearnedOperator.
    """

    def __init__(self, lifted_action: Literal, lifted_preimage: FrozenSet[Literal]):
        self._lifted_action = lifted_action
        self._lifted_preimage = lifted_preimage
        self._lifted_effect_counts = {}  # Dict[FrozenSet[Literal], count]

        # initialize preconds
        self._preconds = copy.deepcopy(lifted_preimage)

        # necessary preconditions from human
        # self._known_preconds = necessary_preconds

        # pddl operators (to be set)
        self._pddl_operator = None

    @property
    def lifted_action(self):
        return self._lifted_action

    @property
    def lifted_preimage(self):
        return self._lifted_preimage

    @property
    def lifted_effect_counts_list(self):
        return list(self._lifted_effect_counts.values())

    @property
    def lifted_effect_counts(self):
        return self._lifted_effect_counts

    @lifted_effect_counts.setter
    def lifted_effect_counts(self, lifted_effect_counts: Dict[FrozenSet[Literal], int]):
        self._lifted_effect_counts = lifted_effect_counts

    @property
    def determinized_effect(self):
        assert len(self._lifted_effect_counts) > 0, "No effect found!"
        return max(self._lifted_effect_counts, key=self._lifted_effect_counts.get)

    @property
    def precond_set(self):
        return set(self._preconds)

    @precond_set.setter
    def precond_set(self, preconds: Set[Literal]):
        self._preconds = frozenset(preconds)

    @property
    def num_transitions(self):
        return sum(self._lifted_effect_counts.values())

    def add_effect(self, lifted_effect: FrozenSet[Literal]):
        merge_number_dict(self._lifted_effect_counts, {lifted_effect: 1})

    def add_effects(self, lifted_effect_counts: Dict[FrozenSet[Literal], int]):
        merge_number_dict(self._lifted_effect_counts, lifted_effect_counts)

    def replace_variables(self, change_from: str, change_to: str):
        """
        Replace all variables with name change_from to change_to.
        """
        # calculate replace variables
        variables = (
            set(self._lifted_action.variables)
            | set([v for lit in self._lifted_preimage for v in lit.variables])
            | set(
                [
                    v
                    for effect in self._lifted_effect_counts.keys()
                    for lit in effect
                    for v in lit.variables
                ]
            )
        )
        assignment = {
            var: TypedEntity(var.name.replace(change_from, change_to), var.var_type)
            for var in variables
        }

        # replace variables in action
        self._lifted_action = lifted_literal(
            self._lifted_action, assignment, assign_unknown_vars=False
        )
        # replace variables in preconditions
        self._preconds = lifted_literals(self._preconds, assignment, assign_unknown_vars=False)
        # replace variables in effects
        self._lifted_effect_counts = transform_effect_counts(self._lifted_effect_counts, assignment)
        # replace variables in preimage
        self._lifted_preimage = lifted_literals(
            self._lifted_preimage, assignment, assign_unknown_vars=False
        )

    def create_pddl_operator(self, op_name: str):
        # effect
        effect_list = list(self.determinized_effect)
        if len(effect_list) == 0:
            effect_list.append(NoChange())
        # precond
        precond_list = list(self._preconds) + [self._lifted_action]
        # variables
        variables = set(v for lit in (precond_list + effect_list) for v in lit.variables)

        self._pddl_operator = Operator(
            op_name,
            variables,
            preconds=LiteralConjunction(precond_list),
            effects=LiteralConjunction(effect_list),
        )
        print(self._pddl_operator)


def compute_entropy(counts: List[int], unnormalize: bool = False):
    """
    Compute entropy given counts.
        If unnormalize: return -\sum_{i} p_i * log(p_i) * sum_count
        Else: -\sum_{i} p_i * log(p_i)
    """
    entropy = 0
    count_sum = sum(counts)
    for count in counts:
        assert count > 0, "Effect count should be positive number."
        prob = count / count_sum
        entropy -= prob * math.log(prob)

    if unnormalize:
        entropy *= count_sum
    return entropy


def unify_effects(effects: Dict[FrozenSet[Literal], int], fixed_vars: Set[TypedEntity] = set()):
    """
    Unify effects by merging effects with the same set of fixed variables.
    In place operation.
    """
    merged_effects = set()
    init_assign = {var: var for var in fixed_vars}
    for effect in list(effects.keys()):
        if effect in merged_effects:
            continue

        for other_effect in set(effects.keys()) - set([effect]):
            if_eq, _ = check_eq_literals(effect, other_effect, init_assignment=init_assign)
            if if_eq:
                effects[effect] += effects[other_effect]
                merged_effects.add(other_effect)
                effects.pop(other_effect)


def transform_effect_counts(
    effect_counts: Dict[FrozenSet[Literal], int],
    assignment: Dict[TypedEntity, TypedEntity],
    reverse: bool = False,
):
    if reverse:
        assignment = {v: k for k, v in assignment.items()}

    new_effect_counts = {}
    for effect, count in effect_counts.items():
        new_effect = lifted_literals(effect, assignment, assign_unknown_vars=False)
        new_effect_counts[new_effect] = count
    return new_effect_counts
