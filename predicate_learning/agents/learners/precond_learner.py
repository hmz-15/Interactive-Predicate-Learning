import logging
from dataclasses import dataclass
from termcolor import colored
from typing import Any, List, Dict, Optional, FrozenSet
from predicate_learning.utils.pddl_util import (
    check_eq_literals,
    check_satisfy_precondition,
    lifted_literal,
    lifted_literals,
    DEFAULT_TYPE,
)
from pddlgym.structs import Predicate, Literal, LiteralConjunction, NoChange
from pddlgym.parser import Operator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class LearnedPreconds:
    lifted_action: Predicate
    lifted_preconds: Optional[FrozenSet[Literal]] = frozenset()


class PrecondLearner:
    """
    Learn action preconditions.
    """

    def __init__(self, action_predicates: List[Predicate]):
        # leared preconditions from human
        self._learned_preconds = {}  # Dict[str, LearnedPreconds]
        for action_predicate in action_predicates:
            lifted_action = action_predicate(
                *[DEFAULT_TYPE(f"?v_{i}") for i in range(action_predicate.arity)]
            )
            self._learned_preconds[action_predicate.name] = LearnedPreconds(
                lifted_action=lifted_action
            )

    @property
    def learned_preconds(self) -> Dict[str, LearnedPreconds]:
        return self._learned_preconds

    @learned_preconds.setter
    def learned_preconds(self, learned_preconds: Dict[str, LearnedPreconds]) -> None:
        self._learned_preconds = learned_preconds

    @property
    def pddl_operators(self) -> Dict[str, Operator]:
        pddl_operators = {}
        for action_name, learned_preconds in self._learned_preconds.items():
            lifted_action = learned_preconds.lifted_action
            lifted_preconds = learned_preconds.lifted_preconds
            pddl_operators[action_name] = Operator(
                name=action_name,
                params=set(v for lit in lifted_preconds | {lifted_action} for v in lit.variables),
                preconds=LiteralConjunction(list(lifted_preconds | {lifted_action})),
                effects=LiteralConjunction([]),
            )
        return pddl_operators

    def add_new_preconds(
        self,
        action: Literal,
        preconds: FrozenSet[Literal],
    ) -> None:
        """
        Learn the preconditions of an action.
        """
        action_name = action.predicate.name
        assign = {}
        if action_name in self._learned_preconds:
            lifted_action = self._learned_preconds[action_name].lifted_action
            satisfy, assign = check_eq_literals([action], [lifted_action])
            assert satisfy, f"{action} != {lifted_action}"
        else:
            lifted_action = lifted_literal(action, assign)
            self._learned_preconds[action_name] = LearnedPreconds(lifted_action=lifted_action)

        # add new preconds
        known_vars = set(assign.keys())
        filtered_preconds = frozenset(
            [lit for lit in preconds if set(lit.variables).issubset(known_vars)]
        )
        lifted_preconds = lifted_literals(filtered_preconds, assign)
        self._learned_preconds[action_name].lifted_preconds |= lifted_preconds

        logger.debug("New preconds!")
        logger.debug(self._learned_preconds[action_name].lifted_action)
        logger.debug(self._learned_preconds[action_name].lifted_preconds)

    def check_action_feasibility(self, action: Literal, current_lits: FrozenSet[Literal]) -> bool:
        """
        Check whether an action is feasible given the current literals.
        """
        action_name = action.predicate.name
        if action_name not in self._learned_preconds:
            return True

        # check whether the action is satisfied
        lifted_action = self._learned_preconds[action_name].lifted_action
        lifted_preconds = self._learned_preconds[action_name].lifted_preconds
        satisfy = check_satisfy_precondition(
            lifted_preconds | {lifted_action}, current_lits | {action}
        )[0]

        return satisfy
