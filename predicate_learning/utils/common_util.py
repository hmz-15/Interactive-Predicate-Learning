import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, FrozenSet, Union
from pddlgym.structs import Literal, Predicate
from pddlgym.parser import Operator


class FailureType:
    Unknown = -1
    PerceptionFailure = 0
    ExecutionFailure = 1
    PreconditionViolation = 2


@dataclass
class AgentQuery:
    failure_explain: Optional[bool] = False
    non_goal_explain: Optional[bool] = False


@dataclass
class AgentAction:
    action: Union[Literal, None]
    query: Optional[AgentQuery] = AgentQuery()
    pddl_plan_time: Optional[float] = 0.0


@dataclass
class AgentState:
    goal_spec: Optional[str] = ""
    failure_explain: Optional[str] = ""
    non_goal_explain: Optional[str] = ""
    goal_reached_feedback: Optional[bool] = False
    num_llm_calls: Optional[int] = 0
    learn_pred_time: Optional[float] = 0.0
    learn_op_time: Optional[float] = 0.0
    parse_goal_time: Optional[float] = 0.0
    learned_predicates: Optional[Dict[str, Predicate]] = field(default_factory=dict)
    learned_operators: Optional[Dict[str, Operator]] = field(default_factory=dict)


@dataclass
class EnvFeedback:
    observation: Dict[str, Any]
    abstract_observation: Optional[FrozenSet[Literal]] = frozenset()
    success: Optional[bool] = True
    failure_type: Optional[int] = -1
    goal_achieved: Optional[bool] = False
    episode_ends: Optional[bool] = False
    failure_explain: Optional[str] = FailureType.Unknown
    non_goal_explain: Optional[str] = ""


@dataclass
class Trajectory:
    goal_text: str
    entities: List[str]
    state_history: Optional[List[EnvFeedback]] = field(default_factory=list)
    action_history: Optional[List[AgentAction]] = field(default_factory=list)
    agent_state_history: Optional[List[AgentState]] = field(default_factory=list)
    goal_achieved: Optional[bool] = False
    goal_lits: Optional[FrozenSet[Literal]] = frozenset()
    predicates: Optional[Dict[str, Predicate]] = field(default_factory=dict)
    available_actions: Optional[List[str]] = field(default_factory=list)

    def is_all_actions_success(self):
        return all([x.success for x in self.state_history]) and all(
            [x.action for x in self.action_history]
        )
