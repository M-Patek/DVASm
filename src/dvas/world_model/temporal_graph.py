"""Temporal graphs for World Model training data.

Provides graph structures for representing temporal sequences:
- TemporalEventGraph: Event sequences and timing relationships
- ObjectStateTransitionGraph: State transitions per object
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """Type of temporal event."""

    ACTION_START = "action_start"
    ACTION_END = "action_end"
    CONTACT_START = "contact_start"
    CONTACT_END = "contact_end"
    STATE_CHANGE = "state_change"
    MOTION_START = "motion_start"
    MOTION_END = "motion_end"
    COLLISION = "collision"
    GOAL_REACHED = "goal_reached"
    FAILURE = "failure"
    CUSTOM = "custom"


class TemporalRelationType(str, Enum):
    """Type of temporal relation between events."""

    BEFORE = "before"  # A happens before B
    AFTER = "after"  # A happens after B
    DURING = "during"  # A happens during B
    CONTAINS = "contains"  # A contains B
    MEETS = "meets"  # A ends when B starts
    OVERLAPS = "overlaps"  # A overlaps with B
    EQUALS = "equals"  # A and B at same time
    CAUSES = "causes"  # A causes B


@dataclass
class TemporalEvent:
    """A temporal event in a sequence.

    Attributes:
        event_id: Unique identifier
        event_type: Type of event
        timestamp: When the event occurs
        duration: Event duration (0 for instantaneous)
        description: Human-readable description
        objects: Object IDs involved
        source: Source of event detection
        attributes: Additional event data
    """

    event_id: str
    event_type: EventType
    timestamp: float = 0.0
    duration: float = 0.0
    description: str = ""
    objects: List[str] = field(default_factory=list)
    source: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        """Get event end time."""
        return self.timestamp + self.duration

    @property
    def is_instantaneous(self) -> bool:
        """Check if event is instantaneous."""
        return self.duration == 0.0

    def overlaps(self, other: TemporalEvent) -> bool:
        """Check if this event overlaps with another."""
        return self.timestamp < other.end_time and other.timestamp < self.end_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "description": self.description,
            "objects": self.objects,
            "source": self.source,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemporalEvent:
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data.get("event_type", "custom")),
            timestamp=data.get("timestamp", 0.0),
            duration=data.get("duration", 0.0),
            description=data.get("description", ""),
            objects=data.get("objects", []),
            source=data.get("source", "unknown"),
            attributes=data.get("attributes", {}),
        )


@dataclass
class TemporalRelation:
    """Relation between two temporal events.

    Attributes:
        event_a_id: First event ID
        event_b_id: Second event ID
        relation_type: Type of temporal relation
        confidence: Confidence in the relation
        attributes: Additional relation data
    """

    event_a_id: str
    event_b_id: str
    relation_type: TemporalRelationType
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_a_id": self.event_a_id,
            "event_b_id": self.event_b_id,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemporalRelation:
        """Create from dictionary."""
        return cls(
            event_a_id=data["event_a_id"],
            event_b_id=data["event_b_id"],
            relation_type=TemporalRelationType(data.get("relation_type", "before")),
            confidence=data.get("confidence", 1.0),
            attributes=data.get("attributes", {}),
        )


class TemporalEventGraph:
    """Graph of temporal events and their relations.

    Represents a sequence of events with temporal relationships,
    enabling reasoning about causality and event ordering.

    Attributes:
        events: Dictionary of event_id -> TemporalEvent
        relations: List of temporal relations between events
    """

    def __init__(
        self,
        events: Optional[Dict[str, TemporalEvent]] = None,
        relations: Optional[List[TemporalRelation]] = None,
    ):
        self.events: Dict[str, TemporalEvent] = events or {}
        self.relations: List[TemporalRelation] = relations or []
        self._adjacency: Optional[Dict[str, List[Tuple[str, TemporalRelationType]]]] = None

    def add_event(self, event: TemporalEvent) -> None:
        """Add an event to the graph."""
        self.events[event.event_id] = event
        self._adjacency = None

    def add_relation(self, relation: TemporalRelation) -> None:
        """Add a temporal relation."""
        if relation.event_a_id in self.events and relation.event_b_id in self.events:
            self.relations.append(relation)
            self._adjacency = None
        else:
            logger.warning(
                "temporal_graph_add_relation_skip",
                event_a=relation.event_a_id,
                event_b=relation.event_b_id,
                reason="event_not_found",
            )

    def get_event(self, event_id: str) -> Optional[TemporalEvent]:
        """Get an event by ID."""
        return self.events.get(event_id)

    def get_events_at_time(
        self,
        timestamp: float,
        event_type: Optional[EventType] = None,
    ) -> List[TemporalEvent]:
        """Get events occurring at a specific time."""
        events = [e for e in self.events.values() if e.timestamp <= timestamp <= e.end_time]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events

    def get_events_in_range(
        self,
        start: float,
        end: float,
        event_type: Optional[EventType] = None,
    ) -> List[TemporalEvent]:
        """Get events in a time range."""
        events = [e for e in self.events.values() if e.timestamp < end and e.end_time > start]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events

    def get_events_involving_object(self, object_id: str) -> List[TemporalEvent]:
        """Get all events involving a specific object."""
        return [e for e in self.events.values() if object_id in e.objects]

    def get_relations_for_event(self, event_id: str) -> List[TemporalRelation]:
        """Get all relations involving an event."""
        return [r for r in self.relations if r.event_a_id == event_id or r.event_b_id == event_id]

    def get_causal_chain(self, start_event_id: str) -> List[str]:
        """Get the causal chain starting from an event.

        Follows CAUSES relations to build a chain of causally
        related events.

        Returns:
            List of event IDs in causal order
        """
        chain = [start_event_id]
        current = start_event_id

        while True:
            # Find event that current causes
            next_events = [
                r.event_b_id
                for r in self.relations
                if r.event_a_id == current and r.relation_type == TemporalRelationType.CAUSES
            ]
            if not next_events:
                break
            next_event = next_events[0]
            if next_event in chain:  # Cycle detection
                break
            chain.append(next_event)
            current = next_event

        return chain

    def get_event_order(self) -> List[str]:
        """Get event IDs sorted by timestamp."""
        return [e.event_id for e in sorted(self.events.values(), key=lambda x: x.timestamp)]

    def get_parallel_events(self) -> List[List[str]]:
        """Group overlapping events into parallel groups.

        Returns:
            List of event ID groups that occur in parallel
        """
        if not self.events:
            return []

        # Sort by start time
        sorted_events = sorted(self.events.values(), key=lambda x: x.timestamp)
        groups: List[List[str]] = []
        current_group: List[str] = []
        group_end = 0.0

        for event in sorted_events:
            if event.timestamp >= group_end:
                # New group
                if current_group:
                    groups.append(current_group)
                current_group = [event.event_id]
                group_end = event.end_time
            else:
                # Add to current group
                current_group.append(event.event_id)
                group_end = max(group_end, event.end_time)

        if current_group:
            groups.append(current_group)

        return groups

    def infer_relations(self) -> List[TemporalRelation]:
        """Infer temporal relations from timestamps.

        Automatically creates BEFORE/AFTER/DURING/OVERLAPS relations
        based on event timing.

        Returns:
            List of inferred relations
        """
        inferred = []
        events_list = list(self.events.values())

        for i, event_a in enumerate(events_list):
            for event_b in events_list[i + 1 :]:
                if event_a.timestamp == event_b.timestamp and event_a.duration == event_b.duration:
                    relation = TemporalRelation(
                        event_a_id=event_a.event_id,
                        event_b_id=event_b.event_id,
                        relation_type=TemporalRelationType.EQUALS,
                        confidence=1.0,
                    )
                elif event_a.end_time == event_b.timestamp:
                    relation = TemporalRelation(
                        event_a_id=event_a.event_id,
                        event_b_id=event_b.event_id,
                        relation_type=TemporalRelationType.MEETS,
                        confidence=1.0,
                    )
                elif event_a.timestamp < event_b.timestamp:
                    if event_a.end_time > event_b.timestamp:
                        relation = TemporalRelation(
                            event_a_id=event_a.event_id,
                            event_b_id=event_b.event_id,
                            relation_type=TemporalRelationType.OVERLAPS,
                            confidence=1.0,
                        )
                    else:
                        relation = TemporalRelation(
                            event_a_id=event_a.event_id,
                            event_b_id=event_b.event_id,
                            relation_type=TemporalRelationType.BEFORE,
                            confidence=1.0,
                        )
                else:
                    continue

                inferred.append(relation)

        return inferred

    def traverse(
        self,
        start_event_id: str,
        relation_types: Optional[List[TemporalRelationType]] = None,
    ) -> Iterator[TemporalEvent]:
        """Traverse the graph from a starting event.

        Args:
            start_event_id: Event to start from
            relation_types: Only follow these relation types (None = all)

        Yields:
            Events reachable from start
        """
        visited = {start_event_id}
        queue = [start_event_id]

        while queue:
            current_id = queue.pop(0)
            current_event = self.events.get(current_id)
            if current_event:
                yield current_event

            # Find connected events
            for relation in self.relations:
                if relation.event_a_id == current_id:
                    if relation_types is None or relation.relation_type in relation_types:
                        if relation.event_b_id not in visited:
                            visited.add(relation.event_b_id)
                            queue.append(relation.event_b_id)

    def get_duration(self) -> float:
        """Get total duration spanned by events."""
        if not self.events:
            return 0.0
        min_time = min(e.timestamp for e in self.events.values())
        max_time = max(e.end_time for e in self.events.values())
        return max_time - min_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "events": {eid: e.to_dict() for eid, e in self.events.items()},
            "relations": [r.to_dict() for r in self.relations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemporalEventGraph:
        """Create from dictionary."""
        events = {
            eid: TemporalEvent.from_dict(edata) for eid, edata in data.get("events", {}).items()
        }
        relations = [TemporalRelation.from_dict(rdata) for rdata in data.get("relations", [])]
        return cls(events=events, relations=relations)

    def merge(self, other: TemporalEventGraph) -> TemporalEventGraph:
        """Merge another graph into this one."""
        merged_events = {**self.events, **other.events}
        merged_relations = self.relations + other.relations
        return TemporalEventGraph(
            events=merged_events,
            relations=merged_relations,
        )


@dataclass
class StateTransition:
    """A single state transition for an object.

    Attributes:
        object_id: Object that changed state
        from_state: Previous state
        to_state: New state
        timestamp: When transition occurred
        duration: Transition duration
        trigger_event_id: Event that caused the transition
        confidence: Confidence in the transition
    """

    object_id: str
    from_state: str
    to_state: str
    timestamp: float = 0.0
    duration: float = 0.0
    trigger_event_id: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "trigger_event_id": self.trigger_event_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateTransition:
        """Create from dictionary."""
        return cls(
            object_id=data["object_id"],
            from_state=data["from_state"],
            to_state=data["to_state"],
            timestamp=data.get("timestamp", 0.0),
            duration=data.get("duration", 0.0),
            trigger_event_id=data.get("trigger_event_id"),
            confidence=data.get("confidence", 1.0),
        )


class ObjectStateTransitionGraph:
    """Graph of state transitions for objects over time.

    Tracks how objects change state throughout a sequence,
    useful for identifying state patterns and predicting
    future states.

    Attributes:
        object_id: ID of the tracked object
        transitions: Chronological list of state transitions
    """

    def __init__(
        self,
        object_id: str,
        transitions: Optional[List[StateTransition]] = None,
    ):
        self.object_id = object_id
        self.transitions: List[StateTransition] = transitions or []

    def add_transition(self, transition: StateTransition) -> None:
        """Add a state transition."""
        if transition.object_id != self.object_id:
            logger.warning(
                "transition_mismatched_object",
                expected=self.object_id,
                actual=transition.object_id,
            )
            return

        # Insert in chronological order
        insert_idx = len(self.transitions)
        for i, t in enumerate(self.transitions):
            if t.timestamp > transition.timestamp:
                insert_idx = i
                break
        self.transitions.insert(insert_idx, transition)

    def get_state_at_time(self, timestamp: float) -> str:
        """Get the object state at a given time.

        Returns:
            State string, or "unknown" if no state recorded
        """
        current_state = "unknown"
        for t in self.transitions:
            if t.timestamp <= timestamp:
                current_state = t.to_state
            else:
                break
        return current_state

    def get_state_sequence(self) -> List[Tuple[float, str]]:
        """Get the sequence of states over time.

        Returns:
            List of (timestamp, state) tuples
        """
        result = [(0.0, "initial")]
        for t in self.transitions:
            result.append((t.timestamp, t.to_state))
        return result

    def get_transitions_between(
        self,
        start_time: float,
        end_time: float,
    ) -> List[StateTransition]:
        """Get transitions in a time range."""
        return [t for t in self.transitions if start_time <= t.timestamp <= end_time]

    def get_transition_counts(self) -> Dict[str, int]:
        """Count occurrences of each state."""
        counts: Dict[str, int] = {}
        for t in self.transitions:
            counts[t.to_state] = counts.get(t.to_state, 0) + 1
        return counts

    def get_transition_matrix(self) -> Dict[Tuple[str, str], int]:
        """Get transition frequency matrix.

        Returns:
            Dictionary mapping (from_state, to_state) -> count
        """
        matrix: Dict[Tuple[str, str], int] = {}
        for t in self.transitions:
            key = (t.from_state, t.to_state)
            matrix[key] = matrix.get(key, 0) + 1
        return matrix

    def find_cycles(self) -> List[List[str]]:
        """Find cycles in the state transition graph.

        Returns:
            List of state sequences that form cycles
        """
        # Build adjacency
        adj: Dict[str, Set[str]] = {}
        for t in self.transitions:
            if t.from_state not in adj:
                adj[t.from_state] = set()
            adj[t.from_state].add(t.to_state)

        # Find cycles using DFS
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.append(node)

            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = rec_stack.index(neighbor)
                    cycle = rec_stack[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            rec_stack.pop()

        for node in adj:
            if node not in visited:
                dfs(node)

        return cycles

    def predict_next_state(self, current_state: str) -> Optional[str]:
        """Predict next state based on transition history.

        Args:
            current_state: Current object state

        Returns:
            Most likely next state, or None if no history
        """
        candidates: Dict[str, int] = {}
        for t in self.transitions:
            if t.from_state == current_state:
                candidates[t.to_state] = candidates.get(t.to_state, 0) + 1

        if candidates:
            return max(candidates, key=candidates.get)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "transitions": [t.to_dict() for t in self.transitions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObjectStateTransitionGraph:
        """Create from dictionary."""
        return cls(
            object_id=data["object_id"],
            transitions=[StateTransition.from_dict(t) for t in data.get("transitions", [])],
        )

    def merge(self, other: ObjectStateTransitionGraph) -> ObjectStateTransitionGraph:
        """Merge another transition graph for the same object.

        Assumes both graphs represent the same object.
        """
        if other.object_id != self.object_id:
            raise ValueError(
                f"Cannot merge graphs for different objects: {self.object_id} vs {other.object_id}"
            )

        merged_transitions = self.transitions + other.transitions
        merged_transitions.sort(key=lambda x: x.timestamp)

        return ObjectStateTransitionGraph(
            object_id=self.object_id,
            transitions=merged_transitions,
        )


class MultiObjectTransitionGraph:
    """Collection of state transition graphs for multiple objects.

    Provides a centralized view of how all objects in a scene
    change state over time.
    """

    def __init__(
        self,
        object_graphs: Optional[Dict[str, ObjectStateTransitionGraph]] = None,
    ):
        self.object_graphs: Dict[str, ObjectStateTransitionGraph] = object_graphs or {}

    def add_object_graph(self, graph: ObjectStateTransitionGraph) -> None:
        """Add a transition graph for an object."""
        self.object_graphs[graph.object_id] = graph

    def get_object_graph(self, object_id: str) -> Optional[ObjectStateTransitionGraph]:
        """Get transition graph for an object."""
        return self.object_graphs.get(object_id)

    def get_all_transitions_at_time(
        self,
        timestamp: float,
    ) -> List[StateTransition]:
        """Get all transitions occurring at a specific time."""
        transitions = []
        for graph in self.object_graphs.values():
            for t in graph.transitions:
                if abs(t.timestamp - timestamp) < 0.001:  # Small epsilon
                    transitions.append(t)
        return transitions

    def get_states_at_time(self, timestamp: float) -> Dict[str, str]:
        """Get all object states at a specific time."""
        return {
            oid: graph.get_state_at_time(timestamp) for oid, graph in self.object_graphs.items()
        }

    def find_correlated_transitions(
        self,
        time_window: float = 0.1,
    ) -> List[List[str]]:
        """Find groups of transitions that occur together.

        Args:
            time_window: Maximum time difference for correlation

        Returns:
            List of transition groups (by object_id)
        """
        # Collect all transitions with timestamps
        all_transitions: List[Tuple[str, float]] = []
        for oid, graph in self.object_graphs.items():
            for t in graph.transitions:
                all_transitions.append((oid, t.timestamp))

        # Group by time proximity
        all_transitions.sort(key=lambda x: x[1])
        groups: List[List[str]] = []
        current_group: List[str] = []
        group_end: float = 0.0

        for oid, ts in all_transitions:
            if not current_group:
                # First item in group
                current_group = [oid]
                group_end = ts
            elif ts - group_end <= time_window:
                # Within window, add to current group
                current_group.append(oid)
                group_end = ts
            else:
                # Outside window, save current and start new
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [oid]
                _ = ts  # group_start assigned for clarity
                group_end = ts

        if len(current_group) > 1:
            groups.append(current_group)

        return groups

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_graphs": {oid: g.to_dict() for oid, g in self.object_graphs.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MultiObjectTransitionGraph:
        """Create from dictionary."""
        return cls(
            object_graphs={
                oid: ObjectStateTransitionGraph.from_dict(gdata)
                for oid, gdata in data.get("object_graphs", {}).items()
            },
        )
