# Lesson 1: Single-Agent Context Architecture

## Introduction

Based on industry research from Cognition AI and others, **single-agent systems with excellent context engineering outperform multi-agent architectures**. This lesson teaches you to design and implement the context management architecture that makes single-agent systems reliable and powerful.

### Why Single-Agent Architecture Wins

**Multi-Agent Problems:**

- **Context Loss**: Information doesn't transfer effectively between agents
- **Decision Conflicts**: Multiple agents make contradictory choices
- **Coordination Overhead**: Management complexity outweighs benefits
- **System Fragility**: Multiple failure points reduce reliability

**Single-Agent Advantages:**

- **Context Continuity**: All information flows through one system
- **Decision Consistency**: Unified reasoning prevents conflicts
- **Simplified Architecture**: Fewer moving parts, easier to debug
- **Better Performance**: No coordination overhead

## Centralized Context Management Patterns

### The Context Hub Architecture

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import datetime
import json

class ContextState(Enum):
    """States in the context lifecycle."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    OPTIMIZING = "optimizing"
    PERSISTING = "persisting"
    ARCHIVED = "archived"

@dataclass
class ContextItem:
    """Individual context item with metadata."""
    id: str
    content: Any
    type: str
    importance: float  # 0.0 to 1.0
    timestamp: datetime.datetime
    access_count: int = 0
    last_accessed: Optional[datetime.datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def access(self):
        """Record access to this context item."""
        self.access_count += 1
        self.last_accessed = datetime.datetime.now()

class CentralizedContextHub:
    """
    Central hub for all context management in single-agent systems.

    This is the single source of truth for context, ensuring
    consistency and preventing context loss.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = ContextState.INITIALIZING

        # Core context stores
        self.active_context: Dict[str, ContextItem] = {}
        self.context_history: List[Dict] = []
        self.decision_log: List[Dict] = []

        # Context management components
        self.optimizer = ContextOptimizer()
        self.validator = ContextValidator()
        self.persister = ContextPersister()

        # Monitoring and metrics
        self.metrics = ContextMetrics()
        self.health_monitor = ContextHealthMonitor()

        self.state = ContextState.ACTIVE

    def add_context(
        self,
        key: str,
        content: Any,
        importance: float = 0.5,
        context_type: str = "general",
        tags: List[str] = None
    ) -> ContextItem:
        """
        Add new context to the hub.

        Args:
            key: Unique identifier for context
            content: The context content
            importance: Importance score (0.0-1.0)
            context_type: Type of context
            tags: Optional tags for categorization

        Returns:
            The created ContextItem
        """
        # Create context item
        item = ContextItem(
            id=key,
            content=content,
            type=context_type,
            importance=importance,
            timestamp=datetime.datetime.now(),
            tags=tags or []
        )

        # Add to active context
        self.active_context[key] = item

        # Log addition
        self.context_history.append({
            'action': 'add',
            'key': key,
            'type': context_type,
            'timestamp': datetime.datetime.now()
        })

        # Update metrics
        self.metrics.record_addition(item)

        # Trigger optimization if needed
        if self._should_optimize():
            self.optimize_context()

        return item

    def get_context(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        """
        Retrieve context by key.

        Automatically records access for optimization.
        """
        item = self.active_context.get(key)
        if item:
            item.access()
            self.metrics.record_access(item)
            return item.content
        return default

    def get_relevant_context(
        self,
        query: str,
        max_items: int = 10
    ) -> List[ContextItem]:
        """
        Get context relevant to a query.

        Uses importance, recency, and semantic similarity.
        """
        # Score all context items
        scored_items = []
        for item in self.active_context.values():
            score = self._calculate_relevance(item, query)
            scored_items.append((score, item))

        # Sort by score and return top items
        scored_items.sort(reverse=True, key=lambda x: x[0])

        # Access items and return
        relevant = []
        for score, item in scored_items[:max_items]:
            item.access()
            relevant.append(item)

        return relevant

    def update_context(
        self,
        key: str,
        content: Any,
        merge: bool = False
    ):
        """
        Update existing context.

        Args:
            key: Context identifier
            content: New content
            merge: If True, merge with existing; otherwise replace
        """
        item = self.active_context.get(key)
        if not item:
            # Create new if doesn't exist
            self.add_context(key, content)
            return

        # Update content
        if merge and isinstance(item.content, dict) and isinstance(content, dict):
            item.content.update(content)
        else:
            item.content = content

        # Update timestamp
        item.timestamp = datetime.datetime.now()

        # Log update
        self.context_history.append({
            'action': 'update',
            'key': key,
            'merged': merge,
            'timestamp': datetime.datetime.now()
        })

    def remove_context(self, key: str) -> bool:
        """Remove context from active store."""
        if key in self.active_context:
            item = self.active_context.pop(key)

            # Log removal
            self.context_history.append({
                'action': 'remove',
                'key': key,
                'timestamp': datetime.datetime.now()
            })

            self.metrics.record_removal(item)
            return True
        return False

    def log_decision(
        self,
        decision: str,
        rationale: str,
        confidence: float,
        context_used: List[str]
    ):
        """
        Log a decision made using context.

        Critical for Cognition AI Principle 2: Actions carry implicit decisions.
        """
        decision_entry = {
            'decision': decision,
            'rationale': rationale,
            'confidence': confidence,
            'context_used': context_used,
            'timestamp': datetime.datetime.now(),
            'context_snapshot': self._create_snapshot(context_used)
        }

        self.decision_log.append(decision_entry)

        # Update context items used in decision
        for key in context_used:
            if key in self.active_context:
                item = self.active_context[key]
                item.metadata['used_in_decisions'] = \
                    item.metadata.get('used_in_decisions', 0) + 1

    def get_decision_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get decision history for maintaining consistency.

        Supports Cognition AI Principle 1: Share context.
        """
        if limit:
            return self.decision_log[-limit:]
        return self.decision_log

    def optimize_context(self):
        """
        Optimize context storage and organization.
        """
        self.state = ContextState.OPTIMIZING

        # Run optimization
        optimized_context = self.optimizer.optimize(
            self.active_context,
            self.metrics,
            self.decision_log
        )

        # Update active context
        self.active_context = optimized_context

        self.state = ContextState.ACTIVE

    def persist_context(self):
        """Persist context to storage."""
        self.state = ContextState.PERSISTING

        self.persister.persist(
            agent_id=self.agent_id,
            active_context=self.active_context,
            history=self.context_history,
            decisions=self.decision_log
        )

        self.state = ContextState.ACTIVE

    def restore_context(self):
        """Restore context from storage."""
        restored = self.persister.restore(self.agent_id)
        if restored:
            self.active_context = restored['active_context']
            self.context_history = restored['history']
            self.decision_log = restored['decisions']

    def get_health_status(self) -> Dict:
        """Get current health status of context system."""
        return self.health_monitor.assess_health(
            self.active_context,
            self.metrics,
            self.state
        )

    def _calculate_relevance(
        self,
        item: ContextItem,
        query: str
    ) -> float:
        """Calculate relevance score for a context item."""
        # Importance score
        importance_score = item.importance * 0.4

        # Recency score
        age = (datetime.datetime.now() - item.timestamp).total_seconds()
        recency_score = max(0, 1 - (age / 3600)) * 0.3  # Decay over 1 hour

        # Access frequency score
        access_score = min(item.access_count / 10, 1.0) * 0.2

        # Semantic similarity (simplified - use embeddings in production)
        semantic_score = self._calculate_semantic_similarity(
            str(item.content),
            query
        ) * 0.1

        return importance_score + recency_score + access_score + semantic_score

    def _should_optimize(self) -> bool:
        """Determine if optimization is needed."""
        # Optimize if context size exceeds threshold
        if len(self.active_context) > 1000:
            return True

        # Optimize if memory usage is high
        if self.metrics.get_memory_usage() > 0.8:  # 80% threshold
            return True

        return False

    def _create_snapshot(self, context_keys: List[str]) -> Dict:
        """Create snapshot of specific context items."""
        snapshot = {}
        for key in context_keys:
            if key in self.active_context:
                item = self.active_context[key]
                snapshot[key] = {
                    'content': item.content,
                    'importance': item.importance,
                    'timestamp': item.timestamp.isoformat()
                }
        return snapshot

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Simplified version - use sentence embeddings in production.
        """
        # Convert to lowercase sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class ContextOptimizer:
    """Optimizes context storage and organization."""

    def optimize(
        self,
        context: Dict[str, ContextItem],
        metrics: 'ContextMetrics',
        decision_log: List[Dict]
    ) -> Dict[str, ContextItem]:
        """
        Optimize context based on usage patterns.
        """
        # Calculate importance scores
        scored_items = []
        for key, item in context.items():
            score = self._calculate_retention_score(item, decision_log)
            scored_items.append((score, key, item))

        # Sort by score
        scored_items.sort(reverse=True, key=lambda x: x[0])

        # Keep top items (adaptive threshold)
        threshold = self._calculate_threshold(len(context))
        optimized = {}

        for score, key, item in scored_items[:threshold]:
            optimized[key] = item

        return optimized

    def _calculate_retention_score(
        self,
        item: ContextItem,
        decision_log: List[Dict]
    ) -> float:
        """Calculate score for whether to retain context item."""
        # Base importance
        score = item.importance * 0.4

        # Recency
        age_hours = (datetime.datetime.now() - item.timestamp).total_seconds() / 3600
        recency_score = max(0, 1 - (age_hours / 24)) * 0.3
        score += recency_score

        # Access frequency
        access_score = min(item.access_count / 20, 1.0) * 0.2
        score += access_score

        # Decision usage
        decision_usage = item.metadata.get('used_in_decisions', 0)
        decision_score = min(decision_usage / 5, 1.0) * 0.1
        score += decision_score

        return score

    def _calculate_threshold(self, current_size: int) -> int:
        """Calculate how many items to keep."""
        # Keep at least 100 items
        min_items = 100

        # Keep at most 1000 items
        max_items = 1000

        # Adaptive threshold based on size
        if current_size < min_items:
            return current_size
        elif current_size > max_items:
            return max_items
        else:
            return int(current_size * 0.8)  # Keep top 80%


class ContextValidator:
    """Validates context integrity and quality."""

    def validate_context_item(self, item: ContextItem) -> tuple[bool, List[str]]:
        """Validate a single context item."""
        errors = []

        # Check required fields
        if not item.id:
            errors.append("Context item missing ID")

        if item.content is None:
            errors.append("Context item missing content")

        # Check importance range
        if not 0.0 <= item.importance <= 1.0:
            errors.append(f"Importance {item.importance} out of range [0,1]")

        # Check timestamp
        if item.timestamp > datetime.datetime.now():
            errors.append("Context timestamp is in the future")

        return len(errors) == 0, errors

    def validate_context_hub(
        self,
        hub: CentralizedContextHub
    ) -> tuple[bool, List[str]]:
        """Validate entire context hub."""
        errors = []

        # Validate all context items
        for key, item in hub.active_context.items():
            is_valid, item_errors = self.validate_context_item(item)
            if not is_valid:
                errors.extend([f"{key}: {err}" for err in item_errors])

        # Check for consistency
        if len(hub.active_context) == 0 and hub.state == ContextState.ACTIVE:
            errors.append("Warning: No active context in ACTIVE state")

        return len(errors) == 0, errors


class ContextPersister:
    """Handles context persistence to storage."""

    def __init__(self, storage_path: str = "./context_storage"):
        self.storage_path = storage_path
        import os
        os.makedirs(storage_path, exist_ok=True)

    def persist(
        self,
        agent_id: str,
        active_context: Dict[str, ContextItem],
        history: List[Dict],
        decisions: List[Dict]
    ):
        """Persist context to storage."""
        import os

        # Create agent directory
        agent_dir = os.path.join(self.storage_path, agent_id)
        os.makedirs(agent_dir, exist_ok=True)

        # Serialize active context
        context_data = {}
        for key, item in active_context.items():
            context_data[key] = {
                'id': item.id,
                'content': item.content,
                'type': item.type,
                'importance': item.importance,
                'timestamp': item.timestamp.isoformat(),
                'access_count': item.access_count,
                'last_accessed': item.last_accessed.isoformat() if item.last_accessed else None,
                'tags': item.tags,
                'metadata': item.metadata
            }

        # Write to file
        context_file = os.path.join(agent_dir, 'active_context.json')
        with open(context_file, 'w') as f:
            json.dump(context_data, f, indent=2)

        # Persist history and decisions similarly
        # (Implementation omitted for brevity)

    def restore(self, agent_id: str) -> Optional[Dict]:
        """Restore context from storage."""
        import os

        agent_dir = os.path.join(self.storage_path, agent_id)
        context_file = os.path.join(agent_dir, 'active_context.json')

        if not os.path.exists(context_file):
            return None

        # Load context
        with open(context_file, 'r') as f:
            context_data = json.load(f)

        # Deserialize to ContextItems
        active_context = {}
        for key, data in context_data.items():
            item = ContextItem(
                id=data['id'],
                content=data['content'],
                type=data['type'],
                importance=data['importance'],
                timestamp=datetime.datetime.fromisoformat(data['timestamp']),
                access_count=data['access_count'],
                last_accessed=datetime.datetime.fromisoformat(data['last_accessed']) if data['last_accessed'] else None,
                tags=data['tags'],
                metadata=data['metadata']
            )
            active_context[key] = item

        return {
            'active_context': active_context,
            'history': [],  # Load from file
            'decisions': []  # Load from file
        }


class ContextMetrics:
    """Track context system metrics."""

    def __init__(self):
        self.additions = 0
        self.accesses = 0
        self.removals = 0
        self.memory_usage = 0.0

    def record_addition(self, item: ContextItem):
        self.additions += 1

    def record_access(self, item: ContextItem):
        self.accesses += 1

    def record_removal(self, item: ContextItem):
        self.removals += 1

    def get_memory_usage(self) -> float:
        """Get normalized memory usage (0.0-1.0)."""
        return self.memory_usage

    def get_stats(self) -> Dict:
        return {
            'additions': self.additions,
            'accesses': self.accesses,
            'removals': self.removals,
            'memory_usage': self.memory_usage
        }


class ContextHealthMonitor:
    """Monitor context system health."""

    def assess_health(
        self,
        active_context: Dict[str, ContextItem],
        metrics: ContextMetrics,
        state: ContextState
    ) -> Dict:
        """Assess overall health of context system."""
        health = {
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'metrics': metrics.get_stats()
        }

        # Check context size
        if len(active_context) > 1500:
            health['warnings'].append("Context size very large")
            health['status'] = 'degraded'

        # Check memory usage
        if metrics.get_memory_usage() > 0.9:
            health['issues'].append("Memory usage critical")
            health['status'] = 'unhealthy'

        # Check state
        if state not in [ContextState.ACTIVE, ContextState.OPTIMIZING]:
            health['warnings'].append(f"Unusual state: {state.value}")

        return health
```

## Context State Machines and Transitions

### State Machine Implementation

```python
from enum import Enum, auto
from typing import Callable, Dict, Optional

class ContextEvent(Enum):
    """Events that trigger context state transitions."""
    INITIALIZE = auto()
    ACTIVATE = auto()
    OPTIMIZE_REQUEST = auto()
    OPTIMIZATION_COMPLETE = auto()
    PERSIST_REQUEST = auto()
    PERSISTENCE_COMPLETE = auto()
    ARCHIVE_REQUEST = auto()
    ERROR = auto()
    RECOVERY = auto()

class ContextStateMachine:
    """
    Manages context lifecycle through well-defined states.
    """
    def __init__(self):
        self.current_state = ContextState.INITIALIZING
        self.state_history: List[tuple] = []

        # Define valid transitions
        self.transitions: Dict[tuple, Callable] = {
            (ContextState.INITIALIZING, ContextEvent.ACTIVATE):
                self._transition_to_active,
            (ContextState.ACTIVE, ContextEvent.OPTIMIZE_REQUEST):
                self._transition_to_optimizing,
            (ContextState.OPTIMIZING, ContextEvent.OPTIMIZATION_COMPLETE):
                self._transition_to_active,
            (ContextState.ACTIVE, ContextEvent.PERSIST_REQUEST):
                self._transition_to_persisting,
            (ContextState.PERSISTING, ContextEvent.PERSISTENCE_COMPLETE):
                self._transition_to_active,
            (ContextState.ACTIVE, ContextEvent.ARCHIVE_REQUEST):
                self._transition_to_archived,
        }

        # Error handling transitions
        for state in ContextState:
            self.transitions[(state, ContextEvent.ERROR)] = self._handle_error
            self.transitions[(state, ContextEvent.RECOVERY)] = self._handle_recovery

    def trigger_event(
        self,
        event: ContextEvent,
        context_hub: CentralizedContextHub
    ) -> bool:
        """
        Trigger an event and transition states if valid.

        Returns True if transition successful, False otherwise.
        """
        transition_key = (self.current_state, event)

        if transition_key not in self.transitions:
            print(f"Invalid transition: {self.current_state} -> {event}")
            return False

        # Record history
        self.state_history.append((
            self.current_state,
            event,
            datetime.datetime.now()
        ))

        # Execute transition
        transition_fn = self.transitions[transition_key]
        new_state = transition_fn(context_hub)

        # Update state
        old_state = self.current_state
        self.current_state = new_state
        context_hub.state = new_state

        print(f"State transition: {old_state} -> {new_state} (event: {event})")

        return True

    def _transition_to_active(
        self,
        context_hub: CentralizedContextHub
    ) -> ContextState:
        """Transition to active state."""
        # Validate context before activating
        validator = ContextValidator()
        is_valid, errors = validator.validate_context_hub(context_hub)

        if not is_valid:
            print(f"Validation errors: {errors}")
            return ContextState.INITIALIZING

        return ContextState.ACTIVE

    def _transition_to_optimizing(
        self,
        context_hub: CentralizedContextHub
    ) -> ContextState:
        """Transition to optimizing state."""
        return ContextState.OPTIMIZING

    def _transition_to_persisting(
        self,
        context_hub: CentralizedContextHub
    ) -> ContextState:
        """Transition to persisting state."""
        return ContextState.PERSISTING

    def _transition_to_archived(
        self,
        context_hub: CentralizedContextHub
    ) -> ContextState:
        """Transition to archived state."""
        return ContextState.ARCHIVED

    def _handle_error(
        self,
        context_hub: CentralizedContextHub
    ) -> ContextState:
        """Handle error state."""
        print("Error occurred in context system")
        # Return to safe state
        return ContextState.ACTIVE

    def _handle_recovery(
        self,
        context_hub: CentralizedContextHub
    ) -> ContextState:
        """Handle recovery from error."""
        print("Recovering context system")
        return ContextState.ACTIVE
```

## Key Takeaways

1. **Centralized Architecture**: Single hub manages all context, preventing loss and conflicts

2. **Decision Logging**: Track all decisions for consistency (Cognition AI Principle 2)

3. **State Management**: Clear state machine prevents invalid operations

4. **Optimization**: Automatic context optimization maintains performance

5. **Persistence**: Save and restore context across sessions

6. **Health Monitoring**: Continuous monitoring ensures reliability

## What's Next

In Lesson 2, we'll explore context optimization algorithms that automatically improve context quality and performance over time.

---

**Practice Exercise**: Implement a CentralizedContextHub for your application domain. Test with >1000 context items and measure retrieval performance, optimization effectiveness, and decision consistency.
