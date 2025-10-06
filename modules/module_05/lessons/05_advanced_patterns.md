# Lesson 5: Advanced Memory Patterns

## Introduction

This lesson explores sophisticated memory architectures beyond basic tiered systems: associative networks for relationship-based retrieval, memory-based learning for continuous improvement, context-aware retrieval that adapts to user behavior, and cross-instance synchronization for distributed systems.

## Associative Memory Networks

### Building Association Networks

```python
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import numpy as np
from datetime import datetime

@dataclass
class MemoryAssociation:
    """Association between two memory items."""
    source_id: str
    target_id: str
    association_type: str  # 'causal', 'temporal', 'semantic', 'contextual'
    strength: float  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_activated: datetime = field(default_factory=datetime.now)
    activation_count: int = 0

class AssociativeMemoryNetwork:
    """
    Memory network with associative connections.
    Retrieval spreads activation through related memories.
    """
    def __init__(self, memory_system: 'HierarchicalMemorySystem'):
        self.memory_system = memory_system
        self.associations: Dict[str, List[MemoryAssociation]] = defaultdict(list)
        self.association_index: Dict[Tuple[str, str], MemoryAssociation] = {}

        # Activation parameters
        self.base_activation = 1.0
        self.decay_rate = 0.3
        self.spreading_rate = 0.7
        self.max_spread_depth = 3

    def add_association(
        self,
        source_id: str,
        target_id: str,
        association_type: str,
        strength: float = 0.5
    ):
        """Add association between memories."""
        # Create association
        assoc = MemoryAssociation(
            source_id=source_id,
            target_id=target_id,
            association_type=association_type,
            strength=strength
        )

        # Store in both directions
        self.associations[source_id].append(assoc)
        self.association_index[(source_id, target_id)] = assoc

        # Create reverse association (weaker)
        reverse_assoc = MemoryAssociation(
            source_id=target_id,
            target_id=source_id,
            association_type=f"reverse_{association_type}",
            strength=strength * 0.5
        )
        self.associations[target_id].append(reverse_assoc)
        self.association_index[(target_id, source_id)] = reverse_assoc

    def auto_discover_associations(self):
        """Automatically discover associations between memories."""
        # Get all episodic memories
        if not hasattr(self.memory_system, 'episodic_store'):
            return

        memories = self.memory_system.episodic_store.query(limit=1000)

        # Discover temporal associations (nearby in time)
        self._discover_temporal_associations(memories)

        # Discover semantic associations (similar content)
        self._discover_semantic_associations(memories)

        # Discover contextual associations (shared tags)
        self._discover_contextual_associations(memories)

    def _discover_temporal_associations(self, memories: List['MemoryItem']):
        """Discover associations based on temporal proximity."""
        # Sort by created_at
        sorted_memories = sorted(memories, key=lambda m: m.created_at)

        # Link nearby memories
        for i in range(len(sorted_memories) - 1):
            current = sorted_memories[i]
            next_mem = sorted_memories[i + 1]

            # If within 5 minutes, create temporal association
            time_diff = (next_mem.created_at - current.created_at).total_seconds()
            if time_diff < 300:  # 5 minutes
                strength = max(0.3, 1.0 - (time_diff / 300) * 0.7)

                self.add_association(
                    current.id,
                    next_mem.id,
                    "temporal",
                    strength=strength
                )

    def _discover_semantic_associations(self, memories: List['MemoryItem']):
        """Discover associations based on semantic similarity."""
        # Use simple term overlap for demonstration
        # In production, use embeddings and cosine similarity

        for i, mem1 in enumerate(memories):
            content1_terms = set(str(mem1.content).lower().split())

            for mem2 in memories[i+1:]:
                content2_terms = set(str(mem2.content).lower().split())

                # Calculate Jaccard similarity
                intersection = content1_terms & content2_terms
                union = content1_terms | content2_terms

                if len(union) > 0:
                    similarity = len(intersection) / len(union)

                    # If similar enough, create semantic association
                    if similarity > 0.3:
                        self.add_association(
                            mem1.id,
                            mem2.id,
                            "semantic",
                            strength=similarity
                        )

    def _discover_contextual_associations(self, memories: List['MemoryItem']):
        """Discover associations based on shared context (tags)."""
        # Group by tags
        tag_groups: Dict[str, List['MemoryItem']] = defaultdict(list)

        for mem in memories:
            for tag in mem.tags:
                tag_groups[tag].append(mem)

        # Create associations within tag groups
        for tag, group_memories in tag_groups.items():
            if len(group_memories) < 2:
                continue

            for i, mem1 in enumerate(group_memories):
                for mem2 in group_memories[i+1:]:
                    # Calculate shared tag ratio
                    shared_tags = mem1.tags & mem2.tags
                    all_tags = mem1.tags | mem2.tags

                    if len(all_tags) > 0:
                        strength = len(shared_tags) / len(all_tags)

                        self.add_association(
                            mem1.id,
                            mem2.id,
                            "contextual",
                            strength=strength
                        )

    def spreading_activation_retrieve(
        self,
        seed_id: str,
        max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Retrieve memories using spreading activation.
        Start from seed, spread activation through network.
        """
        # Initialize activation levels
        activation: Dict[str, float] = {seed_id: self.base_activation}
        visited: Set[str] = set()

        # Queue for breadth-first spreading
        queue: List[Tuple[str, int]] = [(seed_id, 0)]  # (memory_id, depth)

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > self.max_spread_depth:
                continue

            visited.add(current_id)
            current_activation = activation.get(current_id, 0)

            # Spread to associated memories
            for assoc in self.associations[current_id]:
                target_id = assoc.target_id

                # Calculate spread activation
                spread = current_activation * self.spreading_rate * assoc.strength
                spread *= (1 - self.decay_rate * depth)  # Decay with depth

                # Update activation
                if target_id not in activation:
                    activation[target_id] = 0
                activation[target_id] += spread

                # Add to queue for further spreading
                if target_id not in visited:
                    queue.append((target_id, depth + 1))

        # Sort by activation level
        sorted_results = sorted(
            activation.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top results (excluding seed)
        return [(item_id, act) for item_id, act in sorted_results[1:max_results+1]]

    def strengthen_association(self, source_id: str, target_id: str, amount: float = 0.1):
        """Strengthen an association through use (Hebbian learning)."""
        key = (source_id, target_id)
        if key in self.association_index:
            assoc = self.association_index[key]
            assoc.strength = min(1.0, assoc.strength + amount)
            assoc.last_activated = datetime.now()
            assoc.activation_count += 1

    def decay_associations(self, decay_amount: float = 0.05):
        """Decay unused associations over time."""
        now = datetime.now()

        for assoc in self.association_index.values():
            # Calculate time since last activation (in days)
            days_inactive = (now - assoc.last_activated).days

            # Decay based on inactivity
            if days_inactive > 0:
                total_decay = decay_amount * days_inactive
                assoc.strength = max(0, assoc.strength - total_decay)
```

## Memory-Based Learning

### Continuous Improvement System

```python
@dataclass
class LearningPattern:
    """Pattern learned from memory usage."""
    pattern_type: str  # 'access', 'sequence', 'context'
    pattern_data: Dict
    confidence: float  # 0.0 to 1.0
    occurrences: int = 1
    success_rate: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

class MemoryBasedLearningSystem:
    """
    Learns from memory access patterns to improve future retrieval.
    """
    def __init__(self, memory_system: 'HierarchicalMemorySystem'):
        self.memory_system = memory_system
        self.patterns: List[LearningPattern] = []
        self.access_history: List[Dict] = []
        self.prediction_cache: Dict[str, List[str]] = {}

    def record_access(
        self,
        item_id: str,
        context: Dict,
        success: bool = True
    ):
        """Record memory access for learning."""
        access_record = {
            'item_id': item_id,
            'context': context,
            'timestamp': datetime.now(),
            'success': success
        }

        self.access_history.append(access_record)

        # Learn patterns periodically
        if len(self.access_history) % 100 == 0:
            self.learn_patterns()

    def learn_patterns(self):
        """Analyze access history to discover patterns."""
        # Learn access patterns
        self._learn_access_patterns()

        # Learn sequential patterns
        self._learn_sequential_patterns()

        # Learn context patterns
        self._learn_context_patterns()

        # Prune weak patterns
        self._prune_patterns()

    def _learn_access_patterns(self):
        """Learn which memories are frequently accessed together."""
        # Group accesses within time windows
        time_window = 300  # 5 minutes

        current_window = []
        last_time = None

        for access in self.access_history[-1000:]:
            timestamp = access['timestamp']

            if last_time is None or (timestamp - last_time).seconds < time_window:
                current_window.append(access['item_id'])
            else:
                # Process completed window
                if len(current_window) > 1:
                    self._record_access_pattern(current_window)
                current_window = [access['item_id']]

            last_time = timestamp

    def _record_access_pattern(self, item_ids: List[str]):
        """Record co-access pattern."""
        # Look for existing pattern
        for pattern in self.patterns:
            if pattern.pattern_type == 'access':
                if set(pattern.pattern_data['items']) == set(item_ids):
                    # Update existing pattern
                    pattern.occurrences += 1
                    pattern.last_seen = datetime.now()
                    pattern.confidence = min(
                        1.0,
                        pattern.confidence + 0.05
                    )
                    return

        # Create new pattern
        new_pattern = LearningPattern(
            pattern_type='access',
            pattern_data={'items': item_ids},
            confidence=0.3
        )
        self.patterns.append(new_pattern)

    def _learn_sequential_patterns(self):
        """Learn common access sequences."""
        # Find repeated sequences of 2-5 items
        for seq_length in range(2, 6):
            sequences: Dict[Tuple, int] = defaultdict(int)

            for i in range(len(self.access_history) - seq_length + 1):
                sequence = tuple(
                    access['item_id']
                    for access in self.access_history[i:i+seq_length]
                )
                sequences[sequence] += 1

            # Record frequent sequences
            for sequence, count in sequences.items():
                if count >= 3:  # Seen at least 3 times
                    self._record_sequential_pattern(list(sequence), count)

    def _record_sequential_pattern(self, sequence: List[str], occurrences: int):
        """Record sequential access pattern."""
        # Check if pattern exists
        for pattern in self.patterns:
            if pattern.pattern_type == 'sequence':
                if pattern.pattern_data['sequence'] == sequence:
                    pattern.occurrences = max(pattern.occurrences, occurrences)
                    pattern.confidence = min(1.0, occurrences / 10)
                    return

        # Create new pattern
        new_pattern = LearningPattern(
            pattern_type='sequence',
            pattern_data={'sequence': sequence},
            confidence=min(1.0, occurrences / 10),
            occurrences=occurrences
        )
        self.patterns.append(new_pattern)

    def _learn_context_patterns(self):
        """Learn context-based access patterns."""
        context_accesses: Dict[str, List[str]] = defaultdict(list)

        # Group accesses by context
        for access in self.access_history[-500:]:
            context_keys = tuple(sorted(access['context'].items()))
            context_str = str(context_keys)
            context_accesses[context_str].append(access['item_id'])

        # Find context patterns
        for context_str, item_ids in context_accesses.items():
            if len(item_ids) >= 3:
                self._record_context_pattern(context_str, item_ids)

    def _record_context_pattern(self, context: str, item_ids: List[str]):
        """Record context-based pattern."""
        # Find most common items
        from collections import Counter
        item_counts = Counter(item_ids)
        common_items = [item for item, count in item_counts.most_common(5)]

        # Check if pattern exists
        for pattern in self.patterns:
            if pattern.pattern_type == 'context':
                if pattern.pattern_data['context'] == context:
                    pattern.pattern_data['items'] = common_items
                    pattern.occurrences = len(item_ids)
                    return

        # Create new pattern
        new_pattern = LearningPattern(
            pattern_type='context',
            pattern_data={
                'context': context,
                'items': common_items
            },
            confidence=0.5,
            occurrences=len(item_ids)
        )
        self.patterns.append(new_pattern)

    def _prune_patterns(self):
        """Remove weak or outdated patterns."""
        now = datetime.now()

        self.patterns = [
            p for p in self.patterns
            if p.confidence > 0.2 and (now - p.last_seen).days < 30
        ]

    def predict_next_access(
        self,
        current_id: str,
        context: Dict = None
    ) -> List[Tuple[str, float]]:
        """Predict likely next memory accesses."""
        predictions: Dict[str, float] = {}

        # Use sequential patterns
        for pattern in self.patterns:
            if pattern.pattern_type == 'sequence':
                sequence = pattern.pattern_data['sequence']
                if current_id in sequence:
                    idx = sequence.index(current_id)
                    if idx < len(sequence) - 1:
                        next_id = sequence[idx + 1]
                        score = pattern.confidence * 0.8
                        predictions[next_id] = max(
                            predictions.get(next_id, 0),
                            score
                        )

        # Use context patterns
        if context:
            context_str = str(tuple(sorted(context.items())))
            for pattern in self.patterns:
                if pattern.pattern_type == 'context':
                    if pattern.pattern_data['context'] == context_str:
                        for item_id in pattern.pattern_data['items']:
                            score = pattern.confidence * 0.6
                            predictions[item_id] = max(
                                predictions.get(item_id, 0),
                                score
                            )

        # Sort by score
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_predictions[:10]

    def prefetch_predicted(
        self,
        current_id: str,
        context: Dict = None
    ):
        """Prefetch predicted memories into higher tiers."""
        predictions = self.predict_next_access(current_id, context)

        # Prefetch top predictions
        for item_id, confidence in predictions[:3]:
            if confidence > 0.5:
                # Try to retrieve into working memory
                item = self.memory_system.retrieve(item_id)
                if item:
                    # Promote to working tier
                    self.memory_system.working.store(item)
```

## Context-Aware Retrieval

### Adaptive Retrieval System

```python
class ContextAwareRetrievalSystem:
    """
    Adapts retrieval strategy based on current context and user behavior.
    """
    def __init__(
        self,
        memory_system: 'HierarchicalMemorySystem',
        learning_system: MemoryBasedLearningSystem
    ):
        self.memory_system = memory_system
        self.learning_system = learning_system
        self.user_profile: Dict = {
            'preferences': {},
            'interaction_history': []
        }

    def context_aware_search(
        self,
        query: str,
        context: Dict,
        max_results: int = 10
    ) -> List['MemoryItem']:
        """Search with context awareness."""
        # Get predictions from learning system
        predicted_items = []
        if 'current_memory_id' in context:
            predictions = self.learning_system.predict_next_access(
                context['current_memory_id'],
                context
            )
            predicted_items = [item_id for item_id, _ in predictions]

        # Perform base search
        base_results = self.memory_system.search(
            query=query,
            tier_preferences={'working': 0.3, 'episodic': 0.7}
        )

        # Boost predicted items
        scored_results = []
        for item in base_results:
            score = 1.0

            # Boost if predicted
            if item.id in predicted_items:
                boost_idx = predicted_items.index(item.id)
                score += (1.0 - boost_idx * 0.1)  # Higher boost for earlier predictions

            # Boost if matches context
            if self._matches_context(item, context):
                score += 0.5

            # Boost if aligns with user preferences
            if self._matches_preferences(item):
                score += 0.3

            scored_results.append((item, score))

        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [item for item, _ in scored_results[:max_results]]

    def _matches_context(self, item: 'MemoryItem', context: Dict) -> bool:
        """Check if item matches current context."""
        # Check tags
        context_tags = set(context.get('tags', []))
        if context_tags & item.tags:
            return True

        # Check time context
        if 'time_range' in context:
            start, end = context['time_range']
            if start <= item.created_at <= end:
                return True

        return False

    def _matches_preferences(self, item: 'MemoryItem') -> bool:
        """Check if item matches user preferences."""
        prefs = self.user_profile['preferences']

        # Check preferred tags
        if 'preferred_tags' in prefs:
            if item.tags & set(prefs['preferred_tags']):
                return True

        # Check preferred importance level
        if 'min_importance' in prefs:
            if item.importance >= prefs['min_importance']:
                return True

        return False

    def update_user_profile(self, interaction: Dict):
        """Update user profile based on interactions."""
        self.user_profile['interaction_history'].append(interaction)

        # Update preferences from recent interactions
        recent = self.user_profile['interaction_history'][-100:]

        # Find commonly accessed tags
        tag_counts = defaultdict(int)
        for interaction in recent:
            if 'item' in interaction:
                for tag in interaction['item'].tags:
                    tag_counts[tag] += 1

        # Update preferred tags
        self.user_profile['preferences']['preferred_tags'] = [
            tag for tag, count in
            sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
```

## Cross-Instance Memory Synchronization

### Distributed Memory System

```python
import json
from pathlib import Path

class CrossInstanceMemorySync:
    """
    Synchronize memory across multiple agent instances.
    """
    def __init__(
        self,
        memory_system: 'HierarchicalMemorySystem',
        instance_id: str,
        sync_dir: str = "memory_sync"
    ):
        self.memory_system = memory_system
        self.instance_id = instance_id
        self.sync_dir = Path(sync_dir)
        self.sync_dir.mkdir(parents=True, exist_ok=True)

        self.sync_log: List[Dict] = []
        self.last_sync: Optional[datetime] = None

    async def export_changes(self, since: Optional[datetime] = None) -> Dict:
        """Export memory changes since last sync."""
        if since is None:
            since = self.last_sync or datetime.min

        changes = {
            'instance_id': self.instance_id,
            'timestamp': datetime.now(),
            'additions': [],
            'modifications': [],
            'deletions': []
        }

        # Get episodic memories created/modified since last sync
        if hasattr(self.memory_system, 'episodic_store'):
            all_memories = self.memory_system.episodic_store.query(limit=10000)

            for item in all_memories:
                if item.created_at > since:
                    changes['additions'].append(self._serialize_item(item))
                elif hasattr(item, 'modified_at') and item.modified_at > since:
                    changes['modifications'].append(self._serialize_item(item))

        return changes

    def _serialize_item(self, item: 'MemoryItem') -> Dict:
        """Serialize memory item for transfer."""
        return {
            'id': item.id,
            'content': str(item.content),
            'importance': item.importance,
            'tags': list(item.tags),
            'created_at': item.created_at.isoformat(),
            'last_accessed': item.last_accessed.isoformat(),
            'access_count': item.access_count,
            'metadata': item.metadata
        }

    async def import_changes(self, changes: Dict):
        """Import changes from another instance."""
        # Import additions
        for item_data in changes['additions']:
            # Check if already exists
            existing = self.memory_system.retrieve(item_data['id'])
            if not existing:
                # Create new memory item
                from module_05.lessons.01_memory_architecture import MemoryItem

                item = MemoryItem(
                    id=item_data['id'],
                    content=item_data['content'],
                    importance=item_data['importance'],
                    tags=set(item_data['tags']),
                    created_at=datetime.fromisoformat(item_data['created_at']),
                    last_accessed=datetime.fromisoformat(item_data['last_accessed']),
                    access_count=item_data['access_count'],
                    metadata=item_data['metadata']
                )

                # Store in episodic memory
                if hasattr(self.memory_system, 'episodic_store'):
                    self.memory_system.episodic_store.store(item)

        # Import modifications
        for item_data in changes['modifications']:
            existing = self.memory_system.retrieve(item_data['id'])
            if existing:
                # Merge changes (prefer newer data)
                modified_at = datetime.fromisoformat(
                    item_data.get('modified_at', item_data['created_at'])
                )

                if not hasattr(existing, 'modified_at') or modified_at > existing.modified_at:
                    # Update existing item
                    existing.content = item_data['content']
                    existing.importance = max(existing.importance, item_data['importance'])
                    existing.tags.update(item_data['tags'])
                    existing.access_count += item_data['access_count']
                    existing.metadata.update(item_data['metadata'])
                    existing.modified_at = modified_at

                    # Store updated item
                    if hasattr(self.memory_system, 'episodic_store'):
                        self.memory_system.episodic_store.store(existing)

        # Log sync
        self.sync_log.append({
            'source_instance': changes['instance_id'],
            'timestamp': datetime.now(),
            'additions': len(changes['additions']),
            'modifications': len(changes['modifications'])
        })

        self.last_sync = datetime.now()

    async def sync_with_peers(self, peer_instances: List[str]):
        """Sync with peer instances."""
        # Export our changes
        our_changes = await self.export_changes()

        # Write to sync directory
        sync_file = self.sync_dir / f"{self.instance_id}_{datetime.now().timestamp()}.json"
        with open(sync_file, 'w') as f:
            # Serialize datetime objects
            serializable = {
                'instance_id': our_changes['instance_id'],
                'timestamp': our_changes['timestamp'].isoformat(),
                'additions': our_changes['additions'],
                'modifications': our_changes['modifications'],
                'deletions': our_changes['deletions']
            }
            json.dump(serializable, f)

        # Import changes from peers
        for sync_file in self.sync_dir.glob("*.json"):
            # Skip our own files
            if sync_file.stem.startswith(self.instance_id):
                continue

            try:
                with open(sync_file, 'r') as f:
                    changes = json.load(f)
                    # Deserialize timestamp
                    changes['timestamp'] = datetime.fromisoformat(changes['timestamp'])

                    # Import if newer than our last sync
                    if self.last_sync is None or changes['timestamp'] > self.last_sync:
                        await self.import_changes(changes)
            except Exception as e:
                print(f"Error importing sync file {sync_file}: {e}")
```

## Key Takeaways

1. **Associative Networks**: Enable relationship-based retrieval through spreading activation

2. **Memory-Based Learning**: System improves automatically from usage patterns

3. **Context-Aware Retrieval**: Adapts to user behavior and current context

4. **Predictive Prefetching**: Anticipates needs based on learned patterns

5. **Cross-Instance Sync**: Share memory across distributed systems

6. **Continuous Improvement**: All patterns strengthen with use, decay without

## Module Summary

You've now mastered comprehensive memory system design:

- **Multi-tier Architecture**: Immediate, working, session, episodic, semantic, archived
- **Persistent Storage**: SQL-based episodic memory with efficient serialization
- **Hierarchical Management**: Automatic consolidation and intelligent promotion/demotion
- **Lifecycle Optimization**: Archiving, compression, deduplication, monitoring
- **Advanced Patterns**: Associative networks, learning systems, context-aware retrieval

These systems form the foundation for long-running, continuously-improving AI agents.

---

**Practice Exercise**: Build a complete memory system integrating all modules: multi-tier hierarchy, persistent storage, automatic consolidation, lifecycle management with archiving, associative network for retrieval, and memory-based learning. Demonstrate the system improves retrieval accuracy over time through pattern learning.
