# Lesson 1: Memory Architecture and Design Patterns

## Introduction

Memory systems are the foundation of intelligent, long-running AI agents. A well-designed memory architecture enables agents to learn from experience, maintain consistency across sessions, and access relevant information efficiently. This lesson teaches you to design multi-tier memory systems optimized for AI agent workloads.

## Multi-Tier Memory System Design

### Core Memory Architecture

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import OrderedDict

class MemoryTier(Enum):
    """Memory system tiers with different characteristics."""
    IMMEDIATE = "immediate"      # <1ms access, very limited size
    WORKING = "working"          # <10ms access, limited size
    SESSION = "session"          # <100ms access, moderate size
    EPISODIC = "episodic"        # <1s access, large size
    SEMANTIC = "semantic"        # <5s access, very large
    ARCHIVED = "archived"        # <30s access, unlimited

@dataclass
class MemoryItem:
    """Represents an item in the memory system."""
    id: str
    content: Any
    tier: MemoryTier
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5
    tags: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)

    def touch(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1

class ImmediateMemory:
    """
    Ultra-fast in-memory cache for current interaction.
    Optimized for <1ms access times.
    """
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.items: OrderedDict[str, MemoryItem] = OrderedDict()

    def store(self, item: MemoryItem) -> bool:
        """Store item in immediate memory."""
        if len(self.items) >= self.capacity:
            # Evict oldest item
            self.items.popitem(last=False)

        self.items[item.id] = item
        self.items.move_to_end(item.id)  # Mark as most recent
        return True

    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve item from immediate memory."""
        if item_id in self.items:
            item = self.items[item_id]
            item.touch()
            self.items.move_to_end(item_id)  # Move to end (most recent)
            return item
        return None

    def get_all(self) -> List[MemoryItem]:
        """Get all items in immediate memory."""
        return list(self.items.values())

class WorkingMemory:
    """
    Fast in-memory storage for active task context.
    Optimized for <10ms access with 100-1000 item capacity.
    """
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.items: Dict[str, MemoryItem] = {}
        self.access_heap: List[Tuple[float, str]] = []  # (score, item_id)

    def store(self, item: MemoryItem) -> bool:
        """Store item in working memory."""
        if len(self.items) >= self.capacity and item.id not in self.items:
            # Evict lowest importance item
            self._evict_lowest_importance()

        self.items[item.id] = item
        self._update_access_heap()
        return True

    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve item from working memory."""
        if item_id in self.items:
            item = self.items[item_id]
            item.touch()
            self._update_access_heap()
            return item
        return None

    def search(
        self,
        query: str,
        tags: Optional[Set[str]] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Search working memory."""
        results = []

        for item in self.items.values():
            # Tag filtering
            if tags and not tags.intersection(item.tags):
                continue

            # Simple content matching (in production, use embeddings)
            if query.lower() in str(item.content).lower():
                results.append(item)

        # Sort by importance and recency
        results.sort(
            key=lambda x: (x.importance, x.last_accessed),
            reverse=True
        )

        return results[:limit]

    def _evict_lowest_importance(self):
        """Evict lowest importance item."""
        if not self.items:
            return

        lowest = min(self.items.values(), key=lambda x: x.importance)
        del self.items[lowest.id]

    def _update_access_heap(self):
        """Update access heap for efficient eviction."""
        import heapq
        self.access_heap = [
            (item.importance + item.access_count * 0.1, item.id)
            for item in self.items.values()
        ]
        heapq.heapify(self.access_heap)

class SessionMemory:
    """
    Memory for current session/conversation.
    Optimized for <100ms access with session-scoped data.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.items: Dict[str, MemoryItem] = {}
        self.conversation_history: List[Dict] = []
        self.session_start: datetime = datetime.now()

    def store(self, item: MemoryItem) -> bool:
        """Store item in session memory."""
        self.items[item.id] = item
        return True

    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve item from session memory."""
        if item_id in self.items:
            item = self.items[item_id]
            item.touch()
            return item
        return None

    def add_conversation_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """Add conversation turn to session history."""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        })

    def get_conversation_history(
        self,
        last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get conversation history."""
        if last_n:
            return self.conversation_history[-last_n:]
        return self.conversation_history

    def get_session_duration(self) -> timedelta:
        """Get session duration."""
        return datetime.now() - self.session_start

    def summarize_session(self) -> Dict:
        """Get session summary."""
        return {
            'session_id': self.session_id,
            'duration': self.get_session_duration(),
            'items_stored': len(self.items),
            'conversation_turns': len(self.conversation_history),
            'start_time': self.session_start
        }

class HierarchicalMemorySystem:
    """
    Complete multi-tier memory system for AI agents.
    Manages automatic promotion/demotion between tiers.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # Initialize all memory tiers
        self.immediate = ImmediateMemory(capacity=10)
        self.working = WorkingMemory(capacity=500)
        self.session = SessionMemory(session_id=f"session_{datetime.now().timestamp()}")

        # Placeholder for persistent tiers (implemented in later lessons)
        self.episodic = None
        self.semantic = None
        self.archived = None

        # Memory management
        self.promotion_queue: List[str] = []
        self.demotion_queue: List[str] = []

        # Statistics
        self.stats = {
            'promotions': 0,
            'demotions': 0,
            'retrievals': 0,
            'stores': 0
        }

    async def store(
        self,
        content: Any,
        tier: MemoryTier = MemoryTier.WORKING,
        importance: float = 0.5,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict] = None
    ) -> MemoryItem:
        """
        Store item in appropriate memory tier.
        """
        item = MemoryItem(
            id=f"mem_{datetime.now().timestamp()}_{hash(str(content)) % 10000}",
            content=content,
            tier=tier,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            importance=importance,
            tags=tags or set(),
            metadata=metadata or {}
        )

        # Store in appropriate tier
        if tier == MemoryTier.IMMEDIATE:
            self.immediate.store(item)
        elif tier == MemoryTier.WORKING:
            self.working.store(item)
        elif tier == MemoryTier.SESSION:
            self.session.store(item)

        self.stats['stores'] += 1

        # Consider automatic promotion based on importance
        if importance > 0.8:
            await self._consider_promotion(item)

        return item

    async def retrieve(
        self,
        item_id: str,
        hint_tier: Optional[MemoryTier] = None
    ) -> Optional[MemoryItem]:
        """
        Retrieve item from memory system.
        Searches all tiers if tier not specified.
        """
        self.stats['retrievals'] += 1

        # If tier hinted, search there first
        if hint_tier:
            item = self._retrieve_from_tier(item_id, hint_tier)
            if item:
                return item

        # Search tiers in order of access speed
        for tier in [MemoryTier.IMMEDIATE, MemoryTier.WORKING, MemoryTier.SESSION]:
            item = self._retrieve_from_tier(item_id, tier)
            if item:
                # Consider promotion if accessed frequently
                if item.access_count > 5:
                    await self._consider_promotion(item)
                return item

        return None

    def _retrieve_from_tier(
        self,
        item_id: str,
        tier: MemoryTier
    ) -> Optional[MemoryItem]:
        """Retrieve item from specific tier."""
        if tier == MemoryTier.IMMEDIATE:
            return self.immediate.retrieve(item_id)
        elif tier == MemoryTier.WORKING:
            return self.working.retrieve(item_id)
        elif tier == MemoryTier.SESSION:
            return self.session.retrieve(item_id)
        return None

    async def _consider_promotion(self, item: MemoryItem):
        """Consider promoting item to faster tier."""
        current_tier = item.tier

        # Promotion criteria
        should_promote = (
            item.access_count > 3 and
            item.importance > 0.6 and
            (datetime.now() - item.created_at).total_seconds() < 3600
        )

        if should_promote:
            target_tier = self._get_promotion_target(current_tier)
            if target_tier:
                await self._promote_item(item, target_tier)

    def _get_promotion_target(self, current_tier: MemoryTier) -> Optional[MemoryTier]:
        """Determine promotion target tier."""
        tier_hierarchy = [
            MemoryTier.ARCHIVED,
            MemoryTier.SEMANTIC,
            MemoryTier.EPISODIC,
            MemoryTier.SESSION,
            MemoryTier.WORKING,
            MemoryTier.IMMEDIATE
        ]

        try:
            current_idx = tier_hierarchy.index(current_tier)
            if current_idx > 0:
                return tier_hierarchy[current_idx - 1]
        except ValueError:
            pass

        return None

    async def _promote_item(self, item: MemoryItem, target_tier: MemoryTier):
        """Promote item to higher tier."""
        # Update item tier
        old_tier = item.tier
        item.tier = target_tier

        # Move to new tier
        if target_tier == MemoryTier.IMMEDIATE:
            self.immediate.store(item)
        elif target_tier == MemoryTier.WORKING:
            self.working.store(item)

        self.stats['promotions'] += 1

    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics."""
        return {
            'agent_id': self.agent_id,
            'immediate_size': len(self.immediate.items),
            'working_size': len(self.working.items),
            'session_size': len(self.session.items),
            'session_duration': self.session.get_session_duration(),
            'operations': self.stats,
            'timestamp': datetime.now()
        }

    def optimize_memory(self):
        """Run memory optimization."""
        # Demote old, low-importance items from working memory
        current_time = datetime.now()

        for item_id, item in list(self.working.items.items()):
            age_hours = (current_time - item.last_accessed).total_seconds() / 3600

            if age_hours > 1 and item.importance < 0.4:
                # Move to session memory
                self.session.store(item)
                del self.working.items[item_id]
                self.stats['demotions'] += 1
```

## Memory Access Patterns and Optimization

### Intelligent Memory Retrieval

```python
class MemoryAccessOptimizer:
    """
    Optimizes memory access patterns and caching.
    """
    def __init__(self, memory_system: HierarchicalMemorySystem):
        self.memory_system = memory_system
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.prediction_cache: Dict[str, List[str]] = {}

    async def retrieve_with_prediction(
        self,
        item_id: str
    ) -> Optional[MemoryItem]:
        """
        Retrieve item and predict next accesses.
        """
        # Retrieve requested item
        item = await self.memory_system.retrieve(item_id)

        if item:
            # Record access pattern
            if item_id not in self.access_patterns:
                self.access_patterns[item_id] = []
            self.access_patterns[item_id].append(datetime.now())

            # Predict and prefetch related items
            predicted_items = self._predict_next_accesses(item_id)
            await self._prefetch_items(predicted_items)

        return item

    def _predict_next_accesses(self, item_id: str) -> List[str]:
        """Predict which items will be accessed next."""
        # Simple prediction based on historical patterns
        # In production, use more sophisticated models

        if item_id in self.prediction_cache:
            return self.prediction_cache[item_id]

        # Analyze co-access patterns
        predictions = []

        # For now, return empty list
        # In production, implement pattern learning
        return predictions

    async def _prefetch_items(self, item_ids: List[str]):
        """Prefetch items into faster memory tiers."""
        for item_id in item_ids:
            # Try to move to working memory
            item = await self.memory_system.retrieve(item_id)
            if item and item.tier not in [MemoryTier.IMMEDIATE, MemoryTier.WORKING]:
                self.memory_system.working.store(item)
```

## Integration with Context Engineering

### Context-Aware Memory System

```python
class ContextAwareMemorySystem:
    """
    Memory system integrated with context engineering principles.
    """
    def __init__(self, agent_id: str):
        self.memory_system = HierarchicalMemorySystem(agent_id)
        self.context_state = {}

    async def store_with_context(
        self,
        content: Any,
        context: Dict,
        importance: Optional[float] = None
    ) -> MemoryItem:
        """
        Store item with context awareness.
        """
        # Calculate importance based on context
        if importance is None:
            importance = self._calculate_contextual_importance(content, context)

        # Extract tags from context
        tags = set(context.get('tags', []))
        if 'domain' in context:
            tags.add(context['domain'])

        # Determine appropriate tier
        tier = self._determine_tier(importance, context)

        # Store with context metadata
        return await self.memory_system.store(
            content=content,
            tier=tier,
            importance=importance,
            tags=tags,
            metadata={'context': context}
        )

    def _calculate_contextual_importance(
        self,
        content: Any,
        context: Dict
    ) -> float:
        """Calculate importance based on context."""
        base_importance = 0.5

        # Boost for critical context
        if context.get('critical', False):
            base_importance += 0.3

        # Boost for user-explicit storage
        if context.get('user_requested', False):
            base_importance += 0.2

        return min(base_importance, 1.0)

    def _determine_tier(self, importance: float, context: Dict) -> MemoryTier:
        """Determine appropriate memory tier."""
        if importance > 0.8:
            return MemoryTier.IMMEDIATE
        elif importance > 0.6:
            return MemoryTier.WORKING
        else:
            return MemoryTier.SESSION
```

## Key Takeaways

1. **Multi-Tier Architecture**: Different tiers optimize for different access patterns and data volumes

2. **Automatic Management**: Items automatically promote/demote based on access patterns

3. **Performance Optimization**: Sub-10ms access for working memory, <1ms for immediate

4. **Context Integration**: Memory system understands and leverages context

5. **Scalable Design**: Architecture supports growth from immediate to archived storage

6. **Statistics Tracking**: Comprehensive metrics for monitoring and optimization

## What's Next

In Lesson 2, we'll implement persistent storage systems that maintain memory across sessions and system restarts.

---

**Practice Exercise**: Build a complete hierarchical memory system. Test with 10,000+ items and verify proper tier management with sub-10ms working memory access times.
