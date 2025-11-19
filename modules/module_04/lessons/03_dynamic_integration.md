# Lesson 3: Dynamic Knowledge Integration

## Introduction

Real-world AI systems must integrate knowledge from multiple dynamic sources, each with different characteristics, reliability levels, and update frequencies. This lesson teaches you to build systems that seamlessly fuse knowledge while handling conflicts, maintaining consistency, and adapting to changing information.

## Multi-Source Knowledge Architecture

### Knowledge Source Management

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio

class SourceType(Enum):
    """Types of knowledge sources."""
    DATABASE = "database"
    API = "api"
    FILE = "file"
    STREAM = "stream"
    WEB = "web"

@dataclass
class KnowledgeSource:
    """Represents a knowledge source."""
    id: str
    name: str
    source_type: SourceType
    reliability: float  # 0-1
    update_frequency: timedelta
    last_updated: datetime
    metadata: Dict = field(default_factory=dict)
    enabled: bool = True

@dataclass
class KnowledgeItem:
    """Item retrieved from a knowledge source."""
    id: str
    content: any
    source_id: str
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

class MultiSourceKnowledgeManager:
    """
    Manages knowledge integration from multiple sources.
    """
    def __init__(self):
        self.sources: Dict[str, KnowledgeSource] = {}
        self.source_adapters: Dict[str, Callable] = {}
        self.knowledge_cache: Dict[str, List[KnowledgeItem]] = {}
        self.update_scheduler = UpdateScheduler()

    def register_source(
        self,
        source: KnowledgeSource,
        adapter: Callable
    ):
        """
        Register a knowledge source with its adapter.

        Args:
            source: KnowledgeSource configuration
            adapter: Callable that fetches data from source
        """
        self.sources[source.id] = source
        self.source_adapters[source.id] = adapter

        # Schedule updates
        self.update_scheduler.schedule_source(source)

    async def query_source(
        self,
        source_id: str,
        query: str,
        context: Optional[Dict] = None
    ) -> List[KnowledgeItem]:
        """Query a specific knowledge source."""
        if source_id not in self.sources:
            raise ValueError(f"Unknown source: {source_id}")

        source = self.sources[source_id]
        if not source.enabled:
            return []

        # Get adapter
        adapter = self.source_adapters[source_id]

        # Execute query
        try:
            results = await adapter(query, context)

            # Wrap in KnowledgeItem
            items = []
            for result in results:
                item = KnowledgeItem(
                    id=f"{source_id}_{result.get('id', 'unknown')}",
                    content=result.get('content'),
                    source_id=source_id,
                    timestamp=datetime.now(),
                    confidence=source.reliability,
                    metadata=result.get('metadata', {})
                )
                items.append(item)

            # Update cache
            cache_key = f"{source_id}:{query}"
            self.knowledge_cache[cache_key] = items

            return items

        except Exception as e:
            print(f"Error querying source {source_id}: {e}")
            return []

    async def query_all_sources(
        self,
        query: str,
        context: Optional[Dict] = None,
        source_filter: Optional[List[str]] = None
    ) -> Dict[str, List[KnowledgeItem]]:
        """Query all enabled sources in parallel."""
        sources_to_query = []

        for source_id, source in self.sources.items():
            if not source.enabled:
                continue
            if source_filter and source_id not in source_filter:
                continue
            sources_to_query.append(source_id)

        # Query all sources in parallel
        tasks = [
            self.query_source(source_id, query, context)
            for source_id in sources_to_query
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize by source
        source_results = {}
        for source_id, result in zip(sources_to_query, results):
            if isinstance(result, Exception):
                source_results[source_id] = []
            else:
                source_results[source_id] = result

        return source_results

class UpdateScheduler:
    """Schedules automatic updates for knowledge sources."""
    def __init__(self):
        self.scheduled_sources: Dict[str, datetime] = {}

    def schedule_source(self, source: KnowledgeSource):
        """Schedule next update for source."""
        next_update = source.last_updated + source.update_frequency
        self.scheduled_sources[source.id] = next_update

    def get_due_updates(self) -> List[str]:
        """Get sources that need updating."""
        now = datetime.now()
        due = []

        for source_id, next_update in self.scheduled_sources.items():
            if next_update <= now:
                due.append(source_id)

        return due
```

## Knowledge Fusion and Conflict Resolution

### Intelligent Knowledge Fusion

```python
from typing import Callable

class KnowledgeFusionEngine:
    """
    Fuses knowledge from multiple sources with conflict resolution.
    """
    def __init__(self):
        self.fusion_strategies: Dict[str, Callable] = {}
        self.conflict_resolvers: Dict[str, Callable] = {}

        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default fusion strategies."""
        self.fusion_strategies['voting'] = self._voting_fusion
        self.fusion_strategies['weighted'] = self._weighted_fusion
        self.fusion_strategies['confidence'] = self._confidence_fusion

        self.conflict_resolvers['latest'] = self._latest_wins
        self.conflict_resolvers['most_reliable'] = self._most_reliable_wins
        self.conflict_resolvers['consensus'] = self._consensus_resolution

    def fuse_knowledge(
        self,
        source_results: Dict[str, List[KnowledgeItem]],
        strategy: str = 'weighted'
    ) -> List[KnowledgeItem]:
        """
        Fuse knowledge from multiple sources.

        Args:
            source_results: Results from each source
            strategy: Fusion strategy to use
        """
        if strategy not in self.fusion_strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        fusion_func = self.fusion_strategies[strategy]
        return fusion_func(source_results)

    def _voting_fusion(
        self,
        source_results: Dict[str, List[KnowledgeItem]]
    ) -> List[KnowledgeItem]:
        """Simple voting-based fusion."""
        # Group similar items
        item_groups = self._group_similar_items(source_results)

        fused_items = []
        for group in item_groups:
            # Majority vote
            if len(group) >= len(source_results) / 2:
                # Take highest confidence item from group
                best_item = max(group, key=lambda x: x.confidence)
                fused_items.append(best_item)

        return fused_items

    def _weighted_fusion(
        self,
        source_results: Dict[str, List[KnowledgeItem]]
    ) -> List[KnowledgeItem]:
        """Weighted fusion based on source reliability."""
        item_groups = self._group_similar_items(source_results)

        fused_items = []
        for group in item_groups:
            # Calculate weighted average confidence
            total_confidence = sum(item.confidence for item in group)
            weighted_confidence = total_confidence / len(source_results)

            if weighted_confidence > 0.5:
                # Create fused item
                best_item = max(group, key=lambda x: x.confidence)
                best_item.confidence = weighted_confidence
                fused_items.append(best_item)

        return fused_items

    def _confidence_fusion(
        self,
        source_results: Dict[str, List[KnowledgeItem]]
    ) -> List[KnowledgeItem]:
        """Fusion based on individual item confidence."""
        all_items = []
        for items in source_results.values():
            all_items.extend(items)

        # Filter by confidence threshold
        high_confidence = [
            item for item in all_items
            if item.confidence > 0.7
        ]

        # Remove duplicates, keeping highest confidence
        unique_items = {}
        for item in high_confidence:
            key = self._get_item_key(item)
            if key not in unique_items or item.confidence > unique_items[key].confidence:
                unique_items[key] = item

        return list(unique_items.values())

    def resolve_conflicts(
        self,
        conflicting_items: List[KnowledgeItem],
        strategy: str = 'most_reliable'
    ) -> KnowledgeItem:
        """
        Resolve conflicts between knowledge items.
        """
        if strategy not in self.conflict_resolvers:
            raise ValueError(f"Unknown resolution strategy: {strategy}")

        resolver = self.conflict_resolvers[strategy]
        return resolver(conflicting_items)

    def _latest_wins(self, items: List[KnowledgeItem]) -> KnowledgeItem:
        """Most recent item wins."""
        return max(items, key=lambda x: x.timestamp)

    def _most_reliable_wins(self, items: List[KnowledgeItem]) -> KnowledgeItem:
        """Highest confidence item wins."""
        return max(items, key=lambda x: x.confidence)

    def _consensus_resolution(self, items: List[KnowledgeItem]) -> KnowledgeItem:
        """Consensus-based resolution."""
        # Group by similar content
        content_counts = {}
        for item in items:
            key = str(item.content)
            if key not in content_counts:
                content_counts[key] = []
            content_counts[key].append(item)

        # Return item with most agreement
        consensus_group = max(content_counts.values(), key=len)
        return max(consensus_group, key=lambda x: x.confidence)

    def _group_similar_items(
        self,
        source_results: Dict[str, List[KnowledgeItem]]
    ) -> List[List[KnowledgeItem]]:
        """Group similar items across sources."""
        all_items = []
        for items in source_results.values():
            all_items.extend(items)

        # Simple grouping by content similarity
        # In production, use semantic similarity
        groups = []
        used = set()

        for item in all_items:
            if id(item) in used:
                continue

            group = [item]
            used.add(id(item))

            # Find similar items
            for other_item in all_items:
                if id(other_item) in used:
                    continue

                if self._are_similar(item, other_item):
                    group.append(other_item)
                    used.add(id(other_item))

            groups.append(group)

        return groups

    def _are_similar(
        self,
        item1: KnowledgeItem,
        item2: KnowledgeItem
    ) -> bool:
        """Check if two items are similar."""
        # Simple heuristic - in production use embeddings
        content1 = str(item1.content).lower()
        content2 = str(item2.content).lower()

        # Check for significant overlap
        words1 = set(content1.split())
        words2 = set(content2.split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        min_size = min(len(words1), len(words2))

        return overlap / min_size > 0.7

    def _get_item_key(self, item: KnowledgeItem) -> str:
        """Generate key for deduplication."""
        return f"{item.source_id}:{hash(str(item.content))}"
```

## Adaptive Knowledge Base Updates

### Dynamic Knowledge Update System

```python
class AdaptiveKnowledgeBase:
    """
    Self-updating knowledge base that adapts to new information.
    """
    def __init__(self):
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.version_history: Dict[str, List[KnowledgeItem]] = {}
        self.staleness_detector = StalenessDetector()
        self.update_policy = UpdatePolicy()

    def add_or_update(self, item: KnowledgeItem):
        """Add new item or update existing."""
        item_key = self._get_item_key(item)

        if item_key in self.knowledge_items:
            # Update existing
            existing_item = self.knowledge_items[item_key]

            # Store version history
            if item_key not in self.version_history:
                self.version_history[item_key] = []
            self.version_history[item_key].append(existing_item)

            # Decide whether to replace
            if self.update_policy.should_update(existing_item, item):
                self.knowledge_items[item_key] = item
        else:
            # Add new
            self.knowledge_items[item_key] = item

    def get_stale_items(self, threshold_hours: int = 24) -> List[KnowledgeItem]:
        """Identify items that may be stale."""
        return self.staleness_detector.detect_stale(
            list(self.knowledge_items.values()),
            threshold_hours
        )

    def refresh_item(self, item_key: str, new_item: KnowledgeItem):
        """Refresh a specific item."""
        self.add_or_update(new_item)

    def get_item_history(self, item_key: str) -> List[KnowledgeItem]:
        """Get version history for an item."""
        return self.version_history.get(item_key, [])

    def _get_item_key(self, item: KnowledgeItem) -> str:
        """Generate consistent key for item."""
        return f"{item.source_id}:{item.id}"

class StalenessDetector:
    """Detects stale knowledge that needs updating."""
    def detect_stale(
        self,
        items: List[KnowledgeItem],
        threshold_hours: int
    ) -> List[KnowledgeItem]:
        """Identify stale items."""
        cutoff = datetime.now() - timedelta(hours=threshold_hours)

        stale_items = []
        for item in items:
            if item.timestamp < cutoff:
                # Check if likely to change
                change_probability = self._estimate_change_probability(item)
                if change_probability > 0.5:
                    stale_items.append(item)

        return stale_items

    def _estimate_change_probability(self, item: KnowledgeItem) -> float:
        """Estimate probability that item has changed."""
        # Factors: age, source type, content type
        age_hours = (datetime.now() - item.timestamp).total_seconds() / 3600

        # Older items more likely to be stale
        age_factor = min(age_hours / 168, 1.0)  # Normalize to week

        # Content-specific factors
        content_factor = 0.5  # Default
        if item.metadata.get('content_type') == 'news':
            content_factor = 0.8
        elif item.metadata.get('content_type') == 'reference':
            content_factor = 0.2

        return age_factor * content_factor

class UpdatePolicy:
    """Policy for updating knowledge items."""
    def should_update(
        self,
        existing: KnowledgeItem,
        new: KnowledgeItem
    ) -> bool:
        """Determine if existing item should be replaced."""
        # Update if new item is more recent
        if new.timestamp > existing.timestamp:
            return True

        # Update if new item has higher confidence
        if new.confidence > existing.confidence * 1.1:  # 10% better
            return True

        return False
```

## Key Takeaways

1. **Multi-Source Integration**: Parallel querying and intelligent fusion

2. **Conflict Resolution**: Multiple strategies for handling disagreements

3. **Dynamic Updates**: Automatic staleness detection and refreshing

4. **Version Control**: Track knowledge evolution over time

5. **Adaptive Policies**: Learn which sources and strategies work best

6. **Scalability**: Async operations for performance

## What's Next

In Lesson 4, we'll explore advanced retrieval strategies combining multiple search methods.

---

**Practice Exercise**: Build a multi-source knowledge system that queries 5+ sources in parallel, fuses results with conflict resolution, and maintains a self-updating knowledge base with 99% consistency.
