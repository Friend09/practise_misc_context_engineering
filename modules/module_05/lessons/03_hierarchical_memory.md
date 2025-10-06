# Lesson 3: Hierarchical Memory Management

## Introduction

Hierarchical memory management enables AI agents to maintain both immediate responsiveness and long-term knowledge. This lesson teaches you to build systems that automatically consolidate memories, optimize across tiers, and provide intelligent cross-memory search capabilities.

## Memory Consolidation Processes

### Automatic Memory Consolidation

````python
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

@dataclass
class ConsolidationRule:
    """Rule for memory consolidation."""
    name: str
    source_tier: 'MemoryTier'
    target_tier: 'MemoryTier'
    condition: callable
    priority: int = 5

class MemoryConsolidationEngine:
    """
    Manages automatic memory consolidation across tiers.
    Simulates sleep-like consolidation processes.
    """
    def __init__(self, memory_system: 'HierarchicalMemorySystem'):
        self.memory_system = memory_system
        self.consolidation_rules: List[ConsolidationRule] = []
        self.consolidation_history: List[Dict] = []

        self._register_default_rules()

    def _register_default_rules(self):
        """Register standard consolidation rules."""
        from modules.module_05.lessons import MemoryTier

        # Rule: Frequently accessed session memories to episodic
        self.consolidation_rules.append(ConsolidationRule(
            name="session_to_episodic",
            source_tier=MemoryTier.SESSION,
            target_tier=MemoryTier.EPISODIC,
            condition=lambda item: (
                item.access_count > 5 and
                item.importance > 0.5
            ),
            priority=7
        ))

        # Rule: High-importance working memory to episodic
        self.consolidation_rules.append(ConsolidationRule(
            name="working_to_episodic",
            source_tier=MemoryTier.WORKING,
            target_tier=MemoryTier.EPISODIC,
            condition=lambda item: (
                item.importance > 0.7 and
                (datetime.now() - item.created_at).total_seconds() > 3600
            ),
            priority=8
        ))

        # Rule: Old, low-importance working memory to session
        self.consolidation_rules.append(ConsolidationRule(
            name="working_to_session",
            source_tier=MemoryTier.WORKING,
            target_tier=MemoryTier.SESSION,
            condition=lambda item: (
                item.importance < 0.4 and
                (datetime.now() - item.last_accessed).total_seconds() > 1800
            ),
            priority=5
        ))

    async def consolidate(self) -> Dict:
        """
        Run consolidation across all memory tiers.
        """
        stats = {
            'consolidations': 0,
            'rules_applied': {},
            'timestamp': datetime.now()
        }

        # Sort rules by priority
        sorted_rules = sorted(
            self.consolidation_rules,
            key=lambda r: r.priority,
            reverse=True
        )

        # Apply each rule
        for rule in sorted_rules:
            applied_count = await self._apply_rule(rule)
            stats['rules_applied'][rule.name] = applied_count
            stats['consolidations'] += applied_count

        # Record consolidation
        self.consolidation_history.append(stats)

        return stats

    async def _apply_rule(self, rule: ConsolidationRule) -> int:
        """Apply a single consolidation rule."""
        consolidation_count = 0

        # Get items from source tier
        source_items = self._get_tier_items(rule.source_tier)

        for item in source_items:
            if rule.condition(item):
                # Consolidate to target tier
                await self._consolidate_item(item, rule.target_tier)
                consolidation_count += 1

        return consolidation_count

    def _get_tier_items(self, tier: 'MemoryTier') -> List['MemoryItem']:
        """Get all items from a tier."""
        from modules.module_05.lessons import MemoryTier

        if tier == MemoryTier.IMMEDIATE:
            return self.memory_system.immediate.get_all()
        elif tier == MemoryTier.WORKING:
            return list(self.memory_system.working.items.values())
        elif tier == MemoryTier.SESSION:
            return list(self.memory_system.session.items.values())

        return []

    async def _consolidate_item(
        self,
        item: 'MemoryItem',
        target_tier: 'MemoryTier'
    ):
        """Consolidate item to target tier."""
        from modules.module_05.lessons import MemoryTier

        # Update item tier
        old_tier = item.tier
        item.tier = target_tier

        # Store in target tier
        if target_tier == MemoryTier.EPISODIC:
            # Store in persistent episodic memory
            if hasattr(self.memory_system, 'episodic_store'):
                self.memory_system.episodic_store.store(item)
        elif target_tier == MemoryTier.SESSION:
            self.memory_system.session.store(item)
        elif target_tier == MemoryTier.WORKING:
            self.memory_system.working.store(item)

        # Remove from old tier if different
        if old_tier != target_tier:
            self._remove_from_tier(item.id, old_tier)

    def _remove_from_tier(self, item_id: str, tier: 'MemoryTier'):
        """Remove item from specified tier."""
        from modules.module_05.lessons import MemoryTier

        if tier == MemoryTier.IMMEDIATE:
            if item_id in self.memory_system.immediate.items:
                del self.memory_system.immediate.items[item_id]
        elif tier == MemoryTier.WORKING:
            if item_id in self.memory_system.working.items:
                del self.memory_system.working.items[item_id]
        elif tier == MemoryTier.SESSION:
            if item_id in self.memory_system.session.items:
                del self.memory_system.session.items[item_id]

    async def run_nightly_consolidation(self):
        """
        Run comprehensive consolidation (simulate sleep consolidation).
        """
        # Phase 1: Consolidate high-importance memories
        await self.consolidate()

        # Phase 2: Create semantic associations
        await self._create_semantic_links()

        # Phase 3: Compress and optimize
        await self._compress_similar_memories()

    async def _create_semantic_links(self):
        """Create semantic associations between memories."""
        # Get recent memories
        session_items = list(self.memory_system.session.items.values())

        # Find semantically related memories
        for i, item1 in enumerate(session_items):
            for item2 in session_items[i+1:]:
                similarity = self._calculate_semantic_similarity(item1, item2)

                if similarity > 0.7:
                    # Create association in episodic store
                    if hasattr(self.memory_system, 'episodic_store'):
                        self.memory_system.episodic_store.add_association(
                            item1.id,
                            item2.id,
                            "semantic",
                            similarity
                        )

    def _calculate_semantic_similarity(
        self,
        item1: 'MemoryItem',
        item2: 'MemoryItem'
    ) -> float:
        """Calculate semantic similarity between memories."""
        # Simple tag-based similarity (in production, use embeddings)
        if not item1.tags or not item2.tags:
            return 0.0

        intersection = item1.tags & item2.tags
        union = item1.tags | item2.tags

        if not union:
            return 0.0

        return len(intersection) / len(union)

    async def _compress_similar_memories(self):
        """Compress similar memories to save space."""
        # Group similar memories
        session_items = list(self.memory_system.session.items.values())

        # Simple clustering by tags
        clusters: Dict[frozenset, List['MemoryItem']] = {}

        for item in session_items:
            key = frozenset(item.tags)
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(item)

        # Merge clusters with multiple items
        for tag_set, items in clusters.items():
            if len(items) > 5:
                # Create summary memory
                summary = self._create_summary_memory(items)

                # Store summary
                await self.memory_system.store(
                    content=summary,
                    tier=MemoryTier.SESSION,
                    importance=np.mean([item.importance for item in items]),
                    tags=tag_set,
                    metadata={'type': 'summary', 'count': len(items)}
                )

    def _create_summary_memory(self, items: List['MemoryItem']) -> Dict:
        """Create summary of multiple memories."""
        return {
            'type': 'summary',
            'count': len(items),
            'timespan': {
                'start': min(item.created_at for item in items),
                'end': max(item.created_at for item in items)
            },
            'avg_importance': np.mean([item.importance for item in items]),
            'total_accesses': sum(item.access_count for item in items)
        }

## Memory Promotion and Demotion Algorithms

### Intelligent Tier Management

```python
class MemoryTierManager:
    """
    Manages memory movement between tiers using sophisticated algorithms.
    """
    def __init__(self, memory_system: 'HierarchicalMemorySystem'):
        self.memory_system = memory_system
        self.tier_stats: Dict = {}

    def calculate_promotion_score(self, item: 'MemoryItem') -> float:
        """
        Calculate score for promoting memory to faster tier.
        """
        score = 0.0

        # Factor 1: Access frequency
        age_hours = (datetime.now() - item.created_at).total_seconds() / 3600
        access_rate = item.access_count / max(age_hours, 1.0)
        score += min(access_rate * 10, 30)

        # Factor 2: Recency of access
        recency_hours = (datetime.now() - item.last_accessed).total_seconds() / 3600
        recency_score = 20 / (1 + recency_hours)
        score += recency_score

        # Factor 3: Importance
        score += item.importance * 30

        # Factor 4: Context relevance (if available)
        if 'context_relevance' in item.metadata:
            score += item.metadata['context_relevance'] * 20

        return score

    def calculate_demotion_score(self, item: 'MemoryItem') -> float:
        """
        Calculate score for demoting memory to slower tier.
        """
        score = 0.0

        # Factor 1: Age without access
        hours_since_access = (datetime.now() - item.last_accessed).total_seconds() / 3600
        score += min(hours_since_access, 100)

        # Factor 2: Low importance
        score += (1.0 - item.importance) * 50

        # Factor 3: Low access rate
        age_hours = (datetime.now() - item.created_at).total_seconds() / 3600
        access_rate = item.access_count / max(age_hours, 1.0)
        score += max(0, 20 - access_rate * 10)

        return score

    async def optimize_tier_distribution(self):
        """
        Optimize memory distribution across tiers.
        """
        optimizations = {
            'promotions': 0,
            'demotions': 0,
            'timestamp': datetime.now()
        }

        # Check working memory for promotions/demotions
        for item_id, item in list(self.memory_system.working.items.items()):
            promotion_score = self.calculate_promotion_score(item)
            demotion_score = self.calculate_demotion_score(item)

            if promotion_score > 70:
                # Promote to immediate
                self.memory_system.immediate.store(item)
                optimizations['promotions'] += 1
            elif demotion_score > 60:
                # Demote to session
                self.memory_system.session.store(item)
                del self.memory_system.working.items[item_id]
                optimizations['demotions'] += 1

        return optimizations

    def get_tier_health(self) -> Dict:
        """Get health metrics for each tier."""
        from modules.module_05.lessons import MemoryTier

        health = {}

        # Immediate memory
        immediate_count = len(self.memory_system.immediate.items)
        immediate_capacity = self.memory_system.immediate.capacity
        health['immediate'] = {
            'utilization': immediate_count / immediate_capacity,
            'count': immediate_count,
            'status': 'healthy' if immediate_count < immediate_capacity * 0.9 else 'near_full'
        }

        # Working memory
        working_count = len(self.memory_system.working.items)
        working_capacity = self.memory_system.working.capacity
        health['working'] = {
            'utilization': working_count / working_capacity,
            'count': working_count,
            'status': 'healthy' if working_count < working_capacity * 0.8 else 'near_full'
        }

        # Session memory
        session_count = len(self.memory_system.session.items)
        health['session'] = {
            'count': session_count,
            'duration': self.memory_system.session.get_session_duration(),
            'status': 'healthy'
        }

        return health
````

## Cross-Memory Search and Retrieval

### Unified Memory Search System

```python
class UnifiedMemorySearch:
    """
    Searches across all memory tiers with intelligent ranking.
    """
    def __init__(self, memory_system: 'HierarchicalMemorySystem'):
        self.memory_system = memory_system

    async def search(
        self,
        query: str,
        tags: Optional[Set[str]] = None,
        tier_preferences: Optional[Dict] = None,
        max_results: int = 20
    ) -> List[Tuple['MemoryItem', float]]:
        """
        Search across all memory tiers.

        Returns:
            List of (item, score) tuples sorted by relevance
        """
        results = []

        # Search immediate memory
        immediate_results = await self._search_immediate(query, tags)
        results.extend(immediate_results)

        # Search working memory
        working_results = await self._search_working(query, tags)
        results.extend(working_results)

        # Search session memory
        session_results = await self._search_session(query, tags)
        results.extend(session_results)

        # Search episodic memory (if available)
        if hasattr(self.memory_system, 'episodic_store'):
            episodic_results = await self._search_episodic(query, tags)
            results.extend(episodic_results)

        # Apply tier preferences
        if tier_preferences:
            results = self._apply_tier_preferences(results, tier_preferences)

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate
        results = self._deduplicate_results(results)

        return results[:max_results]

    async def _search_immediate(
        self,
        query: str,
        tags: Optional[Set[str]]
    ) -> List[Tuple['MemoryItem', float]]:
        """Search immediate memory."""
        results = []

        for item in self.memory_system.immediate.get_all():
            score = self._calculate_relevance(item, query, tags)
            if score > 0:
                # Boost for being in immediate memory
                score *= 1.2
                results.append((item, score))

        return results

    async def _search_working(
        self,
        query: str,
        tags: Optional[Set[str]]
    ) -> List[Tuple['MemoryItem', float]]:
        """Search working memory."""
        results = []

        working_results = self.memory_system.working.search(
            query,
            tags=tags,
            limit=100
        )

        for item in working_results:
            score = self._calculate_relevance(item, query, tags)
            if score > 0:
                # Boost for being in working memory
                score *= 1.1
                results.append((item, score))

        return results

    async def _search_session(
        self,
        query: str,
        tags: Optional[Set[str]]
    ) -> List[Tuple['MemoryItem', float]]:
        """Search session memory."""
        results = []

        for item in self.memory_system.session.items.values():
            if tags and not tags.intersection(item.tags):
                continue

            score = self._calculate_relevance(item, query, tags)
            if score > 0:
                results.append((item, score))

        return results

    async def _search_episodic(
        self,
        query: str,
        tags: Optional[Set[str]]
    ) -> List[Tuple['MemoryItem', float]]:
        """Search episodic memory."""
        results = []

        # Query episodic store
        episodic_items = self.memory_system.episodic_store.query(
            tags=list(tags) if tags else None,
            limit=50
        )

        for item in episodic_items:
            score = self._calculate_relevance(item, query, tags)
            if score > 0:
                # Slight penalty for being in slower storage
                score *= 0.9
                results.append((item, score))

        return results

    def _calculate_relevance(
        self,
        item: 'MemoryItem',
        query: str,
        tags: Optional[Set[str]]
    ) -> float:
        """Calculate relevance score."""
        score = 0.0

        # Content matching (simple keyword for now)
        content_str = str(item.content).lower()
        query_lower = query.lower()

        if query_lower in content_str:
            score += 50
        else:
            # Partial matching
            query_words = set(query_lower.split())
            content_words = set(content_str.split())
            overlap = len(query_words & content_words)
            score += overlap * 5

        # Tag matching
        if tags and item.tags:
            tag_overlap = len(tags & item.tags)
            score += tag_overlap * 20

        # Importance boost
        score += item.importance * 10

        # Recency boost
        hours_old = (datetime.now() - item.created_at).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + hours_old / 24.0)
        score += recency_factor * 10

        return score

    def _apply_tier_preferences(
        self,
        results: List[Tuple['MemoryItem', float]],
        preferences: Dict
    ) -> List[Tuple['MemoryItem', float]]:
        """Apply tier preference weights."""
        adjusted = []

        for item, score in results:
            tier_weight = preferences.get(item.tier.value, 1.0)
            adjusted_score = score * tier_weight
            adjusted.append((item, adjusted_score))

        return adjusted

    def _deduplicate_results(
        self,
        results: List[Tuple['MemoryItem', float]]
    ) -> List[Tuple['MemoryItem', float]]:
        """Remove duplicate items, keeping highest score."""
        seen_ids = {}

        for item, score in results:
            if item.id not in seen_ids or score > seen_ids[item.id][1]:
                seen_ids[item.id] = (item, score)

        return list(seen_ids.values())
```

## Key Takeaways

1. **Automatic Consolidation**: Sleep-like processes optimize memory over time

2. **Intelligent Promotion**: Multi-factor scoring determines tier placement

3. **Semantic Linking**: Associations enable sophisticated retrieval

4. **Unified Search**: Single interface searches all tiers efficiently

5. **Health Monitoring**: Track tier utilization and optimize proactively

6. **Compression**: Reduce memory footprint while preserving information

## What's Next

In Lesson 4, we'll explore memory lifecycle management with archiving and optimization.

---

**Practice Exercise**: Build a hierarchical memory system with automatic consolidation. Demonstrate that nightly consolidation reduces working memory by 40% while maintaining <15ms search times across all tiers.
