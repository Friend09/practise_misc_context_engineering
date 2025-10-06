# Lesson 5: Advanced Memory Management

## Introduction

Long-running single-agent systems require sophisticated memory management to maintain performance and relevance over extended periods. This lesson teaches you to build hierarchical memory systems, efficient indexing, and intelligent lifecycle management for context that persists across sessions.

## Hierarchical Memory Architecture

### Multi-Tier Memory System

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime, timedelta
import json
import pickle

class MemoryTier(Enum):
    """Memory storage tiers with different characteristics."""
    WORKING = "working"      # Fast access, limited size
    SHORT_TERM = "short_term"  # Recent context, moderate size
    LONG_TERM = "long_term"    # Persistent, unlimited size
    ARCHIVE = "archive"        # Cold storage, compressed

@dataclass
class MemoryItem:
    """Item stored in memory system."""
    id: str
    content: any
    tier: MemoryTier
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5
    metadata: Dict = field(default_factory=dict)

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1

class HierarchicalMemoryManager:
    """
    Manages memory across multiple tiers with automatic promotion/demotion.
    """
    def __init__(
        self,
        working_capacity: int = 100,
        short_term_capacity: int = 1000,
        long_term_capacity: Optional[int] = None
    ):
        self.capacities = {
            MemoryTier.WORKING: working_capacity,
            MemoryTier.SHORT_TERM: short_term_capacity,
            MemoryTier.LONG_TERM: long_term_capacity or float('inf')
        }

        self.memory_tiers: Dict[MemoryTier, Dict[str, MemoryItem]] = {
            tier: {} for tier in MemoryTier
        }

        self.access_patterns: Dict[str, List[datetime]] = {}
        self.tier_transitions: List[Dict] = []

    def store(
        self,
        item_id: str,
        content: any,
        importance: float = 0.5,
        tier: MemoryTier = MemoryTier.WORKING,
        metadata: Optional[Dict] = None
    ) -> MemoryItem:
        """
        Store item in specified memory tier.
        """
        memory_item = MemoryItem(
            id=item_id,
            content=content,
            tier=tier,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )

        # Check capacity and make room if needed
        self._ensure_capacity(tier)

        # Store in tier
        self.memory_tiers[tier][item_id] = memory_item

        return memory_item

    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """
        Retrieve item from any tier and promote if needed.
        """
        # Search all tiers
        found_tier = None
        item = None

        for tier in MemoryTier:
            if item_id in self.memory_tiers[tier]:
                found_tier = tier
                item = self.memory_tiers[tier][item_id]
                break

        if not item:
            return None

        # Update access
        item.update_access()

        # Track access pattern
        if item_id not in self.access_patterns:
            self.access_patterns[item_id] = []
        self.access_patterns[item_id].append(datetime.now())

        # Consider promotion based on access pattern
        if self._should_promote(item):
            self._promote_item(item)

        return item

    def _should_promote(self, item: MemoryItem) -> bool:
        """Determine if item should be promoted to higher tier."""
        if item.tier == MemoryTier.WORKING:
            return False

        # Check recent access frequency
        recent_accesses = [
            ts for ts in self.access_patterns.get(item.id, [])
            if (datetime.now() - ts).total_seconds() < 3600
        ]

        # Promote if frequently accessed recently
        if len(recent_accesses) >= 3:
            return True

        # Promote high importance items
        if item.importance > 0.8:
            return True

        return False

    def _promote_item(self, item: MemoryItem):
        """Promote item to higher tier."""
        current_tier = item.tier

        # Determine target tier
        tier_order = [
            MemoryTier.ARCHIVE,
            MemoryTier.LONG_TERM,
            MemoryTier.SHORT_TERM,
            MemoryTier.WORKING
        ]

        current_idx = tier_order.index(current_tier)
        if current_idx >= len(tier_order) - 1:
            return

        target_tier = tier_order[current_idx + 1]

        # Move item
        del self.memory_tiers[current_tier][item.id]
        item.tier = target_tier
        self._ensure_capacity(target_tier)
        self.memory_tiers[target_tier][item.id] = item

        # Log transition
        self.tier_transitions.append({
            'item_id': item.id,
            'from_tier': current_tier.value,
            'to_tier': target_tier.value,
            'timestamp': datetime.now(),
            'reason': 'promotion'
        })

    def _demote_item(self, item: MemoryItem):
        """Demote item to lower tier."""
        current_tier = item.tier

        tier_order = [
            MemoryTier.WORKING,
            MemoryTier.SHORT_TERM,
            MemoryTier.LONG_TERM,
            MemoryTier.ARCHIVE
        ]

        current_idx = tier_order.index(current_tier)
        if current_idx >= len(tier_order) - 1:
            return

        target_tier = tier_order[current_idx + 1]

        # Move item
        del self.memory_tiers[current_tier][item.id]
        item.tier = target_tier
        self.memory_tiers[target_tier][item.id] = item

        # Log transition
        self.tier_transitions.append({
            'item_id': item.id,
            'from_tier': current_tier.value,
            'to_tier': target_tier.value,
            'timestamp': datetime.now(),
            'reason': 'demotion'
        })

    def _ensure_capacity(self, tier: MemoryTier):
        """Ensure tier has capacity for new item."""
        capacity = self.capacities[tier]
        current_size = len(self.memory_tiers[tier])

        if current_size < capacity:
            return

        # Need to make room - demote least important items
        items = list(self.memory_tiers[tier].values())

        # Score items for demotion
        scored_items = []
        for item in items:
            score = self._calculate_retention_score(item)
            scored_items.append((score, item))

        # Sort by score (lowest first)
        scored_items.sort(key=lambda x: x[0])

        # Demote lowest scoring items
        items_to_demote = int(capacity * 0.1)  # Demote 10%
        for _, item in scored_items[:items_to_demote]:
            self._demote_item(item)

    def _calculate_retention_score(self, item: MemoryItem) -> float:
        """Calculate score for retaining item in current tier."""
        score = item.importance

        # Factor in recency
        age_hours = (datetime.now() - item.last_accessed).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + age_hours / 24.0)
        score += recency_factor * 0.3

        # Factor in access frequency
        access_frequency = item.access_count / max(1, age_hours)
        score += min(access_frequency * 0.2, 0.3)

        return score

    def optimize_memory(self):
        """Run optimization across all tiers."""
        optimized_count = 0

        for tier in [MemoryTier.WORKING, MemoryTier.SHORT_TERM]:
            items = list(self.memory_tiers[tier].values())

            for item in items:
                # Check if item should be demoted
                age_hours = (datetime.now() - item.last_accessed).total_seconds() / 3600

                if age_hours > 24 and item.access_count < 2:
                    self._demote_item(item)
                    optimized_count += 1

        return optimized_count

    def get_tier_stats(self) -> Dict:
        """Get statistics for each tier."""
        stats = {}

        for tier in MemoryTier:
            items = self.memory_tiers[tier].values()

            if not items:
                stats[tier.value] = {
                    'count': 0,
                    'capacity': self.capacities.get(tier, 0),
                    'utilization': 0.0
                }
                continue

            capacity = self.capacities.get(tier, float('inf'))

            stats[tier.value] = {
                'count': len(items),
                'capacity': capacity,
                'utilization': len(items) / capacity if capacity != float('inf') else 0,
                'avg_importance': np.mean([i.importance for i in items]),
                'avg_age_hours': np.mean([
                    (datetime.now() - i.created_at).total_seconds() / 3600
                    for i in items
                ])
            }

        return stats
```

## Efficient Memory Indexing

### Multi-Dimensional Memory Index

```python
from typing import Callable

class MemoryIndex:
    """
    Multi-dimensional indexing for efficient memory retrieval.
    """
    def __init__(self):
        self.indexes: Dict[str, Dict] = {
            'by_type': {},
            'by_importance': {},
            'by_time': {},
            'by_tag': {}
        }
        self.custom_indexes: Dict[str, Dict] = {}

    def index_item(self, item: MemoryItem):
        """Add item to all relevant indexes."""
        # Index by type
        item_type = item.metadata.get('type', 'unknown')
        if item_type not in self.indexes['by_type']:
            self.indexes['by_type'][item_type] = []
        self.indexes['by_type'][item_type].append(item.id)

        # Index by importance bucket
        importance_bucket = int(item.importance * 10)
        if importance_bucket not in self.indexes['by_importance']:
            self.indexes['by_importance'][importance_bucket] = []
        self.indexes['by_importance'][importance_bucket].append(item.id)

        # Index by time bucket (day)
        time_bucket = item.created_at.date()
        if time_bucket not in self.indexes['by_time']:
            self.indexes['by_time'][time_bucket] = []
        self.indexes['by_time'][time_bucket].append(item.id)

        # Index by tags
        tags = item.metadata.get('tags', [])
        for tag in tags:
            if tag not in self.indexes['by_tag']:
                self.indexes['by_tag'][tag] = []
            self.indexes['by_tag'][tag].append(item.id)

    def find_by_type(self, item_type: str) -> List[str]:
        """Find items by type."""
        return self.indexes['by_type'].get(item_type, [])

    def find_by_importance_range(
        self,
        min_importance: float,
        max_importance: float
    ) -> List[str]:
        """Find items within importance range."""
        min_bucket = int(min_importance * 10)
        max_bucket = int(max_importance * 10)

        results = []
        for bucket in range(min_bucket, max_bucket + 1):
            results.extend(self.indexes['by_importance'].get(bucket, []))

        return results

    def find_by_tag(self, tag: str) -> List[str]:
        """Find items with specific tag."""
        return self.indexes['by_tag'].get(tag, [])

    def find_recent(self, days: int = 7) -> List[str]:
        """Find items from recent days."""
        cutoff_date = (datetime.now() - timedelta(days=days)).date()

        results = []
        for date, items in self.indexes['by_time'].items():
            if date >= cutoff_date:
                results.extend(items)

        return results

    def create_custom_index(
        self,
        index_name: str,
        key_extractor: Callable[[MemoryItem], str]
    ):
        """Create custom index with key extraction function."""
        self.custom_indexes[index_name] = {
            'data': {},
            'extractor': key_extractor
        }

    def query_custom_index(
        self,
        index_name: str,
        key: str
    ) -> List[str]:
        """Query custom index."""
        if index_name not in self.custom_indexes:
            return []

        return self.custom_indexes[index_name]['data'].get(key, [])
```

## Memory Lifecycle Management

### Automatic Memory Lifecycle System

```python
class MemoryLifecycleManager:
    """
    Manages complete lifecycle of memory items with automatic policies.
    """
    def __init__(self, memory_manager: HierarchicalMemoryManager):
        self.memory_manager = memory_manager
        self.lifecycle_policies: List[Dict] = []
        self.archived_items: Dict[str, Dict] = {}

        # Register default policies
        self._register_default_policies()

    def _register_default_policies(self):
        """Register standard lifecycle policies."""
        # Archive old, low-importance items
        self.add_policy(
            name="archive_old_low_importance",
            condition=lambda item: (
                (datetime.now() - item.last_accessed).days > 30 and
                item.importance < 0.3
            ),
            action=self._archive_item
        )

        # Delete very old archived items
        self.add_policy(
            name="delete_ancient_archives",
            condition=lambda item: (
                item.tier == MemoryTier.ARCHIVE and
                (datetime.now() - item.created_at).days > 365
            ),
            action=self._delete_item
        )

        # Refresh important items
        self.add_policy(
            name="refresh_important",
            condition=lambda item: (
                item.importance > 0.8 and
                (datetime.now() - item.last_accessed).days > 7
            ),
            action=self._refresh_item
        )

    def add_policy(
        self,
        name: str,
        condition: Callable[[MemoryItem], bool],
        action: Callable[[MemoryItem], None]
    ):
        """Add custom lifecycle policy."""
        self.lifecycle_policies.append({
            'name': name,
            'condition': condition,
            'action': action
        })

    def apply_policies(self) -> Dict[str, int]:
        """Apply all lifecycle policies."""
        actions_taken = {}

        # Collect all items
        all_items = []
        for tier_items in self.memory_manager.memory_tiers.values():
            all_items.extend(tier_items.values())

        # Apply each policy
        for policy in self.lifecycle_policies:
            policy_actions = 0

            for item in all_items:
                if policy['condition'](item):
                    policy['action'](item)
                    policy_actions += 1

            actions_taken[policy['name']] = policy_actions

        return actions_taken

    def _archive_item(self, item: MemoryItem):
        """Archive item to cold storage."""
        # Compress content
        compressed = self._compress_content(item.content)

        # Store in archive
        archive_data = {
            'id': item.id,
            'content': compressed,
            'metadata': item.metadata,
            'archived_at': datetime.now(),
            'original_tier': item.tier.value
        }

        self.archived_items[item.id] = archive_data

        # Remove from active memory
        if item.id in self.memory_manager.memory_tiers[item.tier]:
            del self.memory_manager.memory_tiers[item.tier][item.id]

    def _delete_item(self, item: MemoryItem):
        """Permanently delete item."""
        # Remove from memory tiers
        if item.id in self.memory_manager.memory_tiers[item.tier]:
            del self.memory_manager.memory_tiers[item.tier][item.id]

        # Remove from archives
        if item.id in self.archived_items:
            del self.archived_items[item.id]

    def _refresh_item(self, item: MemoryItem):
        """Refresh item to prevent archival."""
        item.last_accessed = datetime.now()
        item.access_count += 1

    def _compress_content(self, content: any) -> bytes:
        """Compress content for archival."""
        import gzip
        serialized = pickle.dumps(content)
        compressed = gzip.compress(serialized)
        return compressed

    def _decompress_content(self, compressed: bytes) -> any:
        """Decompress archived content."""
        import gzip
        decompressed = gzip.decompress(compressed)
        return pickle.loads(decompressed)

    def restore_from_archive(self, item_id: str) -> Optional[MemoryItem]:
        """Restore archived item to active memory."""
        if item_id not in self.archived_items:
            return None

        archive_data = self.archived_items[item_id]

        # Decompress content
        content = self._decompress_content(archive_data['content'])

        # Restore to short-term memory
        restored_item = self.memory_manager.store(
            item_id=item_id,
            content=content,
            tier=MemoryTier.SHORT_TERM,
            metadata=archive_data['metadata']
        )

        # Remove from archives
        del self.archived_items[item_id]

        return restored_item
```

## Cross-Session Memory Persistence

### Persistent Memory Store

```python
import sqlite3
from pathlib import Path

class PersistentMemoryStore:
    """
    Persistent storage for memory across sessions.
    """
    def __init__(self, db_path: str = "memory_store.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                content BLOB,
                tier TEXT,
                importance REAL,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_index (
                item_id TEXT,
                index_type TEXT,
                index_key TEXT,
                FOREIGN KEY (item_id) REFERENCES memory_items(id)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_index_lookup
            ON memory_index(index_type, index_key)
        ''')

        conn.commit()
        conn.close()

    def save_item(self, item: MemoryItem):
        """Save memory item to persistent storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Serialize content
        content_blob = pickle.dumps(item.content)
        metadata_json = json.dumps(item.metadata)

        cursor.execute('''
            INSERT OR REPLACE INTO memory_items
            (id, content, tier, importance, created_at, last_accessed, access_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id,
            content_blob,
            item.tier.value,
            item.importance,
            item.created_at.isoformat(),
            item.last_accessed.isoformat(),
            item.access_count,
            metadata_json
        ))

        conn.commit()
        conn.close()

    def load_item(self, item_id: str) -> Optional[MemoryItem]:
        """Load memory item from persistent storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT content, tier, importance, created_at, last_accessed, access_count, metadata
            FROM memory_items WHERE id = ?
        ''', (item_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        content = pickle.loads(row[0])
        metadata = json.loads(row[6])

        return MemoryItem(
            id=item_id,
            content=content,
            tier=MemoryTier(row[1]),
            importance=row[2],
            created_at=datetime.fromisoformat(row[3]),
            last_accessed=datetime.fromisoformat(row[4]),
            access_count=row[5],
            metadata=metadata
        )

    def save_all(self, memory_manager: HierarchicalMemoryManager):
        """Save all memory items from manager."""
        for tier_items in memory_manager.memory_tiers.values():
            for item in tier_items.values():
                self.save_item(item)

    def load_all(self, memory_manager: HierarchicalMemoryManager):
        """Load all memory items into manager."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM memory_items')
        item_ids = [row[0] for row in cursor.fetchall()]

        conn.close()

        for item_id in item_ids:
            item = self.load_item(item_id)
            if item:
                memory_manager.memory_tiers[item.tier][item_id] = item

    def cleanup_old_items(self, days: int = 365):
        """Remove items older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            DELETE FROM memory_items
            WHERE created_at < ?
        ''', (cutoff.isoformat(),))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted
```

## Complete Example: Production Memory System

```python
class ProductionMemorySystem:
    """
    Complete production-ready memory system.
    """
    def __init__(self, db_path: str = "memory.db"):
        self.memory_manager = HierarchicalMemoryManager(
            working_capacity=100,
            short_term_capacity=1000
        )
        self.index = MemoryIndex()
        self.lifecycle = MemoryLifecycleManager(self.memory_manager)
        self.persistent_store = PersistentMemoryStore(db_path)

        # Load persisted memory
        self.persistent_store.load_all(self.memory_manager)

    def store(
        self,
        item_id: str,
        content: any,
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> MemoryItem:
        """Store item with indexing and persistence."""
        # Store in memory
        item = self.memory_manager.store(
            item_id=item_id,
            content=content,
            importance=importance,
            metadata=metadata
        )

        # Index item
        self.index.index_item(item)

        # Persist to disk
        self.persistent_store.save_item(item)

        return item

    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve item with automatic promotion."""
        item = self.memory_manager.retrieve(item_id)

        if item:
            # Update persistence
            self.persistent_store.save_item(item)

        return item

    def search(
        self,
        by_type: Optional[str] = None,
        by_tag: Optional[str] = None,
        min_importance: Optional[float] = None
    ) -> List[MemoryItem]:
        """Search memory using indexes."""
        result_ids = set()

        if by_type:
            result_ids.update(self.index.find_by_type(by_type))

        if by_tag:
            tag_results = self.index.find_by_tag(by_tag)
            if result_ids:
                result_ids.intersection_update(tag_results)
            else:
                result_ids.update(tag_results)

        if min_importance:
            importance_results = self.index.find_by_importance_range(
                min_importance, 1.0
            )
            if result_ids:
                result_ids.intersection_update(importance_results)
            else:
                result_ids.update(importance_results)

        # Retrieve items
        items = []
        for item_id in result_ids:
            item = self.retrieve(item_id)
            if item:
                items.append(item)

        return items

    def optimize(self):
        """Run full optimization."""
        # Optimize memory tiers
        optimized = self.memory_manager.optimize_memory()

        # Apply lifecycle policies
        policy_actions = self.lifecycle.apply_policies()

        # Persist changes
        self.persistent_store.save_all(self.memory_manager)

        return {
            'optimized_items': optimized,
            'policy_actions': policy_actions
        }

    def get_stats(self) -> Dict:
        """Get comprehensive memory statistics."""
        tier_stats = self.memory_manager.get_tier_stats()

        return {
            'tiers': tier_stats,
            'total_items': sum(
                stats['count'] for stats in tier_stats.values()
            ),
            'archived_items': len(self.lifecycle.archived_items)
        }
```

## Key Takeaways

1. **Hierarchical Architecture**: Multiple memory tiers optimize for different access patterns

2. **Automatic Management**: Items automatically promote/demote based on usage

3. **Efficient Indexing**: Multi-dimensional indexes enable fast retrieval

4. **Lifecycle Policies**: Automated policies manage memory lifecycle

5. **Persistence**: Cross-session persistence ensures continuity

6. **Production-Ready**: Complete system handles real-world requirements

## What's Next

Module 4 will explore RAG (Retrieval-Augmented Generation) and knowledge integration for context-aware systems.

---

**Practice Exercise**: Build a complete memory system that persists across restarts. Test with 10,000+ items and verify sub-100ms retrieval times with proper tier management and automatic archival of old data.
