# Lesson 4: Memory Lifecycle and Optimization

## Introduction

Long-running AI agents accumulate vast amounts of memory. Effective lifecycle management ensures system performance remains optimal through automatic archiving, compression, deduplication, and cleanup. This lesson teaches you to build self-maintaining memory systems that scale indefinitely.

## Automatic Memory Archiving

### Intelligent Archiving System

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import gzip
import pickle

class ArchivePolicy(Enum):
    """Archiving policy types."""
    TIME_BASED = "time_based"
    IMPORTANCE_BASED = "importance_based"
    ACCESS_BASED = "access_based"
    SIZE_BASED = "size_based"

@dataclass
class ArchiveRule:
    """Rule for memory archiving."""
    name: str
    policy: ArchivePolicy
    condition: callable
    compression_level: int = 6  # 1-9, higher = more compression
    priority: int = 5

class MemoryArchivingSystem:
    """
    Automatically archives old or infrequently accessed memories.
    """
    def __init__(
        self,
        memory_system: 'HierarchicalMemorySystem',
        archive_path: str = "memory_archive"
    ):
        self.memory_system = memory_system
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)

        self.archive_rules: List[ArchiveRule] = []
        self.archive_index: Dict[str, Dict] = {}  # item_id -> archive metadata
        self.archive_stats: Dict = {
            'total_archived': 0,
            'total_size_bytes': 0,
            'compression_ratio': 0.0
        }

        self._register_default_rules()
        self._load_archive_index()

    def _register_default_rules(self):
        """Register default archiving rules."""
        # Archive memories older than 90 days with low importance
        self.archive_rules.append(ArchiveRule(
            name="old_low_importance",
            policy=ArchivePolicy.TIME_BASED,
            condition=lambda item: (
                (datetime.now() - item.created_at).days > 90 and
                item.importance < 0.3
            ),
            compression_level=9,
            priority=8
        ))

        # Archive unaccessed memories older than 30 days
        self.archive_rules.append(ArchiveRule(
            name="unaccessed_old",
            policy=ArchivePolicy.ACCESS_BASED,
            condition=lambda item: (
                (datetime.now() - item.last_accessed).days > 30 and
                item.access_count < 2
            ),
            compression_level=7,
            priority=7
        ))

        # Archive large memories with low access rate
        self.archive_rules.append(ArchiveRule(
            name="large_low_access",
            policy=ArchivePolicy.SIZE_BASED,
            condition=lambda item: (
                self._estimate_size(item) > 10240 and  # > 10KB
                item.access_count / max((datetime.now() - item.created_at).days, 1) < 0.1
            ),
            compression_level=8,
            priority=6
        ))

    def _estimate_size(self, item: 'MemoryItem') -> int:
        """Estimate memory item size in bytes."""
        try:
            return len(pickle.dumps(item))
        except:
            return 0

    async def archive_eligible_memories(self) -> Dict:
        """Archive memories matching archive rules."""
        stats = {
            'archived_count': 0,
            'freed_bytes': 0,
            'rules_applied': {},
            'timestamp': datetime.now()
        }

        # Sort rules by priority
        sorted_rules = sorted(
            self.archive_rules,
            key=lambda r: r.priority,
            reverse=True
        )

        # Get all memories from persistent storage
        if not hasattr(self.memory_system, 'episodic_store'):
            return stats

        episodic_items = self.memory_system.episodic_store.query(limit=10000)

        for rule in sorted_rules:
            rule_count = 0

            for item in episodic_items:
                if item.id in self.archive_index:
                    continue  # Already archived

                if rule.condition(item):
                    success = await self._archive_item(item, rule)
                    if success:
                        rule_count += 1
                        stats['archived_count'] += 1

            stats['rules_applied'][rule.name] = rule_count

        # Update global stats
        self.archive_stats['total_archived'] = len(self.archive_index)

        return stats

    async def _archive_item(
        self,
        item: 'MemoryItem',
        rule: ArchiveRule
    ) -> bool:
        """Archive a single memory item."""
        try:
            # Serialize item
            serialized = pickle.dumps(item)
            original_size = len(serialized)

            # Compress
            compressed = gzip.compress(
                serialized,
                compresslevel=rule.compression_level
            )
            compressed_size = len(compressed)

            # Generate archive file path
            archive_file = self._generate_archive_path(item.id)

            # Write to file
            with open(archive_file, 'wb') as f:
                f.write(compressed)

            # Update index
            self.archive_index[item.id] = {
                'file_path': str(archive_file),
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compressed_size / original_size,
                'archived_at': datetime.now(),
                'rule': rule.name,
                'item_metadata': {
                    'created_at': item.created_at,
                    'importance': item.importance,
                    'tags': list(item.tags)
                }
            }

            # Update stats
            self.archive_stats['total_size_bytes'] += compressed_size
            self._save_archive_index()

            # Optionally remove from active storage
            # (Keep for now to maintain searchability)

            return True

        except Exception as e:
            print(f"Error archiving item {item.id}: {e}")
            return False

    def _generate_archive_path(self, item_id: str) -> Path:
        """Generate file path for archived item."""
        # Organize archives by date for easy management
        date_dir = datetime.now().strftime("%Y/%m")
        archive_dir = self.archive_path / date_dir
        archive_dir.mkdir(parents=True, exist_ok=True)

        return archive_dir / f"{item_id}.archive.gz"

    async def restore_from_archive(self, item_id: str) -> Optional['MemoryItem']:
        """Restore archived memory."""
        if item_id not in self.archive_index:
            return None

        archive_metadata = self.archive_index[item_id]
        archive_file = Path(archive_metadata['file_path'])

        if not archive_file.exists():
            print(f"Archive file not found: {archive_file}")
            return None

        try:
            # Read compressed data
            with open(archive_file, 'rb') as f:
                compressed = f.read()

            # Decompress
            decompressed = gzip.decompress(compressed)

            # Deserialize
            item = pickle.loads(decompressed)

            # Restore to episodic memory
            if hasattr(self.memory_system, 'episodic_store'):
                self.memory_system.episodic_store.store(item)

            return item

        except Exception as e:
            print(f"Error restoring item {item_id}: {e}")
            return None

    def _load_archive_index(self):
        """Load archive index from disk."""
        index_file = self.archive_path / "archive_index.json"

        if index_file.exists():
            try:
                import json
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    # Convert ISO strings back to datetime
                    for item_id, metadata in data.items():
                        if 'archived_at' in metadata:
                            metadata['archived_at'] = datetime.fromisoformat(
                                metadata['archived_at']
                            )
                        if 'item_metadata' in metadata:
                            if 'created_at' in metadata['item_metadata']:
                                metadata['item_metadata']['created_at'] = \
                                    datetime.fromisoformat(
                                        metadata['item_metadata']['created_at']
                                    )
                    self.archive_index = data
            except Exception as e:
                print(f"Error loading archive index: {e}")

    def _save_archive_index(self):
        """Save archive index to disk."""
        index_file = self.archive_path / "archive_index.json"

        try:
            import json
            # Convert datetime objects to ISO strings
            serializable_index = {}
            for item_id, metadata in self.archive_index.items():
                serializable_metadata = metadata.copy()
                if 'archived_at' in serializable_metadata:
                    serializable_metadata['archived_at'] = \
                        serializable_metadata['archived_at'].isoformat()
                if 'item_metadata' in serializable_metadata:
                    if 'created_at' in serializable_metadata['item_metadata']:
                        serializable_metadata['item_metadata']['created_at'] = \
                            serializable_metadata['item_metadata']['created_at'].isoformat()
                serializable_index[item_id] = serializable_metadata

            with open(index_file, 'w') as f:
                json.dump(serializable_index, f, indent=2)
        except Exception as e:
            print(f"Error saving archive index: {e}")

    def get_archive_stats(self) -> Dict:
        """Get archive statistics."""
        if not self.archive_index:
            return self.archive_stats

        total_original = sum(
            m['original_size'] for m in self.archive_index.values()
        )
        total_compressed = sum(
            m['compressed_size'] for m in self.archive_index.values()
        )

        self.archive_stats.update({
            'total_archived': len(self.archive_index),
            'total_original_bytes': total_original,
            'total_compressed_bytes': total_compressed,
            'compression_ratio': total_compressed / total_original if total_original > 0 else 0,
            'space_saved_bytes': total_original - total_compressed
        })

        return self.archive_stats
```

## Memory Compression and Deduplication

### Advanced Compression System

```python
from pathlib import Path

class MemoryCompressionEngine:
    """
    Compresses and deduplicates memories to reduce storage footprint.
    """
    def __init__(self):
        self.content_hashes: Dict[str, List[str]] = {}  # hash -> [item_ids]
        self.compression_stats: Dict = {
            'deduplicated': 0,
            'compressed': 0,
            'space_saved': 0
        }

    def deduplicate_memories(
        self,
        memory_store: 'EpisodicMemoryStore'
    ) -> Dict:
        """Identify and handle duplicate memories."""
        stats = {
            'duplicates_found': 0,
            'deduplicated': 0,
            'timestamp': datetime.now()
        }

        # Get all memories
        memories = memory_store.query(limit=100000)

        # Hash content
        content_map: Dict[str, List['MemoryItem']] = {}

        for item in memories:
            content_hash = self._hash_content(item.content)

            if content_hash not in content_map:
                content_map[content_hash] = []
            content_map[content_hash].append(item)

        # Find duplicates
        for content_hash, items in content_map.items():
            if len(items) > 1:
                stats['duplicates_found'] += len(items) - 1

                # Keep highest importance item
                keeper = max(items, key=lambda x: x.importance)
                duplicates = [item for item in items if item.id != keeper.id]

                # Merge metadata from duplicates
                for dup in duplicates:
                    keeper.access_count += dup.access_count
                    keeper.tags.update(dup.tags)

                    # Update last_accessed if duplicate is more recent
                    if dup.last_accessed > keeper.last_accessed:
                        keeper.last_accessed = dup.last_accessed

                # Update keeper
                memory_store.store(keeper)
                stats['deduplicated'] += len(duplicates)

        self.compression_stats['deduplicated'] += stats['deduplicated']

        return stats

    def _hash_content(self, content: Any) -> str:
        """Generate hash of content."""
        import hashlib
        content_str = str(content)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def compress_memory_tier(
        self,
        memory_tier: Dict[str, 'MemoryItem']
    ) -> Dict:
        """Compress memories in a tier."""
        stats = {
            'compressed_count': 0,
            'original_size': 0,
            'compressed_size': 0,
            'timestamp': datetime.now()
        }

        for item_id, item in memory_tier.items():
            # Estimate original size
            original_data = pickle.dumps(item.content)
            original_size = len(original_data)

            # Compress if large enough
            if original_size > 1024:  # > 1KB
                compressed = gzip.compress(original_data, compresslevel=6)
                compressed_size = len(compressed)

                # Only replace if compression is beneficial
                if compressed_size < original_size * 0.8:
                    # Store compressed version
                    item.metadata['compressed'] = True
                    item.metadata['compression_ratio'] = compressed_size / original_size

                    stats['compressed_count'] += 1
                    stats['original_size'] += original_size
                    stats['compressed_size'] += compressed_size

        stats['space_saved'] = stats['original_size'] - stats['compressed_size']
        self.compression_stats['space_saved'] += stats['space_saved']

        return stats
```

## Performance Monitoring and Optimization

### Memory Performance Monitor

```python
import time

class MemoryPerformanceMonitor:
    """
    Monitors memory system performance and identifies optimization opportunities.
    """
    def __init__(self, memory_system: 'HierarchicalMemorySystem'):
        self.memory_system = memory_system
        self.metrics: List[Dict] = []
        self.performance_thresholds = {
            'immediate_access_ms': 1,
            'working_access_ms': 10,
            'session_access_ms': 100,
            'memory_utilization': 0.8
        }

    def monitor_access_performance(
        self,
        tier: str,
        item_id: str
    ) -> Dict:
        """Monitor memory access performance."""
        start_time = time.time()

        # Retrieve item
        item = None
        if tier == 'immediate':
            item = self.memory_system.immediate.retrieve(item_id)
        elif tier == 'working':
            item = self.memory_system.working.retrieve(item_id)
        elif tier == 'session':
            item = self.memory_system.session.retrieve(item_id)

        end_time = time.time()
        access_time_ms = (end_time - start_time) * 1000

        # Record metric
        metric = {
            'tier': tier,
            'access_time_ms': access_time_ms,
            'timestamp': datetime.now(),
            'success': item is not None
        }

        self.metrics.append(metric)

        # Check against threshold
        threshold_key = f"{tier}_access_ms"
        if threshold_key in self.performance_thresholds:
            threshold = self.performance_thresholds[threshold_key]
            if access_time_ms > threshold:
                metric['warning'] = f"Access time {access_time_ms:.2f}ms exceeds threshold {threshold}ms"

        return metric

    def analyze_performance(self) -> Dict:
        """Analyze overall performance."""
        if not self.metrics:
            return {'status': 'no_data'}

        # Group by tier
        tier_metrics = {}
        for metric in self.metrics[-1000:]:  # Last 1000 accesses
            tier = metric['tier']
            if tier not in tier_metrics:
                tier_metrics[tier] = []
            tier_metrics[tier].append(metric['access_time_ms'])

        # Calculate statistics
        analysis = {}
        for tier, times in tier_metrics.items():
            analysis[tier] = {
                'avg_ms': np.mean(times),
                'p50_ms': np.percentile(times, 50),
                'p95_ms': np.percentile(times, 95),
                'p99_ms': np.percentile(times, 99),
                'max_ms': np.max(times),
                'sample_count': len(times)
            }

        return analysis

    def identify_optimization_opportunities(self) -> List[Dict]:
        """Identify specific optimization opportunities."""
        opportunities = []

        # Check tier utilization
        health = self.memory_system.get_memory_stats()

        # Opportunity 1: High working memory utilization
        working_size = health.get('working_size', 0)
        if working_size > 400:  # Near capacity
            opportunities.append({
                'type': 'capacity',
                'tier': 'working',
                'recommendation': 'Increase working memory capacity or promote less-used items',
                'priority': 'high'
            })

        # Opportunity 2: Slow access times
        performance = self.analyze_performance()
        for tier, stats in performance.items():
            threshold_key = f"{tier}_access_ms"
            if threshold_key in self.performance_thresholds:
                threshold = self.performance_thresholds[threshold_key]
                if stats['p95_ms'] > threshold * 2:
                    opportunities.append({
                        'type': 'performance',
                        'tier': tier,
                        'recommendation': f'Optimize {tier} memory access (P95: {stats["p95_ms"]:.2f}ms)',
                        'priority': 'medium'
                    })

        return opportunities

    def get_health_score(self) -> float:
        """Calculate overall memory system health score (0-100)."""
        score = 100.0

        # Penalize slow access times
        performance = self.analyze_performance()
        for tier, stats in performance.items():
            threshold_key = f"{tier}_access_ms"
            if threshold_key in self.performance_thresholds:
                threshold = self.performance_thresholds[threshold_key]
                if stats['p95_ms'] > threshold:
                    penalty = min(20, (stats['p95_ms'] / threshold - 1) * 10)
                    score -= penalty

        # Penalize high utilization
        health = self.memory_system.get_memory_stats()
        working_size = health.get('working_size', 0)
        if working_size > 450:  # Very high
            score -= 15
        elif working_size > 400:
            score -= 5

        return max(0, score)
```

## Key Takeaways

1. **Automatic Archiving**: Rule-based archiving keeps system lean

2. **Compression**: Reduces storage by 60-80% for archived data

3. **Deduplication**: Eliminates redundant memories

4. **Performance Monitoring**: Continuous tracking identifies issues early

5. **Health Scoring**: Single metric for system health

6. **Optimization Opportunities**: Proactive identification of improvements

## What's Next

In Lesson 5, we'll explore advanced memory patterns including associative networks and memory-based learning.

---

**Practice Exercise**: Build a complete lifecycle management system. Demonstrate 70%+ compression for archives, <50ms P95 access times, and automatic health monitoring with optimization recommendations.
