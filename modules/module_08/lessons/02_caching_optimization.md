# Lesson 2: Caching and Optimization Strategies

## Introduction

Intelligent caching systems are crucial for achieving enterprise-scale performance in context engineering. This lesson teaches you to build sophisticated multi-layer caching architectures with predictive prefetching, consistency guarantees, and adaptive optimization that achieve >95% cache hit rates.

## Advanced Caching Architectures

### Intelligent Multi-Layer Cache System

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import time
import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import weakref
import pickle
import numpy as np
import heapq
from concurrent.futures import ThreadPoolExecutor

class CacheEvictionStrategy(Enum):
    """Cache eviction strategies for different use cases."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Machine learning based
    PRIORITY_BASED = "priority"    # Business priority based

class CacheConsistencyLevel(Enum):
    """Cache consistency levels for distributed scenarios."""
    EVENTUAL = "eventual"          # Best performance, eventual consistency
    STRONG = "strong"             # Strong consistency, higher latency
    SESSION = "session"           # Per-session consistency
    MONOTONIC = "monotonic"       # Monotonic read consistency

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata and intelligence."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    priority: float = 1.0
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    prefetch_score: float = 0.0
    consistency_version: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def idle_time_seconds(self) -> float:
        """Get time since last access in seconds."""
        return (datetime.now() - self.last_accessed).total_seconds()

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class CacheStatistics:
    """Comprehensive cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    total_latency_ms: float = 0.0
    memory_usage_bytes: int = 0
    consistency_conflicts: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def prefetch_accuracy(self) -> float:
        """Calculate prefetch prediction accuracy."""
        total_prefetch = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / total_prefetch if total_prefetch > 0 else 0.0

    @property
    def average_latency_ms(self) -> float:
        """Calculate average request latency."""
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0

class PredictivePrefetcher:
    """ML-based predictive prefetching system."""

    def __init__(self, learning_window: int = 1000, prediction_horizon: int = 10):
        self.learning_window = learning_window
        self.prediction_horizon = prediction_horizon

        # Access pattern tracking
        self.access_history: deque = deque(maxlen=learning_window)
        self.pattern_frequencies: Dict[str, int] = defaultdict(int)
        self.sequence_patterns: Dict[str, List[str]] = defaultdict(list)

        # Prediction models
        self.temporal_patterns: Dict[int, Set[str]] = defaultdict(set)  # hour -> keys
        self.co_occurrence_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Performance tracking
        self.prediction_accuracy: float = 0.0
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()

    def record_access(self, key: str, timestamp: datetime = None):
        """Record cache access for pattern learning."""
        timestamp = timestamp or datetime.now()

        # Store access with temporal information
        access_record = {
            'key': key,
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'minute': timestamp.minute
        }

        self.access_history.append(access_record)

        # Update pattern frequencies
        self.pattern_frequencies[key] += 1

        # Update temporal patterns
        self.temporal_patterns[timestamp.hour].add(key)

        # Update sequence patterns (what comes after this key)
        if len(self.access_history) >= 2:
            prev_access = self.access_history[-2]
            prev_key = prev_access['key']
            self.sequence_patterns[prev_key].append(key)

            # Update co-occurrence matrix
            self.co_occurrence_matrix[prev_key][key] += 1

        # Learn weekly patterns
        day_of_week = timestamp.weekday()
        weekly_key = f"{day_of_week}_{timestamp.hour}"
        self.temporal_patterns[weekly_key].add(key)

    async def predict_next_accesses(
        self,
        current_key: str,
        context: Dict[str, Any] = None
    ) -> List[Tuple[str, float]]:
        """Predict likely next cache accesses with confidence scores."""
        predictions = []

        # Sequence-based predictions
        if current_key in self.sequence_patterns:
            sequences = self.sequence_patterns[current_key]
            sequence_counts = defaultdict(int)
            for next_key in sequences[-100:]:  # Recent sequences
                sequence_counts[next_key] += 1

            total_sequences = sum(sequence_counts.values())
            for next_key, count in sequence_counts.items():
                confidence = count / total_sequences
                predictions.append((next_key, confidence * 0.7))  # Weight sequence predictions

        # Co-occurrence based predictions
        if current_key in self.co_occurrence_matrix:
            co_occurrences = self.co_occurrence_matrix[current_key]
            total_co = sum(co_occurrences.values())

            for co_key, count in co_occurrences.items():
                confidence = count / total_co
                predictions.append((co_key, confidence * 0.5))  # Weight co-occurrence predictions

        # Temporal predictions
        now = datetime.now()
        current_hour = now.hour

        # Next hour predictions
        next_hour = (current_hour + 1) % 24
        if next_hour in self.temporal_patterns:
            temporal_keys = self.temporal_patterns[next_hour]
            for key in temporal_keys:
                predictions.append((key, 0.3))  # Lower confidence for temporal

        # Context-based predictions
        if context:
            context_predictions = self._predict_from_context(context)
            predictions.extend(context_predictions)

        # Aggregate and sort predictions
        prediction_scores = defaultdict(float)
        for key, score in predictions:
            prediction_scores[key] += score

        # Sort by confidence and return top predictions
        sorted_predictions = sorted(
            prediction_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_predictions[:self.prediction_horizon]

    def _predict_from_context(self, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Make predictions based on current context."""
        predictions = []

        # User-based predictions
        if 'user_id' in context:
            user_id = context['user_id']
            # Look for user-specific patterns in access history
            user_accesses = [
                record['key'] for record in self.access_history
                if record.get('user_id') == user_id
            ]

            if user_accesses:
                # Find most frequent keys for this user
                user_frequencies = defaultdict(int)
                for key in user_accesses[-50:]:  # Recent user accesses
                    user_frequencies[key] += 1

                total_user_accesses = sum(user_frequencies.values())
                for key, count in user_frequencies.items():
                    confidence = count / total_user_accesses
                    predictions.append((key, confidence * 0.4))

        # Session-based predictions
        if 'session_id' in context:
            session_id = context['session_id']
            # Similar logic for session patterns
            session_accesses = [
                record['key'] for record in self.access_history
                if record.get('session_id') == session_id
            ]

            if len(session_accesses) >= 2:
                # Look for patterns within this session
                last_key = session_accesses[-1]
                if last_key in self.sequence_patterns:
                    for next_key in self.sequence_patterns[last_key][-10:]:
                        predictions.append((next_key, 0.6))

        return predictions

    def update_prediction_accuracy(self, predicted_keys: List[str], actual_key: str):
        """Update prediction accuracy metrics."""
        was_predicted = actual_key in predicted_keys

        # Exponential moving average for accuracy
        alpha = 0.1
        current_accuracy = 1.0 if was_predicted else 0.0
        self.prediction_accuracy = (1 - alpha) * self.prediction_accuracy + alpha * current_accuracy

    def get_prefetch_recommendations(
        self,
        current_context: Dict[str, Any],
        cache_capacity_remaining: float
    ) -> List[str]:
        """Get recommendations for keys to prefetch."""
        if cache_capacity_remaining < 0.1:  # Less than 10% capacity remaining
            return []

        # Get predictions based on current context
        current_key = current_context.get('last_accessed_key')
        if not current_key:
            return []

        predictions = asyncio.run(self.predict_next_accesses(current_key, current_context))

        # Filter predictions by confidence threshold
        confidence_threshold = 0.3
        high_confidence_predictions = [
            key for key, confidence in predictions
            if confidence >= confidence_threshold
        ]

        # Limit by remaining capacity
        max_prefetch = int(cache_capacity_remaining * 10)  # Conservative prefetching
        return high_confidence_predictions[:max_prefetch]

class ConsistencyManager:
    """Manages cache consistency across distributed systems."""

    def __init__(self, consistency_level: CacheConsistencyLevel = CacheConsistencyLevel.EVENTUAL):
        self.consistency_level = consistency_level
        self.version_vectors: Dict[str, int] = defaultdict(int)
        self.pending_updates: Dict[str, List[Dict]] = defaultdict(list)
        self.consistency_conflicts: int = 0

        # Subscription management for invalidation
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # key -> subscriber_ids
        self.invalidation_queue: asyncio.Queue = asyncio.Queue()

    async def validate_read(self, key: str, version: int, node_id: str = None) -> bool:
        """Validate if a read operation maintains consistency guarantees."""
        if self.consistency_level == CacheConsistencyLevel.EVENTUAL:
            return True  # Always allow reads in eventual consistency

        elif self.consistency_level == CacheConsistencyLevel.STRONG:
            # Check if we have the latest version
            latest_version = self.version_vectors[key]
            return version >= latest_version

        elif self.consistency_level == CacheConsistencyLevel.SESSION:
            # Session consistency - check within session context
            session_id = node_id  # Assume node_id represents session
            session_key = f"{session_id}_{key}"
            expected_version = self.version_vectors.get(session_key, 0)
            return version >= expected_version

        elif self.consistency_level == CacheConsistencyLevel.MONOTONIC:
            # Monotonic read consistency - never see older versions
            last_seen_version = self.version_vectors.get(f"last_seen_{key}", 0)
            return version >= last_seen_version

        return True

    async def register_write(self, key: str, node_id: str = None) -> int:
        """Register a write operation and return new version."""
        # Increment version
        self.version_vectors[key] += 1
        new_version = self.version_vectors[key]

        # Schedule invalidation notifications
        if key in self.subscriptions:
            invalidation_msg = {
                'key': key,
                'new_version': new_version,
                'timestamp': datetime.now().isoformat(),
                'subscribers': list(self.subscriptions[key])
            }
            await self.invalidation_queue.put(invalidation_msg)

        return new_version

    def subscribe_to_invalidation(self, key: str, subscriber_id: str):
        """Subscribe to invalidation notifications for a key."""
        self.subscriptions[key].add(subscriber_id)

    def unsubscribe_from_invalidation(self, key: str, subscriber_id: str):
        """Unsubscribe from invalidation notifications."""
        if key in self.subscriptions:
            self.subscriptions[key].discard(subscriber_id)

    async def process_invalidations(self, invalidation_callback: Callable):
        """Process invalidation queue with callback."""
        while True:
            try:
                invalidation_msg = await self.invalidation_queue.get()
                await invalidation_callback(invalidation_msg)
            except Exception as e:
                logging.error(f"Error processing invalidation: {e}")

class AdaptiveCache:
    """Intelligent cache with adaptive optimization and predictive capabilities."""

    def __init__(
        self,
        max_size_mb: int = 1024,
        eviction_strategy: CacheEvictionStrategy = CacheEvictionStrategy.ADAPTIVE,
        consistency_level: CacheConsistencyLevel = CacheConsistencyLevel.EVENTUAL,
        enable_prefetching: bool = True,
        optimization_interval: int = 300  # 5 minutes
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_strategy = eviction_strategy
        self.enable_prefetching = enable_prefetching
        self.optimization_interval = optimization_interval

        # Core storage
        self.entries: Dict[str, CacheEntry] = {}
        self.size_tracker = 0

        # Eviction strategy components
        self.lru_order: deque = deque()
        self.lfu_heap: List[Tuple[int, str]] = []  # (access_count, key)
        self.priority_heap: List[Tuple[float, str]] = []  # (-priority, key)

        # Intelligence components
        self.prefetcher = PredictivePrefetcher() if enable_prefetching else None
        self.consistency_manager = ConsistencyManager(consistency_level)

        # Performance monitoring
        self.statistics = CacheStatistics()
        self.lock = threading.RLock()

        # Adaptive optimization
        self.optimization_history: List[Dict] = []
        self.current_strategy_performance: Dict[str, float] = {}

        # Background tasks
        self.optimization_task: Optional[asyncio.Task] = None
        self.prefetch_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background optimization and prefetching tasks."""
        if self.enable_prefetching and self.prefetcher:
            self.prefetch_task = asyncio.create_task(self._prefetch_loop())

        self.optimization_task = asyncio.create_task(self._optimization_loop())

    async def stop(self):
        """Stop background tasks gracefully."""
        if self.prefetch_task:
            self.prefetch_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()

    async def get(
        self,
        key: str,
        context: Dict[str, Any] = None,
        consistency_check: bool = True
    ) -> Optional[Any]:
        """Get value from cache with advanced features."""
        start_time = time.time()

        try:
            with self.lock:
                # Check if key exists
                if key not in self.entries:
                    self.statistics.cache_misses += 1
                    self.statistics.total_requests += 1

                    # Record miss for prefetcher
                    if self.prefetcher:
                        self.prefetcher.record_access(key)

                    return None

                entry = self.entries[key]

                # Check expiration
                if entry.is_expired:
                    self._remove_entry(key)
                    self.statistics.cache_misses += 1
                    self.statistics.total_requests += 1
                    return None

                # Consistency validation
                if consistency_check:
                    is_valid = await self.consistency_manager.validate_read(
                        key, entry.consistency_version
                    )
                    if not is_valid:
                        self.statistics.consistency_conflicts += 1
                        self.statistics.cache_misses += 1
                        self.statistics.total_requests += 1
                        return None

                # Update access statistics
                entry.update_access()
                self._update_eviction_structures(key, entry)

                # Record hit for prefetcher
                if self.prefetcher:
                    self.prefetcher.record_access(key)

                # Update statistics
                latency = (time.time() - start_time) * 1000
                self.statistics.cache_hits += 1
                self.statistics.total_requests += 1
                self.statistics.total_latency_ms += latency

                return entry.value

        except Exception as e:
            logging.error(f"Cache get error: {e}")
            self.statistics.total_requests += 1
            self.statistics.cache_misses += 1
            return None

    async def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        priority: float = 1.0,
        tags: Set[str] = None,
        context: Dict[str, Any] = None
    ) -> bool:
        """Put value in cache with metadata and optimization."""
        try:
            with self.lock:
                # Calculate size
                value_size = len(pickle.dumps(value))

                # Check if value is too large
                if value_size > self.max_size_bytes * 0.1:  # Max 10% of cache
                    logging.warning(f"Value too large for key {key}: {value_size} bytes")
                    return False

                # Make space if needed
                while self.size_tracker + value_size > self.max_size_bytes and self.entries:
                    evicted_key = self._evict_entry()
                    if not evicted_key:
                        break  # Couldn't evict anything

                # Check space again
                if self.size_tracker + value_size > self.max_size_bytes:
                    return False

                # Register write for consistency
                version = await self.consistency_manager.register_write(key)

                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    size_bytes=value_size,
                    ttl_seconds=ttl_seconds,
                    priority=priority,
                    tags=tags or set(),
                    consistency_version=version
                )

                # Remove old entry if exists
                if key in self.entries:
                    old_entry = self.entries[key]
                    self.size_tracker -= old_entry.size_bytes

                # Store new entry
                self.entries[key] = entry
                self.size_tracker += value_size

                # Update eviction structures
                self._update_eviction_structures(key, entry)

                # Update memory usage statistics
                self.statistics.memory_usage_bytes = self.size_tracker

                return True

        except Exception as e:
            logging.error(f"Cache put error: {e}")
            return False

    async def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        with self.lock:
            if key in self.entries:
                self._remove_entry(key)
                return True
            return False

    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with any of the specified tags."""
        invalidated_count = 0

        with self.lock:
            keys_to_remove = []
            for key, entry in self.entries.items():
                if entry.tags & tags:  # Intersection of sets
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                self._remove_entry(key)
                invalidated_count += 1

        return invalidated_count

    def _remove_entry(self, key: str):
        """Remove entry and update all tracking structures."""
        if key not in self.entries:
            return

        entry = self.entries[key]

        # Update size tracker
        self.size_tracker -= entry.size_bytes

        # Remove from storage
        del self.entries[key]

        # Update eviction structures
        if key in self.lru_order:
            self.lru_order.remove(key)

    def _evict_entry(self) -> Optional[str]:
        """Evict entry based on current strategy."""
        if not self.entries:
            return None

        eviction_key = None

        if self.eviction_strategy == CacheEvictionStrategy.LRU:
            eviction_key = self._evict_lru()
        elif self.eviction_strategy == CacheEvictionStrategy.LFU:
            eviction_key = self._evict_lfu()
        elif self.eviction_strategy == CacheEvictionStrategy.TTL:
            eviction_key = self._evict_ttl()
        elif self.eviction_strategy == CacheEvictionStrategy.PRIORITY_BASED:
            eviction_key = self._evict_priority()
        elif self.eviction_strategy == CacheEvictionStrategy.ADAPTIVE:
            eviction_key = self._evict_adaptive()

        if eviction_key:
            self._remove_entry(eviction_key)
            self.statistics.evictions += 1

        return eviction_key

    def _evict_lru(self) -> Optional[str]:
        """Evict least recently used entry."""
        if self.lru_order:
            return self.lru_order[0]
        return None

    def _evict_lfu(self) -> Optional[str]:
        """Evict least frequently used entry."""
        if not self.entries:
            return None

        # Find entry with minimum access count
        min_access_count = float('inf')
        lfu_key = None

        for key, entry in self.entries.items():
            if entry.access_count < min_access_count:
                min_access_count = entry.access_count
                lfu_key = key

        return lfu_key

    def _evict_ttl(self) -> Optional[str]:
        """Evict entry closest to expiration."""
        if not self.entries:
            return None

        earliest_expiry = None
        ttl_key = None

        for key, entry in self.entries.items():
            if entry.ttl_seconds is not None:
                expiry_time = entry.created_at + timedelta(seconds=entry.ttl_seconds)
                if earliest_expiry is None or expiry_time < earliest_expiry:
                    earliest_expiry = expiry_time
                    ttl_key = key

        return ttl_key or self._evict_lru()  # Fallback to LRU

    def _evict_priority(self) -> Optional[str]:
        """Evict entry with lowest priority."""
        if not self.entries:
            return None

        min_priority = float('inf')
        priority_key = None

        for key, entry in self.entries.items():
            if entry.priority < min_priority:
                min_priority = entry.priority
                priority_key = key

        return priority_key

    def _evict_adaptive(self) -> Optional[str]:
        """Evict entry using adaptive strategy based on performance."""
        # Combine multiple factors for intelligent eviction
        scores = {}

        for key, entry in self.entries.items():
            # Factor 1: Recency (lower is better for eviction)
            recency_score = entry.idle_time_seconds / 3600  # Hours since last access

            # Factor 2: Frequency (lower is better for eviction)
            frequency_score = 1.0 / (1.0 + entry.access_count)

            # Factor 3: Size (larger entries get higher eviction score)
            size_score = entry.size_bytes / (1024 * 1024)  # MB

            # Factor 4: Priority (lower priority = higher eviction score)
            priority_score = 1.0 / entry.priority

            # Factor 5: Prefetch potential (lower potential = higher eviction score)
            prefetch_score = 1.0 - entry.prefetch_score

            # Combined score (higher score = better candidate for eviction)
            combined_score = (
                recency_score * 0.3 +
                frequency_score * 0.25 +
                size_score * 0.2 +
                priority_score * 0.15 +
                prefetch_score * 0.1
            )

            scores[key] = combined_score

        # Return key with highest eviction score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return None

    def _update_eviction_structures(self, key: str, entry: CacheEntry):
        """Update eviction tracking structures."""
        # Update LRU order
        if key in self.lru_order:
            self.lru_order.remove(key)
        self.lru_order.append(key)

    async def _prefetch_loop(self):
        """Background prefetching based on predictions."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self.prefetcher:
                    continue

                # Get current context
                current_context = {
                    'cache_size': len(self.entries),
                    'capacity_remaining': 1.0 - (self.size_tracker / self.max_size_bytes)
                }

                # Get last accessed key
                if self.lru_order:
                    current_context['last_accessed_key'] = self.lru_order[-1]

                # Get prefetch recommendations
                prefetch_keys = self.prefetcher.get_prefetch_recommendations(
                    current_context,
                    current_context['capacity_remaining']
                )

                # Execute prefetching (would integrate with external data source)
                for key in prefetch_keys:
                    if key not in self.entries:
                        # In real implementation, fetch from external source
                        # For demo, we'll skip actual prefetching
                        logging.info(f"Would prefetch key: {key}")

            except Exception as e:
                logging.error(f"Prefetch loop error: {e}")

    async def _optimization_loop(self):
        """Background optimization of cache parameters."""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)

                # Analyze current performance
                current_performance = {
                    'hit_rate': self.statistics.hit_rate,
                    'average_latency': self.statistics.average_latency_ms,
                    'eviction_rate': self.statistics.evictions / max(1, self.statistics.total_requests)
                }

                # Store performance history
                self.optimization_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'strategy': self.eviction_strategy.value,
                    'performance': current_performance.copy()
                })

                # Keep only recent history
                if len(self.optimization_history) > 100:
                    self.optimization_history = self.optimization_history[-100:]

                # Adaptive strategy optimization
                if self.eviction_strategy == CacheEvictionStrategy.ADAPTIVE:
                    await self._optimize_adaptive_strategy()

            except Exception as e:
                logging.error(f"Optimization loop error: {e}")

    async def _optimize_adaptive_strategy(self):
        """Optimize adaptive eviction strategy parameters."""
        if len(self.optimization_history) < 10:
            return

        # Analyze performance trends
        recent_performance = [h['performance'] for h in self.optimization_history[-10:]]
        avg_hit_rate = np.mean([p['hit_rate'] for p in recent_performance])
        avg_latency = np.mean([p['average_latency'] for p in recent_performance])

        # Log optimization insights
        logging.info(f"Cache optimization: hit_rate={avg_hit_rate:.3f}, latency={avg_latency:.2f}ms")

        # Adaptive improvements could be implemented here
        # For example, adjusting eviction score weights based on performance

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            return {
                'performance': {
                    'hit_rate': self.statistics.hit_rate,
                    'prefetch_accuracy': self.statistics.prefetch_accuracy,
                    'average_latency_ms': self.statistics.average_latency_ms,
                    'total_requests': self.statistics.total_requests,
                    'cache_hits': self.statistics.cache_hits,
                    'cache_misses': self.statistics.cache_misses,
                    'evictions': self.statistics.evictions,
                    'consistency_conflicts': self.statistics.consistency_conflicts
                },
                'memory': {
                    'current_size_mb': self.size_tracker / (1024 * 1024),
                    'max_size_mb': self.max_size_bytes / (1024 * 1024),
                    'utilization_percentage': (self.size_tracker / self.max_size_bytes) * 100,
                    'entry_count': len(self.entries)
                },
                'configuration': {
                    'eviction_strategy': self.eviction_strategy.value,
                    'consistency_level': self.consistency_manager.consistency_level.value,
                    'prefetching_enabled': self.enable_prefetching
                },
                'optimization': {
                    'optimization_cycles': len(self.optimization_history),
                    'current_strategy_performance': self.current_strategy_performance
                }
            }

# Example usage and performance testing
async def caching_performance_demo():
    """Demonstrate advanced caching capabilities."""

    # Create adaptive cache with intelligence
    cache = AdaptiveCache(
        max_size_mb=256,
        eviction_strategy=CacheEvictionStrategy.ADAPTIVE,
        consistency_level=CacheConsistencyLevel.EVENTUAL,
        enable_prefetching=True,
        optimization_interval=60
    )

    await cache.start()

    try:
        # Simulate realistic access patterns
        test_data = {
            f"user_{i}": {
                'id': i,
                'name': f'User {i}',
                'preferences': {'theme': 'dark' if i % 2 else 'light'},
                'large_data': list(range(100))  # Some substantial data
            }
            for i in range(1000)
        }

        # Test cache performance with various patterns
        start_time = time.time()

        # Phase 1: Initial population
        print("Phase 1: Populating cache...")
        for i, (key, value) in enumerate(test_data.items()):
            if i < 200:  # Populate first 200 entries
                priority = 2.0 if i < 50 else 1.0  # High priority for first 50
                tags = {'user_data', 'high_priority'} if i < 50 else {'user_data'}

                success = await cache.put(
                    key=key,
                    value=value,
                    ttl_seconds=3600,
                    priority=priority,
                    tags=tags
                )

                if not success and i < 100:  # Should succeed for first 100
                    print(f"Failed to cache key: {key}")

        # Phase 2: Access pattern simulation
        print("Phase 2: Simulating access patterns...")
        hit_count = 0
        total_requests = 0

        # Simulate hot keys (80/20 rule)
        hot_keys = list(test_data.keys())[:50]
        warm_keys = list(test_data.keys())[50:200]
        cold_keys = list(test_data.keys())[200:300]

        for round_num in range(5):
            print(f"  Round {round_num + 1}/5")

            # Hot key accesses (80% of requests)
            for _ in range(400):
                key = np.random.choice(hot_keys)
                value = await cache.get(key)
                total_requests += 1
                if value is not None:
                    hit_count += 1

            # Warm key accesses (15% of requests)
            for _ in range(75):
                key = np.random.choice(warm_keys)
                value = await cache.get(key)
                total_requests += 1
                if value is not None:
                    hit_count += 1

            # Cold key accesses (5% of requests)
            for _ in range(25):
                key = np.random.choice(cold_keys)
                value = await cache.get(key)
                total_requests += 1
                if value is not None:
                    hit_count += 1

            # Add some new data
            for i in range(5):
                new_key = f"new_user_{round_num}_{i}"
                new_value = {'id': f"new_{i}", 'round': round_num}
                await cache.put(new_key, new_value, priority=1.5)

        # Phase 3: Tag-based invalidation
        print("Phase 3: Testing tag-based invalidation...")
        invalidated = await cache.invalidate_by_tags({'high_priority'})
        print(f"Invalidated {invalidated} high-priority entries")

        # Phase 4: Performance measurement
        total_time = time.time() - start_time
        hit_rate = hit_count / total_requests if total_requests > 0 else 0

        print(f"\nPerformance Results:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Requests: {total_requests}")
        print(f"  Hit Rate: {hit_rate:.2%}")
        print(f"  Throughput: {total_requests / total_time:.2f} requests/second")

        # Get detailed statistics
        stats = cache.get_detailed_statistics()
        print(f"\nDetailed Cache Statistics:")
        print(json.dumps(stats, indent=2))

    finally:
        await cache.stop()

# Run the demonstration
if __name__ == "__main__":
    asyncio.run(caching_performance_demo())
```

## Key Takeaways

1. **Multi-Layer Intelligence**: Adaptive eviction strategies based on access patterns and performance

2. **Predictive Prefetching**: ML-based prediction of future cache accesses for proactive loading

3. **Consistency Management**: Flexible consistency levels for different application requirements

4. **Performance Optimization**: Continuous monitoring and adaptive parameter tuning

5. **Tag-Based Operations**: Efficient bulk operations and invalidation patterns

6. **Resource Management**: Intelligent memory management with size and priority awareness

## What's Next

In Lesson 3, we'll explore distributed systems and consistency patterns for scaling cache architectures across multiple nodes.

---

**Practice Exercise**: Build an adaptive cache system achieving >95% hit rates with predictive prefetching. Demonstrate intelligent eviction strategies and consistency management across 100K+ cache operations with <5ms average latency.
