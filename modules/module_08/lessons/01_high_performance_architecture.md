# Lesson 1: High-Performance Context Architecture

## Introduction

Scaling context engineering to handle enterprise workloads requires sophisticated architectural patterns that maintain sub-50ms response times while processing thousands of concurrent operations. This lesson teaches you to build high-performance context architectures that excel under extreme load conditions.

## High-Performance Architecture Patterns

### Distributed Context Engine

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import defaultdict, deque
import weakref
import pickle
import numpy as np

class PerformanceTier(Enum):
    """Performance tiers for context handling."""
    ULTRA_LOW_LATENCY = "ultra_low_latency"    # <10ms
    LOW_LATENCY = "low_latency"                # <50ms
    STANDARD = "standard"                      # <200ms
    BATCH = "batch"                           # <2s

class ContextPriority(Enum):
    """Context processing priority levels."""
    CRITICAL = 1    # Real-time user interactions
    HIGH = 2        # Important business logic
    NORMAL = 3      # Standard operations
    LOW = 4         # Background processing

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    request_count: int = 0
    total_latency: float = 0.0
    max_latency: float = 0.0
    min_latency: float = float('inf')
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    throughput_ops_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0

    def add_request(self, latency: float, is_error: bool = False, cache_hit: bool = False):
        """Add metrics for a single request."""
        self.request_count += 1
        self.total_latency += latency
        self.max_latency = max(self.max_latency, latency)
        self.min_latency = min(self.min_latency, latency)

        if is_error:
            self.error_count += 1

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    @property
    def average_latency(self) -> float:
        return self.total_latency / self.request_count if self.request_count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.request_count if self.request_count > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total_cache_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0

@dataclass
class ContextRequest:
    """High-performance context request with optimization hints."""
    id: str
    context_data: Dict[str, Any]
    operation: str
    priority: ContextPriority
    performance_tier: PerformanceTier
    timeout_ms: int
    cache_hints: List[str] = field(default_factory=list)
    processing_hints: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None

    # Performance tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def processing_time_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    @property
    def total_time_ms(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds() * 1000
        return 0.0

class HighPerformanceCache:
    """Multi-tier caching system optimized for context data."""

    def __init__(
        self,
        l1_size_mb: int = 64,      # In-memory hot cache
        l2_size_mb: int = 256,     # In-memory warm cache
        l3_size_mb: int = 1024,    # Compressed cache
        ttl_seconds: int = 3600
    ):
        self.l1_size_bytes = l1_size_mb * 1024 * 1024
        self.l2_size_bytes = l2_size_mb * 1024 * 1024
        self.l3_size_bytes = l3_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds

        # Cache tiers
        self.l1_cache: Dict[str, Tuple[Any, datetime, int]] = {}  # key -> (value, timestamp, size)
        self.l2_cache: Dict[str, Tuple[Any, datetime, int]] = {}
        self.l3_cache: Dict[str, Tuple[bytes, datetime, int]] = {}  # Compressed storage

        # Access tracking for LRU
        self.l1_access_order: deque = deque()
        self.l2_access_order: deque = deque()
        self.l3_access_order: deque = deque()

        # Cache statistics
        self.stats = PerformanceMetrics()
        self.lock = threading.RLock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with promotion between tiers."""
        start_time = time.time()

        try:
            with self.lock:
                # Check L1 cache first (hottest)
                if key in self.l1_cache:
                    value, timestamp, size = self.l1_cache[key]
                    if self._is_valid(timestamp):
                        self._update_access_order(key, 1)
                        latency = (time.time() - start_time) * 1000
                        self.stats.add_request(latency, cache_hit=True)
                        return value
                    else:
                        self._evict_expired(key, 1)

                # Check L2 cache (warm)
                if key in self.l2_cache:
                    value, timestamp, size = self.l2_cache[key]
                    if self._is_valid(timestamp):
                        # Promote to L1
                        await self._promote_to_l1(key, value, size)
                        latency = (time.time() - start_time) * 1000
                        self.stats.add_request(latency, cache_hit=True)
                        return value
                    else:
                        self._evict_expired(key, 2)

                # Check L3 cache (compressed)
                if key in self.l3_cache:
                    compressed_value, timestamp, size = self.l3_cache[key]
                    if self._is_valid(timestamp):
                        # Decompress and promote to L2
                        value = pickle.loads(compressed_value)
                        await self._promote_to_l2(key, value, size)
                        latency = (time.time() - start_time) * 1000
                        self.stats.add_request(latency, cache_hit=True)
                        return value
                    else:
                        self._evict_expired(key, 3)

                # Cache miss
                latency = (time.time() - start_time) * 1000
                self.stats.add_request(latency, cache_hit=False)
                return None

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.stats.add_request(latency, is_error=True)
            raise

    async def put(self, key: str, value: Any, tier_hint: int = 1) -> bool:
        """Put value in cache with tier preference."""
        try:
            with self.lock:
                value_size = len(pickle.dumps(value))

                if tier_hint == 1 and value_size <= self.l1_size_bytes // 100:  # Max 1% of L1
                    return await self._put_l1(key, value, value_size)
                elif tier_hint <= 2 and value_size <= self.l2_size_bytes // 50:  # Max 2% of L2
                    return await self._put_l2(key, value, value_size)
                else:
                    return await self._put_l3(key, value, value_size)

        except Exception as e:
            logging.error(f"Cache put error: {e}")
            return False

    async def _put_l1(self, key: str, value: Any, size: int) -> bool:
        """Put value in L1 cache with LRU eviction."""
        # Make space if needed
        while self._get_tier_size(1) + size > self.l1_size_bytes and self.l1_cache:
            oldest_key = self.l1_access_order.popleft()
            if oldest_key in self.l1_cache:
                del self.l1_cache[oldest_key]

        # Store value
        self.l1_cache[key] = (value, datetime.now(), size)
        self._update_access_order(key, 1)
        return True

    async def _put_l2(self, key: str, value: Any, size: int) -> bool:
        """Put value in L2 cache with LRU eviction."""
        while self._get_tier_size(2) + size > self.l2_size_bytes and self.l2_cache:
            oldest_key = self.l2_access_order.popleft()
            if oldest_key in self.l2_cache:
                del self.l2_cache[oldest_key]

        self.l2_cache[key] = (value, datetime.now(), size)
        self._update_access_order(key, 2)
        return True

    async def _put_l3(self, key: str, value: Any, size: int) -> bool:
        """Put value in L3 cache with compression."""
        compressed_value = pickle.dumps(value)
        compressed_size = len(compressed_value)

        while self._get_tier_size(3) + compressed_size > self.l3_size_bytes and self.l3_cache:
            oldest_key = self.l3_access_order.popleft()
            if oldest_key in self.l3_cache:
                del self.l3_cache[oldest_key]

        self.l3_cache[key] = (compressed_value, datetime.now(), compressed_size)
        self._update_access_order(key, 3)
        return True

    async def _promote_to_l1(self, key: str, value: Any, size: int):
        """Promote value from L2 to L1."""
        # Remove from L2
        if key in self.l2_cache:
            del self.l2_cache[key]
            self.l2_access_order.remove(key)

        # Add to L1
        await self._put_l1(key, value, size)

    async def _promote_to_l2(self, key: str, value: Any, size: int):
        """Promote value from L3 to L2."""
        # Remove from L3
        if key in self.l3_cache:
            del self.l3_cache[key]
            self.l3_access_order.remove(key)

        # Add to L2
        await self._put_l2(key, value, size)

    def _is_valid(self, timestamp: datetime) -> bool:
        """Check if cached value is still valid."""
        return (datetime.now() - timestamp).total_seconds() < self.ttl_seconds

    def _evict_expired(self, key: str, tier: int):
        """Evict expired entry from specified tier."""
        if tier == 1 and key in self.l1_cache:
            del self.l1_cache[key]
            if key in self.l1_access_order:
                self.l1_access_order.remove(key)
        elif tier == 2 and key in self.l2_cache:
            del self.l2_cache[key]
            if key in self.l2_access_order:
                self.l2_access_order.remove(key)
        elif tier == 3 and key in self.l3_cache:
            del self.l3_cache[key]
            if key in self.l3_access_order:
                self.l3_access_order.remove(key)

    def _update_access_order(self, key: str, tier: int):
        """Update access order for LRU tracking."""
        if tier == 1:
            if key in self.l1_access_order:
                self.l1_access_order.remove(key)
            self.l1_access_order.append(key)
        elif tier == 2:
            if key in self.l2_access_order:
                self.l2_access_order.remove(key)
            self.l2_access_order.append(key)
        elif tier == 3:
            if key in self.l3_access_order:
                self.l3_access_order.remove(key)
            self.l3_access_order.append(key)

    def _get_tier_size(self, tier: int) -> int:
        """Get current size of specified tier."""
        if tier == 1:
            return sum(size for _, _, size in self.l1_cache.values())
        elif tier == 2:
            return sum(size for _, _, size in self.l2_cache.values())
        elif tier == 3:
            return sum(size for _, _, size in self.l3_cache.values())
        return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'performance_metrics': {
                'hit_rate': self.stats.cache_hit_rate,
                'average_latency_ms': self.stats.average_latency,
                'total_requests': self.stats.request_count
            },
            'tier_utilization': {
                'l1_size_mb': self._get_tier_size(1) / (1024 * 1024),
                'l1_entries': len(self.l1_cache),
                'l2_size_mb': self._get_tier_size(2) / (1024 * 1024),
                'l2_entries': len(self.l2_cache),
                'l3_size_mb': self._get_tier_size(3) / (1024 * 1024),
                'l3_entries': len(self.l3_cache)
            }
        }

class ParallelContextProcessor:
    """High-performance parallel processor for context operations."""

    def __init__(
        self,
        max_workers: int = None,
        use_process_pool: bool = False,
        batch_size: int = 100
    ):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.batch_size = batch_size

        # Executor pools
        if use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.active_requests: Dict[str, ContextRequest] = {}
        self.request_lock = threading.Lock()

    async def process_context_batch(
        self,
        requests: List[ContextRequest],
        processor_func: Callable
    ) -> List[Tuple[str, Any, bool]]:
        """Process batch of context requests in parallel."""
        start_time = time.time()

        try:
            # Group requests by performance tier for optimal processing
            tier_groups = defaultdict(list)
            for request in requests:
                tier_groups[request.performance_tier].append(request)

            # Process ultra-low latency requests first
            results = []

            # Ultra-low latency: Process immediately with minimal overhead
            if PerformanceTier.ULTRA_LOW_LATENCY in tier_groups:
                ultra_requests = tier_groups[PerformanceTier.ULTRA_LOW_LATENCY]
                for request in ultra_requests:
                    result = await self._process_single_fast(request, processor_func)
                    results.append(result)

            # Other tiers: Use parallel processing
            remaining_requests = []
            for tier in [PerformanceTier.LOW_LATENCY, PerformanceTier.STANDARD, PerformanceTier.BATCH]:
                if tier in tier_groups:
                    remaining_requests.extend(tier_groups[tier])

            if remaining_requests:
                parallel_results = await self._process_parallel_batch(remaining_requests, processor_func)
                results.extend(parallel_results)

            # Update metrics
            total_time = (time.time() - start_time) * 1000
            for request_id, result, is_error in results:
                self.metrics.add_request(total_time / len(results), is_error)

            return results

        except Exception as e:
            logging.error(f"Batch processing error: {e}")
            total_time = (time.time() - start_time) * 1000
            self.metrics.add_request(total_time, is_error=True)
            raise

    async def _process_single_fast(
        self,
        request: ContextRequest,
        processor_func: Callable
    ) -> Tuple[str, Any, bool]:
        """Process single request with minimal overhead for ultra-low latency."""
        request.started_at = datetime.now()

        try:
            # Direct synchronous processing for minimal latency
            result = processor_func(request.context_data, request.operation)
            request.completed_at = datetime.now()
            return (request.id, result, False)

        except Exception as e:
            request.completed_at = datetime.now()
            logging.error(f"Fast processing error for {request.id}: {e}")
            return (request.id, None, True)

    async def _process_parallel_batch(
        self,
        requests: List[ContextRequest],
        processor_func: Callable
    ) -> List[Tuple[str, Any, bool]]:
        """Process batch of requests in parallel using executor."""
        loop = asyncio.get_event_loop()

        # Create futures for parallel execution
        futures = []
        for request in requests:
            future = loop.run_in_executor(
                self.executor,
                self._process_request_wrapper,
                request,
                processor_func
            )
            futures.append((request.id, future))

        # Wait for completion with timeout handling
        results = []
        for request_id, future in futures:
            try:
                result = await asyncio.wait_for(future, timeout=30.0)  # 30s timeout
                results.append((request_id, result, False))
            except asyncio.TimeoutError:
                logging.error(f"Request {request_id} timed out")
                results.append((request_id, None, True))
            except Exception as e:
                logging.error(f"Request {request_id} failed: {e}")
                results.append((request_id, None, True))

        return results

    def _process_request_wrapper(self, request: ContextRequest, processor_func: Callable) -> Any:
        """Wrapper for processing individual request in executor."""
        request.started_at = datetime.now()

        try:
            with self.request_lock:
                self.active_requests[request.id] = request

            result = processor_func(request.context_data, request.operation)
            request.completed_at = datetime.now()
            return result

        except Exception as e:
            request.completed_at = datetime.now()
            raise
        finally:
            with self.request_lock:
                if request.id in self.active_requests:
                    del self.active_requests[request.id]

    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor performance statistics."""
        with self.request_lock:
            active_count = len(self.active_requests)

        return {
            'performance_metrics': {
                'average_latency_ms': self.metrics.average_latency,
                'throughput_ops_per_sec': self.metrics.throughput_ops_per_sec,
                'error_rate': self.metrics.error_rate,
                'total_requests': self.metrics.request_count
            },
            'resource_utilization': {
                'max_workers': self.max_workers,
                'active_requests': active_count,
                'utilization_percentage': (active_count / self.max_workers) * 100
            }
        }

class HighPerformanceContextEngine:
    """Main high-performance context processing engine."""

    def __init__(
        self,
        cache_config: Dict[str, Any] = None,
        processor_config: Dict[str, Any] = None,
        performance_targets: Dict[PerformanceTier, float] = None
    ):
        # Initialize components
        cache_config = cache_config or {}
        processor_config = processor_config or {}

        self.cache = HighPerformanceCache(**cache_config)
        self.processor = ParallelContextProcessor(**processor_config)

        # Performance targets (in milliseconds)
        self.performance_targets = performance_targets or {
            PerformanceTier.ULTRA_LOW_LATENCY: 10.0,
            PerformanceTier.LOW_LATENCY: 50.0,
            PerformanceTier.STANDARD: 200.0,
            PerformanceTier.BATCH: 2000.0
        }

        # Request queue with priority handling
        self.request_queues: Dict[ContextPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in ContextPriority
        }

        # Performance monitoring
        self.global_metrics = PerformanceMetrics()
        self.monitoring_enabled = True
        self.monitoring_task = None

        # Circuit breaker for failure handling
        self.circuit_breaker = {
            'failure_threshold': 50,  # failures per minute
            'recovery_timeout': 60,   # seconds
            'current_failures': 0,
            'state': 'closed',        # closed, open, half-open
            'last_failure_time': None
        }

    async def start(self):
        """Start the high-performance context engine."""
        if self.monitoring_enabled:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Start request processing workers
        for priority in ContextPriority:
            asyncio.create_task(self._process_priority_queue(priority))

    async def stop(self):
        """Stop the context engine gracefully."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    async def process_context_request(
        self,
        context_data: Dict[str, Any],
        operation: str,
        priority: ContextPriority = ContextPriority.NORMAL,
        performance_tier: PerformanceTier = PerformanceTier.STANDARD,
        timeout_ms: int = None
    ) -> Any:
        """Process a context request with performance optimization."""
        # Generate request ID
        request_id = self._generate_request_id(context_data, operation)

        # Set timeout based on performance tier
        if timeout_ms is None:
            timeout_ms = int(self.performance_targets[performance_tier])

        # Create request
        request = ContextRequest(
            id=request_id,
            context_data=context_data,
            operation=operation,
            priority=priority,
            performance_tier=performance_tier,
            timeout_ms=timeout_ms
        )

        start_time = time.time()

        try:
            # Check circuit breaker
            if not self._check_circuit_breaker():
                raise Exception("Circuit breaker is open - service degraded")

            # Try cache first for read operations
            if operation.startswith('get') or operation.startswith('read'):
                cache_key = self._generate_cache_key(context_data, operation)
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    latency = (time.time() - start_time) * 1000
                    self.global_metrics.add_request(latency, cache_hit=True)
                    return cached_result

            # Process request based on performance tier
            if performance_tier == PerformanceTier.ULTRA_LOW_LATENCY:
                result = await self._process_ultra_low_latency(request)
            else:
                # Queue for batch processing
                result = await self._queue_and_process(request)

            # Cache result for future requests
            if operation.startswith('get') or operation.startswith('read'):
                cache_key = self._generate_cache_key(context_data, operation)
                tier_hint = 1 if performance_tier == PerformanceTier.ULTRA_LOW_LATENCY else 2
                await self.cache.put(cache_key, result, tier_hint)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.global_metrics.add_request(latency)

            # Check performance target compliance
            target_latency = self.performance_targets[performance_tier]
            if latency > target_latency:
                logging.warning(f"Performance target missed: {latency:.2f}ms > {target_latency}ms")

            return result

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.global_metrics.add_request(latency, is_error=True)
            self._record_circuit_breaker_failure()
            raise

    async def _process_ultra_low_latency(self, request: ContextRequest) -> Any:
        """Process ultra-low latency request with minimal overhead."""
        # Use dedicated fast-path processing
        processor_func = self._get_processor_function(request.operation)
        result = await self.processor._process_single_fast(request, processor_func)
        return result[1]  # Extract result from tuple

    async def _queue_and_process(self, request: ContextRequest) -> Any:
        """Queue request for batch processing."""
        # Add to appropriate priority queue
        await self.request_queues[request.priority].put(request)

        # Wait for result (implement with future/callback mechanism)
        result_future = asyncio.Future()
        request.callback = lambda r: result_future.set_result(r)

        try:
            return await asyncio.wait_for(result_future, timeout=request.timeout_ms / 1000)
        except asyncio.TimeoutError:
            raise Exception(f"Request {request.id} timed out after {request.timeout_ms}ms")

    async def _process_priority_queue(self, priority: ContextPriority):
        """Process requests from a specific priority queue."""
        queue = self.request_queues[priority]
        batch = []

        while True:
            try:
                # Collect batch based on priority
                if priority == ContextPriority.CRITICAL:
                    # Process immediately
                    request = await queue.get()
                    batch = [request]
                else:
                    # Wait for batch to form or timeout
                    try:
                        request = await asyncio.wait_for(queue.get(), timeout=0.01)
                        batch.append(request)

                        # Collect more requests for batch
                        while len(batch) < self.processor.batch_size:
                            try:
                                request = await asyncio.wait_for(queue.get(), timeout=0.001)
                                batch.append(request)
                            except asyncio.TimeoutError:
                                break
                    except asyncio.TimeoutError:
                        continue

                if batch:
                    await self._process_batch(batch)
                    batch = []

            except Exception as e:
                logging.error(f"Error processing priority queue {priority}: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retrying

    async def _process_batch(self, requests: List[ContextRequest]):
        """Process a batch of requests."""
        if not requests:
            return

        processor_func = self._get_processor_function(requests[0].operation)
        results = await self.processor.process_context_batch(requests, processor_func)

        # Deliver results to callbacks
        result_map = {request_id: (result, is_error) for request_id, result, is_error in results}

        for request in requests:
            if request.callback:
                result, is_error = result_map.get(request.id, (None, True))
                if not is_error:
                    request.callback(result)
                else:
                    request.callback(Exception(f"Processing failed for {request.id}"))

    def _get_processor_function(self, operation: str) -> Callable:
        """Get appropriate processor function for operation."""
        # This would be implemented based on your specific operations
        # For demo purposes, return a simple processor
        def default_processor(context_data: Dict[str, Any], operation: str) -> Any:
            # Simulate some processing work
            time.sleep(0.001)  # 1ms simulated work
            return f"Processed {operation} with {len(context_data)} context items"

        return default_processor

    def _generate_request_id(self, context_data: Dict[str, Any], operation: str) -> str:
        """Generate unique request ID."""
        content = f"{json.dumps(context_data, sort_keys=True)}_{operation}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _generate_cache_key(self, context_data: Dict[str, Any], operation: str) -> str:
        """Generate cache key for context data and operation."""
        content = f"{json.dumps(context_data, sort_keys=True)}_{operation}"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests."""
        now = time.time()

        if self.circuit_breaker['state'] == 'open':
            # Check if recovery timeout has passed
            if (now - self.circuit_breaker['last_failure_time']) > self.circuit_breaker['recovery_timeout']:
                self.circuit_breaker['state'] = 'half-open'
                self.circuit_breaker['current_failures'] = 0
                return True
            return False

        return True  # closed or half-open

    def _record_circuit_breaker_failure(self):
        """Record failure for circuit breaker."""
        self.circuit_breaker['current_failures'] += 1
        self.circuit_breaker['last_failure_time'] = time.time()

        if self.circuit_breaker['current_failures'] >= self.circuit_breaker['failure_threshold']:
            self.circuit_breaker['state'] = 'open'
            logging.warning("Circuit breaker opened due to high failure rate")

    async def _monitoring_loop(self):
        """Continuous monitoring of performance metrics."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute

                # Log performance statistics
                stats = self.get_performance_statistics()
                logging.info(f"Performance Stats: {json.dumps(stats, indent=2)}")

                # Check for performance degradation
                if stats['global_metrics']['average_latency_ms'] > 100:  # 100ms threshold
                    logging.warning("Performance degradation detected - high latency")

                if stats['global_metrics']['error_rate'] > 0.05:  # 5% error rate threshold
                    logging.warning("Performance degradation detected - high error rate")

            except Exception as e:
                logging.error(f"Monitoring error: {e}")

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'global_metrics': {
                'average_latency_ms': self.global_metrics.average_latency,
                'max_latency_ms': self.global_metrics.max_latency,
                'min_latency_ms': self.global_metrics.min_latency if self.global_metrics.min_latency != float('inf') else 0,
                'throughput_ops_per_sec': self.global_metrics.throughput_ops_per_sec,
                'error_rate': self.global_metrics.error_rate,
                'total_requests': self.global_metrics.request_count
            },
            'cache_statistics': self.cache.get_cache_stats(),
            'processor_statistics': self.processor.get_processor_stats(),
            'circuit_breaker': {
                'state': self.circuit_breaker['state'],
                'current_failures': self.circuit_breaker['current_failures'],
                'failure_threshold': self.circuit_breaker['failure_threshold']
            },
            'queue_status': {
                priority.name: queue.qsize()
                for priority, queue in self.request_queues.items()
            }
        }

# Example usage and performance testing
async def performance_test_example():
    """Example demonstrating high-performance context engine."""

    # Configure high-performance engine
    cache_config = {
        'l1_size_mb': 128,
        'l2_size_mb': 512,
        'l3_size_mb': 2048,
        'ttl_seconds': 3600
    }

    processor_config = {
        'max_workers': 16,
        'use_process_pool': False,
        'batch_size': 50
    }

    performance_targets = {
        PerformanceTier.ULTRA_LOW_LATENCY: 5.0,   # 5ms target
        PerformanceTier.LOW_LATENCY: 25.0,        # 25ms target
        PerformanceTier.STANDARD: 100.0,          # 100ms target
        PerformanceTier.BATCH: 1000.0             # 1s target
    }

    # Create engine
    engine = HighPerformanceContextEngine(
        cache_config=cache_config,
        processor_config=processor_config,
        performance_targets=performance_targets
    )

    await engine.start()

    try:
        # Test different performance tiers
        test_contexts = [
            {
                'context': {'user_id': f'user_{i}', 'session_data': {'actions': i % 10}},
                'operation': 'get_recommendations',
                'tier': PerformanceTier.ULTRA_LOW_LATENCY if i < 10 else PerformanceTier.LOW_LATENCY,
                'priority': ContextPriority.CRITICAL if i < 5 else ContextPriority.NORMAL
            }
            for i in range(100)
        ]

        # Execute requests and measure performance
        start_time = time.time()
        results = []

        for test_ctx in test_contexts:
            try:
                result = await engine.process_context_request(
                    context_data=test_ctx['context'],
                    operation=test_ctx['operation'],
                    priority=test_ctx['priority'],
                    performance_tier=test_ctx['tier']
                )
                results.append((True, result))
            except Exception as e:
                results.append((False, str(e)))

        total_time = time.time() - start_time

        # Calculate statistics
        successful_requests = sum(1 for success, _ in results if success)
        success_rate = successful_requests / len(results)
        throughput = len(results) / total_time

        print(f"Performance Test Results:")
        print(f"  Total Requests: {len(results)}")
        print(f"  Successful Requests: {successful_requests}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} requests/second")

        # Get detailed statistics
        stats = engine.get_performance_statistics()
        print(f"\nDetailed Engine Statistics:")
        print(json.dumps(stats, indent=2))

    finally:
        await engine.stop()

# Run the performance test
if __name__ == "__main__":
    import os
    asyncio.run(performance_test_example())
```

## Key Takeaways

1. **Multi-Tier Architecture**: L1/L2/L3 caching with intelligent promotion and eviction

2. **Parallel Processing**: Optimized thread/process pools with performance tier handling

3. **Circuit Breaker Pattern**: Automatic failure detection and recovery mechanisms

4. **Performance Monitoring**: Real-time metrics and adaptive optimization

5. **Priority Queue System**: Handle critical requests with dedicated fast paths

6. **Resource Management**: Efficient memory and CPU utilization tracking

## What's Next

In Lesson 2, we'll explore advanced caching strategies and optimization techniques for maximum performance gains.

---

**Practice Exercise**: Build a complete high-performance context engine achieving <10ms P99 latency for ultra-low latency tier while handling 1000+ concurrent requests. Demonstrate >95% cache hit rates and <1% error rates under load.
