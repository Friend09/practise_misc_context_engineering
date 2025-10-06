# Lesson 2: Context Optimization Algorithms

## Introduction

Context optimization is the process of maintaining peak performance as context accumulates over time. Without optimization, context systems degrade—becoming slow, bloated, and ineffective. This lesson teaches you to build algorithms that automatically optimize context for quality, performance, and relevance.

## The Optimization Challenge

### Why Context Needs Optimization

As AI agents interact over time:

- **Context accumulates**: Every interaction adds more information
- **Relevance decays**: Older context becomes less relevant
- **Performance degrades**: Large context slows processing
- **Quality varies**: Not all context is equally valuable

**Without optimization**: Systems slow down, responses degrade, costs increase

**With optimization**: Maintain performance, quality, and efficiency indefinitely

## Context Quality Assessment and Scoring

### Multi-Dimensional Quality Scoring

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta

@dataclass
class QualityScore:
    """Multi-dimensional quality assessment."""
    relevance: float  # 0.0-1.0
    importance: float  # 0.0-1.0
    recency: float  # 0.0-1.0
    utility: float  # 0.0-1.0
    coherence: float  # 0.0-1.0
    overall: float  # Computed weighted average

    @classmethod
    def compute(cls, **scores):
        """Compute overall score from components."""
        weights = {
            'relevance': 0.3,
            'importance': 0.25,
            'recency': 0.2,
            'utility': 0.15,
            'coherence': 0.1
        }

        overall = sum(scores.get(k, 0.5) * v for k, v in weights.items())

        return cls(
            relevance=scores.get('relevance', 0.5),
            importance=scores.get('importance', 0.5),
            recency=scores.get('recency', 0.5),
            utility=scores.get('utility', 0.5),
            coherence=scores.get('coherence', 0.5),
            overall=overall
        )

class ContextQualityAssessor:
    """
    Assess context quality across multiple dimensions.
    """
    def __init__(self):
        self.baseline_scores: Dict[str, float] = {}
        self.assessment_history: List[Dict] = []

    def assess_context_item(
        self,
        item: 'ContextItem',
        current_task: Optional[str] = None,
        context_hub: Optional['CentralizedContextHub'] = None
    ) -> QualityScore:
        """
        Comprehensive quality assessment of a context item.
        """
        # Relevance: How relevant to current task
        relevance = self._assess_relevance(item, current_task)

        # Importance: Intrinsic importance
        importance = item.importance

        # Recency: How recent is the context
        recency = self._assess_recency(item)

        # Utility: How useful has it been
        utility = self._assess_utility(item, context_hub)

        # Coherence: How well it fits with other context
        coherence = self._assess_coherence(item, context_hub)

        # Compute overall score
        score = QualityScore.compute(
            relevance=relevance,
            importance=importance,
            recency=recency,
            utility=utility,
            coherence=coherence
        )

        # Record assessment
        self.assessment_history.append({
            'item_id': item.id,
            'score': score,
            'timestamp': datetime.now()
        })

        return score

    def _assess_relevance(
        self,
        item: 'ContextItem',
        current_task: Optional[str]
    ) -> float:
        """
        Assess relevance to current task.
        """
        if not current_task:
            return 0.5  # Neutral if no task specified

        # Simple keyword matching (use embeddings in production)
        task_words = set(current_task.lower().split())
        content_words = set(str(item.content).lower().split())

        # Jaccard similarity
        intersection = len(task_words & content_words)
        union = len(task_words | content_words)

        if union == 0:
            return 0.0

        return intersection / union

    def _assess_recency(self, item: 'ContextItem') -> float:
        """
        Assess recency with exponential decay.
        """
        age = datetime.now() - item.timestamp
        age_hours = age.total_seconds() / 3600

        # Exponential decay with 24-hour half-life
        decay_rate = np.log(2) / 24  # Half-life = 24 hours
        recency_score = np.exp(-decay_rate * age_hours)

        return float(recency_score)

    def _assess_utility(
        self,
        item: 'ContextItem',
        context_hub: Optional['CentralizedContextHub']
    ) -> float:
        """
        Assess utility based on usage patterns.
        """
        # Access frequency
        access_score = min(item.access_count / 10, 1.0) * 0.4

        # Decision usage
        decision_usage = item.metadata.get('used_in_decisions', 0)
        decision_score = min(decision_usage / 5, 1.0) * 0.4

        # Recent access
        if item.last_accessed:
            recent_access_age = datetime.now() - item.last_accessed
            recent_hours = recent_access_age.total_seconds() / 3600
            recent_score = max(0, 1 - (recent_hours / 12)) * 0.2
        else:
            recent_score = 0.0

        return access_score + decision_score + recent_score

    def _assess_coherence(
        self,
        item: 'ContextItem',
        context_hub: Optional['CentralizedContextHub']
    ) -> float:
        """
        Assess how well item fits with overall context.
        """
        if not context_hub:
            return 0.5

        # Check if item has related context through tags
        related_count = 0
        total_items = len(context_hub.active_context)

        if total_items == 0:
            return 0.5

        for other_item in context_hub.active_context.values():
            if other_item.id == item.id:
                continue

            # Check tag overlap
            common_tags = set(item.tags) & set(other_item.tags)
            if common_tags:
                related_count += 1

        # Coherence = proportion of related items
        return related_count / max(total_items - 1, 1)

    def assess_context_hub(
        self,
        context_hub: 'CentralizedContextHub',
        current_task: Optional[str] = None
    ) -> Dict[str, QualityScore]:
        """
        Assess all items in context hub.
        """
        scores = {}
        for key, item in context_hub.active_context.items():
            scores[key] = self.assess_context_item(
                item,
                current_task,
                context_hub
            )
        return scores

    def get_quality_summary(
        self,
        scores: Dict[str, QualityScore]
    ) -> Dict:
        """
        Get summary statistics of quality scores.
        """
        if not scores:
            return {
                'count': 0,
                'avg_overall': 0.0,
                'avg_relevance': 0.0,
                'avg_utility': 0.0
            }

        overall_scores = [s.overall for s in scores.values()]
        relevance_scores = [s.relevance for s in scores.values()]
        utility_scores = [s.utility for s in scores.values()]

        return {
            'count': len(scores),
            'avg_overall': np.mean(overall_scores),
            'std_overall': np.std(overall_scores),
            'min_overall': np.min(overall_scores),
            'max_overall': np.max(overall_scores),
            'avg_relevance': np.mean(relevance_scores),
            'avg_utility': np.mean(utility_scores)
        }
```

## Automated Context Pruning and Compression

### Intelligent Pruning Algorithms

```python
class ContextPruner:
    """
    Intelligently prune context to maintain optimal size and quality.
    """
    def __init__(self, quality_assessor: ContextQualityAssessor):
        self.quality_assessor = quality_assessor
        self.pruning_history: List[Dict] = []

    def prune_context(
        self,
        context_hub: 'CentralizedContextHub',
        target_size: Optional[int] = None,
        quality_threshold: float = 0.4,
        current_task: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Prune context to target size or quality threshold.

        Returns:
            Dict with 'kept' and 'pruned' item IDs
        """
        # Assess all context items
        scores = self.quality_assessor.assess_context_hub(
            context_hub,
            current_task
        )

        # Sort by quality score
        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1].overall,
            reverse=True
        )

        kept = []
        pruned = []

        # Determine what to keep
        if target_size:
            # Keep top N items
            kept = [key for key, score in sorted_items[:target_size]]
            pruned = [key for key, score in sorted_items[target_size:]]
        else:
            # Keep items above quality threshold
            for key, score in sorted_items:
                if score.overall >= quality_threshold:
                    kept.append(key)
                else:
                    pruned.append(key)

        # Execute pruning
        for key in pruned:
            context_hub.remove_context(key)

        # Record pruning operation
        self.pruning_history.append({
            'timestamp': datetime.now(),
            'kept_count': len(kept),
            'pruned_count': len(pruned),
            'quality_threshold': quality_threshold,
            'target_size': target_size
        })

        return {
            'kept': kept,
            'pruned': pruned
        }

    def adaptive_prune(
        self,
        context_hub: 'CentralizedContextHub',
        performance_metric: float,
        current_task: Optional[str] = None
    ) -> Dict:
        """
        Adaptively prune based on performance.

        If performance is good, be conservative.
        If performance is poor, be aggressive.
        """
        # Determine aggressiveness based on performance
        if performance_metric > 0.8:
            # Good performance, minimal pruning
            quality_threshold = 0.3
        elif performance_metric > 0.6:
            # Moderate performance
            quality_threshold = 0.4
        else:
            # Poor performance, aggressive pruning
            quality_threshold = 0.6

        return self.prune_context(
            context_hub,
            quality_threshold=quality_threshold,
            current_task=current_task
        )

    def get_pruning_stats(self) -> Dict:
        """Get statistics on pruning operations."""
        if not self.pruning_history:
            return {'total_operations': 0}

        kept_counts = [h['kept_count'] for h in self.pruning_history]
        pruned_counts = [h['pruned_count'] for h in self.pruning_history]

        return {
            'total_operations': len(self.pruning_history),
            'avg_kept': np.mean(kept_counts),
            'avg_pruned': np.mean(pruned_counts),
            'total_pruned': sum(pruned_counts)
        }

class ContextCompressor:
    """
    Compress context while preserving essential information.
    """
    def __init__(self):
        self.compression_history: List[Dict] = []

    def compress_context(
        self,
        context_hub: 'CentralizedContextHub',
        compression_ratio: float = 0.5
    ) -> Dict:
        """
        Compress context to reduce size while preserving information.

        Args:
            compression_ratio: Target size as fraction of original (0.5 = 50%)

        Returns:
            Compression statistics
        """
        original_size = len(context_hub.active_context)
        target_size = int(original_size * compression_ratio)

        # Strategy 1: Merge similar context items
        merged = self._merge_similar_items(context_hub)

        # Strategy 2: Summarize verbose items
        summarized = self._summarize_verbose_items(context_hub)

        # Strategy 3: Deduplicate redundant information
        deduplicated = self._deduplicate_content(context_hub)

        # Record compression
        final_size = len(context_hub.active_context)
        compression_achieved = 1 - (final_size / original_size)

        result = {
            'original_size': original_size,
            'final_size': final_size,
            'target_size': target_size,
            'compression_ratio': compression_achieved,
            'merged_count': merged,
            'summarized_count': summarized,
            'deduplicated_count': deduplicated
        }

        self.compression_history.append({
            **result,
            'timestamp': datetime.now()
        })

        return result

    def _merge_similar_items(
        self,
        context_hub: 'CentralizedContextHub'
    ) -> int:
        """
        Merge similar context items.
        """
        merged_count = 0
        items = list(context_hub.active_context.items())

        i = 0
        while i < len(items) - 1:
            key1, item1 = items[i]

            for j in range(i + 1, len(items)):
                key2, item2 = items[j]

                # Check similarity
                if self._are_similar(item1, item2):
                    # Merge item2 into item1
                    merged_content = self._merge_content(
                        item1.content,
                        item2.content
                    )
                    item1.content = merged_content

                    # Update importance
                    item1.importance = max(item1.importance, item2.importance)

                    # Combine tags
                    item1.tags = list(set(item1.tags + item2.tags))

                    # Remove item2
                    context_hub.remove_context(key2)
                    items.pop(j)
                    merged_count += 1
                    break

            i += 1

        return merged_count

    def _are_similar(
        self,
        item1: 'ContextItem',
        item2: 'ContextItem',
        threshold: float = 0.7
    ) -> bool:
        """Check if two items are similar enough to merge."""
        # Check type
        if item1.type != item2.type:
            return False

        # Check tag overlap
        if item1.tags and item2.tags:
            common_tags = set(item1.tags) & set(item2.tags)
            if len(common_tags) / max(len(item1.tags), len(item2.tags)) < 0.5:
                return False

        # Check content similarity (simplified)
        content1 = str(item1.content).lower().split()
        content2 = str(item2.content).lower().split()

        common = set(content1) & set(content2)
        union = set(content1) | set(content2)

        similarity = len(common) / len(union) if union else 0

        return similarity >= threshold

    def _merge_content(self, content1: any, content2: any) -> any:
        """Merge two pieces of content."""
        # If both are strings, concatenate
        if isinstance(content1, str) and isinstance(content2, str):
            return f"{content1}\n{content2}"

        # If both are dicts, merge
        if isinstance(content1, dict) and isinstance(content2, dict):
            merged = content1.copy()
            merged.update(content2)
            return merged

        # If both are lists, combine
        if isinstance(content1, list) and isinstance(content2, list):
            return content1 + content2

        # Otherwise, keep first
        return content1

    def _summarize_verbose_items(
        self,
        context_hub: 'CentralizedContextHub',
        length_threshold: int = 1000
    ) -> int:
        """
        Summarize context items that are too verbose.
        """
        summarized_count = 0

        for key, item in context_hub.active_context.items():
            content_str = str(item.content)

            if len(content_str) > length_threshold:
                # Summarize (simplified - use LLM in production)
                summary = self._create_summary(content_str)
                item.content = summary
                item.metadata['summarized'] = True
                item.metadata['original_length'] = len(content_str)
                summarized_count += 1

        return summarized_count

    def _create_summary(self, text: str, max_length: int = 200) -> str:
        """
        Create summary of text.

        Simplified version - use LLM for production.
        """
        # Take first N characters + ellipsis
        if len(text) <= max_length:
            return text

        return text[:max_length] + "..."

    def _deduplicate_content(
        self,
        context_hub: 'CentralizedContextHub'
    ) -> int:
        """
        Remove duplicate context items.
        """
        deduplicated_count = 0
        seen_content = {}
        items_to_remove = []

        for key, item in context_hub.active_context.items():
            content_hash = hash(str(item.content))

            if content_hash in seen_content:
                # Duplicate found
                original_key = seen_content[content_hash]
                original_item = context_hub.active_context[original_key]

                # Keep the one with higher importance
                if item.importance > original_item.importance:
                    items_to_remove.append(original_key)
                    seen_content[content_hash] = key
                else:
                    items_to_remove.append(key)

                deduplicated_count += 1
            else:
                seen_content[content_hash] = key

        # Remove duplicates
        for key in items_to_remove:
            context_hub.remove_context(key)

        return deduplicated_count
```

## Performance-Based Context Adaptation

### Adaptive Optimization Engine

```python
class AdaptiveContextOptimizer:
    """
    Automatically adapt context optimization based on performance.
    """
    def __init__(self):
        self.quality_assessor = ContextQualityAssessor()
        self.pruner = ContextPruner(self.quality_assessor)
        self.compressor = ContextCompressor()

        self.performance_history: List[Dict] = []
        self.optimization_strategy = 'balanced'

    def optimize(
        self,
        context_hub: 'CentralizedContextHub',
        performance_metrics: Dict[str, float],
        current_task: Optional[str] = None
    ) -> Dict:
        """
        Adaptively optimize context based on performance.
        """
        # Assess current state
        quality_scores = self.quality_assessor.assess_context_hub(
            context_hub,
            current_task
        )
        quality_summary = self.quality_assessor.get_quality_summary(
            quality_scores
        )

        # Determine optimization strategy
        strategy = self._select_strategy(
            performance_metrics,
            quality_summary,
            len(context_hub.active_context)
        )

        # Execute strategy
        result = self._execute_strategy(
            strategy,
            context_hub,
            performance_metrics,
            current_task
        )

        # Record optimization
        self.performance_history.append({
            'timestamp': datetime.now(),
            'strategy': strategy,
            'performance_metrics': performance_metrics,
            'quality_summary': quality_summary,
            'result': result
        })

        return result

    def _select_strategy(
        self,
        performance_metrics: Dict[str, float],
        quality_summary: Dict,
        context_size: int
    ) -> str:
        """
        Select optimization strategy based on metrics.
        """
        # Extract key metrics
        response_time = performance_metrics.get('response_time', 0.5)
        accuracy = performance_metrics.get('accuracy', 0.8)
        avg_quality = quality_summary.get('avg_overall', 0.5)

        # Decision logic
        if response_time > 2.0 and context_size > 500:
            return 'aggressive_compression'
        elif accuracy < 0.7:
            return 'quality_focused'
        elif avg_quality < 0.4:
            return 'quality_improvement'
        elif context_size > 1000:
            return 'size_reduction'
        else:
            return 'balanced'

    def _execute_strategy(
        self,
        strategy: str,
        context_hub: 'CentralizedContextHub',
        performance_metrics: Dict[str, float],
        current_task: Optional[str]
    ) -> Dict:
        """Execute selected optimization strategy."""
        if strategy == 'aggressive_compression':
            # Compress to 40% of original size
            compression_result = self.compressor.compress_context(
                context_hub,
                compression_ratio=0.4
            )
            # Then prune low quality
            prune_result = self.pruner.prune_context(
                context_hub,
                quality_threshold=0.5,
                current_task=current_task
            )
            return {
                'strategy': strategy,
                'compression': compression_result,
                'pruning': prune_result
            }

        elif strategy == 'quality_focused':
            # Aggressive quality-based pruning
            prune_result = self.pruner.prune_context(
                context_hub,
                quality_threshold=0.6,
                current_task=current_task
            )
            return {
                'strategy': strategy,
                'pruning': prune_result
            }

        elif strategy == 'quality_improvement':
            # Remove very low quality items only
            prune_result = self.pruner.prune_context(
                context_hub,
                quality_threshold=0.3,
                current_task=current_task
            )
            return {
                'strategy': strategy,
                'pruning': prune_result
            }

        elif strategy == 'size_reduction':
            # Target 60% of current size
            target = int(len(context_hub.active_context) * 0.6)
            prune_result = self.pruner.prune_context(
                context_hub,
                target_size=target,
                current_task=current_task
            )
            return {
                'strategy': strategy,
                'pruning': prune_result
            }

        else:  # balanced
            # Moderate compression and pruning
            compression_result = self.compressor.compress_context(
                context_hub,
                compression_ratio=0.7
            )
            prune_result = self.pruner.prune_context(
                context_hub,
                quality_threshold=0.4,
                current_task=current_task
            )
            return {
                'strategy': strategy,
                'compression': compression_result,
                'pruning': prune_result
            }

    def get_optimization_report(self) -> Dict:
        """Generate report on optimization history."""
        if not self.performance_history:
            return {'total_optimizations': 0}

        strategies_used = {}
        for entry in self.performance_history:
            strategy = entry['strategy']
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1

        return {
            'total_optimizations': len(self.performance_history),
            'strategies_used': strategies_used,
            'most_common_strategy': max(
                strategies_used.items(),
                key=lambda x: x[1]
            )[0] if strategies_used else None
        }
```

## Real-Time Context Optimization Techniques

### Continuous Optimization System

```python
class ContinuousContextOptimizer:
    """
    Continuously optimize context in real-time.
    """
    def __init__(
        self,
        optimization_interval: int = 300,  # seconds
        enable_auto_optimization: bool = True
    ):
        self.optimization_interval = optimization_interval
        self.enable_auto_optimization = enable_auto_optimization

        self.adaptive_optimizer = AdaptiveContextOptimizer()
        self.last_optimization = datetime.now()

        self.real_time_metrics = {
            'response_times': [],
            'quality_scores': [],
            'context_sizes': []
        }

    def should_optimize_now(
        self,
        context_hub: 'CentralizedContextHub',
        current_performance: Dict[str, float]
    ) -> bool:
        """
        Determine if optimization should run now.
        """
        if not self.enable_auto_optimization:
            return False

        # Time-based trigger
        time_since_last = (datetime.now() - self.last_optimization).total_seconds()
        if time_since_last >= self.optimization_interval:
            return True

        # Performance-based trigger
        response_time = current_performance.get('response_time', 0)
        if response_time > 3.0:  # 3 second threshold
            return True

        # Size-based trigger
        if len(context_hub.active_context) > 1500:
            return True

        # Quality-based trigger
        avg_quality = current_performance.get('avg_quality', 1.0)
        if avg_quality < 0.3:
            return True

        return False

    def optimize_if_needed(
        self,
        context_hub: 'CentralizedContextHub',
        current_performance: Dict[str, float],
        current_task: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Run optimization if needed.
        """
        if self.should_optimize_now(context_hub, current_performance):
            result = self.adaptive_optimizer.optimize(
                context_hub,
                current_performance,
                current_task
            )

            self.last_optimization = datetime.now()

            return result

        return None

    def record_metrics(
        self,
        response_time: float,
        quality_score: float,
        context_size: int
    ):
        """Record real-time metrics for optimization decisions."""
        self.real_time_metrics['response_times'].append(response_time)
        self.real_time_metrics['quality_scores'].append(quality_score)
        self.real_time_metrics['context_sizes'].append(context_size)

        # Keep only recent metrics (last 100)
        for key in self.real_time_metrics:
            if len(self.real_time_metrics[key]) > 100:
                self.real_time_metrics[key] = self.real_time_metrics[key][-100:]

    def get_current_trends(self) -> Dict:
        """Analyze current performance trends."""
        if not any(self.real_time_metrics.values()):
            return {}

        return {
            'avg_response_time': np.mean(self.real_time_metrics['response_times']) if self.real_time_metrics['response_times'] else 0,
            'avg_quality': np.mean(self.real_time_metrics['quality_scores']) if self.real_time_metrics['quality_scores'] else 0,
            'avg_context_size': np.mean(self.real_time_metrics['context_sizes']) if self.real_time_metrics['context_sizes'] else 0,
            'quality_trend': self._calculate_trend(self.real_time_metrics['quality_scores']),
            'size_trend': self._calculate_trend(self.real_time_metrics['context_sizes'])
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return 'stable'

        # Compare first half to second half
        mid = len(values) // 2
        first_half = np.mean(values[:mid])
        second_half = np.mean(values[mid:])

        diff = (second_half - first_half) / first_half if first_half > 0 else 0

        if diff > 0.1:
            return 'increasing'
        elif diff < -0.1:
            return 'decreasing'
        else:
            return 'stable'
```

## Key Takeaways

1. **Multi-Dimensional Quality**: Assess context across relevance, importance, recency, utility, and coherence

2. **Adaptive Strategies**: Choose optimization approach based on current performance and needs

3. **Automated Pruning**: Remove low-quality context automatically while preserving valuable information

4. **Compression Techniques**: Merge, summarize, and deduplicate to reduce size without losing meaning

5. **Continuous Optimization**: Monitor and optimize in real-time to maintain peak performance

6. **Performance-Driven**: Let actual performance metrics guide optimization decisions

## What's Next

In Lesson 3, we'll explore adaptive context strategies that learn from usage patterns and continuously improve over time.

---

**Practice Exercise**: Implement a complete optimization system for your context hub. Run with 1000+ context items and measure optimization effectiveness. Target: Maintain <10ms retrieval time while keeping quality score >0.7.
