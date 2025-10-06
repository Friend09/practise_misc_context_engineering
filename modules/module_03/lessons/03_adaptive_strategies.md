# Lesson 3: Adaptive Context Strategies

## Introduction

The most sophisticated context management systems don't just optimize—they **learn and adapt** over time. Adaptive strategies enable AI systems to continuously improve their context management based on usage patterns, user feedback, and performance data. This lesson teaches you to build self-improving context systems.

## Learning from Context Usage Patterns

### Pattern Recognition System

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

@dataclass
class UsagePattern:
    """Represents a discovered usage pattern."""
    pattern_id: str
    pattern_type: str  # 'sequential', 'co-occurrence', 'temporal'
    items: List[str]
    frequency: int
    confidence: float
    last_seen: datetime
    metadata: Dict = field(default_factory=dict)

class ContextPatternAnalyzer:
    """
    Analyze and learn patterns from context usage.
    """
    def __init__(self):
        self.access_sequences: List[List[str]] = []
        self.co_occurrence_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        self.temporal_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.discovered_patterns: List[UsagePattern] = []

    def record_access(
        self,
        item_id: str,
        timestamp: Optional[datetime] = None
    ):
        """Record a context item access."""
        timestamp = timestamp or datetime.now()

        # Record for temporal analysis
        self.temporal_patterns[item_id].append(timestamp)

        # Add to current sequence
        if not self.access_sequences or len(self.access_sequences[-1]) > 10:
            self.access_sequences.append([])

        self.access_sequences[-1].append(item_id)

        # Update co-occurrence if there's a previous item
        if len(self.access_sequences[-1]) > 1:
            prev_item = self.access_sequences[-1][-2]
            self.co_occurrence_matrix[(prev_item, item_id)] += 1

    def analyze_patterns(
        self,
        min_frequency: int = 3,
        min_confidence: float = 0.6
    ) -> List[UsagePattern]:
        """
        Analyze usage data to discover patterns.
        """
        patterns = []

        # Find sequential patterns
        sequential = self._find_sequential_patterns(
            min_frequency,
            min_confidence
        )
        patterns.extend(sequential)

        # Find co-occurrence patterns
        co_occur = self._find_cooccurrence_patterns(
            min_frequency,
            min_confidence
        )
        patterns.extend(co_occur)

        # Find temporal patterns
        temporal = self._find_temporal_patterns(min_frequency)
        patterns.extend(temporal)

        self.discovered_patterns = patterns
        return patterns

    def _find_sequential_patterns(
        self,
        min_frequency: int,
        min_confidence: float
    ) -> List[UsagePattern]:
        """
        Find sequential access patterns (A → B → C).
        """
        patterns = []
        sequence_counts = defaultdict(int)

        # Count sequences of length 2 and 3
        for sequence in self.access_sequences:
            for i in range(len(sequence) - 1):
                # Length 2
                seq_2 = tuple(sequence[i:i+2])
                sequence_counts[seq_2] += 1

                # Length 3
                if i < len(sequence) - 2:
                    seq_3 = tuple(sequence[i:i+3])
                    sequence_counts[seq_3] += 1

        # Filter by frequency
        for seq, count in sequence_counts.items():
            if count >= min_frequency:
                # Calculate confidence (P(sequence) / P(first item))
                first_item_count = sum(
                    1 for s in self.access_sequences
                    if seq[0] in s
                )
                confidence = count / first_item_count if first_item_count > 0 else 0

                if confidence >= min_confidence:
                    patterns.append(UsagePattern(
                        pattern_id=f"seq_{'_'.join(seq)}",
                        pattern_type='sequential',
                        items=list(seq),
                        frequency=count,
                        confidence=confidence,
                        last_seen=datetime.now()
                    ))

        return patterns

    def _find_cooccurrence_patterns(
        self,
        min_frequency: int,
        min_confidence: float
    ) -> List[UsagePattern]:
        """
        Find items that frequently occur together.
        """
        patterns = []

        for (item1, item2), count in self.co_occurrence_matrix.items():
            if count >= min_frequency:
                # Calculate confidence
                item1_total = sum(
                    c for (i1, i2), c in self.co_occurrence_matrix.items()
                    if i1 == item1
                )
                confidence = count / item1_total if item1_total > 0 else 0

                if confidence >= min_confidence:
                    patterns.append(UsagePattern(
                        pattern_id=f"cooc_{item1}_{item2}",
                        pattern_type='co-occurrence',
                        items=[item1, item2],
                        frequency=count,
                        confidence=confidence,
                        last_seen=datetime.now()
                    ))

        return patterns

    def _find_temporal_patterns(
        self,
        min_frequency: int
    ) -> List[UsagePattern]:
        """
        Find temporal patterns (e.g., accessed every Monday).
        """
        patterns = []

        for item_id, timestamps in self.temporal_patterns.items():
            if len(timestamps) < min_frequency:
                continue

            # Analyze time of day
            hours = [t.hour for t in timestamps]
            hour_mode = max(set(hours), key=hours.count)
            hour_freq = hours.count(hour_mode)

            if hour_freq >= min_frequency:
                patterns.append(UsagePattern(
                    pattern_id=f"temporal_{item_id}_hour_{hour_mode}",
                    pattern_type='temporal',
                    items=[item_id],
                    frequency=hour_freq,
                    confidence=hour_freq / len(timestamps),
                    last_seen=timestamps[-1],
                    metadata={'hour_of_day': hour_mode}
                ))

            # Analyze day of week
            weekdays = [t.weekday() for t in timestamps]
            day_mode = max(set(weekdays), key=weekdays.count)
            day_freq = weekdays.count(day_mode)

            if day_freq >= min_frequency:
                patterns.append(UsagePattern(
                    pattern_id=f"temporal_{item_id}_day_{day_mode}",
                    pattern_type='temporal',
                    items=[item_id],
                    frequency=day_freq,
                    confidence=day_freq / len(timestamps),
                    last_seen=timestamps[-1],
                    metadata={'day_of_week': day_mode}
                ))

        return patterns

    def predict_next_access(
        self,
        recent_items: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict next likely context access based on patterns.
        """
        predictions = defaultdict(float)

        # Use sequential patterns
        for pattern in self.discovered_patterns:
            if pattern.pattern_type != 'sequential':
                continue

            # Check if recent items match pattern prefix
            pattern_items = pattern.items
            if len(recent_items) >= len(pattern_items) - 1:
                recent_subset = recent_items[-(len(pattern_items)-1):]
                if recent_subset == pattern_items[:-1]:
                    # Pattern matches! Predict last item
                    predicted_item = pattern_items[-1]
                    predictions[predicted_item] += pattern.confidence

        # Use co-occurrence patterns
        if recent_items:
            last_item = recent_items[-1]
            for pattern in self.discovered_patterns:
                if pattern.pattern_type != 'co-occurrence':
                    continue

                if pattern.items[0] == last_item:
                    predicted_item = pattern.items[1]
                    predictions[predicted_item] += pattern.confidence * 0.8

        # Sort by score and return top k
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_predictions[:top_k]
```

## Dynamic Context Strategy Selection

### Strategy Selection Engine

```python
class ContextStrategySelector:
    """
    Dynamically select optimal context strategies based on situation.
    """
    def __init__(self):
        self.available_strategies = {
            'aggressive_caching': AggressiveCachingStrategy(),
            'quality_focused': QualityFocusedStrategy(),
            'recency_based': RecencyBasedStrategy(),
            'pattern_based': PatternBasedStrategy(),
            'balanced': BalancedStrategy()
        }

        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.current_strategy = 'balanced'

    def select_strategy(
        self,
        context_state: Dict,
        performance_metrics: Dict[str, float],
        user_context: Optional[Dict] = None
    ) -> str:
        """
        Select optimal strategy for current situation.
        """
        # Analyze situation
        situation_features = self._extract_situation_features(
            context_state,
            performance_metrics,
            user_context
        )

        # Score each strategy for this situation
        strategy_scores = {}
        for strategy_name, strategy in self.available_strategies.items():
            score = strategy.score_suitability(situation_features)

            # Adjust based on historical performance
            if self.strategy_performance[strategy_name]:
                avg_performance = np.mean(
                    self.strategy_performance[strategy_name][-10:]
                )
                score *= (0.7 + 0.3 * avg_performance)

            strategy_scores[strategy_name] = score

        # Select best strategy
        best_strategy = max(
            strategy_scores.items(),
            key=lambda x: x[1]
        )[0]

        self.current_strategy = best_strategy
        return best_strategy

    def record_strategy_performance(
        self,
        strategy_name: str,
        performance_score: float
    ):
        """Record how well a strategy performed."""
        self.strategy_performance[strategy_name].append(performance_score)

    def _extract_situation_features(
        self,
        context_state: Dict,
        performance_metrics: Dict[str, float],
        user_context: Optional[Dict]
    ) -> Dict:
        """Extract features describing current situation."""
        return {
            'context_size': context_state.get('size', 0),
            'avg_quality': context_state.get('avg_quality', 0.5),
            'response_time': performance_metrics.get('response_time', 0.5),
            'accuracy': performance_metrics.get('accuracy', 0.8),
            'user_active_time': user_context.get('active_time', 0) if user_context else 0,
            'task_complexity': user_context.get('task_complexity', 'medium') if user_context else 'medium'
        }

class ContextStrategy:
    """Base class for context strategies."""

    def score_suitability(self, situation_features: Dict) -> float:
        """
        Score how suitable this strategy is for the situation.
        Returns score 0.0-1.0.
        """
        raise NotImplementedError

    def apply(
        self,
        context_hub: 'CentralizedContextHub',
        current_task: Optional[str] = None
    ) -> Dict:
        """Apply this strategy to the context hub."""
        raise NotImplementedError

class AggressiveCachingStrategy(ContextStrategy):
    """Strategy that aggressively caches frequently used context."""

    def score_suitability(self, situation_features: Dict) -> float:
        score = 0.5

        # Good for large context with slow response times
        if situation_features['context_size'] > 800:
            score += 0.2
        if situation_features['response_time'] > 1.5:
            score += 0.3

        return min(score, 1.0)

    def apply(
        self,
        context_hub: 'CentralizedContextHub',
        current_task: Optional[str] = None
    ) -> Dict:
        # Identify most accessed items
        sorted_by_access = sorted(
            context_hub.active_context.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )

        # Cache top 20%
        cache_size = max(10, len(sorted_by_access) // 5)
        cached_items = [key for key, item in sorted_by_access[:cache_size]]

        return {
            'strategy': 'aggressive_caching',
            'cached_items': len(cached_items)
        }

class QualityFocusedStrategy(ContextStrategy):
    """Strategy that prioritizes context quality over quantity."""

    def score_suitability(self, situation_features: Dict) -> float:
        score = 0.5

        # Good when quality is low or accuracy is suffering
        if situation_features['avg_quality'] < 0.5:
            score += 0.3
        if situation_features['accuracy'] < 0.75:
            score += 0.2

        return min(score, 1.0)

    def apply(
        self,
        context_hub: 'CentralizedContextHub',
        current_task: Optional[str] = None
    ) -> Dict:
        from modules.module_03.lessons.lesson_02_context_optimization import ContextQualityAssessor, ContextPruner

        assessor = ContextQualityAssessor()
        pruner = ContextPruner(assessor)

        # Aggressive quality-based pruning
        result = pruner.prune_context(
            context_hub,
            quality_threshold=0.6,
            current_task=current_task
        )

        return {
            'strategy': 'quality_focused',
            **result
        }

class RecencyBasedStrategy(ContextStrategy):
    """Strategy that prioritizes recent context."""

    def score_suitability(self, situation_features: Dict) -> float:
        score = 0.5

        # Good for active users with complex tasks
        if situation_features['user_active_time'] > 30:  # 30 min
            score += 0.2
        if situation_features['task_complexity'] == 'high':
            score += 0.2

        return min(score, 1.0)

    def apply(
        self,
        context_hub: 'CentralizedContextHub',
        current_task: Optional[str] = None
    ) -> Dict:
        # Keep only recent context (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        removed_count = 0

        items_to_remove = []
        for key, item in context_hub.active_context.items():
            if item.timestamp < cutoff:
                items_to_remove.append(key)

        for key in items_to_remove:
            context_hub.remove_context(key)
            removed_count += 1

        return {
            'strategy': 'recency_based',
            'removed_old_items': removed_count
        }

class PatternBasedStrategy(ContextStrategy):
    """Strategy that uses learned patterns to optimize."""

    def __init__(self):
        self.pattern_analyzer = ContextPatternAnalyzer()

    def score_suitability(self, situation_features: Dict) -> float:
        # Good when we have enough data to learn patterns
        if len(self.pattern_analyzer.access_sequences) > 50:
            return 0.8
        return 0.3

    def apply(
        self,
        context_hub: 'CentralizedContextHub',
        current_task: Optional[str] = None
    ) -> Dict:
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze_patterns()

        # Preload items likely to be accessed next
        recent_access = []
        if self.pattern_analyzer.access_sequences:
            recent_access = self.pattern_analyzer.access_sequences[-1][-5:]

        predictions = self.pattern_analyzer.predict_next_access(recent_access)

        return {
            'strategy': 'pattern_based',
            'patterns_found': len(patterns),
            'predictions': predictions
        }

class BalancedStrategy(ContextStrategy):
    """Balanced strategy suitable for general use."""

    def score_suitability(self, situation_features: Dict) -> float:
        # Always moderately suitable
        return 0.6

    def apply(
        self,
        context_hub: 'CentralizedContextHub',
        current_task: Optional[str] = None
    ) -> Dict:
        from modules.module_03.lessons.lesson_02_context_optimization import AdaptiveContextOptimizer

        optimizer = AdaptiveContextOptimizer()

        # Use adaptive optimizer with balanced settings
        result = optimizer.optimize(
            context_hub,
            {'response_time': 1.0, 'accuracy': 0.8},
            current_task
        )

        return {
            'strategy': 'balanced',
            **result
        }
```

## User Behavior Adaptation and Personalization

### Personalized Context Manager

```python
class PersonalizedContextManager:
    """
    Manage context with personalization based on user behavior.
    """
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.pattern_analyzer = ContextPatternAnalyzer()

    def get_or_create_profile(self, user_id: str) -> 'UserProfile':
        """Get or create user profile."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        return self.user_profiles[user_id]

    def personalize_context(
        self,
        user_id: str,
        context_hub: 'CentralizedContextHub',
        current_task: Optional[str] = None
    ) -> Dict:
        """
        Personalize context management for specific user.
        """
        profile = self.get_or_create_profile(user_id)

        # Update profile with current interaction
        profile.record_interaction(current_task)

        # Apply user-specific preferences
        result = {}

        # 1. Adjust importance weights based on user preferences
        if profile.preferences.get('context_verbosity') == 'concise':
            self._apply_concise_preference(context_hub)
            result['applied_concise'] = True

        # 2. Prioritize context types user engages with most
        preferred_types = profile.get_preferred_context_types()
        if preferred_types:
            self._boost_preferred_types(context_hub, preferred_types)
            result['boosted_types'] = preferred_types

        # 3. Apply personalized retention policies
        retention_policy = profile.get_retention_policy()
        self._apply_retention_policy(context_hub, retention_policy)
        result['retention_policy'] = retention_policy

        return result

    def _apply_concise_preference(
        self,
        context_hub: 'CentralizedContextHub'
    ):
        """Apply concise context preference."""
        for item in context_hub.active_context.values():
            # Increase importance of concise items
            content_length = len(str(item.content))
            if content_length < 200:
                item.importance = min(item.importance * 1.2, 1.0)

    def _boost_preferred_types(
        self,
        context_hub: 'CentralizedContextHub',
        preferred_types: List[str]
    ):
        """Boost importance of preferred context types."""
        for item in context_hub.active_context.values():
            if item.type in preferred_types:
                item.importance = min(item.importance * 1.3, 1.0)

    def _apply_retention_policy(
        self,
        context_hub: 'CentralizedContextHub',
        policy: Dict
    ):
        """Apply user-specific retention policy."""
        max_age_hours = policy.get('max_age_hours', 24)
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        items_to_remove = []
        for key, item in context_hub.active_context.items():
            if item.timestamp < cutoff and item.importance < 0.7:
                items_to_remove.append(key)

        for key in items_to_remove:
            context_hub.remove_context(key)

@dataclass
class UserProfile:
    """Profile capturing user behavior and preferences."""
    user_id: str
    preferences: Dict = field(default_factory=dict)
    interaction_history: List[Dict] = field(default_factory=list)
    context_type_engagement: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_interaction(self, task: Optional[str] = None):
        """Record user interaction."""
        self.interaction_history.append({
            'timestamp': datetime.now(),
            'task': task
        })

    def get_preferred_context_types(self, top_k: int = 3) -> List[str]:
        """Get user's most engaged context types."""
        if not self.context_type_engagement:
            return []

        sorted_types = sorted(
            self.context_type_engagement.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [t for t, count in sorted_types[:top_k]]

    def get_retention_policy(self) -> Dict:
        """Get personalized retention policy."""
        # Base policy
        policy = {
            'max_age_hours': 24,
            'min_importance': 0.3
        }

        # Adjust based on user activity level
        recent_interactions = sum(
            1 for i in self.interaction_history
            if datetime.now() - i['timestamp'] < timedelta(days=7)
        )

        if recent_interactions > 50:
            # Very active user - keep more context
            policy['max_age_hours'] = 48
            policy['min_importance'] = 0.2
        elif recent_interactions < 10:
            # Less active - more aggressive cleanup
            policy['max_age_hours'] = 12
            policy['min_importance'] = 0.5

        return policy
```

## Context Strategy Evolution and Improvement

### Evolutionary Strategy System

```python
class EvolutionaryStrategySystem:
    """
    Evolve and improve context strategies over time.
    """
    def __init__(self):
        self.strategy_selector = ContextStrategySelector()
        self.performance_history: List[Dict] = []
        self.strategy_mutations: List[Dict] = []

    def evolve_strategies(self, generations: int = 10):
        """
        Evolve strategies over multiple generations.
        """
        for gen in range(generations):
            # Evaluate current strategies
            evaluations = self._evaluate_all_strategies()

            # Select best performers
            best_strategies = self._select_best(evaluations, top_k=3)

            # Create variations of best strategies
            new_strategies = self._mutate_strategies(best_strategies)

            # Add to available strategies
            for name, strategy in new_strategies.items():
                self.strategy_selector.available_strategies[name] = strategy

            # Record generation
            self.strategy_mutations.append({
                'generation': gen,
                'best_strategies': [s['name'] for s in best_strategies],
                'new_strategies': list(new_strategies.keys()),
                'timestamp': datetime.now()
            })

    def _evaluate_all_strategies(self) -> List[Dict]:
        """Evaluate performance of all strategies."""
        evaluations = []

        for name, strategy in self.strategy_selector.available_strategies.items():
            perf_history = self.strategy_selector.strategy_performance.get(name, [])

            if perf_history:
                avg_performance = np.mean(perf_history[-20:])
                consistency = 1.0 - np.std(perf_history[-20:])

                evaluations.append({
                    'name': name,
                    'strategy': strategy,
                    'avg_performance': avg_performance,
                    'consistency': consistency,
                    'score': avg_performance * 0.7 + consistency * 0.3
                })

        return evaluations

    def _select_best(
        self,
        evaluations: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:
        """Select top performing strategies."""
        sorted_evals = sorted(
            evaluations,
            key=lambda x: x['score'],
            reverse=True
        )
        return sorted_evals[:top_k]

    def _mutate_strategies(
        self,
        best_strategies: List[Dict]
    ) -> Dict[str, ContextStrategy]:
        """
        Create mutations of best strategies.
        """
        new_strategies = {}

        for i, strategy_info in enumerate(best_strategies):
            base_name = strategy_info['name']
            base_strategy = strategy_info['strategy']

            # Create mutation
            mutation_name = f"{base_name}_evolved_{i}"
            new_strategies[mutation_name] = self._create_mutation(
                base_strategy
            )

        return new_strategies

    def _create_mutation(
        self,
        base_strategy: ContextStrategy
    ) -> ContextStrategy:
        """
        Create mutated version of strategy.

        In production, this would modify strategy parameters.
        For now, return a copy of the base strategy.
        """
        # This is simplified - in production you would:
        # 1. Clone the strategy
        # 2. Modify parameters (e.g., thresholds, weights)
        # 3. Test variations
        return base_strategy
```

## Key Takeaways

1. **Pattern Learning**: Analyze usage patterns to predict and optimize context access

2. **Dynamic Selection**: Choose strategies based on current situation and performance

3. **Personalization**: Adapt context management to individual user behaviors and preferences

4. **Continuous Evolution**: Strategies improve over time through performance feedback

5. **User Profiles**: Build and leverage user profiles for personalized context management

6. **Performance-Driven**: Let actual performance data drive strategy evolution

## What's Next

In Lesson 4, we'll explore context validation and monitoring systems to ensure reliability and quality in production environments.

---

**Practice Exercise**: Implement an adaptive context system that learns from usage patterns. Test with diverse user behaviors and measure improvement in response quality and speed over time. Target: >20% improvement after 100 interactions.
