# Lesson 4: Self-Improving Single-Agent Systems

## Introduction

The ultimate goal of context engineering is creating agents that continuously improve their performance through experience. This lesson teaches you to build self-improving single-agent systems that learn from context patterns, adapt their strategies, and optimize their decision-making processes over time.

## Self-Improvement Architecture

### Core Learning Engine

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
import pickle
import hashlib
from abc import ABC, abstractmethod

class LearningStrategy(Enum):
    """Different learning approaches for agent improvement."""
    REINFORCEMENT = "reinforcement"
    PATTERN_RECOGNITION = "pattern_recognition"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CONTEXT_EFFICIENCY = "context_efficiency"
    USER_FEEDBACK = "user_feedback"

class PerformanceMetric(Enum):
    """Performance metrics for learning evaluation."""
    SUCCESS_RATE = "success_rate"
    EXECUTION_TIME = "execution_time"
    CONTEXT_EFFICIENCY = "context_efficiency"
    USER_SATISFACTION = "user_satisfaction"
    RESOURCE_UTILIZATION = "resource_utilization"

@dataclass
class LearningExperience:
    """Represents a learning experience from agent execution."""
    id: str
    timestamp: datetime
    context_snapshot: Dict[str, Any]
    action_taken: str
    outcome: Dict[str, Any]
    performance_scores: Dict[PerformanceMetric, float]
    context_pattern: str
    decision_quality: float
    learning_value: float = 0.0

    def __post_init__(self):
        """Calculate learning value after initialization."""
        self.learning_value = self._calculate_learning_value()

    def _calculate_learning_value(self) -> float:
        """Calculate how valuable this experience is for learning."""
        # Base value from performance
        base_value = sum(self.performance_scores.values()) / len(self.performance_scores)

        # Bonus for novel situations
        novelty_bonus = 0.2 if self.context_pattern not in ["common", "routine"] else 0.0

        # Bonus for high-impact decisions
        impact_bonus = 0.3 if self.decision_quality > 0.8 else 0.0

        return min(1.0, base_value + novelty_bonus + impact_bonus)

@dataclass
class AdaptationRule:
    """Rule for adapting agent behavior based on learning."""
    id: str
    condition: str  # Condition when rule applies
    adaptation: Dict[str, Any]  # What to change
    confidence: float  # Confidence in this rule
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None

class ContextPattern:
    """Represents a recognized pattern in context."""
    def __init__(self, pattern_id: str, features: Dict[str, Any]):
        self.pattern_id = pattern_id
        self.features = features
        self.occurrence_count = 1
        self.success_rate = 0.0
        self.average_performance = 0.0
        self.last_seen = datetime.now()
        self.adaptations: List[str] = []

    def update_statistics(self, success: bool, performance: float):
        """Update pattern statistics with new observation."""
        self.occurrence_count += 1
        self.last_seen = datetime.now()

        # Update success rate (exponential moving average)
        alpha = 0.1
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)

        # Update average performance
        self.average_performance = (
            (1 - alpha) * self.average_performance + alpha * performance
        )

class SelfImprovingAgent:
    """
    Single-agent system that continuously learns and improves performance.
    """

    def __init__(
        self,
        learning_strategies: List[LearningStrategy] = None,
        memory_size: int = 10000,
        adaptation_threshold: float = 0.7,
        learning_rate: float = 0.1
    ):
        self.learning_strategies = learning_strategies or [
            LearningStrategy.PATTERN_RECOGNITION,
            LearningStrategy.PERFORMANCE_OPTIMIZATION,
            LearningStrategy.CONTEXT_EFFICIENCY
        ]
        self.memory_size = memory_size
        self.adaptation_threshold = adaptation_threshold
        self.learning_rate = learning_rate

        # Learning components
        self.experience_memory: deque = deque(maxlen=memory_size)
        self.context_patterns: Dict[str, ContextPattern] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.performance_baselines: Dict[str, float] = {}

        # Learning statistics
        self.learning_stats = {
            'total_experiences': 0,
            'patterns_discovered': 0,
            'adaptations_applied': 0,
            'performance_improvements': 0,
            'learning_cycles': 0
        }

        # Active adaptations
        self.active_adaptations: Dict[str, Any] = {}
        self.adaptation_history: List[Dict] = []

    def record_experience(
        self,
        context: Dict[str, Any],
        action: str,
        outcome: Dict[str, Any],
        performance_scores: Dict[PerformanceMetric, float]
    ) -> LearningExperience:
        """Record a new learning experience."""
        experience_id = self._generate_experience_id(context, action)

        # Extract context pattern
        context_pattern = self._extract_context_pattern(context)

        # Calculate decision quality
        decision_quality = self._assess_decision_quality(action, outcome, performance_scores)

        # Create experience
        experience = LearningExperience(
            id=experience_id,
            timestamp=datetime.now(),
            context_snapshot=context.copy(),
            action_taken=action,
            outcome=outcome,
            performance_scores=performance_scores,
            context_pattern=context_pattern,
            decision_quality=decision_quality
        )

        # Store experience
        self.experience_memory.append(experience)
        self.learning_stats['total_experiences'] += 1

        # Update context pattern statistics
        self._update_pattern_statistics(experience)

        return experience

    def learn_and_adapt(self) -> Dict[str, Any]:
        """Perform learning cycle and generate adaptations."""
        learning_results = {
            'new_patterns': [],
            'new_adaptations': [],
            'performance_insights': [],
            'adaptation_recommendations': []
        }

        # Apply each learning strategy
        for strategy in self.learning_strategies:
            strategy_results = self._apply_learning_strategy(strategy)

            # Merge results
            for key, value in strategy_results.items():
                if key in learning_results:
                    learning_results[key].extend(value)

        # Generate adaptations from learned patterns
        new_adaptations = self._generate_adaptations(learning_results['new_patterns'])
        learning_results['new_adaptations'].extend(new_adaptations)

        # Apply high-confidence adaptations
        applied_adaptations = self._apply_adaptations(learning_results['new_adaptations'])
        learning_results['applied_adaptations'] = applied_adaptations

        # Update learning statistics
        self.learning_stats['learning_cycles'] += 1
        self.learning_stats['patterns_discovered'] += len(learning_results['new_patterns'])
        self.learning_stats['adaptations_applied'] += len(applied_adaptations)

        return learning_results

    def _apply_learning_strategy(self, strategy: LearningStrategy) -> Dict[str, List]:
        """Apply specific learning strategy."""
        if strategy == LearningStrategy.PATTERN_RECOGNITION:
            return self._learn_context_patterns()
        elif strategy == LearningStrategy.PERFORMANCE_OPTIMIZATION:
            return self._optimize_performance()
        elif strategy == LearningStrategy.CONTEXT_EFFICIENCY:
            return self._improve_context_efficiency()
        elif strategy == LearningStrategy.USER_FEEDBACK:
            return self._learn_from_feedback()
        else:
            return {'new_patterns': [], 'performance_insights': []}

    def _learn_context_patterns(self) -> Dict[str, List]:
        """Learn new context patterns from experience."""
        if len(self.experience_memory) < 50:  # Need sufficient data
            return {'new_patterns': [], 'performance_insights': []}

        # Group experiences by context similarity
        context_groups = self._group_similar_contexts()
        new_patterns = []

        for group_id, experiences in context_groups.items():
            if len(experiences) >= 5:  # Minimum occurrences for pattern
                pattern = self._extract_pattern_from_group(group_id, experiences)
                if pattern and pattern.pattern_id not in self.context_patterns:
                    self.context_patterns[pattern.pattern_id] = pattern
                    new_patterns.append({
                        'pattern_id': pattern.pattern_id,
                        'features': pattern.features,
                        'occurrence_count': pattern.occurrence_count,
                        'success_rate': pattern.success_rate
                    })

        return {
            'new_patterns': new_patterns,
            'performance_insights': [
                f"Discovered {len(new_patterns)} new context patterns",
                f"Total patterns: {len(self.context_patterns)}"
            ]
        }

    def _optimize_performance(self) -> Dict[str, List]:
        """Learn performance optimization strategies."""
        performance_insights = []
        new_patterns = []

        # Analyze performance trends
        recent_experiences = list(self.experience_memory)[-100:]  # Last 100 experiences

        if not recent_experiences:
            return {'new_patterns': [], 'performance_insights': []}

        # Group by action type
        action_performance = defaultdict(list)
        for exp in recent_experiences:
            avg_performance = sum(exp.performance_scores.values()) / len(exp.performance_scores)
            action_performance[exp.action_taken].append(avg_performance)

        # Find performance patterns
        for action, performances in action_performance.items():
            if len(performances) >= 5:
                avg_perf = np.mean(performances)
                std_perf = np.std(performances)

                if action not in self.performance_baselines:
                    self.performance_baselines[action] = avg_perf
                else:
                    # Check for improvement
                    baseline = self.performance_baselines[action]
                    if avg_perf > baseline + 0.05:  # 5% improvement threshold
                        self.learning_stats['performance_improvements'] += 1
                        performance_insights.append(
                            f"Performance improved for {action}: {baseline:.3f} -> {avg_perf:.3f}"
                        )
                        self.performance_baselines[action] = avg_perf

                # Create performance pattern
                if std_perf < 0.1:  # Consistent performance
                    pattern_data = {
                        'pattern_id': f"performance_{action}_{int(avg_perf * 100)}",
                        'action': action,
                        'expected_performance': avg_perf,
                        'consistency': 1.0 - std_perf
                    }
                    new_patterns.append(pattern_data)

        return {
            'new_patterns': new_patterns,
            'performance_insights': performance_insights
        }

    def _improve_context_efficiency(self) -> Dict[str, List]:
        """Learn to use context more efficiently."""
        efficiency_insights = []

        if len(self.experience_memory) < 20:
            return {'new_patterns': [], 'performance_insights': efficiency_insights}

        # Analyze context usage patterns
        context_usage = []
        for exp in self.experience_memory:
            context_size = len(json.dumps(exp.context_snapshot))
            avg_performance = sum(exp.performance_scores.values()) / len(exp.performance_scores)
            context_usage.append((context_size, avg_performance))

        # Find optimal context size ranges
        if len(context_usage) >= 10:
            context_sizes, performances = zip(*context_usage)

            # Calculate correlation between context size and performance
            correlation = np.corrcoef(context_sizes, performances)[0, 1]

            if abs(correlation) > 0.3:  # Significant correlation
                if correlation < 0:
                    efficiency_insights.append(
                        "Large context sizes may be reducing performance - consider compression"
                    )
                else:
                    efficiency_insights.append(
                        "Larger context sizes are improving performance - consider expansion"
                    )

            # Find optimal context size range
            sorted_usage = sorted(context_usage, key=lambda x: x[1], reverse=True)
            top_performers = sorted_usage[:len(sorted_usage)//3]  # Top third

            if top_performers:
                optimal_sizes = [size for size, _ in top_performers]
                optimal_range = (min(optimal_sizes), max(optimal_sizes))
                efficiency_insights.append(
                    f"Optimal context size range: {optimal_range[0]}-{optimal_range[1]} characters"
                )

        return {
            'new_patterns': [],
            'performance_insights': efficiency_insights
        }

    def _learn_from_feedback(self) -> Dict[str, List]:
        """Learn from user feedback when available."""
        feedback_insights = []

        # Look for user feedback in experiences
        feedback_experiences = [
            exp for exp in self.experience_memory
            if PerformanceMetric.USER_SATISFACTION in exp.performance_scores
        ]

        if len(feedback_experiences) >= 10:
            # Analyze feedback patterns
            high_satisfaction = [
                exp for exp in feedback_experiences
                if exp.performance_scores[PerformanceMetric.USER_SATISFACTION] > 0.8
            ]

            low_satisfaction = [
                exp for exp in feedback_experiences
                if exp.performance_scores[PerformanceMetric.USER_SATISFACTION] < 0.5
            ]

            if high_satisfaction:
                # Find common patterns in high satisfaction cases
                common_actions = defaultdict(int)
                for exp in high_satisfaction:
                    common_actions[exp.action_taken] += 1

                most_satisfying = max(common_actions.items(), key=lambda x: x[1])
                feedback_insights.append(
                    f"Action '{most_satisfying[0]}' consistently receives high user satisfaction"
                )

            if low_satisfaction:
                # Find problematic patterns
                problematic_actions = defaultdict(int)
                for exp in low_satisfaction:
                    problematic_actions[exp.action_taken] += 1

                most_problematic = max(problematic_actions.items(), key=lambda x: x[1])
                feedback_insights.append(
                    f"Action '{most_problematic[0]}' often receives low user satisfaction"
                )

        return {
            'new_patterns': [],
            'performance_insights': feedback_insights
        }

    def _generate_adaptations(self, new_patterns: List[Dict]) -> List[AdaptationRule]:
        """Generate adaptation rules from discovered patterns."""
        adaptations = []

        for pattern in new_patterns:
            pattern_id = pattern['pattern_id']

            # Generate adaptation based on pattern type
            if 'performance_' in pattern_id:
                # Performance-based adaptation
                action = pattern['action']
                expected_perf = pattern['expected_performance']

                if expected_perf > 0.8:  # High performance action
                    adaptation = AdaptationRule(
                        id=f"prefer_{action}_{int(expected_perf * 100)}",
                        condition=f"context_similar_to_high_performance_{action}",
                        adaptation={
                            'action_preference': action,
                            'weight_increase': 0.2,
                            'reason': f'Action {action} consistently performs well'
                        },
                        confidence=expected_perf
                    )
                    adaptations.append(adaptation)

            elif pattern.get('features'):
                # Feature-based adaptation
                features = pattern['features']
                success_rate = pattern.get('success_rate', 0.5)

                if success_rate > 0.8:
                    adaptation = AdaptationRule(
                        id=f"context_adaptation_{pattern_id}",
                        condition=f"context_matches_{pattern_id}",
                        adaptation={
                            'context_optimization': features,
                            'expected_success_rate': success_rate,
                            'reason': f'Context pattern {pattern_id} shows high success rate'
                        },
                        confidence=success_rate
                    )
                    adaptations.append(adaptation)

        return adaptations

    def _apply_adaptations(self, adaptations: List[AdaptationRule]) -> List[str]:
        """Apply high-confidence adaptations to agent behavior."""
        applied = []

        for adaptation in adaptations:
            if adaptation.confidence >= self.adaptation_threshold:
                # Store adaptation rule
                self.adaptation_rules[adaptation.id] = adaptation

                # Apply to active adaptations
                self.active_adaptations[adaptation.id] = adaptation.adaptation

                # Record in history
                self.adaptation_history.append({
                    'adaptation_id': adaptation.id,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': adaptation.confidence,
                    'adaptation': adaptation.adaptation
                })

                applied.append(adaptation.id)

        return applied

    def get_recommended_action(
        self,
        context: Dict[str, Any],
        available_actions: List[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Get recommended action based on learned patterns and adaptations."""
        action_scores = {}
        reasoning = {
            'pattern_matches': [],
            'adaptations_applied': [],
            'confidence_factors': []
        }

        # Base scores for all actions
        base_score = 1.0 / len(available_actions)
        for action in available_actions:
            action_scores[action] = base_score

        # Apply pattern-based scoring
        context_pattern = self._extract_context_pattern(context)

        # Check for matching patterns
        for pattern_id, pattern in self.context_patterns.items():
            if self._context_matches_pattern(context, pattern):
                reasoning['pattern_matches'].append({
                    'pattern_id': pattern_id,
                    'success_rate': pattern.success_rate,
                    'occurrence_count': pattern.occurrence_count
                })

                # Boost scores for actions that worked well with this pattern
                pattern_boost = pattern.success_rate * 0.3
                for action in available_actions:
                    if action in pattern.adaptations:
                        action_scores[action] += pattern_boost

        # Apply active adaptations
        for adaptation_id, adaptation in self.active_adaptations.items():
            if 'action_preference' in adaptation:
                preferred_action = adaptation['action_preference']
                if preferred_action in action_scores:
                    weight_increase = adaptation.get('weight_increase', 0.1)
                    action_scores[preferred_action] += weight_increase

                    reasoning['adaptations_applied'].append({
                        'adaptation_id': adaptation_id,
                        'preferred_action': preferred_action,
                        'weight_increase': weight_increase,
                        'reason': adaptation.get('reason', 'No reason provided')
                    })

        # Apply performance baseline adjustments
        for action in available_actions:
            if action in self.performance_baselines:
                baseline_perf = self.performance_baselines[action]
                performance_boost = (baseline_perf - 0.5) * 0.2  # Center around 0.5
                action_scores[action] += performance_boost

                reasoning['confidence_factors'].append({
                    'action': action,
                    'baseline_performance': baseline_perf,
                    'performance_boost': performance_boost
                })

        # Select action with highest score
        best_action = max(action_scores.items(), key=lambda x: x[1])
        recommended_action = best_action[0]
        confidence = min(1.0, best_action[1])

        reasoning['recommended_action'] = recommended_action
        reasoning['confidence'] = confidence
        reasoning['action_scores'] = action_scores

        return recommended_action, reasoning

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of learning progress."""
        return {
            'learning_statistics': self.learning_stats,
            'memory_utilization': {
                'experiences_stored': len(self.experience_memory),
                'memory_capacity': self.memory_size,
                'utilization_percentage': len(self.experience_memory) / self.memory_size * 100
            },
            'pattern_analysis': {
                'total_patterns': len(self.context_patterns),
                'pattern_details': {
                    pattern_id: {
                        'occurrence_count': pattern.occurrence_count,
                        'success_rate': pattern.success_rate,
                        'average_performance': pattern.average_performance,
                        'last_seen': pattern.last_seen.isoformat()
                    }
                    for pattern_id, pattern in self.context_patterns.items()
                }
            },
            'adaptation_status': {
                'total_rules': len(self.adaptation_rules),
                'active_adaptations': len(self.active_adaptations),
                'recent_adaptations': self.adaptation_history[-5:]  # Last 5 adaptations
            },
            'performance_trends': {
                'baseline_performances': self.performance_baselines,
                'improvement_count': self.learning_stats['performance_improvements']
            }
        }

    # Helper methods
    def _generate_experience_id(self, context: Dict[str, Any], action: str) -> str:
        """Generate unique ID for experience."""
        content = f"{json.dumps(context, sort_keys=True)}_{action}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _extract_context_pattern(self, context: Dict[str, Any]) -> str:
        """Extract pattern identifier from context."""
        # Simplified pattern extraction
        pattern_features = []

        if 'task_type' in context:
            pattern_features.append(f"task_{context['task_type']}")

        if 'data_size' in context:
            size = context['data_size']
            if size < 1000:
                pattern_features.append("small_data")
            elif size < 10000:
                pattern_features.append("medium_data")
            else:
                pattern_features.append("large_data")

        if 'user_preferences' in context:
            pattern_features.append("user_customized")

        return "_".join(pattern_features) if pattern_features else "generic"

    def _assess_decision_quality(
        self,
        action: str,
        outcome: Dict[str, Any],
        performance_scores: Dict[PerformanceMetric, float]
    ) -> float:
        """Assess the quality of a decision."""
        # Base quality from performance scores
        avg_performance = sum(performance_scores.values()) / len(performance_scores)

        # Adjust based on outcome
        success_bonus = 0.2 if outcome.get('success', True) else -0.2

        # Adjust based on efficiency
        efficiency_bonus = 0.1 if outcome.get('efficient', True) else -0.1

        quality = avg_performance + success_bonus + efficiency_bonus
        return max(0.0, min(1.0, quality))

    def _update_pattern_statistics(self, experience: LearningExperience):
        """Update statistics for context patterns."""
        pattern_id = experience.context_pattern

        if pattern_id not in self.context_patterns:
            # Create new pattern
            pattern = ContextPattern(pattern_id, self._extract_pattern_features(experience))
            self.context_patterns[pattern_id] = pattern
        else:
            pattern = self.context_patterns[pattern_id]

        # Update pattern statistics
        success = experience.decision_quality > 0.6
        avg_performance = sum(experience.performance_scores.values()) / len(experience.performance_scores)
        pattern.update_statistics(success, avg_performance)

        # Record successful adaptations
        if success and experience.action_taken not in pattern.adaptations:
            pattern.adaptations.append(experience.action_taken)

    def _extract_pattern_features(self, experience: LearningExperience) -> Dict[str, Any]:
        """Extract features that define a context pattern."""
        features = {}
        context = experience.context_snapshot

        # Extract key context features
        for key in ['task_type', 'data_size', 'complexity', 'priority']:
            if key in context:
                features[key] = context[key]

        # Add derived features
        features['context_size'] = len(json.dumps(context))
        features['decision_quality_range'] = self._categorize_quality(experience.decision_quality)

        return features

    def _categorize_quality(self, quality: float) -> str:
        """Categorize decision quality into ranges."""
        if quality >= 0.8:
            return "high"
        elif quality >= 0.6:
            return "medium"
        else:
            return "low"

    def _group_similar_contexts(self) -> Dict[str, List[LearningExperience]]:
        """Group experiences by context similarity."""
        groups = defaultdict(list)

        for experience in self.experience_memory:
            group_key = experience.context_pattern
            groups[group_key].append(experience)

        return dict(groups)

    def _extract_pattern_from_group(self, group_id: str, experiences: List[LearningExperience]) -> Optional[ContextPattern]:
        """Extract common pattern from group of experiences."""
        if len(experiences) < 3:
            return None

        # Calculate average performance and success rate
        performances = []
        successes = []

        for exp in experiences:
            avg_perf = sum(exp.performance_scores.values()) / len(exp.performance_scores)
            performances.append(avg_perf)
            successes.append(exp.decision_quality > 0.6)

        avg_performance = np.mean(performances)
        success_rate = np.mean(successes)

        # Extract common features
        common_features = self._find_common_features(experiences)

        # Create pattern
        pattern = ContextPattern(group_id, common_features)
        pattern.occurrence_count = len(experiences)
        pattern.success_rate = success_rate
        pattern.average_performance = avg_performance

        return pattern

    def _find_common_features(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Find features common across experiences."""
        if not experiences:
            return {}

        # Start with first experience features
        common_features = {}
        first_exp = experiences[0]

        # Check each feature from first experience
        for key, value in first_exp.context_snapshot.items():
            # Check if this feature is common across all experiences
            is_common = all(
                exp.context_snapshot.get(key) == value
                for exp in experiences[1:]
            )

            if is_common:
                common_features[key] = value

        return common_features

    def _context_matches_pattern(self, context: Dict[str, Any], pattern: ContextPattern) -> bool:
        """Check if context matches a learned pattern."""
        # Check if all pattern features are present in context
        for feature_key, feature_value in pattern.features.items():
            if context.get(feature_key) != feature_value:
                return False

        return True

# Example usage
async def example_self_improving_agent():
    """Example of self-improving agent in action."""

    # Create self-improving agent
    agent = SelfImprovingAgent(
        learning_strategies=[
            LearningStrategy.PATTERN_RECOGNITION,
            LearningStrategy.PERFORMANCE_OPTIMIZATION,
            LearningStrategy.CONTEXT_EFFICIENCY
        ],
        memory_size=1000,
        adaptation_threshold=0.7,
        learning_rate=0.1
    )

    # Simulate agent experiences
    contexts = [
        {'task_type': 'data_analysis', 'data_size': 5000, 'complexity': 'medium'},
        {'task_type': 'data_analysis', 'data_size': 15000, 'complexity': 'high'},
        {'task_type': 'report_generation', 'data_size': 2000, 'complexity': 'low'},
        {'task_type': 'data_analysis', 'data_size': 4800, 'complexity': 'medium'},
        {'task_type': 'optimization', 'data_size': 8000, 'complexity': 'high'}
    ]

    actions = ['analyze_thoroughly', 'quick_scan', 'detailed_report', 'optimize_performance']

    # Record experiences
    for i, context in enumerate(contexts * 20):  # Repeat for learning
        # Agent selects action
        recommended_action, reasoning = agent.get_recommended_action(context, actions)

        # Simulate action execution and outcome
        performance_scores = {
            PerformanceMetric.SUCCESS_RATE: np.random.beta(2, 1),  # Biased towards success
            PerformanceMetric.EXECUTION_TIME: np.random.beta(1, 2),  # Biased towards fast
            PerformanceMetric.CONTEXT_EFFICIENCY: np.random.beta(1.5, 1.5),  # Balanced
        }

        outcome = {
            'success': performance_scores[PerformanceMetric.SUCCESS_RATE] > 0.6,
            'efficient': performance_scores[PerformanceMetric.EXECUTION_TIME] > 0.5
        }

        # Record experience
        agent.record_experience(context, recommended_action, outcome, performance_scores)

        # Periodic learning
        if (i + 1) % 10 == 0:
            learning_results = agent.learn_and_adapt()
            print(f"Learning cycle {(i + 1) // 10}:")
            print(f"  New patterns: {len(learning_results['new_patterns'])}")
            print(f"  New adaptations: {len(learning_results['new_adaptations'])}")
            print(f"  Applied adaptations: {len(learning_results.get('applied_adaptations', []))}")

    # Final learning summary
    summary = agent.get_learning_summary()
    print("\nFinal Learning Summary:")
    print(json.dumps(summary, indent=2, default=str))

# Run the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(example_self_improving_agent())
```

## Key Takeaways

1. **Continuous Learning**: Agents that learn from every interaction and continuously improve

2. **Pattern Recognition**: Identify and leverage successful context patterns for better decisions

3. **Adaptive Behavior**: Automatically adjust strategies based on accumulated experience

4. **Performance Optimization**: Learn what works best and optimize for better outcomes

5. **Self-Reflection**: Analyze own performance and identify improvement opportunities

6. **Knowledge Accumulation**: Build and maintain knowledge that improves over time

## Module 7 Conclusion

You've now mastered advanced single-agent systems with sophisticated context management. These systems excel at:

- **Context Compression**: Efficiently managing large-scale context without losing critical information
- **Sequential Orchestration**: Maintaining context continuity across complex task sequences
- **Self-Improvement**: Learning and adapting from experience to continuously enhance performance

Next, we'll explore performance optimization and scaling strategies in Module 8.

---

**Practice Exercise**: Build a complete self-improving agent that learns from 1000+ interactions, discovers meaningful patterns, and demonstrates measurable performance improvements over time. Show adaptation capabilities and context efficiency optimization.
