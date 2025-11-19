# Lesson 2: Context Compression for Long-Running Tasks

## Introduction

Long-running AI agents accumulate vast amounts of context that can exceed memory limits and degrade performance. Effective context compression preserves essential information while reducing storage requirements and maintaining decision quality. This lesson teaches you to build intelligent compression systems that maintain context coherency across extended interactions.

## The Context Compression Challenge

### Context Growth Problem

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from collections import defaultdict

class ContextImportance(Enum):
    """Context importance levels for compression decisions."""
    CRITICAL = "critical"        # Must never be compressed away
    HIGH = "high"               # Compress carefully, preserve key details
    MEDIUM = "medium"           # Moderate compression acceptable
    LOW = "low"                 # Aggressive compression acceptable
    DISPOSABLE = "disposable"   # Can be removed entirely

@dataclass
class ContextMoment:
    """Represents a significant moment in context history."""
    id: str
    timestamp: datetime
    context_snapshot: Dict[str, Any]
    decisions_made: List[Dict]
    importance: ContextImportance
    compression_ratio: float = 1.0  # 1.0 = no compression, 0.5 = 50% compression
    summary: Optional[str] = None
    key_insights: List[str] = field(default_factory=list)

class ContextCompressionEngine:
    """
    Intelligent context compression system for long-running agents.
    """
    def __init__(self, max_context_size: int = 10000000):  # 10MB default
        self.max_context_size = max_context_size
        self.compression_rules: List[Dict] = []
        self.context_moments: List[ContextMoment] = []

        # Compression statistics
        self.compression_stats = {
            'total_compressed': 0,
            'space_saved': 0,
            'compression_operations': 0
        }

        # Initialize compression rules
        self._setup_compression_rules()

    def _setup_compression_rules(self):
        """Setup default compression rules."""
        self.compression_rules = [
            {
                'name': 'time_based_decay',
                'condition': lambda context: (datetime.now() - context['timestamp']).days > 7,
                'compression_factor': 0.3,
                'preserve_keys': ['decisions', 'key_insights', 'user_preferences']
            },
            {
                'name': 'repetitive_content',
                'condition': lambda context: self._is_repetitive_content(context),
                'compression_factor': 0.1,
                'preserve_keys': ['unique_insights', 'decisions']
            },
            {
                'name': 'low_access_frequency',
                'condition': lambda context: context.get('access_count', 0) < 2,
                'compression_factor': 0.5,
                'preserve_keys': ['critical_decisions', 'user_feedback']
            },
            {
                'name': 'high_importance_preserve',
                'condition': lambda context: context.get('importance') == ContextImportance.CRITICAL,
                'compression_factor': 1.0,  # No compression
                'preserve_keys': []  # Preserve everything
            }
        ]

    def compress_context(
        self,
        context_data: Dict[str, Any],
        target_compression: float = 0.5
    ) -> Dict[str, Any]:
        """Compress context data while preserving essential information."""
        original_size = len(json.dumps(context_data))

        # Identify key moments and decisions
        key_moments = self._extract_key_moments(context_data)

        # Create compressed representation
        compressed_context = {
            'metadata': {
                'original_size': original_size,
                'compression_timestamp': datetime.now().isoformat(),
                'compression_version': '1.0'
            },
            'key_moments': key_moments,
            'decision_summary': self._summarize_decisions(context_data),
            'critical_context': self._extract_critical_context(context_data),
            'compressed_interactions': self._compress_interactions(context_data)
        }

        # Calculate actual compression ratio
        compressed_size = len(json.dumps(compressed_context))
        actual_compression = compressed_size / original_size

        # Update statistics
        self.compression_stats['total_compressed'] += 1
        self.compression_stats['space_saved'] += original_size - compressed_size
        self.compression_stats['compression_operations'] += 1

        # Add compression metadata
        compressed_context['metadata']['compression_ratio'] = actual_compression
        compressed_context['metadata']['target_ratio'] = target_compression

        return compressed_context

    def _extract_key_moments(self, context_data: Dict[str, Any]) -> List[Dict]:
        """Extract key decision moments and important interactions."""
        key_moments = []

        # Look for decision points
        decisions = context_data.get('decision_history', [])
        for decision in decisions:
            if self._is_key_decision(decision):
                moment = {
                    'type': 'decision',
                    'timestamp': decision.get('timestamp'),
                    'summary': self._summarize_decision(decision),
                    'impact_score': self._calculate_decision_impact(decision),
                    'context_snapshot': self._extract_relevant_context(decision)
                }
                key_moments.append(moment)

        # Look for significant user interactions
        interactions = context_data.get('conversation_history', [])
        for interaction in interactions:
            if self._is_significant_interaction(interaction):
                moment = {
                    'type': 'interaction',
                    'timestamp': interaction.get('timestamp'),
                    'summary': self._summarize_interaction(interaction),
                    'user_intent': self._extract_user_intent(interaction),
                    'agent_response_type': interaction.get('response_type')
                }
                key_moments.append(moment)

        # Sort by importance and timestamp
        key_moments.sort(key=lambda m: (m.get('impact_score', 0), m.get('timestamp')), reverse=True)

        return key_moments[:20]  # Keep top 20 key moments

    def _is_key_decision(self, decision: Dict) -> bool:
        """Determine if a decision is significant enough to preserve."""
        # High-impact decisions
        if decision.get('impact_score', 0) > 0.8:
            return True

        # Decisions that changed agent behavior
        if decision.get('behavior_change', False):
            return True

        # User-initiated decisions
        if decision.get('source') == 'user_request':
            return True

        # Decisions with long-term consequences
        if 'long_term' in decision.get('tags', []):
            return True

        return False

    def _is_significant_interaction(self, interaction: Dict) -> bool:
        """Determine if an interaction is significant enough to preserve."""
        # User feedback or corrections
        if 'feedback' in interaction.get('type', ''):
            return True

        # Complex queries requiring substantial reasoning
        if len(interaction.get('user_input', '')) > 200:
            return True

        # Interactions that led to major responses
        if len(interaction.get('agent_response', '')) > 500:
            return True

        # High user satisfaction scores
        if interaction.get('satisfaction_score', 0) > 0.8:
            return True

        return False

    def _summarize_decisions(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of all decisions made."""
        decisions = context_data.get('decision_history', [])

        # Group decisions by type
        decision_types = defaultdict(list)
        for decision in decisions:
            decision_type = decision.get('type', 'unknown')
            decision_types[decision_type].append(decision)

        # Create summary for each type
        summary = {
            'total_decisions': len(decisions),
            'decision_types': {},
            'major_decisions': [],
            'decision_patterns': self._identify_decision_patterns(decisions)
        }

        for decision_type, type_decisions in decision_types.items():
            summary['decision_types'][decision_type] = {
                'count': len(type_decisions),
                'success_rate': self._calculate_success_rate(type_decisions),
                'common_patterns': self._extract_common_patterns(type_decisions)
            }

        # Extract major decisions
        major_decisions = [d for d in decisions if self._is_major_decision(d)]
        summary['major_decisions'] = [
            {
                'type': d.get('type'),
                'timestamp': d.get('timestamp'),
                'summary': self._summarize_decision(d),
                'outcome': d.get('outcome')
            }
            for d in major_decisions
        ]

        return summary

    def _extract_critical_context(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context that must be preserved."""
        critical_context = {}

        # User preferences and settings
        if 'user_preferences' in context_data:
            critical_context['user_preferences'] = context_data['user_preferences']

        # Long-term goals and objectives
        if 'goals' in context_data:
            critical_context['goals'] = context_data['goals']

        # Important constraints and limitations
        if 'constraints' in context_data:
            critical_context['constraints'] = context_data['constraints']

        # Critical failures and lessons learned
        critical_events = []
        for event in context_data.get('execution_trace', []):
            if not event.get('success', True) and event.get('severity') == 'critical':
                critical_events.append({
                    'type': 'critical_failure',
                    'timestamp': event.get('timestamp'),
                    'description': event.get('error'),
                    'lesson_learned': event.get('lesson_learned')
                })
        critical_context['critical_events'] = critical_events

        # High-value user feedback
        valuable_feedback = []
        for feedback in context_data.get('user_feedback', []):
            if feedback.get('value_score', 0) > 0.8:
                valuable_feedback.append({
                    'feedback': feedback.get('content'),
                    'timestamp': feedback.get('timestamp'),
                    'applied': feedback.get('applied', False)
                })
        critical_context['valuable_feedback'] = valuable_feedback

        return critical_context

    def _compress_interactions(self, context_data: Dict[str, Any]) -> List[Dict]:
        """Compress conversation history intelligently."""
        interactions = context_data.get('conversation_history', [])

        if len(interactions) <= 50:  # Keep all if small
            return interactions

        compressed_interactions = []

        # Always keep recent interactions (last 20)
        compressed_interactions.extend(interactions[-20:])

        # Keep significant interactions from earlier
        older_interactions = interactions[:-20]
        significant_older = [
            interaction for interaction in older_interactions
            if self._is_significant_interaction(interaction)
        ]

        # Compress non-significant interactions by grouping
        non_significant = [
            interaction for interaction in older_interactions
            if not self._is_significant_interaction(interaction)
        ]

        # Group non-significant interactions by time windows
        grouped_interactions = self._group_interactions_by_time(non_significant)

        for group in grouped_interactions:
            if len(group) > 1:
                # Create summary for group
                summary_interaction = {
                    'type': 'compressed_group',
                    'start_time': group[0].get('timestamp'),
                    'end_time': group[-1].get('timestamp'),
                    'interaction_count': len(group),
                    'topics_discussed': self._extract_topics(group),
                    'summary': self._create_group_summary(group)
                }
                compressed_interactions.append(summary_interaction)
            else:
                compressed_interactions.append(group[0])

        # Add significant older interactions
        compressed_interactions.extend(significant_older)

        # Sort by timestamp
        compressed_interactions.sort(key=lambda x: x.get('timestamp', datetime.min))

        return compressed_interactions

    def _group_interactions_by_time(
        self,
        interactions: List[Dict],
        window_hours: int = 2
    ) -> List[List[Dict]]:
        """Group interactions by time windows."""
        if not interactions:
            return []

        groups = []
        current_group = [interactions[0]]

        for interaction in interactions[1:]:
            current_time = interaction.get('timestamp', datetime.now())
            last_time = current_group[-1].get('timestamp', datetime.now())

            if isinstance(current_time, str):
                current_time = datetime.fromisoformat(current_time)
            if isinstance(last_time, str):
                last_time = datetime.fromisoformat(last_time)

            if (current_time - last_time).total_seconds() < window_hours * 3600:
                current_group.append(interaction)
            else:
                groups.append(current_group)
                current_group = [interaction]

        groups.append(current_group)
        return groups

    def _create_group_summary(self, group: List[Dict]) -> str:
        """Create summary for a group of interactions."""
        topics = self._extract_topics(group)
        user_intents = [self._extract_user_intent(interaction) for interaction in group]

        summary = f"Group of {len(group)} interactions covering topics: {', '.join(topics[:3])}"
        if len(topics) > 3:
            summary += f" and {len(topics) - 3} others"

        return summary

    def _extract_topics(self, interactions: List[Dict]) -> List[str]:
        """Extract main topics from interactions."""
        # Simplified topic extraction
        topics = set()

        for interaction in interactions:
            user_input = interaction.get('user_input', '').lower()
            # Simple keyword-based topic extraction
            if 'data' in user_input or 'database' in user_input:
                topics.add('data_management')
            if 'analysis' in user_input or 'analyze' in user_input:
                topics.add('analysis')
            if 'plan' in user_input or 'planning' in user_input:
                topics.add('planning')
            if 'help' in user_input or 'how' in user_input:
                topics.add('assistance')

        return list(topics)

    def decompress_context(self, compressed_context: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct usable context from compressed representation."""
        decompressed = {
            'metadata': {
                'decompressed_at': datetime.now().isoformat(),
                'original_compression_ratio': compressed_context.get('metadata', {}).get('compression_ratio', 1.0)
            }
        }

        # Reconstruct critical context
        critical_context = compressed_context.get('critical_context', {})
        decompressed.update(critical_context)

        # Reconstruct key moments as pseudo-history
        key_moments = compressed_context.get('key_moments', [])
        reconstructed_decisions = []
        reconstructed_interactions = []

        for moment in key_moments:
            if moment.get('type') == 'decision':
                reconstructed_decisions.append({
                    'timestamp': moment.get('timestamp'),
                    'summary': moment.get('summary'),
                    'impact_score': moment.get('impact_score'),
                    'reconstructed': True
                })
            elif moment.get('type') == 'interaction':
                reconstructed_interactions.append({
                    'timestamp': moment.get('timestamp'),
                    'summary': moment.get('summary'),
                    'user_intent': moment.get('user_intent'),
                    'reconstructed': True
                })

        decompressed['decision_history'] = reconstructed_decisions
        decompressed['conversation_history'] = reconstructed_interactions

        # Add compressed interactions
        compressed_interactions = compressed_context.get('compressed_interactions', [])
        decompressed['conversation_history'].extend(compressed_interactions)

        # Add decision summary
        decompressed['decision_summary'] = compressed_context.get('decision_summary', {})

        return decompressed

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        return {
            'total_compressions': self.compression_stats['total_compressed'],
            'total_space_saved_bytes': self.compression_stats['space_saved'],
            'average_compression_ratio': self._calculate_average_compression_ratio(),
            'compression_efficiency': self._calculate_compression_efficiency()
        }

    def _calculate_average_compression_ratio(self) -> float:
        """Calculate average compression ratio achieved."""
        if not self.context_moments:
            return 1.0

        total_ratio = sum(moment.compression_ratio for moment in self.context_moments)
        return total_ratio / len(self.context_moments)

    def _calculate_compression_efficiency(self) -> float:
        """Calculate compression efficiency metric."""
        if self.compression_stats['compression_operations'] == 0:
            return 0.0

        return self.compression_stats['space_saved'] / self.compression_stats['compression_operations']

    # Helper methods for decision analysis
    def _is_repetitive_content(self, context: Dict) -> bool:
        """Check if context contains repetitive content."""
        content_hash = hashlib.md5(str(context).encode()).hexdigest()
        # In production, maintain hash database to detect repetition
        return False  # Simplified

    def _summarize_decision(self, decision: Dict) -> str:
        """Create concise summary of a decision."""
        decision_type = decision.get('type', 'unknown')
        outcome = decision.get('outcome', 'unknown')
        return f"{decision_type} decision with {outcome} outcome"

    def _calculate_decision_impact(self, decision: Dict) -> float:
        """Calculate impact score for a decision."""
        # Simplified impact calculation
        base_score = 0.5

        if decision.get('user_initiated', False):
            base_score += 0.2

        if decision.get('changed_behavior', False):
            base_score += 0.3

        return min(1.0, base_score)

    def _extract_relevant_context(self, decision: Dict) -> Dict:
        """Extract context relevant to a decision."""
        return {
            'context_keys': decision.get('context_used', []),
            'reasoning': decision.get('reasoning', ''),
            'alternatives_considered': decision.get('alternatives', [])
        }

    def _summarize_interaction(self, interaction: Dict) -> str:
        """Create summary of an interaction."""
        user_input = interaction.get('user_input', '')[:100]
        return f"User query about: {user_input}..."

    def _extract_user_intent(self, interaction: Dict) -> str:
        """Extract user intent from interaction."""
        user_input = interaction.get('user_input', '').lower()

        if any(word in user_input for word in ['help', 'how', 'what']):
            return 'information_seeking'
        elif any(word in user_input for word in ['create', 'make', 'generate']):
            return 'creation_request'
        elif any(word in user_input for word in ['analyze', 'review', 'check']):
            return 'analysis_request'
        else:
            return 'general_interaction'

    def _identify_decision_patterns(self, decisions: List[Dict]) -> List[str]:
        """Identify patterns in decision making."""
        patterns = []

        # Decision frequency patterns
        decision_types = [d.get('type') for d in decisions]
        type_counts = defaultdict(int)
        for dt in decision_types:
            type_counts[dt] += 1

        most_common = max(type_counts.items(), key=lambda x: x[1]) if type_counts else None
        if most_common and most_common[1] > len(decisions) * 0.3:
            patterns.append(f"frequent_{most_common[0]}_decisions")

        return patterns

    def _calculate_success_rate(self, decisions: List[Dict]) -> float:
        """Calculate success rate for decisions."""
        if not decisions:
            return 0.0

        successful = sum(1 for d in decisions if d.get('success', True))
        return successful / len(decisions)

    def _extract_common_patterns(self, decisions: List[Dict]) -> List[str]:
        """Extract common patterns in decisions of same type."""
        patterns = []

        # Look for common reasoning patterns
        reasoning_words = []
        for decision in decisions:
            reasoning = decision.get('reasoning', '').lower().split()
            reasoning_words.extend(reasoning)

        # Simple frequency analysis
        word_counts = defaultdict(int)
        for word in reasoning_words:
            if len(word) > 3:  # Skip short words
                word_counts[word] += 1

        common_words = [word for word, count in word_counts.items() if count > 2]
        if common_words:
            patterns.append(f"common_reasoning_terms: {', '.join(common_words[:3])}")

        return patterns

    def _is_major_decision(self, decision: Dict) -> bool:
        """Determine if decision is major enough to preserve in summary."""
        return (
            decision.get('impact_score', 0) > 0.7 or
            decision.get('user_initiated', False) or
            decision.get('changed_behavior', False)
        )
```

## Key Takeaways

1. **Intelligent Compression**: Preserve essential information while reducing storage

2. **Key Moment Extraction**: Identify and preserve critical decisions and interactions

3. **Hierarchical Importance**: Different compression strategies based on content importance

4. **Reconstruction Capability**: Decompress context while maintaining usability

5. **Performance Monitoring**: Track compression effectiveness and optimize

6. **Pattern Recognition**: Learn from decision patterns to improve compression

## What's Next

In Lesson 3, we'll explore sequential task orchestration for maintaining context continuity across complex workflows.

---

**Practice Exercise**: Build a complete context compression system handling 10MB+ context data. Demonstrate 70%+ compression ratios while preserving decision quality. Show reconstruction capability maintaining >90% context utility for subsequent operations.
