# Lesson 4: Context Window Management

## Overview

Context window management is one of the most critical challenges in modern AI systems. This lesson teaches you practical techniques for efficiently managing limited context windows while preserving essential information and maintaining system performance.

## Understanding Context Window Limitations

### Current Model Limitations (2025)

| Model         | Context Window | Practical Limit | Cost per 1K tokens |
| ------------- | -------------- | --------------- | ------------------ |
| GPT-4 Turbo   | 128K tokens    | ~100K tokens    | $0.01 - $0.03      |
| Claude-3 Opus | 200K tokens    | ~180K tokens    | $0.015 - $0.075    |
| Gemini Pro    | 1M tokens      | ~800K tokens    | $0.001 - $0.002    |
| GPT-3.5 Turbo | 16K tokens     | ~12K tokens     | $0.0015 - $0.002   |

### Token Consumption Patterns

Different content types consume tokens at different rates:

```python
def estimate_token_usage(content_type, content_length):
    """Estimate token usage based on content type"""

    token_ratios = {
        'english_text': 0.75,      # ~4 chars per token
        'code_python': 0.5,        # ~2 chars per token
        'code_javascript': 0.6,    # ~2.4 chars per token
        'json_structured': 0.4,    # ~1.6 chars per token
        'xml_markup': 0.3,         # ~1.2 chars per token
        'conversation_history': 0.8 # ~3.2 chars per token
    }

    ratio = token_ratios.get(content_type, 0.75)
    estimated_tokens = int(content_length * ratio)

    return estimated_tokens

# Examples
print(estimate_token_usage('english_text', 1000))    # ~750 tokens
print(estimate_token_usage('code_python', 1000))     # ~500 tokens
print(estimate_token_usage('json_structured', 1000)) # ~400 tokens
```

## Context Compression and Summarization

### Hierarchical Summarization Strategy

```python
from datetime import datetime

class HierarchicalSummarizer:
    """Multi-level context summarization system"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.compression_levels = {
            'light': {'ratio': 0.7, 'detail': 'high'},
            'medium': {'ratio': 0.4, 'detail': 'medium'},
            'heavy': {'ratio': 0.2, 'detail': 'low'},
            'extreme': {'ratio': 0.1, 'detail': 'minimal'}
        }

    def compress_context(self, context, target_tokens, current_tokens):
        """Compress context to fit within target token limit"""
        compression_needed = 1 - (target_tokens / current_tokens)

        if compression_needed <= 0:
            return context  # No compression needed

        # Select appropriate compression level
        level = self.select_compression_level(compression_needed)

        # Apply compression strategy
        if compression_needed < 0.3:
            return self.light_compression(context, level)
        elif compression_needed < 0.6:
            return self.medium_compression(context, level)
        else:
            return self.heavy_compression(context, level)

    def select_compression_level(self, compression_needed):
        """Select appropriate compression level based on compression ratio needed"""
        if compression_needed < 0.3:
            return self.compression_levels['light']
        elif compression_needed < 0.6:
            return self.compression_levels['medium']
        elif compression_needed < 0.8:
            return self.compression_levels['heavy']
        else:
            return self.compression_levels['extreme']

    def light_compression(self, context, level):
        """Light compression: remove redundancy, keep all key info"""
        compressed = {
            'metadata': context['metadata'],
            'system_instructions': context['system_instructions'],
            'current_context': context['current_context'],
            'conversation_summary': self.summarize_conversation(
                context.get('conversation_history', {}),
                detail_level=level['detail']
            ),
            'key_knowledge': self.extract_key_knowledge(
                context.get('knowledge_base', {})
            ),
            'active_tools': context.get('tools_available', [])
        }
        return compressed

    def medium_compression(self, context, level):
        """Medium compression: significant summarization"""
        compressed = {
            'metadata': {
                'context_id': context['metadata']['context_id'],
                'compressed_at': datetime.now().isoformat(),
                'compression_level': 'medium'
            },
            'system_role': context['system_instructions']['role'],
            'current_task': context['current_context'],
            'conversation_essence': self.create_conversation_essence(
                context.get('conversation_history', {})
            ),
            'relevant_knowledge': self.get_most_relevant_knowledge(
                context.get('knowledge_base', {}), max_items=3
            )
        }
        return compressed

    def heavy_compression(self, context, level):
        """Heavy compression: only essential information"""
        compressed = {
            'context_id': context['metadata']['context_id'],
            'system_role': context['system_instructions']['role'],
            'current_query': context['current_context']['user_query'],
            'key_insights': self.extract_key_insights(context),
            'compression_level': 'heavy'
        }
        return compressed

    def summarize_conversation(self, conversation, detail_level='medium'):
        """Summarize conversation history with specified detail level"""
        if not conversation or not conversation.get('recent_turns'):
            return "No conversation history"

        turns = conversation['recent_turns']

        if detail_level == 'high':
            # Preserve more detail
            summary_prompt = f"""
            Summarize this conversation preserving key details and context:

            {self.format_turns(turns)}

            Include:
            - Main topics discussed
            - Key decisions made
            - User's specific requests
            - Assistant's main recommendations
            """
        elif detail_level == 'medium':
            # Balanced summarization
            summary_prompt = f"""
            Create a concise summary of this conversation:

            {self.format_turns(turns)}

            Focus on:
            - Primary objective
            - Key information exchanged
            - Current status
            """
        else:  # low detail
            # Minimal summary
            summary_prompt = f"""
            Provide a brief summary of this conversation in 1-2 sentences:

            {self.format_turns(turns)}
            """

        return self.llm.generate(summary_prompt, max_tokens=200)

    def format_turns(self, turns):
        """Format conversation turns for summarization"""
        formatted = []
        for turn in turns[-10:]:  # Last 10 turns
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def create_conversation_essence(self, conversation):
        """Extract the essence of the conversation"""
        turns = conversation.get('recent_turns', [])
        if not turns:
            return "No conversation history"

        prompt = f"Extract the core essence in 2-3 sentences: {self.format_turns(turns)}"
        return self.llm.generate(prompt, max_tokens=100)

    def extract_key_knowledge(self, knowledge_base):
        """Extract key knowledge items"""
        if not knowledge_base:
            return []
        return knowledge_base.get('key_items', [])[:5]

    def get_most_relevant_knowledge(self, knowledge_base, max_items=3):
        """Get most relevant knowledge items"""
        items = knowledge_base.get('items', [])
        # Sort by relevance score if available
        sorted_items = sorted(
            items,
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )
        return sorted_items[:max_items]

    def extract_key_insights(self, context):
        """Extract key insights from context"""
        insights = []
        if 'current_context' in context:
            insights.append(context['current_context'].get('main_goal', ''))
        if 'conversation_history' in context:
            last_turn = context['conversation_history'].get('recent_turns', [{}])[-1]
            insights.append(last_turn.get('content', '')[:100])
        return insights
```

### Intelligent Content Prioritization

```python
class ContentPrioritizer:
    """Prioritize context components based on relevance and importance"""

    def __init__(self):
        self.priority_weights = {
            'current_context': 1.0,      # Always highest priority
            'system_instructions': 0.9,  # Nearly always needed
            'recent_conversation': 0.8,  # Recent is more important
            'user_preferences': 0.7,     # Important for personalization
            'tool_definitions': 0.6,     # Needed for actions
            'knowledge_base': 0.5,       # Can be retrieved as needed
            'historical_context': 0.3    # Least critical
        }

    def prioritize_context_components(self, context, available_tokens):
        """Prioritize and allocate tokens to context components"""

        # Estimate token usage for each component
        component_tokens = {}
        for component, content in context.items():
            component_tokens[component] = self.estimate_tokens(content)

        # Calculate total tokens needed
        total_tokens = sum(component_tokens.values())

        if total_tokens <= available_tokens:
            return context  # All content fits

        # Prioritize components
        prioritized_components = sorted(
            component_tokens.items(),
            key=lambda x: self.priority_weights.get(x[0], 0.4),
            reverse=True
        )

        # Allocate tokens based on priority
        allocated_context = {}
        remaining_tokens = available_tokens

        for component, token_count in prioritized_components:
            if token_count <= remaining_tokens:
                # Include full component
                allocated_context[component] = context[component]
                remaining_tokens -= token_count
            elif remaining_tokens > 50:  # Minimum viable size
                # Include compressed version
                allocated_context[component] = self.compress_component(
                    context[component], remaining_tokens
                )
                remaining_tokens = 0
            else:
                # Skip component
                continue

        return allocated_context

    def estimate_tokens(self, content):
        """Estimate token count for content"""
        if isinstance(content, str):
            return len(content) // 4  # Rough estimate: 4 chars per token
        elif isinstance(content, dict):
            return sum(self.estimate_tokens(str(v)) for v in content.values())
        elif isinstance(content, list):
            return sum(self.estimate_tokens(str(item)) for item in content)
        return 0

    def compress_component(self, component, target_tokens):
        """Compress a single component to fit within token limit"""
        if isinstance(component, dict):
            return self.compress_dict_component(component, target_tokens)
        elif isinstance(component, list):
            return self.compress_list_component(component, target_tokens)
        else:
            return self.compress_text_component(component, target_tokens)

    def compress_dict_component(self, component, target_tokens):
        """Compress dictionary component"""
        # Keep most important keys based on priority
        compressed = {}
        remaining_tokens = target_tokens
        for key, value in component.items():
            value_tokens = self.estimate_tokens(value)
            if value_tokens <= remaining_tokens:
                compressed[key] = value
                remaining_tokens -= value_tokens
        return compressed

    def compress_list_component(self, component, target_tokens):
        """Compress list component"""
        # Take most recent items that fit
        compressed = []
        remaining_tokens = target_tokens
        for item in reversed(component):
            item_tokens = self.estimate_tokens(item)
            if item_tokens <= remaining_tokens:
                compressed.insert(0, item)
                remaining_tokens -= item_tokens
            else:
                break
        return compressed

    def compress_text_component(self, component, target_tokens):
        """Compress text component"""
        # Truncate to fit target tokens (4 chars per token estimate)
        max_chars = target_tokens * 4
        text = str(component)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
```

### Context Sliding Window Strategy

```python
from datetime import datetime

class SlidingWindowManager:
    """Manage context using sliding window approach"""

    def __init__(self, window_size=10000, overlap_size=2000):
        self.window_size = window_size      # Total window size in tokens
        self.overlap_size = overlap_size    # Overlap between windows
        self.context_windows = [[]]
        self.current_window_index = 0

    def add_context(self, new_context, context_type='conversation_turn'):
        """Add new context using sliding window management"""

        # Estimate token size of new context
        new_tokens = self.estimate_tokens(new_context)

        # Get current window
        current_window = self.get_current_window()
        current_tokens = self.estimate_tokens(current_window)

        # Check if new context fits in current window
        if current_tokens + new_tokens <= self.window_size:
            # Add to current window
            current_window.append({
                'content': new_context,
                'type': context_type,
                'timestamp': datetime.now(),
                'tokens': new_tokens
            })
        else:
            # Create new window with overlap
            self.create_new_window(new_context, context_type)

    def create_new_window(self, new_context, context_type):
        """Create new window with overlap from previous window"""

        # Extract overlap content from current window
        current_window = self.get_current_window()
        overlap_content = self.extract_overlap(current_window)

        # Create new window
        new_window = overlap_content + [{
            'content': new_context,
            'type': context_type,
            'timestamp': datetime.now(),
            'tokens': self.estimate_tokens(new_context)
        }]

        self.context_windows.append(new_window)
        self.current_window_index = len(self.context_windows) - 1

    def extract_overlap(self, window):
        """Extract most important content for overlap"""
        if not window:
            return []

        # Sort by importance (recency, type priority, etc.)
        sorted_content = sorted(
            window,
            key=lambda x: self.calculate_importance_score(x),
            reverse=True
        )

        # Select content that fits in overlap size
        overlap_content = []
        overlap_tokens = 0

        for content_item in sorted_content:
            if overlap_tokens + content_item['tokens'] <= self.overlap_size:
                overlap_content.append(content_item)
                overlap_tokens += content_item['tokens']
            else:
                break

        return overlap_content

    def get_relevant_context(self, query, max_tokens=8000):
        """Retrieve most relevant context for a query"""

        # Score all context items for relevance to query
        all_context_items = []
        for window in self.context_windows:
            for item in window:
                relevance_score = self.calculate_relevance(query, item)
                all_context_items.append({
                    'item': item,
                    'relevance': relevance_score
                })

        # Sort by relevance
        sorted_items = sorted(
            all_context_items,
            key=lambda x: x['relevance'],
            reverse=True
        )

        # Select items that fit within token limit
        selected_context = []
        total_tokens = 0

        for context_item in sorted_items:
            item_tokens = context_item['item']['tokens']
            if total_tokens + item_tokens <= max_tokens:
                selected_context.append(context_item['item'])
                total_tokens += item_tokens
            else:
                break

        return selected_context

    def estimate_tokens(self, content):
        """Estimate token count"""
        if isinstance(content, dict):
            return sum(len(str(v)) for v in content.values()) // 4
        elif isinstance(content, list):
            return sum(len(str(item)) for item in content) // 4
        return len(str(content)) // 4

    def get_current_window(self):
        """Get current window, creating if necessary"""
        if not self.context_windows:
            self.context_windows = [[]]
        return self.context_windows[self.current_window_index]

    def calculate_importance_score(self, content_item):
        """Calculate importance score for content"""
        score = 0.0
        # Recency score (more recent = higher score)
        timestamp = content_item.get('timestamp')
        if timestamp:
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            score += max(0, 1.0 - (age_minutes / 1440))  # Decay over 24 hours

        # Type priority score
        type_priorities = {
            'user_query': 1.0,
            'assistant_response': 0.8,
            'tool_result': 0.7,
            'system_message': 0.6,
            'conversation_turn': 0.5
        }
        content_type = content_item.get('type', 'conversation_turn')
        score += type_priorities.get(content_type, 0.5)

        return score

    def calculate_relevance(self, query, content_item):
        """Calculate relevance score between query and content"""
        # Simple keyword-based relevance (in practice, use embeddings)
        query_lower = str(query).lower()
        content_lower = str(content_item.get('content', '')).lower()

        # Count matching words
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        if not query_words:
            return 0.0

        matching_words = query_words.intersection(content_words)
        relevance = len(matching_words) / len(query_words)

        # Boost by importance score
        importance = self.calculate_importance_score(content_item)

        return (relevance * 0.7) + (importance * 0.3)
```

## Dynamic Context Management

### Adaptive Context Allocation

```python
class AdaptiveContextManager:
    """Dynamically adjust context allocation based on task requirements"""

    def __init__(self):
        self.task_templates = {
            'code_review': {
                'system_instructions': 0.15,
                'current_code': 0.40,
                'conversation_history': 0.20,
                'relevant_docs': 0.15,
                'tools': 0.10
            },
            'creative_writing': {
                'system_instructions': 0.10,
                'current_context': 0.30,
                'conversation_history': 0.35,
                'reference_materials': 0.20,
                'tools': 0.05
            },
            'data_analysis': {
                'system_instructions': 0.12,
                'dataset_info': 0.35,
                'analysis_context': 0.25,
                'conversation_history': 0.18,
                'tools': 0.10
            },
            'customer_support': {
                'system_instructions': 0.20,
                'customer_info': 0.25,
                'conversation_history': 0.30,
                'knowledge_base': 0.20,
                'tools': 0.05
            }
        }

    def allocate_context_dynamically(self, context_components, task_type, total_tokens):
        """Dynamically allocate context based on task requirements"""

        # Get allocation template for task type
        allocation_template = self.task_templates.get(
            task_type,
            self.task_templates['customer_support']  # default
        )

        allocated_context = {}

        for component, content in context_components.items():
            # Calculate target tokens for this component
            allocation_ratio = allocation_template.get(component, 0.1)
            target_tokens = int(total_tokens * allocation_ratio)

            # Compress or truncate content to fit allocation
            if self.estimate_tokens(content) > target_tokens:
                allocated_context[component] = self.fit_to_token_limit(
                    content, target_tokens
                )
            else:
                allocated_context[component] = content

        return allocated_context

    def fit_to_token_limit(self, content, token_limit):
        """Fit content within specified token limit"""

        current_tokens = self.estimate_tokens(content)

        if current_tokens <= token_limit:
            return content

        # Calculate compression ratio needed
        compression_ratio = token_limit / current_tokens

        if compression_ratio > 0.7:
            # Light compression: remove redundancy
            return self.remove_redundancy(content)
        elif compression_ratio > 0.4:
            # Medium compression: summarize
            return self.summarize_content(content, token_limit)
        else:
            # Heavy compression: extract key points
            return self.extract_key_points(content, token_limit)

    def estimate_tokens(self, content):
        """Estimate token count"""
        if isinstance(content, str):
            return len(content) // 4
        elif isinstance(content, dict):
            return sum(len(str(v)) for v in content.values()) // 4
        elif isinstance(content, list):
            return sum(len(str(item)) for item in content) // 4
        return 0

    def remove_redundancy(self, content):
        """Remove redundant information from content"""
        if isinstance(content, str):
            lines = content.split('\n')
            unique_lines = []
            seen = set()
            for line in lines:
                normalized = line.strip().lower()
                if normalized and normalized not in seen:
                    unique_lines.append(line)
                    seen.add(normalized)
            return '\n'.join(unique_lines)
        return content

    def summarize_content(self, content, token_limit):
        """Summarize content to fit token limit"""
        # Truncate to approximate token limit
        max_chars = token_limit * 4
        text = str(content)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "... [truncated]"

    def extract_key_points(self, content, token_limit):
        """Extract key points from content"""
        text = str(content)
        # Split into sentences and take first few
        sentences = text.split('. ')
        key_points = []
        total_chars = 0
        max_chars = token_limit * 4

        for sentence in sentences:
            if total_chars + len(sentence) <= max_chars:
                key_points.append(sentence)
                total_chars += len(sentence)
            else:
                break

        return '. '.join(key_points)
```

### Context Quality Monitoring

```python
class ContextQualityMonitor:
    """Monitor and maintain context quality during window management"""

    def __init__(self):
        self.quality_metrics = {
            'completeness': 0.0,     # How much essential info is preserved
            'coherence': 0.0,        # How well context flows together
            'relevance': 0.0,        # How relevant context is to current task
            'freshness': 0.0,        # How recent/current the context is
            'efficiency': 0.0        # Token usage efficiency
        }

    def assess_context_quality(self, context, task_requirements):
        """Assess overall context quality"""

        scores = {}

        # Completeness: Check if essential components are present
        scores['completeness'] = self.assess_completeness(context, task_requirements)

        # Coherence: Check if context elements work together
        scores['coherence'] = self.assess_coherence(context)

        # Relevance: Check relevance to current task
        scores['relevance'] = self.assess_relevance(context, task_requirements)

        # Freshness: Check how recent the context is
        scores['freshness'] = self.assess_freshness(context)

        # Efficiency: Check token usage efficiency
        scores['efficiency'] = self.assess_efficiency(context)

        # Calculate overall quality score
        overall_score = sum(scores.values()) / len(scores)

        return {
            'overall_quality': overall_score,
            'component_scores': scores,
            'recommendations': self.generate_recommendations(scores)
        }

    def generate_recommendations(self, scores):
        """Generate recommendations for improving context quality"""
        recommendations = []

        if scores['completeness'] < 0.7:
            recommendations.append({
                'issue': 'Missing essential context components',
                'action': 'Increase token allocation for critical components',
                'priority': 'high'
            })

        if scores['relevance'] < 0.6:
            recommendations.append({
                'issue': 'Low relevance to current task',
                'action': 'Improve context filtering and selection algorithms',
                'priority': 'high'
            })

        if scores['efficiency'] < 0.5:
            recommendations.append({
                'issue': 'Inefficient token usage',
                'action': 'Apply better compression or remove redundancy',
                'priority': 'medium'
            })

        return recommendations

    def assess_completeness(self, context, task_requirements):
        """Assess if all required context components are present"""
        required_components = task_requirements.get('required_components', [])
        if not required_components:
            return 1.0

        present_components = [c for c in required_components if c in context]
        return len(present_components) / len(required_components)

    def assess_coherence(self, context):
        """Assess how well context elements work together"""
        # Simple coherence check based on structure
        if not isinstance(context, dict):
            return 0.5

        # Check for conflicting information or gaps
        has_system = 'system_instructions' in context or 'system_role' in context
        has_current = 'current_context' in context or 'current_task' in context

        coherence_score = 0.0
        if has_system:
            coherence_score += 0.5
        if has_current:
            coherence_score += 0.5

        return coherence_score

    def assess_relevance(self, context, task_requirements):
        """Assess relevance to current task"""
        task_type = task_requirements.get('task_type', '')
        if not task_type:
            return 0.7  # Default moderate relevance

        # Check if context contains task-relevant information
        relevant_keys = task_requirements.get('relevant_context_keys', [])
        if not relevant_keys:
            return 0.7

        present_keys = [k for k in relevant_keys if k in context]
        return len(present_keys) / len(relevant_keys) if relevant_keys else 0.7

    def assess_freshness(self, context):
        """Assess how recent the context is"""
        if not isinstance(context, dict):
            return 0.5

        # Look for timestamp information
        timestamps = []
        for key, value in context.items():
            if isinstance(value, dict) and 'timestamp' in value:
                timestamps.append(value['timestamp'])

        if not timestamps:
            return 0.5  # Unknown freshness

        # Calculate average age
        from datetime import datetime
        now = datetime.now()
        total_freshness = 0.0

        for ts in timestamps:
            if isinstance(ts, datetime):
                age_hours = (now - ts).total_seconds() / 3600
                freshness = max(0, 1.0 - (age_hours / 24))  # Decay over 24 hours
                total_freshness += freshness

        return total_freshness / len(timestamps) if timestamps else 0.5

    def assess_efficiency(self, context):
        """Assess token usage efficiency"""
        if not context:
            return 0.0

        # Estimate information density
        total_tokens = self.estimate_tokens(context)
        if total_tokens == 0:
            return 0.0

        # Count unique information pieces
        unique_items = set()
        if isinstance(context, dict):
            for value in context.values():
                unique_items.add(str(value)[:100])  # First 100 chars as proxy

        # Efficiency = unique items per 100 tokens
        efficiency = (len(unique_items) / total_tokens) * 100
        return min(1.0, efficiency)  # Cap at 1.0

    def estimate_tokens(self, content):
        """Estimate token count"""
        if isinstance(content, str):
            return len(content) // 4
        elif isinstance(content, dict):
            return sum(len(str(v)) for v in content.values()) // 4
        elif isinstance(content, list):
            return sum(len(str(item)) for item in content) // 4
        return 0
```

## Advanced Window Management Patterns

### 1. Attention-Based Context Selection

```python
class AttentionBasedContextManager:
    """Use attention mechanisms to select most relevant context"""

    def __init__(self, attention_model):
        self.attention_model = attention_model

    def select_context_with_attention(self, query, context_pool, max_tokens):
        """Select context using attention-based relevance scoring"""

        # Calculate attention scores for all context items
        attention_scores = self.attention_model.calculate_attention(
            query, context_pool
        )

        # Sort by attention scores
        sorted_context = sorted(
            zip(context_pool, attention_scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Select context within token limit
        selected_context = []
        total_tokens = 0

        for context_item, attention_score in sorted_context:
            item_tokens = self.estimate_tokens(context_item)
            if total_tokens + item_tokens <= max_tokens:
                selected_context.append({
                    'content': context_item,
                    'attention_score': attention_score
                })
                total_tokens += item_tokens

        return selected_context

    def estimate_tokens(self, content):
        """Estimate token count"""
        if isinstance(content, dict):
            return len(str(content)) // 4
        return len(str(content)) // 4
```

### 2. Hierarchical Context Windows

```python
class HierarchicalWindowManager:
    """Manage context in hierarchical windows"""

    def __init__(self):
        self.immediate_window = {}    # Current interaction (1K tokens)
        self.working_window = {}      # Current task (5K tokens)
        self.session_window = {}      # Current session (20K tokens)
        self.persistent_window = {}   # Cross-session (stored externally)

    def add_context_hierarchically(self, content, content_type):
        """Add content to appropriate hierarchical level"""

        # Immediate window: always current
        self.immediate_window[content_type] = content

        # Working window: task-relevant content
        if self.is_task_relevant(content, content_type):
            self.add_to_working_window(content, content_type)

        # Session window: session-relevant content
        if self.is_session_relevant(content, content_type):
            self.add_to_session_window(content, content_type)

        # Persistent window: long-term relevant content
        if self.is_persistent_relevant(content, content_type):
            self.add_to_persistent_window(content, content_type)

    def is_task_relevant(self, content, content_type):
        """Check if content is relevant to current task"""
        task_relevant_types = ['current_context', 'user_query', 'tool_result']
        return content_type in task_relevant_types

    def is_session_relevant(self, content, content_type):
        """Check if content is relevant to current session"""
        session_relevant_types = ['conversation_turn', 'task_result', 'user_preference']
        return content_type in session_relevant_types

    def is_persistent_relevant(self, content, content_type):
        """Check if content should be stored persistently"""
        persistent_types = ['user_profile', 'learned_preference', 'important_decision']
        return content_type in persistent_types

    def add_to_working_window(self, content, content_type):
        """Add content to working window"""
        if content_type not in self.working_window:
            self.working_window[content_type] = []
        self.working_window[content_type].append(content)

    def add_to_session_window(self, content, content_type):
        """Add content to session window"""
        if content_type not in self.session_window:
            self.session_window[content_type] = []
        self.session_window[content_type].append(content)

    def add_to_persistent_window(self, content, content_type):
        """Add content to persistent window"""
        if content_type not in self.persistent_window:
            self.persistent_window[content_type] = []
        self.persistent_window[content_type].append(content)
```

## Key Takeaways

1. **Proactive Management**: Don't wait for context overflow - manage windows proactively
2. **Quality Over Quantity**: Better to have relevant, well-structured context than maximum content
3. **Task-Aware Allocation**: Different tasks need different context allocation strategies
4. **Continuous Monitoring**: Regularly assess and improve context quality
5. **Hierarchical Thinking**: Use different window sizes for different types of information

Effective context window management is essential for building AI systems that maintain high performance across extended interactions while working within the practical constraints of current AI models.
