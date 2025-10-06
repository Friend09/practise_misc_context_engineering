# Lesson 1: Context-Aware Prompt Architecture

## Introduction

While Module 1 established the foundations of context engineering, this lesson bridges context engineering with prompt engineering to create **context-aware prompts**—prompts that dynamically adapt to and leverage the rich contextual information you've engineered.

### The Evolution: From Static to Context-Aware

**Traditional Static Prompts:**

```
"Write a summary of this document."
```

**Context-Aware Dynamic Prompts:**

```
Given the user's preference for technical depth (level 8/10),
their background in machine learning, and their previous 3 questions
about neural architectures, provide a summary that:
- Emphasizes technical details relevant to ML practitioners
- References concepts from their previous questions
- Uses terminology consistent with their expertise level
```

The difference? Context-aware prompts **leverage engineered context** to produce dramatically better results.

## The Context-Prompt Integration Framework

### Understanding the Relationship

```
┌─────────────────────────────────────────┐
│     Context Engineering Layer           │
│  (Information Architecture)             │
│  • Memory systems                       │
│  • Knowledge bases                      │
│  • State management                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     Context-Aware Prompt Layer          │
│  (Dynamic Instruction Generation)       │
│  • Context injection                    │
│  • Dynamic adaptation                   │
│  • Relevance filtering                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│          AI Model                       │
│  (Execution with Full Context)          │
└─────────────────────────────────────────┘
```

### Key Principle: Separation of Concerns

- **Context Engineering**: Manages WHAT information is available
- **Prompt Engineering**: Determines HOW to use that information
- **Context-Aware Prompts**: Bridge between the two, adapting prompts based on available context

## Core Patterns for Context-Aware Prompts

### Pattern 1: Context Injection

**Concept**: Systematically inject relevant context into prompts based on current state and requirements.

**Implementation Strategy:**

```python
def build_context_aware_prompt(
    task: str,
    user_context: dict,
    conversation_history: list,
    system_state: dict
) -> str:
    """
    Build a prompt that intelligently injects relevant context.
    """
    prompt_parts = []

    # 1. System role and capabilities
    prompt_parts.append(f"You are {system_state['role']}.")

    # 2. User profile context
    if user_context.get('expertise_level'):
        prompt_parts.append(
            f"The user has {user_context['expertise_level']} expertise."
        )

    # 3. Historical context (filtered for relevance)
    if conversation_history:
        relevant_history = filter_relevant_history(
            conversation_history,
            task
        )
        if relevant_history:
            prompt_parts.append(
                f"Previous relevant context:\n{relevant_history}"
            )

    # 4. Current task
    prompt_parts.append(f"\nCurrent task: {task}")

    # 5. Context-specific instructions
    if user_context.get('preferences'):
        prompt_parts.append(
            format_preferences(user_context['preferences'])
        )

    return "\n\n".join(prompt_parts)
```

**Key Considerations:**

1. **Relevance Filtering**: Don't inject all available context—only what's relevant
2. **Context Ordering**: Place most important context closest to the task
3. **Token Budget**: Monitor and manage token usage to avoid context overflow
4. **Graceful Degradation**: Have fallback strategies when context is incomplete

### Pattern 2: Dynamic Adaptation

**Concept**: Adapt prompt structure, complexity, and content based on context state.

**Adaptation Dimensions:**

1. **Complexity Scaling**

   ```python
   def adapt_prompt_complexity(task: str, context: dict) -> str:
       complexity_level = determine_complexity(task, context)

       if complexity_level == "simple":
           return f"Task: {task}\nProvide a direct answer."
       elif complexity_level == "moderate":
           return f"""
           Task: {task}

           Please:
           1. Analyze the requirements
           2. Provide your solution
           3. Explain your reasoning
           """
       else:  # complex
           return f"""
           Task: {task}

           Context: {context['relevant_info']}

           Please follow this structured approach:
           1. Break down the problem
           2. Consider alternatives
           3. Evaluate trade-offs
           4. Provide recommendation with rationale
           5. Suggest validation steps
           """
   ```

2. **Tone and Style Adaptation**

   ```python
   def adapt_communication_style(
       base_prompt: str,
       user_preferences: dict
   ) -> str:
       style_modifiers = []

       if user_preferences.get('formality') == 'formal':
           style_modifiers.append(
               "Use professional, formal language."
           )

       if user_preferences.get('verbosity') == 'concise':
           style_modifiers.append(
               "Be concise and direct."
           )

       if user_preferences.get('technical_depth') == 'high':
           style_modifiers.append(
               "Include technical details and specifics."
           )

       return f"{base_prompt}\n\nStyle: {' '.join(style_modifiers)}"
   ```

3. **Format Adaptation**

   ```python
   def adapt_output_format(task: str, context: dict) -> str:
       output_format = determine_optimal_format(context)

       format_instructions = {
           'structured': "Provide response in structured sections.",
           'code': "Respond with executable code and comments.",
           'markdown': "Format response in markdown with headers.",
           'json': "Return response as valid JSON object.",
       }

       return f"{task}\n\n{format_instructions[output_format]}"
   ```

### Pattern 3: Contextual Constraints and Guardrails

**Concept**: Use context to dynamically apply constraints and safety measures.

**Implementation:**

```python
def apply_contextual_constraints(
    prompt: str,
    context: dict
) -> str:
    constraints = []

    # User-specific constraints
    if context.get('user_restrictions'):
        constraints.extend(context['user_restrictions'])

    # Task-specific constraints
    if context.get('task_type') == 'sensitive':
        constraints.append(
            "Ensure privacy and data protection compliance."
        )

    # System-level constraints
    if context.get('token_budget_remaining') < 1000:
        constraints.append(
            "Provide a concise response due to context limits."
        )

    if constraints:
        constraint_text = "\n".join(f"- {c}" for c in constraints)
        return f"{prompt}\n\nConstraints:\n{constraint_text}"

    return prompt
```

### Pattern 4: Progressive Context Building

**Concept**: Build context progressively through multi-turn interactions, with prompts that reference and build upon previous exchanges.

**Implementation:**

```python
class ProgressiveContextBuilder:
    def __init__(self):
        self.context_stack = []
        self.decision_log = []

    def generate_next_prompt(
        self,
        task: str,
        previous_response: str
    ) -> str:
        # Add previous decision to log
        if previous_response:
            self.decision_log.append({
                'response': previous_response,
                'timestamp': datetime.now(),
                'key_decisions': extract_decisions(previous_response)
            })

        # Build prompt that references previous context
        prompt = f"Task: {task}\n\n"

        # Reference previous decisions (Cognition AI Principle 2)
        if self.decision_log:
            recent_decisions = self.decision_log[-3:]  # Last 3
            prompt += "Previous decisions to maintain consistency:\n"
            for i, decision in enumerate(recent_decisions, 1):
                prompt += f"{i}. {decision['key_decisions']}\n"

        prompt += f"\nProceed with this context in mind."

        return prompt
```

## Context Injection Strategies

### Strategy 1: Selective Injection (Recommended)

**Principle**: Inject only context that's relevant to the current task.

```python
def selective_context_injection(
    task: str,
    available_context: dict,
    relevance_threshold: float = 0.7
) -> str:
    """
    Inject only relevant context based on semantic similarity.
    """
    # Calculate relevance scores
    context_items = []
    for key, value in available_context.items():
        relevance = calculate_relevance(task, value)
        if relevance >= relevance_threshold:
            context_items.append((relevance, key, value))

    # Sort by relevance
    context_items.sort(reverse=True, key=lambda x: x[0])

    # Build prompt with top N relevant items
    prompt = f"Task: {task}\n\n"
    prompt += "Relevant Context:\n"
    for relevance, key, value in context_items[:5]:  # Top 5
        prompt += f"- {key}: {value}\n"

    return prompt
```

### Strategy 2: Hierarchical Injection

**Principle**: Structure context in layers from general to specific.

```python
def hierarchical_context_injection(
    task: str,
    context_hierarchy: dict
) -> str:
    """
    Inject context in hierarchical layers.
    """
    prompt_layers = [
        # Layer 1: System-level context
        f"System Context: {context_hierarchy.get('system', {})}",

        # Layer 2: User-level context
        f"User Context: {context_hierarchy.get('user', {})}",

        # Layer 3: Session-level context
        f"Session Context: {context_hierarchy.get('session', {})}",

        # Layer 4: Task-level context
        f"Task: {task}"
    ]

    return "\n\n".join(prompt_layers)
```

### Strategy 3: Token-Aware Injection

**Principle**: Manage token budget while maximizing context value.

```python
def token_aware_injection(
    task: str,
    context_items: list,
    max_tokens: int
) -> str:
    """
    Inject context while respecting token limits.
    """
    # Estimate tokens for base prompt
    base_prompt = f"Task: {task}\n\n"
    available_tokens = max_tokens - estimate_tokens(base_prompt)

    # Prioritize context items by importance
    sorted_items = sorted(
        context_items,
        key=lambda x: x['importance'],
        reverse=True
    )

    # Add context until token budget exhausted
    injected_context = []
    for item in sorted_items:
        item_tokens = estimate_tokens(str(item['content']))
        if available_tokens >= item_tokens:
            injected_context.append(item['content'])
            available_tokens -= item_tokens
        else:
            break

    return base_prompt + "Context:\n" + "\n".join(injected_context)
```

## Balancing Prompt Clarity with Context Richness

### The Clarity-Context Trade-off

Adding more context can improve results but may also:

- Increase token usage and cost
- Add noise and reduce clarity
- Slow down processing
- Confuse the model with irrelevant information

**Optimization Strategies:**

1. **Context Summarization**

   ```python
   def summarize_context(
       full_context: str,
       target_length: int
   ) -> str:
       """
       Compress context while preserving key information.
       """
       if len(full_context) <= target_length:
           return full_context

       # Use summarization model or extractive techniques
       summary = extract_key_points(full_context, target_length)
       return summary
   ```

2. **Context Chunking**

   ```python
   def chunk_context(
       context: str,
       chunk_size: int
   ) -> list:
       """
       Break context into logical chunks for progressive processing.
       """
       chunks = []
       sentences = split_into_sentences(context)

       current_chunk = []
       current_size = 0

       for sentence in sentences:
           sentence_size = estimate_tokens(sentence)
           if current_size + sentence_size > chunk_size:
               chunks.append(" ".join(current_chunk))
               current_chunk = [sentence]
               current_size = sentence_size
           else:
               current_chunk.append(sentence)
               current_size += sentence_size

       if current_chunk:
           chunks.append(" ".join(current_chunk))

       return chunks
   ```

3. **Clarity Metrics**
   ```python
   def measure_prompt_clarity(prompt: str) -> dict:
       """
       Measure prompt clarity and structure quality.
       """
       return {
           'length': len(prompt),
           'token_count': estimate_tokens(prompt),
           'structure_score': measure_structure(prompt),
           'redundancy_score': measure_redundancy(prompt),
           'clarity_score': measure_clarity(prompt),
       }
   ```

## Best Practices for Context-Aware Prompts

### 1. Context Validation

Always validate context before injection:

```python
def validate_context(context: dict) -> tuple[bool, list]:
    """
    Validate context completeness and quality.
    """
    errors = []

    # Check required fields
    required_fields = ['user_id', 'session_id', 'task_type']
    for field in required_fields:
        if field not in context:
            errors.append(f"Missing required field: {field}")

    # Check data quality
    if context.get('user_id') and len(context['user_id']) < 1:
        errors.append("Invalid user_id")

    # Check token limits
    total_tokens = estimate_total_tokens(context)
    if total_tokens > MAX_CONTEXT_TOKENS:
        errors.append(f"Context exceeds token limit: {total_tokens}")

    return len(errors) == 0, errors
```

### 2. Context Monitoring

Track context usage and effectiveness:

```python
class ContextMonitor:
    def __init__(self):
        self.metrics = {
            'total_injections': 0,
            'token_usage': [],
            'relevance_scores': [],
            'performance_impact': []
        }

    def log_injection(
        self,
        context_size: int,
        relevance: float,
        response_quality: float
    ):
        self.metrics['total_injections'] += 1
        self.metrics['token_usage'].append(context_size)
        self.metrics['relevance_scores'].append(relevance)
        self.metrics['performance_impact'].append(response_quality)

    def get_insights(self) -> dict:
        return {
            'avg_tokens': np.mean(self.metrics['token_usage']),
            'avg_relevance': np.mean(self.metrics['relevance_scores']),
            'avg_quality': np.mean(self.metrics['performance_impact']),
            'efficiency_score': self.calculate_efficiency()
        }
```

### 3. Fallback Strategies

Implement graceful degradation:

```python
def generate_prompt_with_fallback(
    task: str,
    context: dict,
    fallback_level: int = 3
) -> str:
    """
    Generate prompt with multiple fallback levels.
    """
    try:
        # Level 1: Full context-aware prompt
        return generate_full_context_prompt(task, context)
    except ContextOverflowError:
        if fallback_level >= 2:
            # Level 2: Summarized context
            return generate_summarized_prompt(task, context)
        else:
            raise
    except MissingContextError:
        if fallback_level >= 3:
            # Level 3: Basic prompt without context
            return f"Task: {task}\nProvide your best response."
        else:
            raise
```

## Real-World Examples

### Example 1: Customer Support Bot

```python
def generate_support_prompt(
    customer_query: str,
    customer_context: dict
) -> str:
    """
    Context-aware prompt for customer support.
    """
    prompt = f"You are a customer support specialist.\n\n"

    # Inject customer history
    if customer_context.get('previous_issues'):
        prompt += "Customer's recent issues:\n"
        for issue in customer_context['previous_issues'][-3:]:
            prompt += f"- {issue['description']} (Status: {issue['status']})\n"
        prompt += "\n"

    # Inject account information
    if customer_context.get('account_tier'):
        prompt += f"Account tier: {customer_context['account_tier']}\n"
        prompt += f"Customer since: {customer_context['member_since']}\n\n"

    # Inject sentiment
    sentiment = analyze_sentiment(customer_query)
    if sentiment == 'frustrated':
        prompt += "Note: Customer appears frustrated. Be extra empathetic.\n\n"

    # Current query
    prompt += f"Current Query: {customer_query}\n\n"

    # Instructions
    prompt += """
    Provide a response that:
    1. Acknowledges their history and status
    2. Addresses their specific concern
    3. References relevant previous issues if applicable
    4. Maintains appropriate tone based on sentiment
    """

    return prompt
```

### Example 2: Code Review Assistant

````python
def generate_code_review_prompt(
    code: str,
    project_context: dict,
    developer_context: dict
) -> str:
    """
    Context-aware prompt for code review.
    """
    prompt = "You are an expert code reviewer.\n\n"

    # Project standards
    if project_context.get('coding_standards'):
        prompt += f"Project Standards:\n{project_context['coding_standards']}\n\n"

    # Developer experience level
    experience = developer_context.get('experience_level', 'intermediate')
    if experience == 'junior':
        prompt += "Developer is junior-level. Provide educational feedback.\n\n"

    # Recent patterns
    if developer_context.get('common_issues'):
        prompt += "Watch for these patterns the developer tends toward:\n"
        for issue in developer_context['common_issues']:
            prompt += f"- {issue}\n"
        prompt += "\n"

    # Code to review
    prompt += f"Code to Review:\n```\n{code}\n```\n\n"

    # Instructions
    prompt += """
    Provide a review that:
    1. Checks compliance with project standards
    2. Identifies potential bugs or issues
    3. Suggests improvements appropriate for developer's level
    4. Highlights patterns that may need attention
    """

    return prompt
````

## Key Takeaways

1. **Context-Aware ≠ Context-Overloaded**: More context isn't always better. Selective, relevant injection is key.

2. **Dynamic Adaptation**: Prompts should adapt to context state, not remain static across all scenarios.

3. **Cognition AI Principles Apply**: Share context fully and ensure actions carry consistent decisions (Principles 1 & 2).

4. **Monitor and Optimize**: Track context usage and effectiveness to continuously improve.

5. **Graceful Degradation**: Always have fallback strategies for incomplete or overflowing context.

## What's Next

In Lesson 2, we'll explore advanced prompt patterns like chain-of-thought reasoning and few-shot learning, all enhanced with context awareness to create even more powerful AI interactions.

---

**Practice Exercise**: Take a simple prompt you use regularly and transform it into a context-aware version using the patterns from this lesson. Consider: What context would make it better? How can you inject that context systematically?
