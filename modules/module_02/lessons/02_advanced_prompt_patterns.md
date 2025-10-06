# Lesson 2: Advanced Prompt Patterns with Context

## Introduction

Advanced prompt patterns—such as chain-of-thought reasoning, few-shot learning, and structured reasoning—become exponentially more powerful when combined with context engineering. This lesson explores how to integrate sophisticated prompting techniques with the context systems you've built.

## Chain-of-Thought Reasoning with Context Integration

### Understanding Chain-of-Thought (CoT)

Chain-of-thought prompting encourages models to break down complex problems into intermediate reasoning steps. When combined with context, CoT becomes **context-aware reasoning** that builds on historical decisions and accumulated knowledge.

### Basic Chain-of-Thought

**Traditional CoT (without context):**

```
Question: If John has 5 apples and gives 2 to Mary, then buys 3 more, how many does he have?

Let's think step by step:
1. John starts with 5 apples
2. He gives 2 to Mary: 5 - 2 = 3 apples
3. He buys 3 more: 3 + 3 = 6 apples
4. Final answer: 6 apples
```

### Context-Aware Chain-of-Thought

**Enhanced CoT (with context integration):**

```python
def generate_context_aware_cot_prompt(
    question: str,
    context: dict,
    reasoning_history: list
) -> str:
    """
    Generate CoT prompt that leverages context and reasoning history.
    """
    prompt = f"Question: {question}\n\n"

    # Inject relevant previous reasoning (Principle 1: Share context)
    if reasoning_history:
        prompt += "Previous reasoning patterns to maintain consistency:\n"
        for i, past_reasoning in enumerate(reasoning_history[-3:], 1):
            prompt += f"{i}. {past_reasoning['approach']}: {past_reasoning['outcome']}\n"
        prompt += "\n"

    # Add domain-specific context
    if context.get('domain_knowledge'):
        prompt += f"Relevant domain knowledge:\n{context['domain_knowledge']}\n\n"

    # Add constraints from context
    if context.get('constraints'):
        prompt += f"Constraints to consider:\n"
        for constraint in context['constraints']:
            prompt += f"- {constraint}\n"
        prompt += "\n"

    # CoT instruction
    prompt += """
    Let's approach this step by step, considering:
    1. The established patterns from previous reasoning
    2. The relevant domain knowledge
    3. The specified constraints

    Please show your reasoning for each step.
    """

    return prompt
```

### Advanced CoT Pattern: Decision Logging

Following Cognition AI's Principle 2 (Actions carry implicit decisions), log decisions during reasoning:

```python
class ContextAwareCoTEngine:
    def __init__(self):
        self.decision_log = []
        self.reasoning_trace = []

    def reason_through_problem(
        self,
        problem: str,
        context: dict
    ) -> dict:
        """
        Execute chain-of-thought with decision logging.
        """
        # Generate initial reasoning prompt
        prompt = self._build_reasoning_prompt(problem, context)

        # Execute reasoning
        reasoning_steps = self._execute_cot(prompt)

        # Log decisions made during reasoning
        for step in reasoning_steps:
            decision = self._extract_decision(step)
            if decision:
                self.decision_log.append({
                    'problem': problem,
                    'step': step,
                    'decision': decision,
                    'rationale': step['explanation'],
                    'timestamp': datetime.now()
                })

        # Store full reasoning trace for future context
        self.reasoning_trace.append({
            'problem': problem,
            'steps': reasoning_steps,
            'final_answer': reasoning_steps[-1]['result'],
            'decisions_made': len(self.decision_log)
        })

        return {
            'answer': reasoning_steps[-1]['result'],
            'reasoning': reasoning_steps,
            'decisions': self.decision_log
        }

    def _build_reasoning_prompt(
        self,
        problem: str,
        context: dict
    ) -> str:
        """
        Build prompt that includes previous decision context.
        """
        prompt = f"Problem: {problem}\n\n"

        # Include recent decisions to maintain consistency
        if self.decision_log:
            recent_decisions = self.decision_log[-5:]
            prompt += "Previous decisions to maintain consistency:\n"
            for decision in recent_decisions:
                prompt += f"- {decision['decision']}: {decision['rationale']}\n"
            prompt += "\n"

        prompt += """
        Solve this problem step by step. For each step:
        1. State what you're doing
        2. Explain why (considering previous decisions)
        3. Show your work
        4. Note any new decisions or assumptions
        """

        return prompt
```

### Multi-Step Reasoning with Context Preservation

**Key Challenge**: Maintain context coherence across multiple reasoning steps.

```python
class MultiStepReasoner:
    def __init__(self):
        self.context_stack = []
        self.step_history = []

    def execute_multi_step_reasoning(
        self,
        main_task: str,
        context: dict
    ) -> dict:
        """
        Execute multi-step reasoning while preserving context.
        """
        # Break down into steps
        steps = self._decompose_task(main_task)

        results = []
        accumulated_context = context.copy()

        for i, step in enumerate(steps):
            # Build step-specific prompt with accumulated context
            prompt = self._build_step_prompt(
                step,
                accumulated_context,
                previous_steps=self.step_history
            )

            # Execute step
            step_result = self._execute_step(prompt)

            # Update accumulated context with step results
            # This ensures each step builds on previous work
            accumulated_context[f'step_{i}_result'] = step_result

            # Log for history
            self.step_history.append({
                'step_number': i,
                'step_description': step,
                'result': step_result,
                'context_at_step': accumulated_context.copy()
            })

            results.append(step_result)

        return {
            'final_result': results[-1],
            'all_steps': results,
            'reasoning_trace': self.step_history
        }

    def _build_step_prompt(
        self,
        current_step: str,
        context: dict,
        previous_steps: list
    ) -> str:
        """
        Build prompt that maintains context from previous steps.
        """
        prompt = f"Current Step: {current_step}\n\n"

        # Include results from previous steps
        if previous_steps:
            prompt += "Previous Steps Completed:\n"
            for prev in previous_steps[-3:]:  # Last 3 steps
                prompt += f"- Step {prev['step_number']}: {prev['step_description']}\n"
                prompt += f"  Result: {prev['result']}\n"
            prompt += "\n"

        # Include relevant context
        if context:
            prompt += "Context:\n"
            for key, value in context.items():
                if key.startswith('step_'):
                    prompt += f"- {key}: {value}\n"
            prompt += "\n"

        prompt += "Execute this step building on previous results."

        return prompt
```

## Few-Shot Learning with Contextual Examples

### Traditional Few-Shot Learning

**Basic few-shot prompt:**

```
Examples:
Input: "The movie was amazing!"
Output: Positive

Input: "Terrible waste of time."
Output: Negative

Input: "It was okay, nothing special."
Output: Neutral

Now classify: "Best film I've seen this year!"
```

### Context-Aware Few-Shot Learning

**Enhanced with context:**

```python
class ContextualFewShotLearner:
    def __init__(self):
        self.example_bank = []
        self.user_preferences = {}

    def generate_contextual_few_shot_prompt(
        self,
        task: str,
        user_context: dict,
        domain_context: dict
    ) -> str:
        """
        Generate few-shot prompt with contextually relevant examples.
        """
        # Select examples based on context
        relevant_examples = self._select_contextual_examples(
            task,
            user_context,
            domain_context
        )

        prompt = f"Task: {task}\n\n"

        # Add context-specific guidance
        if user_context.get('expertise_level'):
            prompt += f"Note: User has {user_context['expertise_level']} expertise.\n"

        if domain_context.get('special_considerations'):
            prompt += f"Domain considerations: {domain_context['special_considerations']}\n"

        prompt += "\nExamples:\n"

        # Include contextually relevant examples
        for i, example in enumerate(relevant_examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n"

            # Explain why this example is relevant (meta-learning)
            if example.get('relevance_reason'):
                prompt += f"(Relevant because: {example['relevance_reason']})\n"

        prompt += f"\n\nNow apply this to: {task}"

        return prompt

    def _select_contextual_examples(
        self,
        task: str,
        user_context: dict,
        domain_context: dict,
        num_examples: int = 3
    ) -> list:
        """
        Select most relevant examples based on context.
        """
        scored_examples = []

        for example in self.example_bank:
            # Score based on multiple context factors
            relevance_score = 0.0

            # Task similarity
            relevance_score += self._calculate_task_similarity(
                task,
                example['task']
            ) * 0.4

            # User level match
            if user_context.get('expertise_level'):
                if example.get('complexity_level') == user_context['expertise_level']:
                    relevance_score += 0.3

            # Domain match
            if domain_context.get('domain'):
                if example.get('domain') == domain_context['domain']:
                    relevance_score += 0.3

            scored_examples.append((relevance_score, example))

        # Sort by relevance and return top examples
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        return [ex for score, ex in scored_examples[:num_examples]]
```

### Dynamic Example Generation

Create examples on-the-fly based on context:

```python
def generate_dynamic_examples(
    task_type: str,
    context: dict,
    num_examples: int = 3
) -> list:
    """
    Generate examples dynamically based on context.
    """
    examples = []

    # Use context to determine example characteristics
    complexity = context.get('required_complexity', 'medium')
    domain = context.get('domain', 'general')
    style = context.get('preferred_style', 'formal')

    for i in range(num_examples):
        example = create_example(
            task_type=task_type,
            complexity=complexity,
            domain=domain,
            style=style,
            variation=i  # Ensure diversity
        )
        examples.append(example)

    return examples
```

## Error Handling and Fallback Patterns

### Robust Error Handling with Context

```python
class RobustContextAwarePromptExecutor:
    def __init__(self):
        self.error_history = []
        self.fallback_strategies = []

    def execute_with_fallbacks(
        self,
        prompt: str,
        context: dict,
        max_retries: int = 3
    ) -> dict:
        """
        Execute prompt with sophisticated error handling.
        """
        attempt = 0
        last_error = None

        while attempt < max_retries:
            try:
                # Attempt execution
                result = self._execute_prompt(prompt, context)

                # Validate result
                if self._validate_result(result, context):
                    return {
                        'success': True,
                        'result': result,
                        'attempts': attempt + 1
                    }
                else:
                    # Result invalid, treat as error
                    raise ValidationError("Result validation failed")

            except Exception as e:
                last_error = e
                attempt += 1

                # Log error with context
                self.error_history.append({
                    'error': str(e),
                    'context': context,
                    'attempt': attempt,
                    'timestamp': datetime.now()
                })

                # Apply fallback strategy
                if attempt < max_retries:
                    prompt, context = self._apply_fallback(
                        prompt,
                        context,
                        error=e,
                        attempt=attempt
                    )

        # All retries exhausted
        return {
            'success': False,
            'error': str(last_error),
            'attempts': max_retries,
            'fallback_result': self._get_safe_fallback(context)
        }

    def _apply_fallback(
        self,
        original_prompt: str,
        context: dict,
        error: Exception,
        attempt: int
    ) -> tuple:
        """
        Apply progressive fallback strategies.
        """
        if attempt == 1:
            # Strategy 1: Simplify prompt
            simplified_prompt = self._simplify_prompt(original_prompt)
            return simplified_prompt, context

        elif attempt == 2:
            # Strategy 2: Reduce context
            reduced_context = self._reduce_context(context)
            return original_prompt, reduced_context

        else:
            # Strategy 3: Use minimal prompt
            minimal_prompt = self._create_minimal_prompt(original_prompt)
            minimal_context = self._get_essential_context(context)
            return minimal_prompt, minimal_context

    def _simplify_prompt(self, prompt: str) -> str:
        """
        Simplify complex prompt to reduce failure probability.
        """
        # Remove advanced instructions
        # Reduce complexity
        # Focus on core task
        simplified = extract_core_instruction(prompt)
        return f"{simplified}\n\nProvide a direct, simple response."

    def _reduce_context(self, context: dict) -> dict:
        """
        Reduce context to essentials to avoid overflow.
        """
        essential_keys = ['user_id', 'task_type', 'required_output']
        return {k: v for k, v in context.items() if k in essential_keys}
```

### Context-Aware Error Recovery

```python
class ContextAwareErrorRecovery:
    def recover_from_error(
        self,
        error: Exception,
        context: dict,
        execution_trace: list
    ) -> dict:
        """
        Recover from errors using context and execution history.
        """
        recovery_strategy = self._determine_recovery_strategy(
            error,
            context,
            execution_trace
        )

        if recovery_strategy == 'retry_with_context':
            # Use execution trace to inform retry
            return self._retry_with_learned_context(
                context,
                execution_trace
            )

        elif recovery_strategy == 'alternative_approach':
            # Try different method based on context
            return self._try_alternative_approach(context)

        elif recovery_strategy == 'graceful_degradation':
            # Provide partial result
            return self._graceful_degradation(context, execution_trace)

        else:
            # Cannot recover
            return self._fail_gracefully(error, context)

    def _retry_with_learned_context(
        self,
        context: dict,
        execution_trace: list
    ) -> dict:
        """
        Retry using lessons from previous attempt.
        """
        # Analyze what went wrong
        failure_analysis = self._analyze_failure(execution_trace)

        # Adjust context based on analysis
        adjusted_context = context.copy()
        adjusted_context['learned_constraints'] = failure_analysis['constraints']
        adjusted_context['avoid_patterns'] = failure_analysis['problematic_patterns']

        # Generate new prompt incorporating lessons
        new_prompt = f"""
        Previous attempt encountered issues. Lessons learned:
        {failure_analysis['lessons']}

        Adjusted approach:
        {failure_analysis['suggestions']}

        Please retry with these considerations in mind.
        """

        return {
            'strategy': 'retry',
            'prompt': new_prompt,
            'context': adjusted_context
        }
```

## Structured Reasoning Patterns

### Tree-of-Thought (ToT) with Context

```python
class ContextAwareTreeOfThought:
    def __init__(self):
        self.thought_tree = {}
        self.evaluation_history = []

    def explore_thought_tree(
        self,
        problem: str,
        context: dict,
        max_depth: int = 3,
        branching_factor: int = 3
    ) -> dict:
        """
        Explore multiple reasoning paths with context awareness.
        """
        # Generate initial thoughts
        initial_thoughts = self._generate_thoughts(
            problem,
            context,
            num_thoughts=branching_factor
        )

        # Build thought tree
        thought_tree = {
            'root': {
                'problem': problem,
                'context': context,
                'children': []
            }
        }

        # Explore each path
        for thought in initial_thoughts:
            # Evaluate thought in context
            evaluation = self._evaluate_thought(thought, context)

            # Expand promising thoughts
            if evaluation['score'] > 0.7:
                subtree = self._expand_thought(
                    thought,
                    context,
                    current_depth=1,
                    max_depth=max_depth
                )
                thought_tree['root']['children'].append(subtree)

        # Select best path through tree
        best_path = self._select_best_path(thought_tree, context)

        return {
            'thought_tree': thought_tree,
            'best_path': best_path,
            'final_answer': best_path[-1]['conclusion']
        }

    def _generate_thoughts(
        self,
        problem: str,
        context: dict,
        num_thoughts: int
    ) -> list:
        """
        Generate diverse thoughts considering context.
        """
        prompt = f"""
        Problem: {problem}

        Context: {context}

        Generate {num_thoughts} different approaches to solve this problem.
        Each approach should:
        1. Consider the given context
        2. Be distinct from others
        3. Be potentially viable

        For each approach, explain:
        - The core idea
        - Why it might work given the context
        - Potential challenges
        """

        thoughts = self._execute_prompt(prompt)
        return thoughts

    def _evaluate_thought(
        self,
        thought: dict,
        context: dict
    ) -> dict:
        """
        Evaluate thought quality in context.
        """
        evaluation_prompt = f"""
        Thought: {thought['idea']}
        Context: {context}

        Evaluate this thought on:
        1. Relevance to context (0-1)
        2. Feasibility (0-1)
        3. Alignment with constraints (0-1)
        4. Potential for success (0-1)

        Provide scores and rationale.
        """

        evaluation = self._execute_prompt(evaluation_prompt)

        # Calculate overall score
        score = (
            evaluation['relevance'] * 0.3 +
            evaluation['feasibility'] * 0.3 +
            evaluation['alignment'] * 0.2 +
            evaluation['potential'] * 0.2
        )

        evaluation['score'] = score
        return evaluation
```

### Iterative Refinement with Context

```python
class IterativeContextualRefinement:
    def refine_iteratively(
        self,
        initial_output: str,
        context: dict,
        quality_threshold: float = 0.9,
        max_iterations: int = 5
    ) -> dict:
        """
        Iteratively refine output using context feedback.
        """
        current_output = initial_output
        iteration = 0
        refinement_history = []

        while iteration < max_iterations:
            # Evaluate current output
            quality_score = self._evaluate_quality(
                current_output,
                context
            )

            refinement_history.append({
                'iteration': iteration,
                'output': current_output,
                'quality': quality_score
            })

            # Check if quality threshold met
            if quality_score >= quality_threshold:
                return {
                    'final_output': current_output,
                    'iterations': iteration + 1,
                    'quality_score': quality_score,
                    'history': refinement_history
                }

            # Generate refinement prompt with context
            refinement_prompt = self._generate_refinement_prompt(
                current_output,
                quality_score,
                context,
                refinement_history
            )

            # Refine output
            current_output = self._execute_refinement(refinement_prompt)
            iteration += 1

        # Max iterations reached
        return {
            'final_output': current_output,
            'iterations': max_iterations,
            'quality_score': quality_score,
            'history': refinement_history,
            'note': 'Max iterations reached'
        }

    def _generate_refinement_prompt(
        self,
        current_output: str,
        quality_score: float,
        context: dict,
        history: list
    ) -> str:
        """
        Generate refinement prompt based on context and history.
        """
        prompt = f"Current Output:\n{current_output}\n\n"
        prompt += f"Quality Score: {quality_score:.2f}\n\n"

        # Identify weaknesses from context
        weaknesses = self._identify_weaknesses(
            current_output,
            context,
            quality_score
        )

        prompt += "Areas for Improvement:\n"
        for weakness in weaknesses:
            prompt += f"- {weakness}\n"
        prompt += "\n"

        # Learn from history
        if history:
            prompt += "Previous iterations showed:\n"
            for entry in history[-2:]:
                prompt += f"- Iteration {entry['iteration']}: Quality {entry['quality']:.2f}\n"
            prompt += "\n"

        prompt += """
        Please refine the output addressing the identified weaknesses.
        Maintain strengths from the current version while improving weak areas.
        """

        return prompt
```

## Key Takeaways

1. **Context Amplifies Patterns**: Advanced prompt patterns become significantly more powerful when integrated with context.

2. **Decision Logging is Critical**: Following Cognition AI's principles, log all decisions to maintain consistency across reasoning steps.

3. **Context Preservation**: Multi-step reasoning must preserve and build upon context from previous steps.

4. **Intelligent Fallbacks**: Use context to inform error recovery and fallback strategies.

5. **Iterative Improvement**: Leverage context and history for iterative refinement of outputs.

## What's Next

In Lesson 3, we'll explore template systems that make these advanced patterns reusable and scalable across different contexts and use cases.

---

**Practice Exercise**: Implement a context-aware chain-of-thought system that maintains a decision log across multiple reasoning tasks. Test how consistent decisions improve output quality over time.
