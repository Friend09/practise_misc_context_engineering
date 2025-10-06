# Lesson 4: Structured Reasoning and Context Flow

## Introduction

The pinnacle of context-aware prompting is **structured reasoning with context flow**—designing systems where AI reasoning builds progressively, maintains consistency across steps, and leverages accumulated knowledge. This lesson integrates all previous concepts to create sophisticated, reliable reasoning frameworks.

## The Context Flow Challenge

### Understanding Context Flow

**Context Flow** is the continuous, coherent movement of information through a reasoning process, ensuring that:

1. **Each step builds on previous steps** (Cognition AI Principle 1: Share context)
2. **Decisions remain consistent** (Cognition AI Principle 2: Actions carry implicit decisions)
3. **No critical information is lost** during transitions
4. **The system maintains reliability** over extended interactions

### Why Context Flow Matters

From the Cognition AI research, we know that multi-agent systems fail primarily due to:

- **Context loss** between agents
- **Decision conflicts** from fragmented reasoning
- **Coordination overhead** that introduces fragility

**Solution**: Single-threaded reasoning with excellent context flow.

## Designing Reasoning Frameworks

### The Reasoning Architecture

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
import datetime

class ReasoningStage(Enum):
    """Stages in the reasoning process."""
    ANALYSIS = "analysis"
    HYPOTHESIS = "hypothesis"
    EVALUATION = "evaluation"
    DECISION = "decision"
    EXECUTION = "execution"
    REFLECTION = "reflection"

@dataclass
class ReasoningState:
    """
    Complete state of reasoning at any point.
    """
    stage: ReasoningStage
    context: Dict
    decisions: List[Dict] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    intermediate_results: List[Dict] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def add_decision(
        self,
        decision: str,
        rationale: str,
        confidence: float
    ):
        """Log a decision with rationale."""
        self.decisions.append({
            'decision': decision,
            'rationale': rationale,
            'confidence': confidence,
            'timestamp': datetime.datetime.now(),
            'stage': self.stage.value
        })

    def add_assumption(self, assumption: str):
        """Add an assumption to track."""
        self.assumptions.append(assumption)

    def get_context_summary(self) -> str:
        """Get a summary of current reasoning context."""
        return f"""
        Stage: {self.stage.value}
        Decisions Made: {len(self.decisions)}
        Assumptions: {len(self.assumptions)}
        Constraints: {len(self.constraints)}
        """

class ReasoningFramework:
    """
    Framework for structured reasoning with context flow.
    """
    def __init__(self):
        self.state = None
        self.stage_handlers: Dict[ReasoningStage, Callable] = {}
        self.transition_rules: Dict[tuple, Callable] = {}
        self.context_compressor = ContextCompressor()

    def register_stage_handler(
        self,
        stage: ReasoningStage,
        handler: Callable
    ):
        """Register handler for a reasoning stage."""
        self.stage_handlers[stage] = handler

    def execute_reasoning(
        self,
        problem: str,
        context: Dict
    ) -> Dict:
        """
        Execute structured reasoning process.
        """
        # Initialize reasoning state
        self.state = ReasoningState(
            stage=ReasoningStage.ANALYSIS,
            context=context
        )

        # Execute reasoning stages
        stages = [
            ReasoningStage.ANALYSIS,
            ReasoningStage.HYPOTHESIS,
            ReasoningStage.EVALUATION,
            ReasoningStage.DECISION,
            ReasoningStage.EXECUTION,
            ReasoningStage.REFLECTION
        ]

        results = {}
        for stage in stages:
            # Execute stage
            stage_result = self._execute_stage(stage, problem)
            results[stage.value] = stage_result

            # Update state
            self.state.stage = stage
            self.state.intermediate_results.append({
                'stage': stage.value,
                'result': stage_result,
                'timestamp': datetime.datetime.now()
            })

            # Check if context compression needed
            if self._should_compress_context():
                self._compress_context()

        return {
            'final_result': results[ReasoningStage.EXECUTION.value],
            'full_trace': results,
            'decisions': self.state.decisions,
            'state': self.state
        }

    def _execute_stage(
        self,
        stage: ReasoningStage,
        problem: str
    ) -> Dict:
        """Execute a specific reasoning stage."""
        handler = self.stage_handlers.get(stage)
        if not handler:
            raise ValueError(f"No handler for stage {stage}")

        # Build context-aware prompt for stage
        prompt = self._build_stage_prompt(stage, problem)

        # Execute
        result = handler(prompt, self.state)

        return result

    def _build_stage_prompt(
        self,
        stage: ReasoningStage,
        problem: str
    ) -> str:
        """
        Build prompt for stage with full context flow.
        """
        prompt = f"Problem: {problem}\n\n"
        prompt += f"Current Stage: {stage.value}\n\n"

        # Include previous decisions (Principle 1: Share context)
        if self.state.decisions:
            prompt += "Previous Decisions (maintain consistency):\n"
            for i, decision in enumerate(self.state.decisions, 1):
                prompt += f"{i}. {decision['decision']}\n"
                prompt += f"   Rationale: {decision['rationale']}\n"
                prompt += f"   Confidence: {decision['confidence']:.2f}\n"
            prompt += "\n"

        # Include assumptions
        if self.state.assumptions:
            prompt += "Working Assumptions:\n"
            for assumption in self.state.assumptions:
                prompt += f"- {assumption}\n"
            prompt += "\n"

        # Include constraints
        if self.state.constraints:
            prompt += "Constraints:\n"
            for constraint in self.state.constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"

        # Include relevant intermediate results
        if self.state.intermediate_results:
            prompt += "Previous Stage Results:\n"
            for result in self.state.intermediate_results[-2:]:  # Last 2
                prompt += f"- {result['stage']}: {result['result']}\n"
            prompt += "\n"

        # Stage-specific instructions
        prompt += self._get_stage_instructions(stage)

        return prompt

    def _get_stage_instructions(self, stage: ReasoningStage) -> str:
        """Get instructions specific to reasoning stage."""
        instructions = {
            ReasoningStage.ANALYSIS: """
            Analyze the problem:
            1. Break down key components
            2. Identify relevant information
            3. Note any ambiguities
            4. State assumptions clearly
            """,

            ReasoningStage.HYPOTHESIS: """
            Generate hypotheses:
            1. Propose potential approaches
            2. Consider alternatives
            3. Note trade-offs
            4. Estimate confidence for each
            """,

            ReasoningStage.EVALUATION: """
            Evaluate options:
            1. Compare against requirements
            2. Consider constraints
            3. Assess feasibility
            4. Identify risks
            """,

            ReasoningStage.DECISION: """
            Make decision:
            1. Select best approach
            2. Justify choice clearly
            3. Note confidence level
            4. State key assumptions
            """,

            ReasoningStage.EXECUTION: """
            Execute decision:
            1. Implement chosen approach
            2. Document key steps
            3. Note any issues
            4. Validate results
            """,

            ReasoningStage.REFLECTION: """
            Reflect on process:
            1. Assess outcome quality
            2. Identify lessons learned
            3. Note improvements for future
            4. Update confidence scores
            """
        }

        return instructions.get(stage, "")

    def _should_compress_context(self) -> bool:
        """Determine if context compression is needed."""
        # Estimate total context size
        total_tokens = self._estimate_context_tokens()

        # Compress if approaching limit
        return total_tokens > 6000  # Conservative threshold

    def _compress_context(self):
        """Compress context while preserving critical information."""
        # Compress intermediate results
        if len(self.state.intermediate_results) > 5:
            # Keep first 2 and last 3
            compressed = (
                self.state.intermediate_results[:2] +
                self.state.intermediate_results[-3:]
            )
            self.state.intermediate_results = compressed

        # Compress decision log (keep most important)
        if len(self.state.decisions) > 10:
            # Sort by confidence, keep top 10
            sorted_decisions = sorted(
                self.state.decisions,
                key=lambda x: x['confidence'],
                reverse=True
            )
            self.state.decisions = sorted_decisions[:10]
```

## Context-Aware Decision Trees

### Building Decision Trees with Context

```python
class ContextAwareDecisionTree:
    """
    Decision tree that maintains context throughout navigation.
    """
    def __init__(self):
        self.root = None
        self.current_node = None
        self.navigation_history = []
        self.context_stack = []

    def navigate(
        self,
        problem: str,
        context: Dict
    ) -> Dict:
        """
        Navigate decision tree with context preservation.
        """
        # Start at root
        self.current_node = self.root
        self.context_stack = [context.copy()]

        while not self.current_node.is_leaf():
            # Build decision prompt with full context
            prompt = self._build_decision_prompt(
                self.current_node,
                problem,
                self.context_stack[-1]
            )

            # Make decision
            decision = self._make_decision(prompt)

            # Log navigation
            self.navigation_history.append({
                'node': self.current_node.id,
                'decision': decision,
                'context': self.context_stack[-1].copy(),
                'timestamp': datetime.datetime.now()
            })

            # Update context with decision
            updated_context = self.context_stack[-1].copy()
            updated_context['last_decision'] = decision
            updated_context['decision_path'] = [
                h['decision'] for h in self.navigation_history
            ]
            self.context_stack.append(updated_context)

            # Move to next node
            self.current_node = self.current_node.get_child(decision)

        # Reached leaf - final answer
        return {
            'answer': self.current_node.value,
            'path': [h['decision'] for h in self.navigation_history],
            'full_context': self.context_stack[-1]
        }

    def _build_decision_prompt(
        self,
        node: DecisionNode,
        problem: str,
        context: Dict
    ) -> str:
        """
        Build prompt for decision point with context.
        """
        prompt = f"Problem: {problem}\n\n"

        # Current decision point
        prompt += f"Decision Point: {node.question}\n\n"

        # Previous decisions (maintain consistency)
        if self.navigation_history:
            prompt += "Previous Decisions:\n"
            for i, step in enumerate(self.navigation_history, 1):
                prompt += f"{i}. {step['decision']}\n"
            prompt += "\n"

        # Available options
        prompt += "Options:\n"
        for option in node.options:
            prompt += f"- {option}\n"
        prompt += "\n"

        # Context-specific guidance
        if context.get('preferences'):
            prompt += f"User preferences: {context['preferences']}\n\n"

        prompt += "Select the best option considering all previous decisions."

        return prompt

@dataclass
class DecisionNode:
    """Node in decision tree."""
    id: str
    question: str
    options: List[str]
    children: Dict[str, 'DecisionNode']
    value: Optional[str] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_child(self, decision: str) -> 'DecisionNode':
        return self.children.get(decision)
```

## Integrating External Knowledge with Contextual Reasoning

### Knowledge-Enhanced Reasoning

```python
class KnowledgeEnhancedReasoner:
    """
    Reasoning system that integrates external knowledge with context.
    """
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.query_history = []

    def reason_with_knowledge(
        self,
        problem: str,
        context: Dict
    ) -> Dict:
        """
        Reason about problem using context and external knowledge.
        """
        # Phase 1: Analyze what knowledge is needed
        knowledge_requirements = self._identify_knowledge_needs(
            problem,
            context
        )

        # Phase 2: Retrieve relevant knowledge
        retrieved_knowledge = []
        for requirement in knowledge_requirements:
            knowledge = self.knowledge_base.query(
                requirement,
                context=context
            )
            retrieved_knowledge.append({
                'requirement': requirement,
                'knowledge': knowledge,
                'timestamp': datetime.datetime.now()
            })
            self.query_history.append(requirement)

        # Phase 3: Integrate knowledge with context
        integrated_context = self._integrate_knowledge(
            context,
            retrieved_knowledge
        )

        # Phase 4: Reason with integrated context
        reasoning_result = self._execute_reasoning(
            problem,
            integrated_context
        )

        return {
            'result': reasoning_result,
            'knowledge_used': retrieved_knowledge,
            'enhanced_context': integrated_context
        }

    def _identify_knowledge_needs(
        self,
        problem: str,
        context: Dict
    ) -> List[str]:
        """
        Identify what knowledge is needed to solve problem.
        """
        prompt = f"""
        Problem: {problem}

        Current Context: {context}

        What external knowledge would help solve this problem?
        List specific knowledge requirements.
        """

        # Execute prompt to identify needs
        needs = self._execute_prompt(prompt)
        return needs

    def _integrate_knowledge(
        self,
        context: Dict,
        knowledge: List[Dict]
    ) -> Dict:
        """
        Integrate retrieved knowledge with existing context.
        """
        integrated = context.copy()

        # Add knowledge section
        integrated['external_knowledge'] = []

        for item in knowledge:
            integrated['external_knowledge'].append({
                'topic': item['requirement'],
                'content': item['knowledge'],
                'retrieved_at': item['timestamp']
            })

        # Update context with key facts
        integrated['key_facts'] = self._extract_key_facts(knowledge)

        return integrated

    def _extract_key_facts(self, knowledge: List[Dict]) -> List[str]:
        """Extract key facts from retrieved knowledge."""
        facts = []
        for item in knowledge:
            # Extract most relevant facts
            item_facts = extract_facts(item['knowledge'])
            facts.extend(item_facts)
        return facts
```

## Building Explanatory and Transparent AI Responses

### Transparent Reasoning System

```python
class TransparentReasoner:
    """
    Reasoning system that provides full explanations and transparency.
    """
    def __init__(self):
        self.trace = []
        self.explanations = []

    def reason_transparently(
        self,
        problem: str,
        context: Dict
    ) -> Dict:
        """
        Execute reasoning with full transparency.
        """
        # Phase 1: Explain the problem understanding
        problem_understanding = self._explain_problem(problem, context)
        self._add_explanation(
            "Problem Understanding",
            problem_understanding
        )

        # Phase 2: Explain the approach
        approach = self._explain_approach(problem, context)
        self._add_explanation(
            "Chosen Approach",
            approach
        )

        # Phase 3: Execute with step-by-step explanations
        steps = self._execute_with_explanations(problem, context)

        # Phase 4: Explain the final decision
        final_explanation = self._explain_decision(steps)
        self._add_explanation(
            "Final Decision",
            final_explanation
        )

        return {
            'result': steps[-1]['result'],
            'explanations': self.explanations,
            'trace': self.trace,
            'transparency_score': self._calculate_transparency()
        }

    def _explain_problem(
        self,
        problem: str,
        context: Dict
    ) -> str:
        """
        Generate explanation of problem understanding.
        """
        prompt = f"""
        Problem: {problem}
        Context: {context}

        Explain your understanding of:
        1. What is being asked
        2. Why it's being asked (from context)
        3. Key constraints or requirements
        4. Success criteria

        Be clear and explicit.
        """

        explanation = self._execute_prompt(prompt)
        return explanation

    def _explain_approach(
        self,
        problem: str,
        context: Dict
    ) -> str:
        """
        Explain why a particular approach is chosen.
        """
        prompt = f"""
        Given the problem and context, explain:
        1. What approach will be used
        2. Why this approach is suitable
        3. What alternatives were considered
        4. Why those alternatives were rejected

        Be specific about reasoning.
        """

        explanation = self._execute_prompt(prompt)
        return explanation

    def _execute_with_explanations(
        self,
        problem: str,
        context: Dict
    ) -> List[Dict]:
        """
        Execute reasoning with step-by-step explanations.
        """
        steps = []

        # Break down into steps
        step_plan = self._plan_steps(problem, context)

        for i, step_description in enumerate(step_plan, 1):
            # Execute step
            step_result = self._execute_step(
                step_description,
                context,
                previous_steps=steps
            )

            # Explain step
            explanation = self._explain_step(
                step_description,
                step_result,
                steps
            )

            steps.append({
                'step_number': i,
                'description': step_description,
                'result': step_result,
                'explanation': explanation
            })

            self._add_explanation(f"Step {i}", explanation)

        return steps

    def _explain_step(
        self,
        step: str,
        result: str,
        previous_steps: List[Dict]
    ) -> str:
        """
        Explain a reasoning step.
        """
        prompt = f"""
        Step: {step}
        Result: {result}

        Previous Steps: {previous_steps}

        Explain:
        1. What was done in this step
        2. Why this was necessary
        3. How it builds on previous steps
        4. What the result means
        """

        explanation = self._execute_prompt(prompt)
        return explanation

    def _add_explanation(self, stage: str, explanation: str):
        """Add explanation to trace."""
        self.explanations.append({
            'stage': stage,
            'explanation': explanation,
            'timestamp': datetime.datetime.now()
        })

    def _calculate_transparency(self) -> float:
        """
        Calculate transparency score based on explanations.
        """
        # Check completeness of explanations
        required_stages = [
            "Problem Understanding",
            "Chosen Approach",
            "Final Decision"
        ]

        covered_stages = [e['stage'] for e in self.explanations]
        coverage = len(
            set(required_stages) & set(covered_stages)
        ) / len(required_stages)

        # Check explanation quality (length, clarity)
        avg_length = np.mean([
            len(e['explanation']) for e in self.explanations
        ])
        length_score = min(avg_length / 500, 1.0)  # Target 500 chars

        # Combined score
        return (coverage * 0.7 + length_score * 0.3)
```

## Meta-Reasoning: Reasoning About Reasoning

### Self-Monitoring Reasoner

```python
class MetaReasoner:
    """
    Reasoner that monitors and improves its own reasoning.
    """
    def __init__(self):
        self.reasoning_history = []
        self.performance_metrics = []

    def reason_with_monitoring(
        self,
        problem: str,
        context: Dict
    ) -> Dict:
        """
        Execute reasoning with self-monitoring.
        """
        # Execute reasoning
        initial_result = self._execute_reasoning(problem, context)

        # Monitor quality
        quality_assessment = self._assess_reasoning_quality(
            problem,
            initial_result,
            context
        )

        # If quality insufficient, iterate
        if quality_assessment['score'] < 0.8:
            improved_result = self._improve_reasoning(
                problem,
                initial_result,
                quality_assessment,
                context
            )
            final_result = improved_result
        else:
            final_result = initial_result

        # Log for learning
        self._log_reasoning(problem, final_result, quality_assessment)

        return {
            'result': final_result,
            'quality': quality_assessment,
            'iterations': quality_assessment.get('iterations', 1)
        }

    def _assess_reasoning_quality(
        self,
        problem: str,
        reasoning: Dict,
        context: Dict
    ) -> Dict:
        """
        Assess quality of reasoning process.
        """
        prompt = f"""
        Problem: {problem}
        Reasoning: {reasoning}
        Context: {context}

        Assess this reasoning on:
        1. Logical consistency (0-1)
        2. Completeness (0-1)
        3. Relevance to problem (0-1)
        4. Use of context (0-1)

        Provide scores and specific areas for improvement.
        """

        assessment = self._execute_prompt(prompt)

        # Calculate overall score
        assessment['score'] = (
            assessment['logical_consistency'] * 0.3 +
            assessment['completeness'] * 0.3 +
            assessment['relevance'] * 0.2 +
            assessment['context_use'] * 0.2
        )

        return assessment

    def _improve_reasoning(
        self,
        problem: str,
        initial_reasoning: Dict,
        assessment: Dict,
        context: Dict
    ) -> Dict:
        """
        Improve reasoning based on quality assessment.
        """
        prompt = f"""
        Original Problem: {problem}
        Initial Reasoning: {initial_reasoning}
        Quality Issues: {assessment['issues']}

        Context: {context}

        Improve the reasoning addressing:
        {assessment['improvement_suggestions']}

        Maintain strengths while fixing weaknesses.
        """

        improved = self._execute_prompt(prompt)
        assessment['iterations'] = assessment.get('iterations', 1) + 1

        return improved
```

## Key Takeaways

1. **Context Flow is Critical**: Maintain continuous, coherent information flow throughout reasoning.

2. **Decision Consistency**: Log and reference all decisions to maintain consistency (Cognition AI Principle 2).

3. **Structured Stages**: Break reasoning into clear stages with explicit transitions.

4. **Transparent Reasoning**: Provide explanations at every step for interpretability and debugging.

5. **Self-Monitoring**: Implement meta-reasoning to continuously improve reasoning quality.

6. **Knowledge Integration**: Systematically integrate external knowledge with contextual reasoning.

## Integration with Previous Lessons

This lesson brings together:

- **Lesson 1**: Context injection and management
- **Lesson 2**: Advanced patterns like chain-of-thought
- **Lesson 3**: Template systems for scalable reasoning

Together, these create production-ready reasoning systems that maintain reliability over extended interactions—the goal of modern context engineering.

## What's Next

You're now ready to apply these concepts in the hands-on exercises where you'll build:

1. Context-aware prompt generation systems
2. Advanced reasoning engines with decision logging
3. Production-ready template systems

These practical exercises will solidify your understanding and prepare you for real-world applications.

---

**Practice Exercise**: Design a structured reasoning framework for a domain you work in. Implement decision logging, context flow, and transparency. Test with complex, multi-step problems and analyze how context preservation improves results.
