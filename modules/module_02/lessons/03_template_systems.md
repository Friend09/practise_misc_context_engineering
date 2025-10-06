# Lesson 3: Template Systems and Dynamic Generation

## Introduction

Template systems transform context-aware prompting from an art into an engineering discipline. A well-designed template system provides **scalability, consistency, and maintainability** while allowing prompts to adapt dynamically to context. This lesson explores building production-ready prompt template systems.

## Why Template Systems Matter

### The Scalability Problem

**Without templates:**

- Every prompt is hand-crafted
- Inconsistent patterns across use cases
- Difficult to maintain and update
- No systematic reuse

**With template systems:**

- Systematic prompt generation
- Consistent patterns and best practices
- Easy updates and improvements
- Scalable across contexts and use cases

### Industry Context: Production AI Systems

Leading AI companies like Cognition AI build sophisticated template systems because:

1. **Consistency**: Templates ensure all prompts follow best practices
2. **Maintainability**: Update patterns in one place, affect all usage
3. **Context Integration**: Templates systematically inject context
4. **Performance**: Tested, optimized patterns perform reliably

## Scalable Prompt Template Architecture

### Core Architecture Components

```python
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class TemplateType(Enum):
    """Types of prompt templates."""
    SIMPLE = "simple"
    CONTEXTUAL = "contextual"
    MULTI_STEP = "multi_step"
    ADAPTIVE = "adaptive"

@dataclass
class Template:
    """
    Core template structure.
    """
    id: str
    name: str
    type: TemplateType
    base_prompt: str
    required_context: List[str]
    optional_context: List[str]
    validation_rules: List[Callable]
    metadata: Dict

class TemplateEngine:
    """
    Central template engine for prompt generation.
    """
    def __init__(self):
        self.templates: Dict[str, Template] = {}
        self.context_processors: Dict[str, Callable] = {}
        self.validators: Dict[str, Callable] = {}

    def register_template(self, template: Template):
        """Register a new template."""
        self.templates[template.id] = template

    def register_context_processor(
        self,
        name: str,
        processor: Callable
    ):
        """Register a context processing function."""
        self.context_processors[name] = processor

    def generate_prompt(
        self,
        template_id: str,
        context: Dict,
        options: Optional[Dict] = None
    ) -> str:
        """
        Generate a prompt from template and context.
        """
        # Get template
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")

        # Validate context
        self._validate_context(template, context)

        # Process context
        processed_context = self._process_context(
            template,
            context,
            options
        )

        # Build prompt
        prompt = self._build_prompt(template, processed_context)

        # Validate prompt
        self._validate_prompt(template, prompt)

        return prompt
```

### Template Registry Pattern

```python
class TemplateRegistry:
    """
    Centralized template registry with versioning.
    """
    def __init__(self):
        self._templates: Dict[str, Dict[str, Template]] = {}
        self._latest_versions: Dict[str, str] = {}

    def register(
        self,
        template: Template,
        version: str = "1.0.0"
    ):
        """
        Register a template with version.
        """
        if template.id not in self._templates:
            self._templates[template.id] = {}

        self._templates[template.id][version] = template

        # Update latest version
        if (template.id not in self._latest_versions or
            self._is_newer_version(version, self._latest_versions[template.id])):
            self._latest_versions[template.id] = version

    def get(
        self,
        template_id: str,
        version: Optional[str] = None
    ) -> Template:
        """
        Get template by ID and optional version.
        """
        if template_id not in self._templates:
            raise ValueError(f"Template {template_id} not found")

        # Use specified version or latest
        version = version or self._latest_versions[template_id]

        if version not in self._templates[template_id]:
            raise ValueError(
                f"Version {version} not found for template {template_id}"
            )

        return self._templates[template_id][version]

    def list_templates(self) -> List[str]:
        """List all available template IDs."""
        return list(self._templates.keys())

    def list_versions(self, template_id: str) -> List[str]:
        """List all versions for a template."""
        if template_id not in self._templates:
            return []
        return list(self._templates[template_id].keys())
```

## Context-Driven Template Selection

### Intelligent Template Selection

```python
class TemplateSelector:
    """
    Intelligently select templates based on context.
    """
    def __init__(self, registry: TemplateRegistry):
        self.registry = registry
        self.selection_rules: List[SelectionRule] = []

    def select_template(
        self,
        task: str,
        context: Dict
    ) -> Template:
        """
        Select the best template for the given task and context.
        """
        # Get all available templates
        available_templates = self._get_available_templates()

        # Score each template
        scored_templates = []
        for template in available_templates:
            score = self._score_template(template, task, context)
            scored_templates.append((score, template))

        # Sort by score and return best
        scored_templates.sort(reverse=True, key=lambda x: x[0])

        if not scored_templates:
            raise ValueError("No suitable template found")

        return scored_templates[0][1]

    def _score_template(
        self,
        template: Template,
        task: str,
        context: Dict
    ) -> float:
        """
        Score template suitability for task and context.
        """
        score = 0.0

        # Check task type match
        if template.metadata.get('task_type') in task.lower():
            score += 0.3

        # Check context completeness
        required_fields = set(template.required_context)
        available_fields = set(context.keys())
        completeness = len(required_fields & available_fields) / len(required_fields)
        score += completeness * 0.4

        # Check complexity match
        task_complexity = estimate_task_complexity(task)
        template_complexity = template.metadata.get('complexity', 'medium')
        if task_complexity == template_complexity:
            score += 0.2

        # Check user preferences
        if context.get('user_preferences'):
            prefs = context['user_preferences']
            if template.metadata.get('style') == prefs.get('style'):
                score += 0.1

        return score

@dataclass
class SelectionRule:
    """
    Rule for template selection.
    """
    name: str
    condition: Callable[[str, Dict], bool]
    template_id: str
    priority: int
```

### Adaptive Template System

```python
class AdaptiveTemplateSystem:
    """
    Template system that learns from usage and adapts.
    """
    def __init__(self):
        self.registry = TemplateRegistry()
        self.selector = TemplateSelector(self.registry)
        self.usage_stats: Dict[str, UsageStats] = {}
        self.performance_tracker = PerformanceTracker()

    def generate_adaptive_prompt(
        self,
        task: str,
        context: Dict
    ) -> str:
        """
        Generate prompt with adaptive template selection.
        """
        # Select template based on context and performance history
        template = self._select_adaptive_template(task, context)

        # Generate prompt
        engine = TemplateEngine()
        prompt = engine.generate_prompt(
            template.id,
            context,
            options={'adaptive': True}
        )

        # Track usage
        self._track_usage(template, context)

        return prompt

    def _select_adaptive_template(
        self,
        task: str,
        context: Dict
    ) -> Template:
        """
        Select template considering historical performance.
        """
        # Get candidate templates
        candidates = self.selector.get_candidates(task, context)

        # Score based on historical performance
        scored_candidates = []
        for template in candidates:
            # Base score from selector
            base_score = self.selector._score_template(
                template,
                task,
                context
            )

            # Adjust based on performance history
            performance_score = self._get_performance_score(
                template,
                context
            )

            # Combined score
            final_score = base_score * 0.6 + performance_score * 0.4
            scored_candidates.append((final_score, template))

        # Return best performing template
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        return scored_candidates[0][1]

    def _get_performance_score(
        self,
        template: Template,
        context: Dict
    ) -> float:
        """
        Get historical performance score for template in similar contexts.
        """
        stats = self.usage_stats.get(template.id)
        if not stats:
            return 0.5  # Neutral for unknown templates

        # Find similar contexts
        similar_contexts = self._find_similar_contexts(
            context,
            stats.contexts
        )

        if not similar_contexts:
            return stats.overall_score

        # Average performance in similar contexts
        scores = [ctx['performance'] for ctx in similar_contexts]
        return sum(scores) / len(scores)

    def record_performance(
        self,
        template_id: str,
        context: Dict,
        performance: float
    ):
        """
        Record performance for continuous learning.
        """
        if template_id not in self.usage_stats:
            self.usage_stats[template_id] = UsageStats(template_id)

        self.usage_stats[template_id].add_performance(
            context,
            performance
        )

        # Update performance tracker
        self.performance_tracker.record(
            template_id,
            context,
            performance
        )

@dataclass
class UsageStats:
    """Track template usage and performance."""
    template_id: str
    usage_count: int = 0
    contexts: List[Dict] = None
    overall_score: float = 0.5

    def __post_init__(self):
        if self.contexts is None:
            self.contexts = []

    def add_performance(self, context: Dict, performance: float):
        self.usage_count += 1
        self.contexts.append({
            'context': context,
            'performance': performance,
            'timestamp': datetime.now()
        })
        # Update overall score (moving average)
        self.overall_score = (
            self.overall_score * 0.9 + performance * 0.1
        )
```

## Dynamic Content Insertion and Formatting

### Content Insertion Strategies

````python
class ContentInserter:
    """
    Intelligently insert content into templates.
    """
    def __init__(self):
        self.formatters: Dict[str, Callable] = {}
        self.filters: Dict[str, Callable] = {}

    def insert_content(
        self,
        template: str,
        context: Dict,
        insertion_rules: Dict
    ) -> str:
        """
        Insert content into template with intelligent formatting.
        """
        result = template

        for placeholder, rules in insertion_rules.items():
            # Get content from context
            content = self._extract_content(
                context,
                rules.get('source')
            )

            # Apply filters
            if rules.get('filters'):
                for filter_name in rules['filters']:
                    filter_fn = self.filters.get(filter_name)
                    if filter_fn:
                        content = filter_fn(content)

            # Apply formatting
            if rules.get('formatter'):
                formatter = self.formatters.get(rules['formatter'])
                if formatter:
                    content = formatter(content)

            # Insert into template
            result = result.replace(placeholder, str(content))

        return result

    def register_formatter(self, name: str, formatter: Callable):
        """Register a content formatter."""
        self.formatters[name] = formatter

    def register_filter(self, name: str, filter_fn: Callable):
        """Register a content filter."""
        self.filters[name] = filter_fn

# Example formatters
def format_as_list(items: List) -> str:
    """Format items as numbered list."""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))

def format_as_table(data: Dict) -> str:
    """Format data as markdown table."""
    if not data:
        return ""

    headers = list(data.keys())
    values = list(data.values())

    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    value_row = "| " + " | ".join(str(v) for v in values) + " |"

    return f"{header_row}\n{separator}\n{value_row}"

def format_as_code_block(code: str, language: str = "python") -> str:
    """Format as code block."""
    return f"```{language}\n{code}\n```"

# Example filters
def filter_relevant_only(content: List, context: Dict) -> List:
    """Filter to include only relevant items."""
    relevance_threshold = context.get('relevance_threshold', 0.7)
    return [
        item for item in content
        if calculate_relevance(item, context) >= relevance_threshold
    ]

def filter_by_token_limit(content: str, max_tokens: int) -> str:
    """Filter content to fit within token limit."""
    tokens = estimate_tokens(content)
    if tokens <= max_tokens:
        return content

    # Truncate intelligently
    return truncate_to_tokens(content, max_tokens)
````

### Dynamic Formatting Based on Context

````python
class DynamicFormatter:
    """
    Format content dynamically based on context.
    """
    def format_response(
        self,
        content: str,
        context: Dict
    ) -> str:
        """
        Format content based on context preferences.
        """
        # Determine output format from context
        format_type = self._determine_format(context)

        if format_type == 'markdown':
            return self._format_as_markdown(content, context)
        elif format_type == 'json':
            return self._format_as_json(content, context)
        elif format_type == 'code':
            return self._format_as_code(content, context)
        else:
            return content

    def _determine_format(self, context: Dict) -> str:
        """Determine best format from context."""
        # Check explicit preference
        if context.get('output_format'):
            return context['output_format']

        # Infer from context
        if context.get('task_type') == 'coding':
            return 'code'
        elif context.get('structured_output'):
            return 'json'
        else:
            return 'markdown'

    def _format_as_markdown(
        self,
        content: str,
        context: Dict
    ) -> str:
        """Format content as markdown with appropriate structure."""
        sections = self._parse_sections(content)

        formatted = ""
        for section in sections:
            # Add appropriate header level
            header_level = section.get('level', 2)
            formatted += f"{'#' * header_level} {section['title']}\n\n"
            formatted += f"{section['content']}\n\n"

        return formatted

    def _format_as_json(
        self,
        content: str,
        context: Dict
    ) -> str:
        """Format content as structured JSON."""
        # Parse content into structured format
        structured = self._structure_content(content, context)

        # Convert to JSON
        import json
        return json.dumps(structured, indent=2)

    def _format_as_code(
        self,
        content: str,
        context: Dict
    ) -> str:
        """Format content as code with appropriate language."""
        language = context.get('language', 'python')

        return f"```{language}\n{content}\n```"
````

## Template Validation and Optimization

### Comprehensive Template Validation

```python
class TemplateValidator:
    """
    Validate templates for correctness and quality.
    """
    def __init__(self):
        self.validation_rules: List[ValidationRule] = []
        self.quality_metrics: List[QualityMetric] = []

    def validate_template(self, template: Template) -> ValidationResult:
        """
        Comprehensive template validation.
        """
        errors = []
        warnings = []
        suggestions = []

        # Check required fields
        field_check = self._validate_required_fields(template)
        errors.extend(field_check['errors'])

        # Check context requirements
        context_check = self._validate_context_requirements(template)
        errors.extend(context_check['errors'])
        warnings.extend(context_check['warnings'])

        # Check prompt quality
        quality_check = self._check_prompt_quality(template)
        warnings.extend(quality_check['warnings'])
        suggestions.extend(quality_check['suggestions'])

        # Check for common issues
        issue_check = self._check_common_issues(template)
        warnings.extend(issue_check['warnings'])

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def _validate_required_fields(
        self,
        template: Template
    ) -> Dict:
        """Validate required template fields."""
        errors = []

        if not template.id:
            errors.append("Template ID is required")

        if not template.base_prompt:
            errors.append("Base prompt is required")

        if not template.type:
            errors.append("Template type is required")

        return {'errors': errors}

    def _validate_context_requirements(
        self,
        template: Template
    ) -> Dict:
        """Validate context requirements are reasonable."""
        errors = []
        warnings = []

        # Check for too many required fields
        if len(template.required_context) > 10:
            warnings.append(
                f"Template requires {len(template.required_context)} context fields. "
                "Consider reducing for flexibility."
            )

        # Check for undefined placeholders
        placeholders = extract_placeholders(template.base_prompt)
        required_set = set(template.required_context)
        optional_set = set(template.optional_context)

        for placeholder in placeholders:
            if placeholder not in required_set and placeholder not in optional_set:
                errors.append(
                    f"Placeholder '{placeholder}' not defined in "
                    "required or optional context"
                )

        return {'errors': errors, 'warnings': warnings}

    def _check_prompt_quality(self, template: Template) -> Dict:
        """Check prompt quality metrics."""
        warnings = []
        suggestions = []

        prompt = template.base_prompt

        # Check length
        token_count = estimate_tokens(prompt)
        if token_count > 2000:
            warnings.append(
                f"Prompt is long ({token_count} tokens). "
                "Consider using context compression."
            )

        # Check clarity
        clarity_score = measure_clarity(prompt)
        if clarity_score < 0.7:
            suggestions.append(
                "Prompt clarity could be improved. "
                "Consider simpler language and clearer structure."
            )

        # Check for ambiguity
        ambiguity_score = measure_ambiguity(prompt)
        if ambiguity_score > 0.3:
            warnings.append(
                "Prompt contains ambiguous instructions. "
                "Be more specific."
            )

        return {'warnings': warnings, 'suggestions': suggestions}

    def _check_common_issues(self, template: Template) -> Dict:
        """Check for common template issues."""
        warnings = []

        prompt = template.base_prompt

        # Check for conflicting instructions
        if has_conflicting_instructions(prompt):
            warnings.append("Prompt contains conflicting instructions")

        # Check for redundancy
        redundancy_score = measure_redundancy(prompt)
        if redundancy_score > 0.3:
            warnings.append("Prompt contains redundant information")

        # Check for missing error handling
        if not has_error_handling(prompt):
            warnings.append(
                "Consider adding error handling instructions"
            )

        return {'warnings': warnings}

@dataclass
class ValidationResult:
    """Result of template validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

    def print_report(self):
        """Print validation report."""
        print(f"Validation: {'PASS' if self.is_valid else 'FAIL'}")

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  ❌ {error}")

        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")

        if self.suggestions:
            print("\nSuggestions:")
            for suggestion in self.suggestions:
                print(f"  💡 {suggestion}")
```

### Template Optimization

```python
class TemplateOptimizer:
    """
    Optimize templates for performance and quality.
    """
    def __init__(self):
        self.optimization_strategies: List[OptimizationStrategy] = []

    def optimize_template(
        self,
        template: Template,
        performance_data: Dict
    ) -> Template:
        """
        Optimize template based on performance data.
        """
        optimized = template

        # Apply optimization strategies
        for strategy in self.optimization_strategies:
            if strategy.is_applicable(template, performance_data):
                optimized = strategy.apply(optimized, performance_data)

        return optimized

    def suggest_optimizations(
        self,
        template: Template,
        performance_data: Dict
    ) -> List[str]:
        """
        Suggest potential optimizations.
        """
        suggestions = []

        # Check token usage
        if performance_data.get('avg_tokens') > 3000:
            suggestions.append(
                "Consider context compression to reduce token usage"
            )

        # Check response quality
        if performance_data.get('avg_quality_score', 1.0) < 0.8:
            suggestions.append(
                "Consider adding more specific instructions or examples"
            )

        # Check consistency
        if performance_data.get('consistency_score', 1.0) < 0.9:
            suggestions.append(
                "Add explicit consistency requirements to improve reliability"
            )

        # Check failure rate
        if performance_data.get('failure_rate', 0) > 0.05:
            suggestions.append(
                "Add error handling and fallback instructions"
            )

        return suggestions
```

## Best Practices for Template Systems

### 1. Template Design Principles

```python
# Good: Clear, structured, modular
GOOD_TEMPLATE = """
Role: {role}

Context:
{context}

Task: {task}

Instructions:
{instructions}

Output Format:
{output_format}
"""

# Bad: Unclear structure, mixed concerns
BAD_TEMPLATE = """
You are a {role} and you need to {task} using {context}
and make sure to {instructions} and format it as {output_format}
"""
```

### 2. Version Control for Templates

```python
class VersionedTemplate:
    """Template with version control."""
    def __init__(self, template: Template):
        self.current = template
        self.history: List[TemplateVersion] = []

    def update(
        self,
        new_template: Template,
        change_description: str
    ):
        """Update template with version history."""
        # Save current as historical version
        self.history.append(TemplateVersion(
            template=self.current,
            timestamp=datetime.now(),
            description="Previous version"
        ))

        # Update to new version
        self.current = new_template

        # Log change
        self.history.append(TemplateVersion(
            template=new_template,
            timestamp=datetime.now(),
            description=change_description
        ))
```

### 3. A/B Testing for Templates

```python
class TemplateABTester:
    """A/B test different template versions."""
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}

    def create_experiment(
        self,
        name: str,
        template_a: Template,
        template_b: Template,
        traffic_split: float = 0.5
    ):
        """Create new A/B test."""
        self.experiments[name] = Experiment(
            name=name,
            template_a=template_a,
            template_b=template_b,
            traffic_split=traffic_split,
            results_a=[],
            results_b=[]
        )

    def get_template(self, experiment_name: str) -> Template:
        """Get template for experiment (random selection)."""
        experiment = self.experiments[experiment_name]

        if random.random() < experiment.traffic_split:
            return experiment.template_a
        else:
            return experiment.template_b

    def record_result(
        self,
        experiment_name: str,
        template_id: str,
        performance: float
    ):
        """Record experiment result."""
        experiment = self.experiments[experiment_name]

        if template_id == experiment.template_a.id:
            experiment.results_a.append(performance)
        else:
            experiment.results_b.append(performance)

    def get_winner(self, experiment_name: str) -> Template:
        """Determine winning template."""
        experiment = self.experiments[experiment_name]

        avg_a = np.mean(experiment.results_a)
        avg_b = np.mean(experiment.results_b)

        if avg_a > avg_b:
            return experiment.template_a
        else:
            return experiment.template_b
```

## Key Takeaways

1. **Template Systems Enable Scale**: Transform prompting from art to engineering through systematic templates.

2. **Context-Driven Selection**: Choose templates dynamically based on task and context characteristics.

3. **Validation is Critical**: Always validate templates before deployment to catch issues early.

4. **Continuous Optimization**: Use performance data to continuously improve templates.

5. **Version Control**: Track template changes and maintain history for reliability and debugging.

## What's Next

In Lesson 4, we'll explore structured reasoning and context flow—how to design reasoning frameworks that maintain consistency and build sophisticated AI interactions over extended conversations.

---

**Practice Exercise**: Build a template system for your domain with at least 3 template types, context-driven selection, and validation. Test it with different contexts and measure performance improvements.
