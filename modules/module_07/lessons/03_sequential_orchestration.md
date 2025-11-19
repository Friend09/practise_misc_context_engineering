# Lesson 3: Sequential Task Orchestration

## Introduction

Single-agent systems excel at sequential task orchestration when context is properly managed across task boundaries. This lesson teaches you to build sophisticated workflow engines that maintain context continuity, preserve decision history, and adapt execution strategies based on accumulated knowledge.

## Sequential Task Architecture

### Core Orchestration Engine

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import logging
from abc import ABC, abstractmethod

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class TaskContext:
    """Context passed between tasks in a sequence."""
    task_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_results: List[Dict] = field(default_factory=list)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[Dict] = field(default_factory=list)

    def add_result(self, task_id: str, result: Any, metadata: Dict = None):
        """Add result from a completed task."""
        self.previous_results.append({
            'task_id': task_id,
            'result': result,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        })

    def get_result(self, task_id: str) -> Optional[Any]:
        """Get result from a specific task."""
        for result in self.previous_results:
            if result['task_id'] == task_id:
                return result['result']
        return None

    def update_shared_state(self, key: str, value: Any):
        """Update shared state accessible to all tasks."""
        self.shared_state[key] = value

    def log_execution(self, task_id: str, action: str, details: Dict = None):
        """Log execution step for debugging and analysis."""
        self.execution_trace.append({
            'task_id': task_id,
            'action': action,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })

@dataclass
class Task:
    """Individual task in a sequence."""
    id: str
    name: str
    handler: Callable
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    context_requirements: List[str] = field(default_factory=list)
    output_context_keys: List[str] = field(default_factory=list)

    # Execution state
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Any = None

class ContextPreservationStrategy(ABC):
    """Abstract base class for context preservation strategies."""

    @abstractmethod
    def preserve_context(self, context: TaskContext, task: Task) -> TaskContext:
        """Preserve context before task execution."""
        pass

    @abstractmethod
    def restore_context(self, context: TaskContext, task: Task) -> TaskContext:
        """Restore context after task execution."""
        pass

class IncrementalContextStrategy(ContextPreservationStrategy):
    """Preserve context incrementally, building knowledge over time."""

    def __init__(self, max_context_items: int = 1000):
        self.max_context_items = max_context_items
        self.context_importance_scores: Dict[str, float] = {}

    def preserve_context(self, context: TaskContext, task: Task) -> TaskContext:
        """Preserve context with incremental learning."""
        # Calculate importance score for current context
        importance_score = self._calculate_importance(context, task)
        context_key = f"{task.id}_input"
        self.context_importance_scores[context_key] = importance_score

        # Create preservation snapshot
        preservation_data = {
            'input_data': context.data.copy(),
            'shared_state_snapshot': context.shared_state.copy(),
            'task_dependencies': task.dependencies,
            'context_requirements': task.context_requirements,
            'importance_score': importance_score
        }

        context.metadata[f"{task.id}_preservation"] = preservation_data
        context.log_execution(task.id, "context_preserved", preservation_data)

        return context

    def restore_context(self, context: TaskContext, task: Task) -> TaskContext:
        """Restore and enrich context after task execution."""
        # Get preservation data
        preservation_key = f"{task.id}_preservation"
        preservation_data = context.metadata.get(preservation_key)

        if preservation_data:
            # Enhance context with execution results
            enhanced_context = {
                'pre_execution_state': preservation_data['shared_state_snapshot'],
                'post_execution_state': context.shared_state.copy(),
                'task_output': task.result,
                'execution_duration': self._calculate_duration(task),
                'success': task.status == TaskStatus.COMPLETED
            }

            context.metadata[f"{task.id}_enhanced"] = enhanced_context
            context.log_execution(task.id, "context_restored", enhanced_context)

        # Prune old context if necessary
        self._prune_context(context)

        return context

    def _calculate_importance(self, context: TaskContext, task: Task) -> float:
        """Calculate importance score for context preservation."""
        score = 0.5  # Base score

        # High priority tasks are more important
        if task.priority == TaskPriority.CRITICAL:
            score += 0.3
        elif task.priority == TaskPriority.HIGH:
            score += 0.2

        # Tasks with many dependencies are important
        score += min(0.2, len(task.dependencies) * 0.05)

        # Tasks that modify shared state are important
        if len(context.shared_state) > 0:
            score += 0.1

        # Tasks with specific context requirements are important
        score += min(0.1, len(task.context_requirements) * 0.02)

        return min(1.0, score)

    def _calculate_duration(self, task: Task) -> float:
        """Calculate task execution duration."""
        if task.start_time and task.end_time:
            return (task.end_time - task.start_time).total_seconds()
        return 0.0

    def _prune_context(self, context: TaskContext):
        """Remove low-importance context to stay within limits."""
        if len(context.metadata) <= self.max_context_items:
            return

        # Sort context items by importance
        items_with_scores = []
        for key, value in context.metadata.items():
            if key.endswith('_preservation'):
                task_id = key.replace('_preservation', '')
                score = self.context_importance_scores.get(f"{task_id}_input", 0.0)
                items_with_scores.append((key, score))

        # Remove lowest importance items
        items_with_scores.sort(key=lambda x: x[1])
        items_to_remove = len(items_with_scores) - self.max_context_items + 100  # Keep buffer

        for i in range(items_to_remove):
            key_to_remove = items_with_scores[i][0]
            if key_to_remove in context.metadata:
                del context.metadata[key_to_remove]

class SequentialTaskOrchestrator:
    """
    Orchestrates sequential task execution with context preservation.
    """

    def __init__(
        self,
        context_strategy: ContextPreservationStrategy = None,
        max_concurrent_tasks: int = 1,  # Sequential by default
        enable_adaptive_execution: bool = True
    ):
        self.tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []
        self.context_strategy = context_strategy or IncrementalContextStrategy()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_adaptive_execution = enable_adaptive_execution

        # Execution state
        self.current_context: Optional[TaskContext] = None
        self.execution_stats: Dict[str, Any] = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'context_preservation_overhead': 0.0
        }

        # Adaptive execution learning
        self.task_performance_history: Dict[str, List[float]] = {}
        self.context_efficiency_scores: Dict[str, float] = {}

    def add_task(
        self,
        task_id: str,
        name: str,
        handler: Callable,
        dependencies: List[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> Task:
        """Add a task to the orchestration sequence."""
        task = Task(
            id=task_id,
            name=name,
            handler=handler,
            dependencies=dependencies or [],
            priority=priority,
            **kwargs
        )

        self.tasks[task_id] = task
        self._update_execution_order()

        return task

    def _update_execution_order(self):
        """Update execution order based on dependencies and priorities."""
        # Topological sort with priority consideration
        in_degree = {task_id: 0 for task_id in self.tasks}
        graph = {task_id: [] for task_id in self.tasks}

        # Build dependency graph
        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                if dep in self.tasks:
                    graph[dep].append(task_id)
                    in_degree[task_id] += 1

        # Priority queue for tasks with no dependencies
        ready_tasks = []
        for task_id in self.tasks:
            if in_degree[task_id] == 0:
                ready_tasks.append((self.tasks[task_id].priority.value, task_id))

        ready_tasks.sort()  # Sort by priority
        execution_order = []

        while ready_tasks:
            _, current_task_id = ready_tasks.pop(0)
            execution_order.append(current_task_id)

            # Update dependencies
            for next_task_id in graph[current_task_id]:
                in_degree[next_task_id] -= 1
                if in_degree[next_task_id] == 0:
                    priority = self.tasks[next_task_id].priority.value
                    ready_tasks.append((priority, next_task_id))
                    ready_tasks.sort()

        self.execution_order = execution_order

    async def execute_sequence(
        self,
        initial_context: Dict[str, Any],
        sequence_id: str = None
    ) -> TaskContext:
        """Execute the complete task sequence."""
        sequence_id = sequence_id or f"sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize context
        self.current_context = TaskContext(
            task_id=sequence_id,
            data=initial_context.copy(),
            metadata={
                'sequence_id': sequence_id,
                'start_time': datetime.now().isoformat(),
                'orchestrator_config': {
                    'max_concurrent_tasks': self.max_concurrent_tasks,
                    'adaptive_execution': self.enable_adaptive_execution
                }
            }
        )

        try:
            # Execute tasks in order
            for task_id in self.execution_order:
                task = self.tasks[task_id]

                # Check dependencies
                if not self._dependencies_satisfied(task):
                    task.status = TaskStatus.FAILED
                    task.error_message = f"Dependencies not satisfied: {task.dependencies}"
                    continue

                # Execute task with context preservation
                await self._execute_task_with_context(task)

                # Update execution stats
                self._update_execution_stats(task)

                # Adaptive execution optimization
                if self.enable_adaptive_execution:
                    self._learn_from_execution(task)

                # Stop on critical task failure
                if task.status == TaskStatus.FAILED and task.priority == TaskPriority.CRITICAL:
                    break

            # Finalize context
            self.current_context.metadata['end_time'] = datetime.now().isoformat()
            self.current_context.metadata['execution_summary'] = self._create_execution_summary()

            return self.current_context

        except Exception as e:
            logging.error(f"Sequence execution failed: {e}")
            self.current_context.metadata['execution_error'] = str(e)
            raise

    async def _execute_task_with_context(self, task: Task):
        """Execute a single task with full context preservation."""
        preservation_start = datetime.now()

        # Preserve context before execution
        self.current_context = self.context_strategy.preserve_context(
            self.current_context, task
        )

        preservation_time = (datetime.now() - preservation_start).total_seconds()
        self.execution_stats['context_preservation_overhead'] += preservation_time

        # Execute the task
        await self._execute_task(task)

        # Restore and enhance context after execution
        restoration_start = datetime.now()
        self.current_context = self.context_strategy.restore_context(
            self.current_context, task
        )
        restoration_time = (datetime.now() - restoration_start).total_seconds()
        self.execution_stats['context_preservation_overhead'] += restoration_time

        # Add task result to context
        if task.status == TaskStatus.COMPLETED:
            self.current_context.add_result(task.id, task.result, {
                'execution_time': self._calculate_task_duration(task),
                'retry_count': task.retry_count,
                'context_keys_used': task.context_requirements,
                'context_keys_produced': task.output_context_keys
            })

            # Update shared state with task outputs
            for key in task.output_context_keys:
                if hasattr(task.result, 'get') and key in task.result:
                    self.current_context.update_shared_state(key, task.result[key])

    async def _execute_task(self, task: Task):
        """Execute individual task with error handling and retries."""
        for attempt in range(task.max_retries + 1):
            try:
                task.status = TaskStatus.RUNNING
                task.start_time = datetime.now()
                task.retry_count = attempt

                # Prepare task input from context
                task_input = self._prepare_task_input(task)

                # Execute task handler
                if asyncio.iscoroutinefunction(task.handler):
                    task.result = await task.handler(task_input)
                else:
                    task.result = task.handler(task_input)

                task.status = TaskStatus.COMPLETED
                task.end_time = datetime.now()

                self.current_context.log_execution(
                    task.id, "task_completed",
                    {'result': str(task.result)[:200], 'attempt': attempt + 1}
                )
                break

            except Exception as e:
                task.error_message = str(e)

                if attempt < task.max_retries:
                    task.status = TaskStatus.RETRYING
                    self.current_context.log_execution(
                        task.id, "task_retry",
                        {'error': str(e), 'attempt': attempt + 1}
                    )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    task.status = TaskStatus.FAILED
                    task.end_time = datetime.now()
                    self.current_context.log_execution(
                        task.id, "task_failed",
                        {'error': str(e), 'total_attempts': attempt + 1}
                    )

    def _prepare_task_input(self, task: Task) -> Dict[str, Any]:
        """Prepare input for task execution from current context."""
        task_input = {
            'context': self.current_context,
            'shared_state': self.current_context.shared_state,
            'previous_results': self.current_context.previous_results
        }

        # Add specific context requirements
        for req_key in task.context_requirements:
            if req_key in self.current_context.data:
                task_input[req_key] = self.current_context.data[req_key]
            elif req_key in self.current_context.shared_state:
                task_input[req_key] = self.current_context.shared_state[req_key]

        # Add results from dependencies
        for dep_task_id in task.dependencies:
            dep_result = self.current_context.get_result(dep_task_id)
            if dep_result is not None:
                task_input[f"{dep_task_id}_result"] = dep_result

        return task_input

    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.tasks:
                return False

            dep_task = self.tasks[dep_task_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False

        return True

    def _calculate_task_duration(self, task: Task) -> float:
        """Calculate task execution duration."""
        if task.start_time and task.end_time:
            return (task.end_time - task.start_time).total_seconds()
        return 0.0

    def _update_execution_stats(self, task: Task):
        """Update execution statistics."""
        self.execution_stats['total_tasks'] += 1

        if task.status == TaskStatus.COMPLETED:
            self.execution_stats['successful_tasks'] += 1
        elif task.status == TaskStatus.FAILED:
            self.execution_stats['failed_tasks'] += 1

        duration = self._calculate_task_duration(task)
        self.execution_stats['total_execution_time'] += duration

    def _learn_from_execution(self, task: Task):
        """Learn from task execution for adaptive optimization."""
        task_duration = self._calculate_task_duration(task)

        # Update performance history
        if task.id not in self.task_performance_history:
            self.task_performance_history[task.id] = []

        self.task_performance_history[task.id].append(task_duration)

        # Keep only recent history
        if len(self.task_performance_history[task.id]) > 10:
            self.task_performance_history[task.id] = self.task_performance_history[task.id][-10:]

        # Calculate context efficiency
        context_size = len(json.dumps(self.current_context.data))
        context_efficiency = 1.0 / (1.0 + context_size / 10000)  # Penalty for large context
        self.context_efficiency_scores[task.id] = context_efficiency

    def _create_execution_summary(self) -> Dict[str, Any]:
        """Create summary of sequence execution."""
        total_tasks = len(self.tasks)
        successful_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)

        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            'total_execution_time': self.execution_stats['total_execution_time'],
            'context_preservation_overhead': self.execution_stats['context_preservation_overhead'],
            'average_task_duration': (
                self.execution_stats['total_execution_time'] / total_tasks
                if total_tasks > 0 else 0.0
            ),
            'context_efficiency': (
                sum(self.context_efficiency_scores.values()) / len(self.context_efficiency_scores)
                if self.context_efficiency_scores else 0.0
            )
        }

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive execution metrics."""
        return {
            'execution_stats': self.execution_stats,
            'task_performance': self.task_performance_history,
            'context_efficiency': self.context_efficiency_scores,
            'current_context_size': len(json.dumps(self.current_context.data)) if self.current_context else 0,
            'task_status_distribution': {
                status.value: sum(1 for task in self.tasks.values() if task.status == status)
                for status in TaskStatus
            }
        }

# Example usage and implementation
async def example_data_processing_sequence():
    """Example of complex data processing sequence."""

    # Create orchestrator with adaptive execution
    orchestrator = SequentialTaskOrchestrator(
        context_strategy=IncrementalContextStrategy(max_context_items=500),
        enable_adaptive_execution=True
    )

    # Define task handlers
    async def load_data(task_input):
        """Load initial dataset."""
        data_source = task_input.get('data_source', 'default.csv')
        # Simulate data loading
        await asyncio.sleep(0.1)
        return {
            'dataset': f"loaded_data_from_{data_source}",
            'record_count': 10000,
            'columns': ['id', 'name', 'value', 'category']
        }

    async def validate_data(task_input):
        """Validate loaded data."""
        load_result = task_input.get('load_data_result')
        if not load_result:
            raise ValueError("No data to validate")

        # Simulate validation
        await asyncio.sleep(0.05)
        validation_errors = []  # No errors in this example

        return {
            'is_valid': len(validation_errors) == 0,
            'errors': validation_errors,
            'validation_timestamp': datetime.now().isoformat()
        }

    async def transform_data(task_input):
        """Transform validated data."""
        load_result = task_input.get('load_data_result')
        validation_result = task_input.get('validate_data_result')

        if not validation_result.get('is_valid'):
            raise ValueError("Cannot transform invalid data")

        # Simulate transformation
        await asyncio.sleep(0.2)
        return {
            'transformed_dataset': f"transformed_{load_result['dataset']}",
            'transformation_applied': ['normalize_values', 'clean_names'],
            'output_record_count': load_result['record_count']
        }

    async def analyze_data(task_input):
        """Analyze transformed data."""
        transform_result = task_input.get('transform_data_result')

        # Simulate analysis
        await asyncio.sleep(0.15)
        return {
            'analysis_results': {
                'mean_value': 42.5,
                'category_distribution': {'A': 0.3, 'B': 0.4, 'C': 0.3},
                'quality_score': 0.95
            },
            'insights': [
                'Data quality is excellent',
                'Category distribution is balanced',
                'No significant outliers detected'
            ]
        }

    async def generate_report(task_input):
        """Generate final report."""
        analysis_result = task_input.get('analyze_data_result')

        # Simulate report generation
        await asyncio.sleep(0.1)
        return {
            'report_generated': True,
            'report_sections': ['executive_summary', 'detailed_analysis', 'recommendations'],
            'quality_score': analysis_result['analysis_results']['quality_score']
        }

    # Add tasks to orchestrator
    orchestrator.add_task(
        'load_data', 'Load Dataset', load_data,
        priority=TaskPriority.HIGH,
        context_requirements=['data_source'],
        output_context_keys=['dataset_info']
    )

    orchestrator.add_task(
        'validate_data', 'Validate Data', validate_data,
        dependencies=['load_data'],
        priority=TaskPriority.CRITICAL,
        output_context_keys=['validation_status']
    )

    orchestrator.add_task(
        'transform_data', 'Transform Data', transform_data,
        dependencies=['validate_data'],
        priority=TaskPriority.HIGH,
        output_context_keys=['transformed_data_info']
    )

    orchestrator.add_task(
        'analyze_data', 'Analyze Data', analyze_data,
        dependencies=['transform_data'],
        priority=TaskPriority.NORMAL,
        output_context_keys=['analysis_insights']
    )

    orchestrator.add_task(
        'generate_report', 'Generate Report', generate_report,
        dependencies=['analyze_data'],
        priority=TaskPriority.NORMAL
    )

    # Execute sequence
    initial_context = {
        'data_source': 'customer_data.csv',
        'output_format': 'comprehensive_report',
        'quality_threshold': 0.9
    }

    result_context = await orchestrator.execute_sequence(initial_context)

    # Display results
    print("Execution Summary:")
    print(json.dumps(result_context.metadata['execution_summary'], indent=2))

    print("\nFinal Results:")
    for result in result_context.previous_results:
        print(f"Task {result['task_id']}: {result['result']}")

# Run the example
if __name__ == "__main__":
    asyncio.run(example_data_processing_sequence())
```

## Key Takeaways

1. **Context Continuity**: Maintain context across task boundaries while preserving decision history

2. **Dependency Management**: Handle complex task dependencies with priority-based execution

3. **Adaptive Execution**: Learn from execution patterns to optimize future sequences

4. **Error Recovery**: Implement robust error handling with intelligent retry strategies

5. **Performance Monitoring**: Track execution metrics for continuous improvement

6. **Context Preservation**: Balance context richness with performance overhead

## What's Next

In Lesson 4, we'll explore self-improving single-agent systems that learn and adapt their strategies based on accumulated experience.

---

**Practice Exercise**: Build a sequential orchestrator handling 20+ interdependent tasks with context preservation. Demonstrate <100ms task switching overhead while maintaining 95%+ context accuracy across the complete sequence.
