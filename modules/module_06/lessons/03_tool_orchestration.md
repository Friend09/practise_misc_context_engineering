# Lesson 3: Tool Orchestration and Workflow Management

## Introduction

Complex tasks often require coordinating multiple tools in sophisticated workflows. This lesson teaches you to build orchestration systems that manage tool dependencies, execute tools in parallel or sequence, optimize performance, and handle complex multi-step processes reliably.

## Workflow Design and Execution

### Workflow Definition System

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
import asyncio
from datetime import datetime
import uuid

class ExecutionMode(Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowTask:
    """Individual task in a workflow."""
    task_id: str
    tool_id: str
    parameters: Dict[str, Any]

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Execution configuration
    timeout_seconds: float = 60.0
    retry_count: int = 0
    max_retries: int = 3

    # Conditional execution
    condition: Optional[Callable] = None

    # Result handling
    output_mapping: Optional[Dict[str, str]] = None

    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class Workflow:
    """Complete workflow definition."""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]

    # Execution configuration
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout_seconds: float = 300.0

    # Error handling
    continue_on_error: bool = False
    rollback_on_failure: bool = False

    # Context
    input_context: Dict[str, Any] = field(default_factory=dict)
    output_context: Dict[str, Any] = field(default_factory=dict)

    # Status
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class WorkflowEngine:
    """
    Orchestrates workflow execution with dependency management.
    """
    def __init__(self, tool_integration_engine):
        self.tool_engine = tool_integration_engine
        self.active_workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict] = []

    async def execute_workflow(
        self,
        workflow: Workflow,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute complete workflow."""
        workflow.started_at = datetime.now()
        workflow.input_context = context.copy()

        # Store active workflow
        self.active_workflows[workflow.workflow_id] = workflow

        try:
            if workflow.execution_mode == ExecutionMode.SEQUENTIAL:
                result = await self._execute_sequential(workflow, context)
            elif workflow.execution_mode == ExecutionMode.PARALLEL:
                result = await self._execute_parallel(workflow, context)
            elif workflow.execution_mode == ExecutionMode.PIPELINE:
                result = await self._execute_pipeline(workflow, context)
            elif workflow.execution_mode == ExecutionMode.CONDITIONAL:
                result = await self._execute_conditional(workflow, context)
            else:
                raise ValueError(f"Unknown execution mode: {workflow.execution_mode}")

            workflow.completed_at = datetime.now()
            workflow.output_context = result.get('context', {})

            # Record execution
            self._record_execution(workflow, True)

            return result

        except Exception as e:
            workflow.completed_at = datetime.now()

            # Handle rollback
            if workflow.rollback_on_failure:
                await self._rollback_workflow(workflow)

            # Record failed execution
            self._record_execution(workflow, False, str(e))

            return {
                'success': False,
                'error': str(e),
                'workflow_id': workflow.workflow_id
            }

        finally:
            # Clean up
            if workflow.workflow_id in self.active_workflows:
                del self.active_workflows[workflow.workflow_id]

    async def _execute_sequential(
        self,
        workflow: Workflow,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tasks sequentially."""
        execution_context = context.copy()

        for task in workflow.tasks:
            # Check dependencies
            if not self._dependencies_satisfied(task, workflow.tasks):
                task.status = TaskStatus.SKIPPED
                continue

            # Check condition
            if task.condition and not task.condition(execution_context):
                task.status = TaskStatus.SKIPPED
                continue

            # Execute task
            task_result = await self._execute_task(task, execution_context)

            if not task_result['success']:
                if not workflow.continue_on_error:
                    raise Exception(f"Task {task.task_id} failed: {task_result.get('error')}")
            else:
                # Update context with task output
                if task.output_mapping:
                    for output_key, context_key in task.output_mapping.items():
                        if output_key in task_result.get('result', {}):
                            execution_context[context_key] = task_result['result'][output_key]

        return {
            'success': True,
            'context': execution_context,
            'task_results': {task.task_id: task.result for task in workflow.tasks}
        }

    async def _execute_parallel(
        self,
        workflow: Workflow,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute independent tasks in parallel."""
        # Group tasks by dependency level
        dependency_levels = self._compute_dependency_levels(workflow.tasks)
        execution_context = context.copy()

        for level, tasks_at_level in dependency_levels.items():
            # Execute all tasks at this level in parallel
            tasks_to_execute = []

            for task in tasks_at_level:
                # Check condition
                if task.condition and not task.condition(execution_context):
                    task.status = TaskStatus.SKIPPED
                    continue

                tasks_to_execute.append(self._execute_task(task, execution_context))

            if tasks_to_execute:
                # Wait for all tasks at this level
                results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)

                # Process results
                for i, result in enumerate(results):
                    task = tasks_at_level[i]

                    if isinstance(result, Exception):
                        if not workflow.continue_on_error:
                            raise result
                    elif result['success']:
                        # Update context
                        if task.output_mapping:
                            for output_key, context_key in task.output_mapping.items():
                                if output_key in result.get('result', {}):
                                    execution_context[context_key] = result['result'][output_key]

        return {
            'success': True,
            'context': execution_context,
            'task_results': {task.task_id: task.result for task in workflow.tasks}
        }

    async def _execute_pipeline(
        self,
        workflow: Workflow,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tasks as data pipeline."""
        execution_context = context.copy()
        pipeline_data = execution_context.get('pipeline_input')

        for task in workflow.tasks:
            # Check condition
            if task.condition and not task.condition(execution_context):
                task.status = TaskStatus.SKIPPED
                continue

            # Pass pipeline data as parameter
            task_params = task.parameters.copy()
            task_params['input_data'] = pipeline_data

            # Execute task
            original_params = task.parameters
            task.parameters = task_params

            task_result = await self._execute_task(task, execution_context)

            task.parameters = original_params

            if not task_result['success']:
                if not workflow.continue_on_error:
                    raise Exception(f"Pipeline task {task.task_id} failed")
            else:
                # Output becomes input for next task
                pipeline_data = task_result.get('result')

        execution_context['pipeline_output'] = pipeline_data

        return {
            'success': True,
            'context': execution_context,
            'pipeline_output': pipeline_data
        }

    async def _execute_conditional(
        self,
        workflow: Workflow,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tasks based on conditions and branching."""
        execution_context = context.copy()
        executed_tasks = set()

        # Keep executing until no more tasks can be executed
        while True:
            progress_made = False

            for task in workflow.tasks:
                if task.task_id in executed_tasks:
                    continue

                # Check dependencies
                if not self._dependencies_satisfied(task, workflow.tasks, executed_tasks):
                    continue

                # Check condition
                if task.condition and not task.condition(execution_context):
                    task.status = TaskStatus.SKIPPED
                    executed_tasks.add(task.task_id)
                    progress_made = True
                    continue

                # Execute task
                task_result = await self._execute_task(task, execution_context)
                executed_tasks.add(task.task_id)
                progress_made = True

                if not task_result['success']:
                    if not workflow.continue_on_error:
                        raise Exception(f"Task {task.task_id} failed")
                else:
                    # Update context
                    if task.output_mapping:
                        for output_key, context_key in task.output_mapping.items():
                            if output_key in task_result.get('result', {}):
                                execution_context[context_key] = task_result['result'][output_key]

            if not progress_made:
                break

        return {
            'success': True,
            'context': execution_context,
            'task_results': {task.task_id: task.result for task in workflow.tasks}
        }

    async def _execute_task(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single task with retry logic."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        for attempt in range(task.max_retries + 1):
            try:
                # Execute tool
                result = await asyncio.wait_for(
                    self.tool_engine.execute_tool(
                        tool_id=task.tool_id,
                        parameters=task.parameters,
                        context=context
                    ),
                    timeout=task.timeout_seconds
                )

                if result['success']:
                    task.status = TaskStatus.COMPLETED
                    task.result = result.get('result')
                    task.completed_at = datetime.now()
                    return result
                else:
                    task.error = result.get('error', 'Unknown error')

            except asyncio.TimeoutError:
                task.error = "Task timeout"
            except Exception as e:
                task.error = str(e)

            # Retry if not last attempt
            if attempt < task.max_retries:
                task.retry_count += 1
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()

        return {
            'success': False,
            'error': task.error,
            'task_id': task.task_id
        }

    def _dependencies_satisfied(
        self,
        task: WorkflowTask,
        all_tasks: List[WorkflowTask],
        completed_tasks: Optional[Set[str]] = None
    ) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.depends_on:
            return True

        if completed_tasks is None:
            completed_tasks = {
                t.task_id for t in all_tasks
                if t.status == TaskStatus.COMPLETED
            }

        return all(dep_id in completed_tasks for dep_id in task.depends_on)

    def _compute_dependency_levels(
        self,
        tasks: List[WorkflowTask]
    ) -> Dict[int, List[WorkflowTask]]:
        """Compute dependency levels for parallel execution."""
        levels = {}
        task_levels = {}
        task_map = {t.task_id: t for t in tasks}

        def compute_level(task_id: str) -> int:
            if task_id in task_levels:
                return task_levels[task_id]

            task = task_map[task_id]
            if not task.depends_on:
                level = 0
            else:
                level = max(compute_level(dep) for dep in task.depends_on) + 1

            task_levels[task_id] = level
            return level

        # Compute levels for all tasks
        for task in tasks:
            level = compute_level(task.task_id)
            if level not in levels:
                levels[level] = []
            levels[level].append(task)

        return levels

    async def _rollback_workflow(self, workflow: Workflow):
        """Rollback completed tasks in reverse order."""
        completed_tasks = [
            task for task in reversed(workflow.tasks)
            if task.status == TaskStatus.COMPLETED
        ]

        for task in completed_tasks:
            # Check if tool supports rollback
            tool = self.tool_engine.registry.get_tool(task.tool_id)
            if tool and hasattr(tool.executor, 'rollback'):
                try:
                    await tool.executor.rollback(task.result)
                except Exception as e:
                    print(f"Rollback failed for task {task.task_id}: {e}")

    def _record_execution(
        self,
        workflow: Workflow,
        success: bool,
        error: Optional[str] = None
    ):
        """Record workflow execution."""
        execution_time = None
        if workflow.started_at and workflow.completed_at:
            execution_time = (workflow.completed_at - workflow.started_at).total_seconds()

        self.execution_history.append({
            'workflow_id': workflow.workflow_id,
            'workflow_name': workflow.name,
            'success': success,
            'error': error,
            'execution_time_seconds': execution_time,
            'task_count': len(workflow.tasks),
            'completed_tasks': len([t for t in workflow.tasks if t.status == TaskStatus.COMPLETED]),
            'failed_tasks': len([t for t in workflow.tasks if t.status == TaskStatus.FAILED]),
            'timestamp': datetime.now()
        })
```

## Dependency Management

### Advanced Dependency Resolver

```python
from typing import Tuple, Union

class DependencyResolver:
    """
    Resolves complex dependencies between workflow tasks.
    """
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_graph: Dict[str, Set[str]] = {}

    def add_dependency(self, task_id: str, depends_on: str):
        """Add dependency relationship."""
        if task_id not in self.dependency_graph:
            self.dependency_graph[task_id] = set()
        self.dependency_graph[task_id].add(depends_on)

        if depends_on not in self.reverse_graph:
            self.reverse_graph[depends_on] = set()
        self.reverse_graph[depends_on].add(task_id)

    def detect_cycles(self) -> List[List[str]]:
        """Detect dependency cycles."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.dependency_graph.get(node, []):
                if dfs(neighbor):
                    return True

            rec_stack.remove(node)
            path.pop()
            return False

        for node in self.dependency_graph:
            if node not in visited:
                dfs(node)

        return cycles

    def topological_sort(self) -> List[str]:
        """Get topological order of tasks."""
        in_degree = {}

        # Calculate in-degrees
        for node in self.dependency_graph:
            in_degree[node] = len(self.dependency_graph[node])

        # Add nodes with no dependencies
        for node in self.reverse_graph:
            if node not in in_degree:
                in_degree[node] = 0

        # Find nodes with no incoming edges
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree of dependent nodes
            for dependent in self.reverse_graph.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result

    def find_critical_path(self, task_durations: Dict[str, float]) -> Tuple[List[str], float]:
        """Find critical path for workflow optimization."""
        # Use longest path algorithm
        distances = {node: 0 for node in self.dependency_graph}
        predecessors = {}

        # Relax edges in topological order
        for node in self.topological_sort():
            for dependent in self.reverse_graph.get(node, []):
                new_distance = distances[node] + task_durations.get(node, 0)
                if new_distance > distances[dependent]:
                    distances[dependent] = new_distance
                    predecessors[dependent] = node

        # Find end node with maximum distance
        end_node = max(distances, key=distances.get)
        max_distance = distances[end_node]

        # Reconstruct path
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = predecessors.get(current)

        path.reverse()
        return path, max_distance

class ResourceManager:
    """
    Manages resource allocation for workflow execution.
    """
    def __init__(self):
        self.resource_pools: Dict[str, int] = {}
        self.allocated_resources: Dict[str, int] = {}
        self.waiting_tasks: List[Dict] = []

    def register_resource_pool(self, resource_type: str, capacity: int):
        """Register resource pool."""
        self.resource_pools[resource_type] = capacity
        self.allocated_resources[resource_type] = 0

    async def acquire_resources(
        self,
        task_id: str,
        required_resources: Dict[str, int]
    ) -> bool:
        """Acquire resources for task execution."""
        # Check if resources are available
        for resource_type, amount in required_resources.items():
            if resource_type not in self.resource_pools:
                return False

            available = self.resource_pools[resource_type] - self.allocated_resources[resource_type]
            if available < amount:
                # Add to waiting queue
                self.waiting_tasks.append({
                    'task_id': task_id,
                    'resources': required_resources,
                    'timestamp': datetime.now()
                })
                return False

        # Allocate resources
        for resource_type, amount in required_resources.items():
            self.allocated_resources[resource_type] += amount

        return True

    def release_resources(
        self,
        task_id: str,
        released_resources: Dict[str, int]
    ):
        """Release resources after task completion."""
        # Release resources
        for resource_type, amount in released_resources.items():
            if resource_type in self.allocated_resources:
                self.allocated_resources[resource_type] -= amount
                self.allocated_resources[resource_type] = max(0, self.allocated_resources[resource_type])

        # Check waiting tasks
        self._process_waiting_tasks()

    def _process_waiting_tasks(self):
        """Process tasks waiting for resources."""
        ready_tasks = []

        for i, waiting_task in enumerate(self.waiting_tasks):
            can_allocate = True

            for resource_type, amount in waiting_task['resources'].items():
                available = self.resource_pools[resource_type] - self.allocated_resources[resource_type]
                if available < amount:
                    can_allocate = False
                    break

            if can_allocate:
                ready_tasks.append(i)

        # Remove ready tasks from waiting list (in reverse order)
        for i in reversed(ready_tasks):
            self.waiting_tasks.pop(i)
```

## Performance Optimization

### Workflow Optimizer

```python
class WorkflowOptimizer:
    """
    Optimizes workflow execution for performance.
    """
    def __init__(self):
        self.execution_stats: Dict[str, List[float]] = {}
        self.optimization_history: List[Dict] = []

    def analyze_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Analyze workflow for optimization opportunities."""
        analysis = {
            'total_tasks': len(workflow.tasks),
            'dependency_levels': len(self._compute_dependency_levels(workflow.tasks)),
            'parallel_opportunities': 0,
            'bottlenecks': [],
            'optimization_suggestions': []
        }

        # Find parallel opportunities
        levels = self._compute_dependency_levels(workflow.tasks)
        for level, tasks in levels.items():
            if len(tasks) > 1:
                analysis['parallel_opportunities'] += len(tasks) - 1

        # Identify bottlenecks
        for task in workflow.tasks:
            avg_time = self._get_average_execution_time(task.tool_id)
            if avg_time > 5000:  # > 5 seconds
                analysis['bottlenecks'].append({
                    'task_id': task.task_id,
                    'tool_id': task.tool_id,
                    'avg_time_ms': avg_time
                })

        # Generate suggestions
        if analysis['parallel_opportunities'] > 0:
            analysis['optimization_suggestions'].append(
                f"Consider parallel execution mode for {analysis['parallel_opportunities']} tasks"
            )

        if analysis['bottlenecks']:
            analysis['optimization_suggestions'].append(
                f"Optimize {len(analysis['bottlenecks'])} slow tasks or increase timeouts"
            )

        return analysis

    def optimize_workflow(self, workflow: Workflow) -> Workflow:
        """Optimize workflow configuration."""
        optimized = workflow

        # Suggest optimal execution mode
        levels = self._compute_dependency_levels(workflow.tasks)
        max_parallel = max(len(tasks) for tasks in levels.values())

        if max_parallel > 1 and workflow.execution_mode == ExecutionMode.SEQUENTIAL:
            optimized.execution_mode = ExecutionMode.PARALLEL

        # Optimize timeouts
        for task in optimized.tasks:
            avg_time = self._get_average_execution_time(task.tool_id)
            if avg_time > 0:
                # Set timeout to 3x average time, minimum 30 seconds
                optimal_timeout = max(30, avg_time / 1000 * 3)
                task.timeout_seconds = optimal_timeout

        # Record optimization
        self.optimization_history.append({
            'workflow_id': workflow.workflow_id,
            'original_mode': workflow.execution_mode,
            'optimized_mode': optimized.execution_mode,
            'timestamp': datetime.now()
        })

        return optimized

    def _get_average_execution_time(self, tool_id: str) -> float:
        """Get average execution time for tool."""
        if tool_id not in self.execution_stats:
            return 1000.0  # Default 1 second

        times = self.execution_stats[tool_id]
        return sum(times) / len(times) if times else 1000.0

    def _compute_dependency_levels(self, tasks: List[WorkflowTask]) -> Dict[int, List[WorkflowTask]]:
        """Compute dependency levels."""
        # Same implementation as in WorkflowEngine
        levels = {}
        task_levels = {}
        task_map = {t.task_id: t for t in tasks}

        def compute_level(task_id: str) -> int:
            if task_id in task_levels:
                return task_levels[task_id]

            task = task_map[task_id]
            if not task.depends_on:
                level = 0
            else:
                level = max(compute_level(dep) for dep in task.depends_on) + 1

            task_levels[task_id] = level
            return level

        for task in tasks:
            level = compute_level(task.task_id)
            if level not in levels:
                levels[level] = []
            levels[level].append(task)

        return levels
```

## Key Takeaways

1. **Workflow Definition**: Comprehensive task and dependency modeling

2. **Multiple Execution Modes**: Sequential, parallel, pipeline, conditional

3. **Dependency Resolution**: Automatic topological sorting and cycle detection

4. **Resource Management**: Pool-based resource allocation and queuing

5. **Performance Optimization**: Critical path analysis and automatic tuning

6. **Error Handling**: Retry logic, rollback, and graceful degradation

## What's Next

In Lesson 4, we'll explore comprehensive error handling and resilience patterns for robust tool integration.

---

**Practice Exercise**: Build a complete workflow orchestration system supporting all execution modes. Create workflows with 10+ tasks, complex dependencies, and resource constraints. Demonstrate parallel execution achieving 60%+ performance improvement over sequential execution for appropriate workflows.
