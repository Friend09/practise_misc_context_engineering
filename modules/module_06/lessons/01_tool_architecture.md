# Lesson 1: Tool Architecture and Context Integration

## Introduction

AI agents gain real power when they can interact with external systems through tools. This lesson teaches you to build sophisticated tool architectures that select and integrate tools based on current context, creating agents that can access databases, call APIs, execute code, and interact with services intelligently.

## Tool Registry and Discovery

### Tool Capability Modeling

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import inspect

class ToolCategory(Enum):
    """Categories of tools."""
    DATA_RETRIEVAL = "data_retrieval"
    DATA_PROCESSING = "data_processing"
    EXTERNAL_API = "external_api"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATIONS = "file_operations"
    DATABASE = "database"
    COMMUNICATION = "communication"

@dataclass
class ToolParameter:
    """Parameter specification for a tool."""
    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None
    validation: Optional[Callable] = None

@dataclass
class ToolCapability:
    """Comprehensive tool capability description."""
    tool_id: str
    name: str
    description: str
    category: ToolCategory

    # Input/output specifications
    parameters: List[ToolParameter]
    return_type: type

    # Context requirements
    required_context: List[str] = field(default_factory=list)
    optional_context: List[str] = field(default_factory=list)

    # Performance characteristics
    avg_execution_time_ms: float = 100.0
    max_execution_time_ms: float = 5000.0
    reliability_score: float = 0.95  # 0.0 to 1.0

    # Cost profile
    cost_per_execution: float = 0.0
    requires_authentication: bool = False
    rate_limit_per_minute: Optional[int] = None

    # Metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class Tool:
    """Executable tool with capability specification."""
    capability: ToolCapability
    executor: Callable

    # Execution state
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time_ms: float = 0.0
    last_executed: Optional[datetime] = None

class ToolRegistry:
    """
    Central registry for all available tools with discovery capabilities.
    """
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.category_index: Dict[ToolCategory, List[str]] = {}
        self.tag_index: Dict[str, List[str]] = {}
        self.context_index: Dict[str, List[str]] = {}

    def register_tool(
        self,
        capability: ToolCapability,
        executor: Callable
    ) -> Tool:
        """Register a new tool."""
        tool = Tool(capability=capability, executor=executor)

        # Store in main registry
        self.tools[capability.tool_id] = tool

        # Index by category
        if capability.category not in self.category_index:
            self.category_index[capability.category] = []
        self.category_index[capability.category].append(capability.tool_id)

        # Index by tags
        for tag in capability.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(capability.tool_id)

        # Index by context requirements
        for context_key in capability.required_context:
            if context_key not in self.context_index:
                self.context_index[context_key] = []
            self.context_index[context_key].append(capability.tool_id)

        return tool

    def discover_tools(
        self,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None,
        context_available: Optional[List[str]] = None
    ) -> List[Tool]:
        """Discover tools matching criteria."""
        candidate_ids = set(self.tools.keys())

        # Filter by category
        if category:
            candidate_ids &= set(self.category_index.get(category, []))

        # Filter by tags
        if tags:
            for tag in tags:
                candidate_ids &= set(self.tag_index.get(tag, []))

        # Filter by context availability
        if context_available:
            available_set = set(context_available)
            filtered_ids = set()

            for tool_id in candidate_ids:
                tool = self.tools[tool_id]
                required = set(tool.capability.required_context)

                # Check if all required context is available
                if required.issubset(available_set):
                    filtered_ids.add(tool_id)

            candidate_ids = filtered_ids

        return [self.tools[tool_id] for tool_id in candidate_ids]

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get tool by ID."""
        return self.tools.get(tool_id)

    def get_tool_stats(self, tool_id: str) -> Dict:
        """Get execution statistics for a tool."""
        tool = self.tools.get(tool_id)
        if not tool:
            return {}

        success_rate = 0.0
        if tool.total_executions > 0:
            success_rate = tool.successful_executions / tool.total_executions

        avg_time = 0.0
        if tool.successful_executions > 0:
            avg_time = tool.total_execution_time_ms / tool.successful_executions

        return {
            'tool_id': tool_id,
            'total_executions': tool.total_executions,
            'success_rate': success_rate,
            'avg_execution_time_ms': avg_time,
            'last_executed': tool.last_executed
        }
```

## Context-Aware Tool Selection

### Intelligent Selection Algorithm

```python
import numpy as np

class ContextAwareToolSelector:
    """
    Selects optimal tools based on task requirements and current context.
    """
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.selection_history: List[Dict] = []

    def select_tools(
        self,
        task_description: str,
        context: Dict[str, Any],
        max_tools: int = 5
    ) -> List[Tuple[Tool, float]]:
        """
        Select most appropriate tools for task given context.
        Returns list of (Tool, score) tuples.
        """
        # Extract context features
        available_context = list(context.keys())

        # Discover candidate tools
        candidates = self.registry.discover_tools(
            context_available=available_context
        )

        if not candidates:
            return []

        # Score each tool
        scored_tools = []
        for tool in candidates:
            score = self._score_tool(tool, task_description, context)
            scored_tools.append((tool, score))

        # Sort by score
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        # Return top tools
        return scored_tools[:max_tools]

    def _score_tool(
        self,
        tool: Tool,
        task_description: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate appropriateness score for tool."""
        score = 0.0

        # 1. Semantic relevance (simplified keyword matching)
        relevance_score = self._calculate_semantic_relevance(
            tool.capability.description,
            task_description
        )
        score += relevance_score * 0.4

        # 2. Context alignment
        context_score = self._calculate_context_alignment(
            tool.capability,
            context
        )
        score += context_score * 0.25

        # 3. Reliability
        reliability = tool.capability.reliability_score
        score += reliability * 0.2

        # 4. Performance
        if tool.capability.avg_execution_time_ms < 500:
            performance_score = 1.0
        elif tool.capability.avg_execution_time_ms < 2000:
            performance_score = 0.7
        else:
            performance_score = 0.4
        score += performance_score * 0.15

        return score

    def _calculate_semantic_relevance(
        self,
        tool_description: str,
        task_description: str
    ) -> float:
        """Calculate semantic relevance between tool and task."""
        # Simple keyword-based relevance (in production, use embeddings)
        tool_words = set(tool_description.lower().split())
        task_words = set(task_description.lower().split())

        if not task_words:
            return 0.0

        # Jaccard similarity
        intersection = tool_words & task_words
        union = tool_words | task_words

        return len(intersection) / len(union) if union else 0.0

    def _calculate_context_alignment(
        self,
        capability: ToolCapability,
        context: Dict[str, Any]
    ) -> float:
        """Calculate how well context matches tool requirements."""
        # Check required context
        required = set(capability.required_context)
        available = set(context.keys())

        if not required:
            return 1.0  # No requirements, perfect alignment

        # Must have all required context
        if not required.issubset(available):
            return 0.0

        # Bonus for optional context
        optional = set(capability.optional_context)
        available_optional = optional & available

        optional_bonus = len(available_optional) / len(optional) if optional else 0

        return 1.0 + (optional_bonus * 0.2)  # Up to 20% bonus

    def record_selection(
        self,
        task: str,
        context: Dict,
        selected_tool: Tool,
        success: bool,
        execution_time_ms: float
    ):
        """Record tool selection outcome for learning."""
        self.selection_history.append({
            'task': task,
            'context_keys': list(context.keys()),
            'tool_id': selected_tool.capability.tool_id,
            'success': success,
            'execution_time_ms': execution_time_ms,
            'timestamp': datetime.now()
        })
```

## Tool Integration Patterns

### Universal Tool Adapter

```python
from abc import ABC, abstractmethod
import asyncio

class ToolAdapter(ABC):
    """Base class for tool adapters."""

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute tool with parameters and context."""
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters before execution."""
        pass

class SyncFunctionAdapter(ToolAdapter):
    """Adapter for synchronous Python functions."""

    def __init__(self, func: Callable, capability: ToolCapability):
        self.func = func
        self.capability = capability

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute synchronous function."""
        if not self.validate_parameters(parameters):
            raise ValueError("Invalid parameters")

        # Run in executor to not block async loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.func,
            **parameters
        )

        return result

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against capability spec."""
        # Check required parameters
        for param in self.capability.parameters:
            if param.required and param.name not in parameters:
                return False

            # Check type
            if param.name in parameters:
                value = parameters[param.name]
                if not isinstance(value, param.type):
                    return False

                # Custom validation
                if param.validation and not param.validation(value):
                    return False

        return True

class AsyncFunctionAdapter(ToolAdapter):
    """Adapter for async Python functions."""

    def __init__(self, func: Callable, capability: ToolCapability):
        self.func = func
        self.capability = capability

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute async function."""
        if not self.validate_parameters(parameters):
            raise ValueError("Invalid parameters")

        result = await self.func(**parameters)
        return result

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        # Same as SyncFunctionAdapter
        for param in self.capability.parameters:
            if param.required and param.name not in parameters:
                return False

            if param.name in parameters:
                value = parameters[param.name]
                if not isinstance(value, param.type):
                    return False

                if param.validation and not param.validation(value):
                    return False

        return True

class ToolIntegrationEngine:
    """
    Orchestrates tool execution with adapters.
    """
    def __init__(self, registry: ToolRegistry, selector: ContextAwareToolSelector):
        self.registry = registry
        self.selector = selector
        self.adapters: Dict[str, ToolAdapter] = {}

    def register_adapter(self, tool_id: str, adapter: ToolAdapter):
        """Register adapter for a tool."""
        self.adapters[tool_id] = adapter

    async def execute_tool(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool with error handling and tracking."""
        tool = self.registry.get_tool(tool_id)
        if not tool:
            return {'success': False, 'error': 'Tool not found'}

        adapter = self.adapters.get(tool_id)
        if not adapter:
            return {'success': False, 'error': 'No adapter registered'}

        # Track execution
        start_time = datetime.now()
        tool.total_executions += 1

        try:
            result = await adapter.execute(parameters, context)

            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            tool.successful_executions += 1
            tool.total_execution_time_ms += execution_time
            tool.last_executed = datetime.now()

            # Record selection outcome
            self.selector.record_selection(
                task=context.get('task_description', ''),
                context=context,
                selected_tool=tool,
                success=True,
                execution_time_ms=execution_time
            )

            return {
                'success': True,
                'result': result,
                'execution_time_ms': execution_time
            }

        except Exception as e:
            tool.failed_executions += 1

            return {
                'success': False,
                'error': str(e),
                'tool_id': tool_id
            }

    async def auto_select_and_execute(
        self,
        task_description: str,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Automatically select and execute best tool for task."""
        # Select tools
        selected_tools = self.selector.select_tools(
            task_description=task_description,
            context=context,
            max_tools=3
        )

        if not selected_tools:
            return {'success': False, 'error': 'No suitable tools found'}

        # Try tools in order of score until one succeeds
        for tool, score in selected_tools:
            result = await self.execute_tool(
                tool_id=tool.capability.tool_id,
                parameters=parameters,
                context=context
            )

            if result['success']:
                result['tool_id'] = tool.capability.tool_id
                result['selection_score'] = score
                return result

        return {
            'success': False,
            'error': 'All selected tools failed',
            'attempted_tools': [t.capability.tool_id for t, _ in selected_tools]
        }
```

## Example: Building a Complete Tool System

### Practical Implementation

```python
# Define some example tools

def search_database(query: str, limit: int = 10) -> List[Dict]:
    """Search database for records matching query."""
    # Simulate database search
    return [
        {'id': 1, 'title': 'Result 1', 'relevance': 0.95},
        {'id': 2, 'title': 'Result 2', 'relevance': 0.87}
    ][:limit]

async def call_external_api(endpoint: str, params: Dict) -> Dict:
    """Call external REST API."""
    # Simulate API call
    await asyncio.sleep(0.1)
    return {'status': 'success', 'data': {'result': 'API response'}}

def process_data(data: List[Any], operation: str) -> Any:
    """Process data with specified operation."""
    if operation == 'sum':
        return sum(data)
    elif operation == 'average':
        return sum(data) / len(data) if data else 0
    return data

# Create registry and register tools

registry = ToolRegistry()

# Register database search tool
db_search_capability = ToolCapability(
    tool_id='db_search',
    name='Database Search',
    description='Search database for records matching query',
    category=ToolCategory.DATABASE,
    parameters=[
        ToolParameter('query', str, 'Search query string', required=True),
        ToolParameter('limit', int, 'Maximum results', required=False, default=10)
    ],
    return_type=list,
    required_context=['database_connection'],
    avg_execution_time_ms=50,
    reliability_score=0.98,
    tags=['database', 'search', 'retrieval']
)

db_tool = registry.register_tool(
    capability=db_search_capability,
    executor=search_database
)

# Register API call tool
api_capability = ToolCapability(
    tool_id='external_api',
    name='External API Call',
    description='Call external REST API endpoint',
    category=ToolCategory.EXTERNAL_API,
    parameters=[
        ToolParameter('endpoint', str, 'API endpoint URL', required=True),
        ToolParameter('params', dict, 'Request parameters', required=False, default={})
    ],
    return_type=dict,
    required_context=['api_key'],
    avg_execution_time_ms=200,
    reliability_score=0.92,
    tags=['api', 'external', 'http']
)

api_tool = registry.register_tool(
    capability=api_capability,
    executor=call_external_api
)

# Create selector and integration engine
selector = ContextAwareToolSelector(registry)
engine = ToolIntegrationEngine(registry, selector)

# Register adapters
engine.register_adapter(
    'db_search',
    SyncFunctionAdapter(search_database, db_search_capability)
)
engine.register_adapter(
    'external_api',
    AsyncFunctionAdapter(call_external_api, api_capability)
)

# Example usage
async def example_usage():
    # Context with database connection
    context = {
        'database_connection': 'db://localhost/mydb',
        'api_key': 'sk-test-key-123',
        'task_description': 'Search for user records in database'
    }

    # Auto-select and execute tool
    result = await engine.auto_select_and_execute(
        task_description='Search for user records',
        context=context,
        parameters={'query': 'active users', 'limit': 5}
    )

    print(f"Success: {result['success']}")
    print(f"Tool used: {result.get('tool_id')}")
    print(f"Result: {result.get('result')}")
```

## Key Takeaways

1. **Tool Capability Modeling**: Comprehensive specifications enable intelligent selection

2. **Context-Aware Selection**: Tools chosen based on task requirements and available context

3. **Universal Adapters**: Support both sync and async tools with consistent interface

4. **Execution Tracking**: Monitor performance and reliability for continuous improvement

5. **Automatic Selection**: System selects best tool without manual intervention

6. **Graceful Fallback**: Try multiple tools until one succeeds

## What's Next

In Lesson 2, we'll explore external API and service integration patterns with authentication, rate limiting, and error handling.

---

**Practice Exercise**: Build a complete tool system with 10+ different tools across multiple categories. Implement context-aware selection achieving >85% accuracy in tool selection for various tasks. Add execution tracking and demonstrate automatic tool selection with fallback handling.
