# Lesson 4: Error Handling and Resilience

## Introduction

Production tool integration systems must handle failures gracefully. This lesson teaches you to build resilient systems with sophisticated error handling, retry mechanisms, circuit breakers, fallback strategies, and comprehensive monitoring that maintain functionality even when external services fail.

## Robust Error Handling Strategies

### Comprehensive Error Classification

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Type
from enum import Enum
import asyncio
import time
from datetime import datetime, timedelta
import logging

class ErrorCategory(Enum):
    """Categories of errors for different handling strategies."""
    TRANSIENT = "transient"  # Temporary network issues, rate limits
    CONFIGURATION = "configuration"  # Wrong parameters, missing auth
    RESOURCE = "resource"  # Out of memory, disk space
    EXTERNAL = "external"  # Third-party service down
    VALIDATION = "validation"  # Invalid input data
    SECURITY = "security"  # Authentication, authorization failures
    FATAL = "fatal"  # Unrecoverable errors

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None

    # Context
    tool_id: str = ""
    task_id: str = ""
    workflow_id: str = ""

    # Timing
    occurred_at: datetime = field(default_factory=datetime.now)

    # Recovery
    is_recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3

    # Additional data
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

class ErrorHandler:
    """
    Sophisticated error handling with categorization and recovery strategies.
    """
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.alert_handlers: List[Callable] = []

        # Error patterns
        self.error_patterns: Dict[str, ErrorCategory] = {
            'timeout': ErrorCategory.TRANSIENT,
            'connection': ErrorCategory.TRANSIENT,
            'rate limit': ErrorCategory.TRANSIENT,
            'unauthorized': ErrorCategory.SECURITY,
            'forbidden': ErrorCategory.SECURITY,
            'not found': ErrorCategory.CONFIGURATION,
            'bad request': ErrorCategory.VALIDATION,
            'internal server error': ErrorCategory.EXTERNAL,
            'service unavailable': ErrorCategory.EXTERNAL
        }

        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default recovery strategies."""
        self.recovery_strategies[ErrorCategory.TRANSIENT] = self._handle_transient_error
        self.recovery_strategies[ErrorCategory.CONFIGURATION] = self._handle_configuration_error
        self.recovery_strategies[ErrorCategory.RESOURCE] = self._handle_resource_error
        self.recovery_strategies[ErrorCategory.EXTERNAL] = self._handle_external_error
        self.recovery_strategies[ErrorCategory.VALIDATION] = self._handle_validation_error
        self.recovery_strategies[ErrorCategory.SECURITY] = self._handle_security_error
        self.recovery_strategies[ErrorCategory.FATAL] = self._handle_fatal_error

    def categorize_error(self, exception: Exception, context: Dict = None) -> ErrorCategory:
        """Categorize error based on exception and context."""
        error_message = str(exception).lower()

        # Check patterns
        for pattern, category in self.error_patterns.items():
            if pattern in error_message:
                return category

        # Check exception types
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorCategory.TRANSIENT
        elif isinstance(exception, ValueError):
            return ErrorCategory.VALIDATION
        elif isinstance(exception, PermissionError):
            return ErrorCategory.SECURITY
        elif isinstance(exception, FileNotFoundError):
            return ErrorCategory.CONFIGURATION
        elif isinstance(exception, MemoryError):
            return ErrorCategory.RESOURCE

        # Default to external
        return ErrorCategory.EXTERNAL

    async def handle_error(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle error with appropriate strategy."""
        # Categorize error
        category = self.categorize_error(exception, context)

        # Determine severity
        severity = self._determine_severity(exception, category, context)

        # Create error info
        error_info = ErrorInfo(
            error_id=self._generate_error_id(),
            category=category,
            severity=severity,
            message=str(exception),
            original_exception=exception,
            tool_id=context.get('tool_id', ''),
            task_id=context.get('task_id', ''),
            workflow_id=context.get('workflow_id', ''),
            context=context,
            stack_trace=self._get_stack_trace(exception)
        )

        # Record error
        self.error_history.append(error_info)

        # Apply recovery strategy
        if category in self.recovery_strategies:
            recovery_result = await self.recovery_strategies[category](error_info)
        else:
            recovery_result = {'recovered': False, 'action': 'no_strategy'}

        # Send alerts if needed
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self._send_alerts(error_info)

        return {
            'error_id': error_info.error_id,
            'category': category.value,
            'severity': severity.value,
            'recovered': recovery_result.get('recovered', False),
            'action_taken': recovery_result.get('action', 'none'),
            'retry_recommended': recovery_result.get('retry', False)
        }

    async def _handle_transient_error(self, error_info: ErrorInfo) -> Dict:
        """Handle transient errors with backoff."""
        if error_info.retry_count < error_info.max_retries:
            # Calculate backoff delay
            delay = min(60, (2 ** error_info.retry_count) + (time.time() % 1))

            return {
                'recovered': False,
                'action': f'retry_after_{delay}s',
                'retry': True,
                'delay_seconds': delay
            }

        return {'recovered': False, 'action': 'max_retries_exceeded'}

    async def _handle_configuration_error(self, error_info: ErrorInfo) -> Dict:
        """Handle configuration errors."""
        # Log for manual intervention
        logging.error(f"Configuration error in {error_info.tool_id}: {error_info.message}")

        return {
            'recovered': False,
            'action': 'manual_intervention_required',
            'retry': False
        }

    async def _handle_resource_error(self, error_info: ErrorInfo) -> Dict:
        """Handle resource errors."""
        # Try to free up resources
        import gc
        gc.collect()

        return {
            'recovered': True,
            'action': 'resource_cleanup',
            'retry': True
        }

    async def _handle_external_error(self, error_info: ErrorInfo) -> Dict:
        """Handle external service errors."""
        # Check service health
        service_health = await self._check_service_health(error_info.tool_id)

        if service_health['available']:
            return {
                'recovered': True,
                'action': 'service_recovered',
                'retry': True
            }

        return {
            'recovered': False,
            'action': 'service_unavailable',
            'retry': False,
            'fallback_recommended': True
        }

    async def _handle_validation_error(self, error_info: ErrorInfo) -> Dict:
        """Handle validation errors."""
        # Validation errors typically don't recover with retry
        return {
            'recovered': False,
            'action': 'invalid_input',
            'retry': False
        }

    async def _handle_security_error(self, error_info: ErrorInfo) -> Dict:
        """Handle security errors."""
        # Log security event
        logging.warning(f"Security error: {error_info.message}")

        return {
            'recovered': False,
            'action': 'security_violation',
            'retry': False
        }

    async def _handle_fatal_error(self, error_info: ErrorInfo) -> Dict:
        """Handle fatal errors."""
        # Log critical error
        logging.critical(f"Fatal error: {error_info.message}")

        return {
            'recovered': False,
            'action': 'system_shutdown_recommended',
            'retry': False
        }

    def _determine_severity(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: Dict
    ) -> ErrorSeverity:
        """Determine error severity."""
        # Fatal categories
        if category == ErrorCategory.FATAL:
            return ErrorSeverity.CRITICAL

        # Security issues
        if category == ErrorCategory.SECURITY:
            return ErrorSeverity.HIGH

        # Resource issues
        if category == ErrorCategory.RESOURCE:
            return ErrorSeverity.HIGH

        # Check context for importance
        workflow_critical = context.get('critical_workflow', False)
        if workflow_critical and category == ErrorCategory.EXTERNAL:
            return ErrorSeverity.HIGH

        # Transient and validation errors
        if category in [ErrorCategory.TRANSIENT, ErrorCategory.VALIDATION]:
            return ErrorSeverity.MEDIUM

        return ErrorSeverity.LOW

    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _get_stack_trace(self, exception: Exception) -> str:
        """Get stack trace from exception."""
        import traceback
        return traceback.format_exc()

    async def _check_service_health(self, tool_id: str) -> Dict:
        """Check if external service is healthy."""
        # Simplified health check
        try:
            # In real implementation, ping service endpoint
            await asyncio.sleep(0.1)
            return {'available': True, 'response_time_ms': 100}
        except:
            return {'available': False}

    async def _send_alerts(self, error_info: ErrorInfo):
        """Send alerts for severe errors."""
        alert_data = {
            'error_id': error_info.error_id,
            'severity': error_info.severity.value,
            'message': error_info.message,
            'tool_id': error_info.tool_id,
            'occurred_at': error_info.occurred_at
        }

        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
```

## Circuit Breaker Pattern

### Intelligent Circuit Breaker

```python
class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    monitoring_window_seconds: float = 300.0

class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    """
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED

        # Counters
        self.failure_count = 0
        self.success_count = 0
        self.request_count = 0

        # Timing
        self.last_failure_time: Optional[datetime] = None
        self.state_changed_at = datetime.now()

        # Monitoring
        self.request_history: List[Dict] = []

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        # Record request
        self.request_count += 1

        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.state_changed_at = datetime.now()
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")

        # Execute function
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)

            # Record success
            self._record_success()

            # Update state if half-open
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.state_changed_at = datetime.now()
                    self._reset_counters()

            return result

        except Exception as e:
            # Record failure
            self._record_failure()

            # Update state
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.state_changed_at = datetime.now()
            elif (self.state == CircuitState.CLOSED and
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                self.state_changed_at = datetime.now()

            raise e

        finally:
            # Record request
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self._record_request(success=(self.state != CircuitState.OPEN),
                               execution_time_ms=execution_time)

    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit."""
        if not self.last_failure_time:
            return True

        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout_seconds

    def _record_success(self):
        """Record successful execution."""
        self.success_count += 1
        self.failure_count = max(0, self.failure_count - 1)  # Decay failures

    def _record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = datetime.now()

    def _reset_counters(self):
        """Reset all counters."""
        self.failure_count = 0
        self.success_count = 0

    def _record_request(self, success: bool, execution_time_ms: float):
        """Record request for monitoring."""
        self.request_history.append({
            'timestamp': datetime.now(),
            'success': success,
            'execution_time_ms': execution_time_ms,
            'state': self.state.value
        })

        # Cleanup old records
        cutoff_time = datetime.now() - timedelta(seconds=self.config.monitoring_window_seconds)
        self.request_history = [
            r for r in self.request_history
            if r['timestamp'] >= cutoff_time
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        recent_requests = len(self.request_history)
        recent_failures = sum(1 for r in self.request_history if not r['success'])

        success_rate = 0.0
        if recent_requests > 0:
            success_rate = (recent_requests - recent_failures) / recent_requests

        avg_response_time = 0.0
        if self.request_history:
            avg_response_time = sum(r['execution_time_ms'] for r in self.request_history) / len(self.request_history)

        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'recent_requests': recent_requests,
            'recent_success_rate': success_rate,
            'avg_response_time_ms': avg_response_time,
            'last_state_change': self.state_changed_at
        }

class CircuitBreakerManager:
    """
    Manages circuit breakers for multiple services.
    """
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()

    def get_breaker(
        self,
        service_name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.breakers:
            breaker_config = config or self.default_config
            self.breakers[service_name] = CircuitBreaker(service_name, breaker_config)

        return self.breakers[service_name]

    async def execute_with_breaker(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection."""
        breaker = self.get_breaker(service_name)
        return await breaker.call(func, *args, **kwargs)

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
```

## Fallback Strategies

### Intelligent Fallback System

```python
@dataclass
class FallbackStrategy:
    """Fallback strategy definition."""
    strategy_id: str
    name: str
    description: str

    # Execution
    fallback_func: Callable
    condition: Callable  # When to use this fallback

    # Priority and cost
    priority: int = 5  # Lower = higher priority
    cost_factor: float = 1.0  # Relative cost

    # Success tracking
    usage_count: int = 0
    success_count: int = 0

class FallbackManager:
    """
    Manages fallback strategies for failed tool executions.
    """
    def __init__(self):
        self.strategies: Dict[str, List[FallbackStrategy]] = {}
        self.execution_history: List[Dict] = []

    def register_fallback(
        self,
        tool_id: str,
        strategy: FallbackStrategy
    ):
        """Register fallback strategy for tool."""
        if tool_id not in self.strategies:
            self.strategies[tool_id] = []

        self.strategies[tool_id].append(strategy)

        # Sort by priority
        self.strategies[tool_id].sort(key=lambda s: s.priority)

    async def execute_with_fallback(
        self,
        tool_id: str,
        primary_func: Callable,
        context: Dict[str, Any],
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute function with fallback strategies."""
        # Try primary function first
        try:
            result = await primary_func(*args, **kwargs)
            return {
                'success': True,
                'result': result,
                'execution_path': 'primary'
            }
        except Exception as primary_error:
            # Try fallback strategies
            if tool_id in self.strategies:
                for strategy in self.strategies[tool_id]:
                    # Check if strategy applies
                    if strategy.condition(primary_error, context):
                        try:
                            strategy.usage_count += 1

                            result = await strategy.fallback_func(*args, **kwargs)

                            strategy.success_count += 1

                            # Record successful fallback
                            self._record_execution(
                                tool_id=tool_id,
                                strategy_id=strategy.strategy_id,
                                success=True,
                                primary_error=str(primary_error)
                            )

                            return {
                                'success': True,
                                'result': result,
                                'execution_path': 'fallback',
                                'strategy_used': strategy.strategy_id,
                                'primary_error': str(primary_error)
                            }

                        except Exception as fallback_error:
                            # Record failed fallback
                            self._record_execution(
                                tool_id=tool_id,
                                strategy_id=strategy.strategy_id,
                                success=False,
                                primary_error=str(primary_error),
                                fallback_error=str(fallback_error)
                            )
                            continue

            # All fallbacks failed
            return {
                'success': False,
                'error': str(primary_error),
                'execution_path': 'all_failed',
                'fallback_attempts': len(self.strategies.get(tool_id, []))
            }

    def _record_execution(
        self,
        tool_id: str,
        strategy_id: str,
        success: bool,
        primary_error: str,
        fallback_error: Optional[str] = None
    ):
        """Record fallback execution."""
        self.execution_history.append({
            'tool_id': tool_id,
            'strategy_id': strategy_id,
            'success': success,
            'primary_error': primary_error,
            'fallback_error': fallback_error,
            'timestamp': datetime.now()
        })

    def get_strategy_stats(self, tool_id: str) -> List[Dict]:
        """Get statistics for fallback strategies."""
        if tool_id not in self.strategies:
            return []

        stats = []
        for strategy in self.strategies[tool_id]:
            success_rate = 0.0
            if strategy.usage_count > 0:
                success_rate = strategy.success_count / strategy.usage_count

            stats.append({
                'strategy_id': strategy.strategy_id,
                'name': strategy.name,
                'usage_count': strategy.usage_count,
                'success_count': strategy.success_count,
                'success_rate': success_rate,
                'priority': strategy.priority
            })

        return stats

# Example fallback strategies
async def cache_fallback(query: str, *args, **kwargs) -> Dict:
    """Fallback to cached results."""
    # Simplified cache lookup
    cache_key = f"cache_{hash(query)}"
    # In real implementation, check actual cache
    return {'cached_result': f"Cached data for {query}"}

async def simplified_fallback(query: str, *args, **kwargs) -> Dict:
    """Fallback to simplified processing."""
    # Simplified processing
    return {'simplified_result': f"Basic processing of {query}"}

async def manual_fallback(query: str, *args, **kwargs) -> Dict:
    """Fallback to manual processing queue."""
    # Queue for manual processing
    return {'manual_queue_id': f"manual_{int(time.time())}"}

# Example usage
def create_example_fallbacks(manager: FallbackManager):
    """Create example fallback strategies."""

    # Cache fallback for any error
    cache_strategy = FallbackStrategy(
        strategy_id='cache_fallback',
        name='Cache Fallback',
        description='Use cached results when service fails',
        fallback_func=cache_fallback,
        condition=lambda error, context: True,  # Always try cache
        priority=1,
        cost_factor=0.1
    )

    # Simplified processing for service errors
    simplified_strategy = FallbackStrategy(
        strategy_id='simplified_processing',
        name='Simplified Processing',
        description='Use simplified algorithm when service unavailable',
        fallback_func=simplified_fallback,
        condition=lambda error, context: 'service unavailable' in str(error).lower(),
        priority=2,
        cost_factor=0.5
    )

    # Manual processing as last resort
    manual_strategy = FallbackStrategy(
        strategy_id='manual_processing',
        name='Manual Processing Queue',
        description='Queue for manual processing',
        fallback_func=manual_fallback,
        condition=lambda error, context: True,  # Last resort
        priority=9,
        cost_factor=10.0
    )

    # Register fallbacks for a search tool
    manager.register_fallback('search_tool', cache_strategy)
    manager.register_fallback('search_tool', simplified_strategy)
    manager.register_fallback('search_tool', manual_strategy)
```

## Health Monitoring and Alerting

### Comprehensive Health Monitor

```python
class HealthMetric(Enum):
    """Health metrics to monitor."""
    SUCCESS_RATE = "success_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    THROUGHPUT = "throughput"

@dataclass
class HealthThreshold:
    """Health monitoring threshold."""
    metric: HealthMetric
    warning_threshold: float
    critical_threshold: float
    evaluation_window_minutes: int = 5

class HealthMonitor:
    """
    Monitors tool and system health with alerting.
    """
    def __init__(self):
        self.metrics: Dict[str, List[Dict]] = {}
        self.thresholds: Dict[str, List[HealthThreshold]] = {}
        self.alerts: List[Dict] = []

        # Default thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self):
        """Setup default health thresholds."""
        default_thresholds = [
            HealthThreshold(HealthMetric.SUCCESS_RATE, 0.95, 0.90),
            HealthThreshold(HealthMetric.RESPONSE_TIME, 1000, 5000),  # ms
            HealthThreshold(HealthMetric.ERROR_RATE, 0.05, 0.10)
        ]

        self.thresholds['default'] = default_thresholds

    def record_metric(
        self,
        service_id: str,
        metric: HealthMetric,
        value: float
    ):
        """Record health metric."""
        if service_id not in self.metrics:
            self.metrics[service_id] = []

        self.metrics[service_id].append({
            'metric': metric,
            'value': value,
            'timestamp': datetime.now()
        })

        # Cleanup old metrics
        cutoff = datetime.now() - timedelta(hours=1)
        self.metrics[service_id] = [
            m for m in self.metrics[service_id]
            if m['timestamp'] >= cutoff
        ]

        # Check thresholds
        self._check_thresholds(service_id, metric, value)

    def _check_thresholds(
        self,
        service_id: str,
        metric: HealthMetric,
        current_value: float
    ):
        """Check if metric violates thresholds."""
        # Use service-specific thresholds or default
        thresholds = self.thresholds.get(service_id, self.thresholds['default'])

        for threshold in thresholds:
            if threshold.metric == metric:
                # Calculate metric over evaluation window
                window_start = datetime.now() - timedelta(minutes=threshold.evaluation_window_minutes)

                recent_values = [
                    m['value'] for m in self.metrics.get(service_id, [])
                    if m['metric'] == metric and m['timestamp'] >= window_start
                ]

                if not recent_values:
                    continue

                # Different aggregation for different metrics
                if metric in [HealthMetric.SUCCESS_RATE, HealthMetric.AVAILABILITY]:
                    aggregated_value = sum(recent_values) / len(recent_values)
                elif metric == HealthMetric.RESPONSE_TIME:
                    aggregated_value = sum(recent_values) / len(recent_values)  # Average
                elif metric == HealthMetric.ERROR_RATE:
                    aggregated_value = sum(recent_values) / len(recent_values)
                else:
                    aggregated_value = current_value

                # Check thresholds
                if aggregated_value <= threshold.critical_threshold:
                    self._create_alert(service_id, metric, 'critical', aggregated_value, threshold)
                elif aggregated_value <= threshold.warning_threshold:
                    self._create_alert(service_id, metric, 'warning', aggregated_value, threshold)

    def _create_alert(
        self,
        service_id: str,
        metric: HealthMetric,
        severity: str,
        value: float,
        threshold: HealthThreshold
    ):
        """Create health alert."""
        alert = {
            'alert_id': f"{service_id}_{metric.value}_{int(time.time())}",
            'service_id': service_id,
            'metric': metric.value,
            'severity': severity,
            'current_value': value,
            'threshold': threshold.warning_threshold if severity == 'warning' else threshold.critical_threshold,
            'timestamp': datetime.now(),
            'resolved': False
        }

        self.alerts.append(alert)

        # Log alert
        logging.warning(f"Health alert: {service_id} {metric.value} {severity} - {value}")

    def get_health_status(self, service_id: str) -> Dict[str, Any]:
        """Get current health status for service."""
        if service_id not in self.metrics:
            return {'status': 'unknown', 'metrics': {}}

        # Calculate current metrics
        recent_metrics = {}
        cutoff = datetime.now() - timedelta(minutes=5)

        for metric_record in self.metrics[service_id]:
            if metric_record['timestamp'] >= cutoff:
                metric_name = metric_record['metric'].value
                if metric_name not in recent_metrics:
                    recent_metrics[metric_name] = []
                recent_metrics[metric_name].append(metric_record['value'])

        # Aggregate metrics
        aggregated = {}
        for metric_name, values in recent_metrics.items():
            if values:
                aggregated[metric_name] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

        # Determine overall status
        active_alerts = [
            a for a in self.alerts
            if a['service_id'] == service_id and not a['resolved']
        ]

        if any(a['severity'] == 'critical' for a in active_alerts):
            status = 'critical'
        elif any(a['severity'] == 'warning' for a in active_alerts):
            status = 'warning'
        else:
            status = 'healthy'

        return {
            'status': status,
            'metrics': aggregated,
            'active_alerts': len(active_alerts),
            'last_updated': datetime.now()
        }
```

## Key Takeaways

1. **Error Classification**: Systematic categorization enables appropriate recovery strategies

2. **Circuit Breakers**: Prevent cascading failures with intelligent state management

3. **Fallback Strategies**: Multiple backup approaches for failed operations

4. **Health Monitoring**: Continuous tracking with threshold-based alerting

5. **Recovery Automation**: Self-healing systems reduce manual intervention

6. **Comprehensive Logging**: Detailed error tracking for analysis and improvement

## What's Next

In Lesson 5, we'll explore security and sandbox execution for safe tool operations.

---

**Practice Exercise**: Build a complete resilience system with error handling, circuit breakers, and fallback strategies. Demonstrate handling 1000+ tool executions with various failure scenarios, maintaining >95% overall success rate through fallback mechanisms, and automatic recovery from transient failures.
