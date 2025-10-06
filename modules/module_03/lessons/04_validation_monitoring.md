# Lesson 4: Context Validation and Monitoring

## Introduction

Production context systems require robust validation and monitoring to ensure reliability, detect issues early, and maintain quality over time. This lesson teaches you to build comprehensive validation and monitoring systems that catch problems before they impact users.

## Context Integrity Validation Systems

### Multi-Layer Validation Framework

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Callable
from enum import Enum
from datetime import datetime
import hashlib

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    item_id: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class ContextIntegrityValidator:
    """
    Comprehensive validation system for context integrity.
    """
    def __init__(self):
        self.validation_rules: List[Callable] = []
        self.validation_history: List[Dict] = []
        self.issue_patterns: Dict[str, int] = {}

        # Register default validation rules
        self._register_default_rules()

    def _register_default_rules(self):
        """Register standard validation rules."""
        self.register_rule(self._validate_structure)
        self.register_rule(self._validate_relationships)
        self.register_rule(self._validate_consistency)
        self.register_rule(self._validate_completeness)
        self.register_rule(self._validate_temporal_integrity)

    def register_rule(self, rule_func: Callable):
        """Register a custom validation rule."""
        self.validation_rules.append(rule_func)

    def validate_context_hub(
        self,
        context_hub: 'CentralizedContextHub'
    ) -> 'ValidationReport':
        """
        Perform comprehensive validation of context hub.
        """
        issues = []

        # Run all validation rules
        for rule in self.validation_rules:
            rule_issues = rule(context_hub)
            issues.extend(rule_issues)

        # Create report
        report = ValidationReport(
            timestamp=datetime.now(),
            total_items=len(context_hub.active_context),
            issues=issues,
            passed=len([i for i in issues if i.severity != ValidationSeverity.CRITICAL]) == len(issues)
        )

        # Record validation
        self.validation_history.append({
            'timestamp': report.timestamp,
            'total_items': report.total_items,
            'issue_count': len(issues),
            'passed': report.passed
        })

        # Track issue patterns
        for issue in issues:
            key = f"{issue.category}:{issue.severity.value}"
            self.issue_patterns[key] = self.issue_patterns.get(key, 0) + 1

        return report

    def _validate_structure(
        self,
        context_hub: 'CentralizedContextHub'
    ) -> List[ValidationIssue]:
        """Validate context structure and data types."""
        issues = []

        for key, item in context_hub.active_context.items():
            # Check required fields
            if not item.id:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="structure",
                    message="Context item missing ID",
                    item_id=key
                ))

            if item.content is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="structure",
                    message="Context item missing content",
                    item_id=key
                ))

            # Check importance range
            if not (0.0 <= item.importance <= 1.0):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="structure",
                    message=f"Importance value {item.importance} out of range [0,1]",
                    item_id=key
                ))

            # Check timestamp validity
            if item.timestamp > datetime.now():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="structure",
                    message="Timestamp is in the future",
                    item_id=key
                ))

        return issues

    def _validate_relationships(
        self,
        context_hub: 'CentralizedContextHub'
    ) -> List[ValidationIssue]:
        """Validate relationships between context items."""
        issues = []

        # Check for orphaned references
        all_ids = set(context_hub.active_context.keys())

        for key, item in context_hub.active_context.items():
            # Check if referenced items exist
            references = item.metadata.get('references', [])
            for ref in references:
                if ref not in all_ids:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="relationships",
                        message=f"References non-existent item: {ref}",
                        item_id=key
                    ))

        return issues

    def _validate_consistency(
        self,
        context_hub: 'CentralizedContextHub'
    ) -> List[ValidationIssue]:
        """Validate internal consistency."""
        issues = []

        # Check for contradictory context
        content_by_type: Dict[str, List] = {}
        for key, item in context_hub.active_context.items():
            if item.type not in content_by_type:
                content_by_type[item.type] = []
            content_by_type[item.type].append((key, item))

        # Check for duplicates within types
        for ctx_type, items in content_by_type.items():
            seen_hashes = {}
            for key, item in items:
                content_hash = hash(str(item.content))
                if content_hash in seen_hashes:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="consistency",
                        message=f"Potential duplicate of {seen_hashes[content_hash]}",
                        item_id=key
                    ))
                else:
                    seen_hashes[content_hash] = key

        return issues

    def _validate_completeness(
        self,
        context_hub: 'CentralizedContextHub'
    ) -> List[ValidationIssue]:
        """Validate context completeness."""
        issues = []

        # Check if critical context types are present
        critical_types = {'user_profile', 'session_info'}
        present_types = {item.type for item in context_hub.active_context.values()}

        missing_types = critical_types - present_types
        if missing_types:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="completeness",
                message=f"Missing critical context types: {missing_types}"
            ))

        # Check for empty context
        if len(context_hub.active_context) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="completeness",
                message="Context hub is empty"
            ))

        return issues

    def _validate_temporal_integrity(
        self,
        context_hub: 'CentralizedContextHub'
    ) -> List[ValidationIssue]:
        """Validate temporal aspects of context."""
        issues = []

        # Check for very old context that should have been pruned
        from datetime import timedelta
        old_threshold = datetime.now() - timedelta(days=7)

        old_count = 0
        for key, item in context_hub.active_context.items():
            if item.timestamp < old_threshold and item.importance < 0.5:
                old_count += 1

        if old_count > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="temporal",
                message=f"{old_count} items are very old and low importance",
                metadata={'count': old_count}
            ))

        # Check access patterns
        never_accessed = [
            key for key, item in context_hub.active_context.items()
            if item.access_count == 0 and
            (datetime.now() - item.timestamp).days > 1
        ]

        if len(never_accessed) > len(context_hub.active_context) * 0.3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="temporal",
                message=f"{len(never_accessed)} items never accessed",
                metadata={'count': len(never_accessed)}
            ))

        return issues

    def get_validation_stats(self) -> Dict:
        """Get statistics on validation history."""
        if not self.validation_history:
            return {'total_validations': 0}

        total_validations = len(self.validation_history)
        passed_validations = sum(1 for v in self.validation_history if v['passed'])

        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'pass_rate': passed_validations / total_validations,
            'common_issues': dict(sorted(
                self.issue_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }

@dataclass
class ValidationReport:
    """Report from context validation."""
    timestamp: datetime
    total_items: int
    issues: List[ValidationIssue]
    passed: bool

    def get_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues of specific severity."""
        return [i for i in self.issues if i.severity == severity]

    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get critical issues that need immediate attention."""
        return self.get_by_severity(ValidationSeverity.CRITICAL)

    def print_report(self):
        """Print human-readable validation report."""
        print(f"\n{'='*60}")
        print(f"Context Validation Report - {self.timestamp}")
        print(f"{'='*60}")
        print(f"Total Items: {self.total_items}")
        print(f"Status: {'✓ PASSED' if self.passed else '✗ FAILED'}")
        print(f"\nIssues Found: {len(self.issues)}")

        if self.issues:
            by_severity = {}
            for issue in self.issues:
                severity = issue.severity.value
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(issue)

            for severity in ['critical', 'error', 'warning', 'info']:
                if severity in by_severity:
                    print(f"\n{severity.upper()}: {len(by_severity[severity])}")
                    for issue in by_severity[severity][:5]:  # Show first 5
                        print(f"  - [{issue.category}] {issue.message}")
                    if len(by_severity[severity]) > 5:
                        print(f"  ... and {len(by_severity[severity]) - 5} more")
        print(f"{'='*60}\n")
```

## Real-Time Monitoring and Alerting

### Context Monitoring System

```python
import time
from collections import deque
from typing import Deque

class ContextMonitoringSystem:
    """
    Real-time monitoring system for context health and performance.
    """
    def __init__(self):
        self.metrics: Dict[str, Deque] = {
            'response_times': deque(maxlen=1000),
            'context_sizes': deque(maxlen=1000),
            'quality_scores': deque(maxlen=1000),
            'error_counts': deque(maxlen=1000),
            'optimization_times': deque(maxlen=100)
        }

        self.alerts: List[Dict] = []
        self.alert_thresholds = {
            'response_time': 2.0,  # seconds
            'context_size': 1500,
            'quality_score_min': 0.3,
            'error_rate': 0.05
        }

        self.monitoring_enabled = True

    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """Record a metric value."""
        if not self.monitoring_enabled:
            return

        timestamp = timestamp or datetime.now()

        if metric_name in self.metrics:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': timestamp
            })

            # Check for alert conditions
            self._check_alerts(metric_name, value)

    def _check_alerts(self, metric_name: str, value: float):
        """Check if value triggers any alerts."""
        alerts_triggered = []

        if metric_name == 'response_times':
            if value > self.alert_thresholds['response_time']:
                alerts_triggered.append({
                    'type': 'high_response_time',
                    'severity': 'warning',
                    'message': f'Response time {value:.2f}s exceeds threshold',
                    'value': value
                })

        elif metric_name == 'context_sizes':
            if value > self.alert_thresholds['context_size']:
                alerts_triggered.append({
                    'type': 'large_context',
                    'severity': 'warning',
                    'message': f'Context size {value} exceeds threshold',
                    'value': value
                })

        elif metric_name == 'quality_scores':
            if value < self.alert_thresholds['quality_score_min']:
                alerts_triggered.append({
                    'type': 'low_quality',
                    'severity': 'error',
                    'message': f'Quality score {value:.2f} below threshold',
                    'value': value
                })

        # Record alerts
        for alert in alerts_triggered:
            alert['timestamp'] = datetime.now()
            self.alerts.append(alert)

    def get_current_metrics(self) -> Dict:
        """Get current metric summaries."""
        import numpy as np

        summaries = {}

        for metric_name, values in self.metrics.items():
            if not values:
                continue

            nums = [v['value'] for v in values]

            summaries[metric_name] = {
                'current': nums[-1] if nums else 0,
                'avg': np.mean(nums),
                'min': np.min(nums),
                'max': np.max(nums),
                'p95': np.percentile(nums, 95) if len(nums) > 0 else 0,
                'count': len(nums)
            }

        return summaries

    def get_recent_alerts(
        self,
        minutes: int = 60,
        severity: Optional[str] = None
    ) -> List[Dict]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(minutes=minutes)

        recent = [
            alert for alert in self.alerts
            if alert['timestamp'] > cutoff
        ]

        if severity:
            recent = [a for a in recent if a['severity'] == severity]

        return recent

    def get_health_status(self) -> Dict:
        """Get overall health status."""
        metrics = self.get_current_metrics()
        recent_alerts = self.get_recent_alerts(minutes=15)

        # Determine health status
        critical_alerts = [a for a in recent_alerts if a['severity'] == 'error']
        warning_alerts = [a for a in recent_alerts if a['severity'] == 'warning']

        if critical_alerts:
            status = 'unhealthy'
        elif len(warning_alerts) >= 5:
            status = 'degraded'
        else:
            status = 'healthy'

        return {
            'status': status,
            'metrics': metrics,
            'recent_alerts': len(recent_alerts),
            'critical_alerts': len(critical_alerts),
            'timestamp': datetime.now()
        }

    def print_dashboard(self):
        """Print monitoring dashboard."""
        metrics = self.get_current_metrics()
        health = self.get_health_status()

        print("\n" + "="*70)
        print("CONTEXT MONITORING DASHBOARD")
        print("="*70)
        print(f"Status: {health['status'].upper()}")
        print(f"Timestamp: {health['timestamp']}")
        print("\nMetrics:")
        print("-"*70)

        for metric_name, stats in metrics.items():
            print(f"\n{metric_name}:")
            print(f"  Current: {stats['current']:.2f}")
            print(f"  Average: {stats['avg']:.2f}")
            print(f"  P95: {stats['p95']:.2f}")
            print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")

        print("\n" + "-"*70)
        print(f"\nRecent Alerts: {health['recent_alerts']}")
        print(f"Critical: {health['critical_alerts']}")

        recent_alerts = self.get_recent_alerts(minutes=15)
        if recent_alerts:
            print("\nLatest Alerts:")
            for alert in recent_alerts[-5:]:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")

        print("="*70 + "\n")
```

## Performance Metrics and Analytics

### Context Analytics Engine

```python
class ContextAnalyticsEngine:
    """
    Advanced analytics for context system performance.
    """
    def __init__(self):
        self.event_log: List[Dict] = []
        self.aggregated_metrics: Dict = {}

    def log_event(
        self,
        event_type: str,
        metadata: Optional[Dict] = None
    ):
        """Log an event for analytics."""
        event = {
            'type': event_type,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        self.event_log.append(event)

    def analyze_performance(
        self,
        time_window_hours: int = 24
    ) -> Dict:
        """
        Analyze performance over time window.
        """
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        recent_events = [
            e for e in self.event_log
            if e['timestamp'] > cutoff
        ]

        # Analyze by event type
        event_counts = {}
        for event in recent_events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Calculate rates
        hours_in_window = min(time_window_hours,
                            (datetime.now() - recent_events[0]['timestamp']).total_seconds() / 3600
                            if recent_events else time_window_hours)

        event_rates = {
            event_type: count / hours_in_window
            for event_type, count in event_counts.items()
        }

        return {
            'time_window_hours': time_window_hours,
            'total_events': len(recent_events),
            'event_counts': event_counts,
            'event_rates_per_hour': event_rates
        }

    def analyze_trends(self, metric_name: str, periods: int = 7) -> Dict:
        """
        Analyze trends for a specific metric over periods.
        """
        # Group events by period (day)
        period_metrics = {}

        for event in self.event_log:
            if metric_name not in event.get('metadata', {}):
                continue

            period_key = event['timestamp'].date()
            if period_key not in period_metrics:
                period_metrics[period_key] = []

            period_metrics[period_key].append(
                event['metadata'][metric_name]
            )

        # Calculate trend
        sorted_periods = sorted(period_metrics.keys())[-periods:]
        period_avgs = [
            np.mean(period_metrics[p]) for p in sorted_periods
        ]

        if len(period_avgs) >= 2:
            # Simple linear trend
            x = np.arange(len(period_avgs))
            slope = np.polyfit(x, period_avgs, 1)[0]
            trend_direction = 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
        else:
            trend_direction = 'insufficient_data'

        return {
            'metric': metric_name,
            'periods_analyzed': len(sorted_periods),
            'trend': trend_direction,
            'recent_average': np.mean(period_avgs) if period_avgs else 0,
            'period_averages': dict(zip(sorted_periods, period_avgs))
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive analytics report."""
        return {
            'performance': self.analyze_performance(),
            'total_events_logged': len(self.event_log),
            'timestamp': datetime.now()
        }
```

## Context Debugging and Troubleshooting

### Context Debugger

```python
class ContextDebugger:
    """
    Debugging tools for context systems.
    """
    def __init__(self):
        self.debug_mode = False
        self.trace_log: List[Dict] = []

    def enable_debug_mode(self):
        """Enable detailed debug logging."""
        self.debug_mode = True

    def trace_context_flow(
        self,
        context_hub: 'CentralizedContextHub',
        operation: str,
        details: Optional[Dict] = None
    ):
        """Trace context operations for debugging."""
        if not self.debug_mode:
            return

        trace_entry = {
            'timestamp': datetime.now(),
            'operation': operation,
            'context_size': len(context_hub.active_context),
            'details': details or {},
            'snapshot': self._create_debug_snapshot(context_hub)
        }

        self.trace_log.append(trace_entry)

    def _create_debug_snapshot(
        self,
        context_hub: 'CentralizedContextHub'
    ) -> Dict:
        """Create snapshot for debugging."""
        return {
            'item_count': len(context_hub.active_context),
            'types': list(set(item.type for item in context_hub.active_context.values())),
            'avg_importance': np.mean([item.importance for item in context_hub.active_context.values()]) if context_hub.active_context else 0,
            'state': context_hub.state.value
        }

    def analyze_issue(
        self,
        issue_description: str,
        context_hub: 'CentralizedContextHub'
    ) -> Dict:
        """Analyze a reported issue."""
        analysis = {
            'issue': issue_description,
            'timestamp': datetime.now(),
            'current_state': {},
            'potential_causes': [],
            'recommendations': []
        }

        # Check current state
        analysis['current_state'] = {
            'context_size': len(context_hub.active_context),
            'state': context_hub.state.value,
            'has_decisions': len(context_hub.decision_log) > 0
        }

        # Identify potential causes
        if len(context_hub.active_context) > 1000:
            analysis['potential_causes'].append(
                "Large context size may cause performance issues"
            )
            analysis['recommendations'].append(
                "Run context optimization"
            )

        if context_hub.state != ContextState.ACTIVE:
            analysis['potential_causes'].append(
                f"Context in {context_hub.state.value} state"
            )

        # Check recent trace
        if self.trace_log:
            recent_ops = [t['operation'] for t in self.trace_log[-10:]]
            analysis['recent_operations'] = recent_ops

        return analysis

    def print_trace(self, last_n: int = 20):
        """Print recent trace log."""
        print("\n" + "="*70)
        print("CONTEXT TRACE LOG")
        print("="*70)

        for entry in self.trace_log[-last_n:]:
            print(f"\n[{entry['timestamp']}] {entry['operation']}")
            print(f"  Context Size: {entry['context_size']}")
            if entry['details']:
                print(f"  Details: {entry['details']}")

        print("="*70 + "\n")
```

## Key Takeaways

1. **Multi-Layer Validation**: Validate structure, relationships, consistency, and completeness

2. **Real-Time Monitoring**: Continuously monitor metrics and trigger alerts

3. **Health Assessment**: Regular health checks prevent issues before they escalate

4. **Analytics**: Use data to identify trends and optimization opportunities

5. **Debugging Tools**: Comprehensive debugging capabilities for troubleshooting

6. **Proactive Alerting**: Alert on issues before they impact users

## What's Next

In Lesson 5, we'll explore advanced memory management techniques for long-running single-agent systems.

---

**Practice Exercise**: Build a complete validation and monitoring system. Test with production-like loads and verify that issues are detected within 30 seconds and alerts are actionable.
