# Lesson 4: Performance Monitoring and Analytics

## Introduction

Enterprise-scale context engineering requires sophisticated monitoring and analytics to maintain optimal performance. This lesson teaches you to build comprehensive monitoring systems that provide real-time insights, predictive analytics, and automated optimization for context engineering systems handling millions of operations.

## Advanced Monitoring Architecture

### Comprehensive Performance Monitoring System

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import time
import statistics
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import numpy as np
import heapq
import uuid
from concurrent.futures import ThreadPoolExecutor

class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"      # Distribution of values
    TIMER = "timer"              # Timing measurements
    RATE = "rate"                # Rate of change

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class TimeWindow(Enum):
    """Time windows for aggregation."""
    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    ONE_HOUR = 3600
    ONE_DAY = 86400

@dataclass
class MetricDataPoint:
    """Individual metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.timestamp, self.value, tuple(sorted(self.tags.items()))))

@dataclass
class MetricSeries:
    """Time series of metric data points."""
    metric_name: str
    metric_type: MetricType
    data_points: deque = field(default_factory=lambda: deque(maxlen=10000))
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

    def add_point(self, value: float, timestamp: datetime = None, tags: Dict[str, str] = None):
        """Add a data point to the series."""
        timestamp = timestamp or datetime.now()
        tags = tags or {}

        point = MetricDataPoint(
            timestamp=timestamp,
            value=value,
            tags=tags
        )

        self.data_points.append(point)

    def get_points_in_window(self, start_time: datetime, end_time: datetime) -> List[MetricDataPoint]:
        """Get data points within time window."""
        return [
            point for point in self.data_points
            if start_time <= point.timestamp <= end_time
        ]

    def calculate_statistics(self, window: TimeWindow) -> Dict[str, float]:
        """Calculate statistics for recent time window."""
        cutoff_time = datetime.now() - timedelta(seconds=window.value)
        recent_points = [
            point.value for point in self.data_points
            if point.timestamp >= cutoff_time
        ]

        if not recent_points:
            return {'count': 0}

        return {
            'count': len(recent_points),
            'min': min(recent_points),
            'max': max(recent_points),
            'mean': statistics.mean(recent_points),
            'median': statistics.median(recent_points),
            'stddev': statistics.stdev(recent_points) if len(recent_points) > 1 else 0,
            'p95': np.percentile(recent_points, 95),
            'p99': np.percentile(recent_points, 99)
        }

@dataclass
class Alert:
    """Performance alert definition and state."""
    alert_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "value > 100", "rate_5m > 0.1"
    severity: AlertSeverity
    threshold_value: float
    time_window: TimeWindow

    # State tracking
    is_active: bool = False
    first_triggered: Optional[datetime] = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    # Configuration
    cooldown_seconds: int = 300  # 5 minutes
    escalation_seconds: int = 1800  # 30 minutes
    auto_resolve: bool = True

    def should_trigger(self, current_value: float, metric_stats: Dict[str, float]) -> bool:
        """Check if alert should trigger based on current conditions."""
        # Parse condition (simplified evaluation)
        if ">" in self.condition:
            return current_value > self.threshold_value
        elif "<" in self.condition:
            return current_value < self.threshold_value
        elif "rate_" in self.condition:
            # Rate-based conditions
            time_suffix = self.condition.split("rate_")[1].split()[0]
            if time_suffix == "5m" and "rate_5m" in metric_stats:
                return metric_stats["rate_5m"] > self.threshold_value

        return False

    def trigger(self):
        """Trigger the alert."""
        now = datetime.now()

        if not self.is_active:
            self.first_triggered = now
            self.is_active = True

        self.last_triggered = now
        self.trigger_count += 1

    def resolve(self):
        """Resolve the alert."""
        self.is_active = False

class PerformanceAnalyzer:
    """Advanced performance analysis engine."""

    def __init__(self, analysis_window: int = 3600):  # 1 hour default
        self.analysis_window = analysis_window
        self.anomaly_threshold = 2.0  # Standard deviations
        self.trend_analysis_points = 100

        # Analysis results cache
        self.analysis_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes

        # Pattern recognition
        self.known_patterns: Dict[str, Dict] = {}
        self.seasonal_patterns: Dict[str, List[float]] = {}

    def detect_anomalies(self, metric_series: MetricSeries) -> List[Dict[str, Any]]:
        """Detect anomalies in metric series using statistical methods."""
        if len(metric_series.data_points) < 10:
            return []

        # Get recent data
        recent_points = list(metric_series.data_points)[-100:]
        values = [point.value for point in recent_points]

        # Calculate statistical baseline
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0

        if std_value == 0:
            return []

        anomalies = []

        # Z-score based anomaly detection
        for point in recent_points[-10:]:  # Check last 10 points
            z_score = abs(point.value - mean_value) / std_value

            if z_score > self.anomaly_threshold:
                anomalies.append({
                    'timestamp': point.timestamp.isoformat(),
                    'value': point.value,
                    'expected_value': mean_value,
                    'z_score': z_score,
                    'severity': 'high' if z_score > 3.0 else 'medium',
                    'type': 'statistical_outlier'
                })

        # Trend-based anomaly detection
        if len(values) >= 20:
            trend_anomalies = self._detect_trend_anomalies(recent_points[-20:])
            anomalies.extend(trend_anomalies)

        return anomalies

    def _detect_trend_anomalies(self, points: List[MetricDataPoint]) -> List[Dict[str, Any]]:
        """Detect anomalies based on trend analysis."""
        values = [point.value for point in points]

        # Simple trend detection using linear regression
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]

        anomalies = []

        # Detect sudden trend changes
        if len(values) >= 10:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]

            first_trend = np.polyfit(range(len(first_half)), first_half, 1)[0]
            second_trend = np.polyfit(range(len(second_half)), second_half, 1)[0]

            trend_change = abs(second_trend - first_trend)

            if trend_change > np.std(values):
                anomalies.append({
                    'timestamp': points[-1].timestamp.isoformat(),
                    'value': points[-1].value,
                    'type': 'trend_change',
                    'trend_change': trend_change,
                    'first_trend': first_trend,
                    'second_trend': second_trend,
                    'severity': 'medium'
                })

        return anomalies

    def analyze_performance_trends(self, metric_series: MetricSeries) -> Dict[str, Any]:
        """Analyze performance trends and patterns."""
        cache_key = f"{metric_series.metric_name}_trends"

        # Check cache
        if cache_key in self.analysis_cache:
            cached_result = self.analysis_cache[cache_key]
            if (datetime.now() - cached_result['computed_at']).total_seconds() < self.cache_ttl:
                return cached_result['analysis']

        if len(metric_series.data_points) < self.trend_analysis_points:
            return {'error': 'Insufficient data for trend analysis'}

        recent_points = list(metric_series.data_points)[-self.trend_analysis_points:]
        values = [point.value for point in recent_points]
        timestamps = [point.timestamp for point in recent_points]

        analysis = {
            'trend_direction': self._calculate_trend_direction(values),
            'volatility': self._calculate_volatility(values),
            'seasonal_patterns': self._detect_seasonal_patterns(timestamps, values),
            'performance_degradation': self._detect_performance_degradation(values),
            'capacity_forecast': self._forecast_capacity_needs(values),
            'optimization_recommendations': self._generate_optimization_recommendations(metric_series, values)
        }

        # Cache result
        self.analysis_cache[cache_key] = {
            'analysis': analysis,
            'computed_at': datetime.now()
        }

        return analysis

    def _calculate_trend_direction(self, values: List[float]) -> Dict[str, Any]:
        """Calculate overall trend direction."""
        if len(values) < 2:
            return {'direction': 'unknown', 'confidence': 0}

        x = list(range(len(values)))
        slope, intercept = np.polyfit(x, values, 1)

        # Calculate R-squared for confidence
        y_pred = [slope * xi + intercept for xi in x]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(len(values)))
        ss_tot = sum((val - np.mean(values)) ** 2 for val in values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'

        return {
            'direction': direction,
            'slope': slope,
            'confidence': r_squared,
            'strength': abs(slope) / np.std(values) if np.std(values) > 0 else 0
        }

    def _calculate_volatility(self, values: List[float]) -> Dict[str, float]:
        """Calculate metric volatility."""
        if len(values) < 2:
            return {'volatility': 0, 'coefficient_of_variation': 0}

        std_dev = np.std(values)
        mean_val = np.mean(values)

        return {
            'volatility': std_dev,
            'coefficient_of_variation': std_dev / mean_val if mean_val != 0 else 0,
            'volatility_classification': self._classify_volatility(std_dev / mean_val if mean_val != 0 else 0)
        }

    def _classify_volatility(self, cv: float) -> str:
        """Classify volatility level."""
        if cv < 0.1:
            return 'low'
        elif cv < 0.3:
            return 'medium'
        else:
            return 'high'

    def _detect_seasonal_patterns(self, timestamps: List[datetime], values: List[float]) -> Dict[str, Any]:
        """Detect seasonal patterns in metrics."""
        # Group by hour of day
        hourly_patterns = defaultdict(list)
        for timestamp, value in zip(timestamps, values):
            hourly_patterns[timestamp.hour].append(value)

        hourly_averages = {
            hour: np.mean(vals) for hour, vals in hourly_patterns.items()
            if len(vals) > 0
        }

        # Group by day of week
        daily_patterns = defaultdict(list)
        for timestamp, value in zip(timestamps, values):
            daily_patterns[timestamp.weekday()].append(value)

        daily_averages = {
            day: np.mean(vals) for day, vals in daily_patterns.items()
            if len(vals) > 0
        }

        return {
            'hourly_patterns': hourly_averages,
            'daily_patterns': daily_averages,
            'peak_hour': max(hourly_averages.items(), key=lambda x: x[1])[0] if hourly_averages else None,
            'peak_day': max(daily_averages.items(), key=lambda x: x[1])[0] if daily_averages else None
        }

    def _detect_performance_degradation(self, values: List[float]) -> Dict[str, Any]:
        """Detect performance degradation patterns."""
        if len(values) < 20:
            return {'degradation_detected': False}

        # Compare recent performance to baseline
        baseline_period = values[:len(values)//3]  # First third as baseline
        recent_period = values[-len(values)//3:]   # Last third as recent

        baseline_avg = np.mean(baseline_period)
        recent_avg = np.mean(recent_period)

        degradation_percent = ((recent_avg - baseline_avg) / baseline_avg * 100) if baseline_avg != 0 else 0

        return {
            'degradation_detected': degradation_percent > 10,  # 10% threshold
            'degradation_percent': degradation_percent,
            'baseline_average': baseline_avg,
            'recent_average': recent_avg,
            'severity': 'high' if degradation_percent > 25 else 'medium' if degradation_percent > 10 else 'low'
        }

    def _forecast_capacity_needs(self, values: List[float]) -> Dict[str, Any]:
        """Forecast future capacity needs based on trends."""
        if len(values) < 10:
            return {'forecast_available': False}

        # Simple linear extrapolation
        x = list(range(len(values)))
        slope, intercept = np.polyfit(x, values, 1)

        # Forecast next 24 hours (assuming 1-minute intervals)
        forecast_points = 1440  # 24 * 60
        future_x = list(range(len(values), len(values) + forecast_points))
        forecast_values = [slope * xi + intercept for xi in future_x]

        current_max = max(values)
        forecast_max = max(forecast_values) if forecast_values else current_max

        return {
            'forecast_available': True,
            'current_peak': current_max,
            'forecast_peak_24h': forecast_max,
            'growth_rate': slope,
            'capacity_warning': forecast_max > current_max * 1.5,  # 50% increase warning
            'estimated_saturation_hours': self._estimate_saturation_time(values, slope, intercept)
        }

    def _estimate_saturation_time(self, values: List[float], slope: float, intercept: float) -> Optional[float]:
        """Estimate time to saturation based on trend."""
        if slope <= 0:
            return None  # No growth trend

        current_max = max(values)
        # Assume saturation at 90% of theoretical maximum (rough estimate)
        saturation_threshold = current_max * 2.0  # Assume current max is 50% of capacity

        current_point = len(values)
        current_value = slope * current_point + intercept

        if current_value >= saturation_threshold:
            return 0  # Already at saturation

        # Calculate time to reach saturation
        saturation_point = (saturation_threshold - intercept) / slope
        hours_to_saturation = (saturation_point - current_point) / 60  # Convert minutes to hours

        return max(0, hours_to_saturation)

    def _generate_optimization_recommendations(
        self,
        metric_series: MetricSeries,
        values: List[float]
    ) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        current_avg = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        overall_avg = np.mean(values)
        volatility = np.std(values)

        # High average performance issues
        if current_avg > overall_avg * 1.3:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'recommendation': 'Current performance is 30% above average. Consider scaling up resources.',
                'metric': metric_series.metric_name
            })

        # High volatility issues
        if volatility > overall_avg * 0.5:
            recommendations.append({
                'type': 'stability',
                'priority': 'medium',
                'recommendation': 'High volatility detected. Investigate load balancing and resource allocation.',
                'metric': metric_series.metric_name
            })

        # Trend-based recommendations
        trend = self._calculate_trend_direction(values)
        if trend['direction'] == 'increasing' and trend['confidence'] > 0.7:
            recommendations.append({
                'type': 'capacity',
                'priority': 'medium',
                'recommendation': 'Increasing trend detected. Plan for capacity expansion.',
                'metric': metric_series.metric_name
            })

        return recommendations

class MonitoringDashboard:
    """Real-time monitoring dashboard."""

    def __init__(self):
        self.widgets: Dict[str, Dict] = {}
        self.refresh_interval = 5  # seconds
        self.auto_refresh = True

        # Dashboard configuration
        self.layout = {
            'grid_size': (4, 3),  # 4x3 grid
            'widget_positions': {},
            'themes': ['light', 'dark'],
            'current_theme': 'dark'
        }

    def add_widget(
        self,
        widget_id: str,
        widget_type: str,
        config: Dict[str, Any],
        position: Tuple[int, int] = (0, 0)
    ):
        """Add widget to dashboard."""
        self.widgets[widget_id] = {
            'type': widget_type,
            'config': config,
            'position': position,
            'last_updated': datetime.now(),
            'data': {}
        }

        self.layout['widget_positions'][widget_id] = position

    def update_widget_data(self, widget_id: str, data: Dict[str, Any]):
        """Update widget data."""
        if widget_id in self.widgets:
            self.widgets[widget_id]['data'] = data
            self.widgets[widget_id]['last_updated'] = datetime.now()

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get current dashboard state."""
        return {
            'widgets': self.widgets,
            'layout': self.layout,
            'last_refresh': datetime.now().isoformat(),
            'auto_refresh': self.auto_refresh,
            'refresh_interval': self.refresh_interval
        }

    def generate_summary_report(self, metrics: Dict[str, MetricSeries]) -> Dict[str, Any]:
        """Generate executive summary report."""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'metrics_summary': {},
            'health_status': 'healthy',
            'key_insights': [],
            'action_items': []
        }

        total_metrics = len(metrics)
        healthy_metrics = 0
        warning_metrics = 0
        critical_metrics = 0

        for metric_name, metric_series in metrics.items():
            if len(metric_series.data_points) > 0:
                stats = metric_series.calculate_statistics(TimeWindow.ONE_HOUR)
                latest_value = metric_series.data_points[-1].value

                # Simple health classification
                if 'latency' in metric_name.lower():
                    if latest_value < 50:  # < 50ms
                        healthy_metrics += 1
                    elif latest_value < 200:  # < 200ms
                        warning_metrics += 1
                    else:
                        critical_metrics += 1
                elif 'error' in metric_name.lower():
                    if latest_value < 0.01:  # < 1%
                        healthy_metrics += 1
                    elif latest_value < 0.05:  # < 5%
                        warning_metrics += 1
                    else:
                        critical_metrics += 1
                else:
                    healthy_metrics += 1  # Default to healthy

                summary['metrics_summary'][metric_name] = {
                    'current_value': latest_value,
                    'average': stats.get('mean', 0),
                    'p95': stats.get('p95', 0),
                    'trend': 'stable'  # Simplified
                }

        # Overall health status
        if critical_metrics > 0:
            summary['health_status'] = 'critical'
        elif warning_metrics > total_metrics * 0.3:  # More than 30% warnings
            summary['health_status'] = 'warning'
        else:
            summary['health_status'] = 'healthy'

        # Generate insights
        if critical_metrics > 0:
            summary['key_insights'].append(f"{critical_metrics} metrics are in critical state")
            summary['action_items'].append("Immediate investigation required for critical metrics")

        if warning_metrics > 0:
            summary['key_insights'].append(f"{warning_metrics} metrics require attention")
            summary['action_items'].append("Schedule performance review for warning metrics")

        return summary

class EnterpriseMonitoringSystem:
    """Comprehensive enterprise monitoring system."""

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days

        # Core components
        self.metrics: Dict[str, MetricSeries] = {}
        self.alerts: Dict[str, Alert] = {}
        self.analyzer = PerformanceAnalyzer()
        self.dashboard = MonitoringDashboard()

        # Background processing
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.alert_queue: asyncio.Queue = asyncio.Queue()

        # Performance tracking
        self.system_metrics = {
            'monitoring_overhead_ms': deque(maxlen=1000),
            'alerts_triggered': 0,
            'alerts_resolved': 0,
            'data_points_processed': 0
        }

        # Background tasks
        self.processing_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the monitoring system."""
        self.processing_task = asyncio.create_task(self._processing_loop())
        self.alert_task = asyncio.create_task(self._alert_processing_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Setup default dashboard widgets
        self._setup_default_dashboard()

        logging.info("Enterprise monitoring system started")

    async def stop(self):
        """Stop the monitoring system gracefully."""
        # Cancel background tasks
        for task in [self.processing_task, self.alert_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logging.info("Enterprise monitoring system stopped")

    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Dict[str, str] = None,
        timestamp: datetime = None
    ):
        """Record a metric value."""
        start_time = time.time()

        # Create metric series if it doesn't exist
        if metric_name not in self.metrics:
            self.metrics[metric_name] = MetricSeries(
                metric_name=metric_name,
                metric_type=metric_type
            )

        # Add data point
        self.metrics[metric_name].add_point(value, timestamp, tags)

        # Queue for processing
        self.processing_queue.put_nowait({
            'type': 'metric_update',
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp or datetime.now()
        })

        # Track monitoring overhead
        overhead = (time.time() - start_time) * 1000
        self.system_metrics['monitoring_overhead_ms'].append(overhead)
        self.system_metrics['data_points_processed'] += 1

    def create_alert(
        self,
        alert_id: str,
        name: str,
        metric_name: str,
        condition: str,
        threshold_value: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        time_window: TimeWindow = TimeWindow.FIVE_MINUTES
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            alert_id=alert_id,
            name=name,
            description=f"Alert for {metric_name}: {condition}",
            metric_name=metric_name,
            condition=condition,
            severity=severity,
            threshold_value=threshold_value,
            time_window=time_window
        )

        self.alerts[alert_id] = alert
        return alert

    async def get_real_time_metrics(self, time_window: TimeWindow = TimeWindow.FIVE_MINUTES) -> Dict[str, Any]:
        """Get real-time metrics summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'time_window': time_window.name,
            'metrics': {},
            'system_health': {},
            'active_alerts': []
        }

        # Process each metric
        for metric_name, metric_series in self.metrics.items():
            if len(metric_series.data_points) > 0:
                stats = metric_series.calculate_statistics(time_window)

                # Detect anomalies
                anomalies = self.analyzer.detect_anomalies(metric_series)

                summary['metrics'][metric_name] = {
                    'current_value': metric_series.data_points[-1].value,
                    'statistics': stats,
                    'anomalies': len(anomalies),
                    'trend': self.analyzer._calculate_trend_direction(
                        [p.value for p in list(metric_series.data_points)[-50:]]
                    )
                }

        # System health
        active_alerts = [alert for alert in self.alerts.values() if alert.is_active]
        critical_alerts = [alert for alert in active_alerts if alert.severity == AlertSeverity.CRITICAL]

        summary['system_health'] = {
            'overall_status': 'critical' if critical_alerts else 'warning' if active_alerts else 'healthy',
            'active_alerts_count': len(active_alerts),
            'critical_alerts_count': len(critical_alerts),
            'monitoring_overhead_avg_ms': (
                np.mean(self.system_metrics['monitoring_overhead_ms'])
                if self.system_metrics['monitoring_overhead_ms'] else 0
            )
        }

        # Active alerts summary
        summary['active_alerts'] = [
            {
                'alert_id': alert.alert_id,
                'name': alert.name,
                'severity': alert.severity.value,
                'metric_name': alert.metric_name,
                'triggered_at': alert.first_triggered.isoformat() if alert.first_triggered else None,
                'trigger_count': alert.trigger_count
            }
            for alert in active_alerts
        ]

        return summary

    async def generate_performance_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        report = {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': hours_back
            },
            'executive_summary': {},
            'detailed_metrics': {},
            'performance_trends': {},
            'anomalies': {},
            'recommendations': []
        }

        # Generate executive summary
        report['executive_summary'] = self.dashboard.generate_summary_report(self.metrics)

        # Detailed metrics analysis
        for metric_name, metric_series in self.metrics.items():
            if len(metric_series.data_points) > 0:
                # Get data points in time window
                window_points = metric_series.get_points_in_window(start_time, end_time)

                if window_points:
                    values = [p.value for p in window_points]

                    report['detailed_metrics'][metric_name] = {
                        'data_points': len(window_points),
                        'min_value': min(values),
                        'max_value': max(values),
                        'avg_value': np.mean(values),
                        'p50': np.percentile(values, 50),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99),
                        'stddev': np.std(values)
                    }

                    # Performance trend analysis
                    trend_analysis = self.analyzer.analyze_performance_trends(metric_series)
                    report['performance_trends'][metric_name] = trend_analysis

                    # Anomaly detection
                    anomalies = self.analyzer.detect_anomalies(metric_series)
                    if anomalies:
                        report['anomalies'][metric_name] = anomalies

        # Generate recommendations
        all_recommendations = []
        for metric_name, metric_series in self.metrics.items():
            if len(metric_series.data_points) > 0:
                values = [p.value for p in list(metric_series.data_points)[-100:]]
                recommendations = self.analyzer._generate_optimization_recommendations(
                    metric_series, values
                )
                all_recommendations.extend(recommendations)

        report['recommendations'] = all_recommendations

        return report

    async def _processing_loop(self):
        """Background processing loop."""
        while True:
            try:
                # Process queued items
                try:
                    item = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                    await self._process_queue_item(item)
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                logging.error(f"Processing loop error: {e}")
                await asyncio.sleep(1)

    async def _process_queue_item(self, item: Dict[str, Any]):
        """Process individual queue item."""
        if item['type'] == 'metric_update':
            metric_name = item['metric_name']

            # Check alerts for this metric
            await self._check_alerts_for_metric(metric_name)

            # Update dashboard widgets
            await self._update_dashboard_widgets(metric_name)

    async def _check_alerts_for_metric(self, metric_name: str):
        """Check alerts for specific metric."""
        metric_series = self.metrics.get(metric_name)
        if not metric_series or len(metric_series.data_points) == 0:
            return

        current_value = metric_series.data_points[-1].value
        metric_stats = metric_series.calculate_statistics(TimeWindow.FIVE_MINUTES)

        for alert in self.alerts.values():
            if alert.metric_name == metric_name:
                should_trigger = alert.should_trigger(current_value, metric_stats)

                if should_trigger and not alert.is_active:
                    alert.trigger()
                    await self.alert_queue.put({
                        'type': 'alert_triggered',
                        'alert': alert,
                        'current_value': current_value
                    })
                    self.system_metrics['alerts_triggered'] += 1

                elif not should_trigger and alert.is_active and alert.auto_resolve:
                    alert.resolve()
                    await self.alert_queue.put({
                        'type': 'alert_resolved',
                        'alert': alert,
                        'current_value': current_value
                    })
                    self.system_metrics['alerts_resolved'] += 1

    async def _update_dashboard_widgets(self, metric_name: str):
        """Update dashboard widgets with new metric data."""
        metric_series = self.metrics.get(metric_name)
        if not metric_series:
            return

        # Update relevant widgets
        for widget_id, widget in self.dashboard.widgets.items():
            widget_config = widget.get('config', {})

            if widget_config.get('metric_name') == metric_name:
                # Calculate widget data based on type
                if widget['type'] == 'line_chart':
                    recent_points = list(metric_series.data_points)[-100:]
                    widget_data = {
                        'data_points': [
                            {'x': p.timestamp.isoformat(), 'y': p.value}
                            for p in recent_points
                        ]
                    }
                elif widget['type'] == 'gauge':
                    current_value = metric_series.data_points[-1].value
                    widget_data = {
                        'current_value': current_value,
                        'max_value': widget_config.get('max_value', 100)
                    }
                elif widget['type'] == 'stat':
                    stats = metric_series.calculate_statistics(TimeWindow.ONE_HOUR)
                    widget_data = stats
                else:
                    continue

                self.dashboard.update_widget_data(widget_id, widget_data)

    async def _alert_processing_loop(self):
        """Process alert notifications."""
        while True:
            try:
                alert_item = await self.alert_queue.get()
                await self._handle_alert(alert_item)
            except Exception as e:
                logging.error(f"Alert processing error: {e}")
                await asyncio.sleep(1)

    async def _handle_alert(self, alert_item: Dict[str, Any]):
        """Handle alert notifications."""
        alert_type = alert_item['type']
        alert = alert_item['alert']
        current_value = alert_item['current_value']

        if alert_type == 'alert_triggered':
            logging.warning(
                f"ALERT TRIGGERED: {alert.name} - {alert.metric_name} = {current_value} "
                f"(threshold: {alert.threshold_value})"
            )

            # In production, this would send notifications via email, Slack, PagerDuty, etc.

        elif alert_type == 'alert_resolved':
            logging.info(
                f"ALERT RESOLVED: {alert.name} - {alert.metric_name} = {current_value}"
            )

    async def _cleanup_loop(self):
        """Clean up old data based on retention policy."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                cutoff_time = datetime.now() - timedelta(days=self.retention_days)

                for metric_series in self.metrics.values():
                    # Remove old data points
                    while (metric_series.data_points and
                           metric_series.data_points[0].timestamp < cutoff_time):
                        metric_series.data_points.popleft()

                logging.info(f"Cleaned up data older than {self.retention_days} days")

            except Exception as e:
                logging.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)

    def _setup_default_dashboard(self):
        """Setup default dashboard widgets."""
        # Response time chart
        self.dashboard.add_widget(
            'response_time_chart',
            'line_chart',
            {
                'metric_name': 'response_time_ms',
                'title': 'Response Time (ms)',
                'time_window': '1h'
            },
            position=(0, 0)
        )

        # Throughput gauge
        self.dashboard.add_widget(
            'throughput_gauge',
            'gauge',
            {
                'metric_name': 'requests_per_second',
                'title': 'Requests/Second',
                'max_value': 1000
            },
            position=(1, 0)
        )

        # Error rate stat
        self.dashboard.add_widget(
            'error_rate_stat',
            'stat',
            {
                'metric_name': 'error_rate',
                'title': 'Error Rate',
                'format': 'percentage'
            },
            position=(2, 0)
        )

        # Cache hit rate
        self.dashboard.add_widget(
            'cache_hit_rate',
            'stat',
            {
                'metric_name': 'cache_hit_rate',
                'title': 'Cache Hit Rate',
                'format': 'percentage'
            },
            position=(3, 0)
        )

# Example usage and demonstration
async def monitoring_system_demo():
    """Demonstrate comprehensive monitoring system."""

    # Create and start monitoring system
    monitoring = EnterpriseMonitoringSystem(retention_days=7)
    await monitoring.start()

    try:
        print("Setting up monitoring system...")

        # Create alerts
        monitoring.create_alert(
            'high_latency',
            'High Response Time',
            'response_time_ms',
            'value > 100',
            100.0,
            AlertSeverity.WARNING
        )

        monitoring.create_alert(
            'critical_latency',
            'Critical Response Time',
            'response_time_ms',
            'value > 500',
            500.0,
            AlertSeverity.CRITICAL
        )

        monitoring.create_alert(
            'high_error_rate',
            'High Error Rate',
            'error_rate',
            'value > 0.05',
            0.05,
            AlertSeverity.ERROR
        )

        print("Simulating system metrics...")

        # Simulate various metrics over time
        for i in range(200):
            # Simulate response time with some spikes
            base_latency = 50 + np.random.normal(0, 10)
            if i % 30 == 0:  # Periodic spikes
                base_latency += np.random.uniform(100, 300)

            monitoring.record_metric(
                'response_time_ms',
                max(0, base_latency),
                MetricType.TIMER
            )

            # Simulate throughput
            base_throughput = 100 + np.random.normal(0, 20)
            monitoring.record_metric(
                'requests_per_second',
                max(0, base_throughput),
                MetricType.GAUGE
            )

            # Simulate error rate
            base_error_rate = 0.02 + np.random.normal(0, 0.01)
            if i % 50 == 0:  # Occasional error spikes
                base_error_rate += np.random.uniform(0.05, 0.15)

            monitoring.record_metric(
                'error_rate',
                max(0, min(1, base_error_rate)),
                MetricType.GAUGE
            )

            # Simulate cache hit rate
            cache_hit_rate = 0.95 + np.random.normal(0, 0.02)
            monitoring.record_metric(
                'cache_hit_rate',
                max(0, min(1, cache_hit_rate)),
                MetricType.GAUGE
            )

            # Simulate CPU utilization
            cpu_util = 0.6 + np.random.normal(0, 0.1)
            monitoring.record_metric(
                'cpu_utilization',
                max(0, min(1, cpu_util)),
                MetricType.GAUGE
            )

            # Brief delay to simulate real-time data
            await asyncio.sleep(0.01)

        print("Generating real-time metrics report...")

        # Get real-time metrics
        real_time_report = await monitoring.get_real_time_metrics(TimeWindow.FIVE_MINUTES)
        print(json.dumps(real_time_report, indent=2))

        print("\nGenerating performance analysis report...")

        # Generate performance report
        performance_report = await monitoring.generate_performance_report(hours_back=1)

        # Print summary sections
        print(f"Executive Summary:")
        print(json.dumps(performance_report['executive_summary'], indent=2))

        print(f"\nPerformance Trends:")
        for metric_name, trends in performance_report['performance_trends'].items():
            if 'trend_direction' in trends:
                print(f"  {metric_name}: {trends['trend_direction']}")

        print(f"\nAnomalies Detected:")
        for metric_name, anomalies in performance_report['anomalies'].items():
            print(f"  {metric_name}: {len(anomalies)} anomalies")

        print(f"\nRecommendations:")
        for rec in performance_report['recommendations']:
            print(f"  [{rec['priority']}] {rec['recommendation']}")

        # Dashboard state
        print(f"\nDashboard State:")
        dashboard_state = monitoring.dashboard.get_dashboard_state()
        print(f"  Widgets: {len(dashboard_state['widgets'])}")
        print(f"  Last Refresh: {dashboard_state['last_refresh']}")

    finally:
        await monitoring.stop()

# Run the demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(monitoring_system_demo())
```

## Key Takeaways

1. **Comprehensive Metrics**: Track performance, availability, and business metrics with intelligent aggregation

2. **Real-Time Analytics**: Provide instant insights with anomaly detection and trend analysis

3. **Predictive Monitoring**: Forecast capacity needs and performance degradation before they impact users

4. **Intelligent Alerting**: Smart alert management with escalation and auto-resolution capabilities

5. **Executive Reporting**: Generate actionable insights for both technical teams and business stakeholders

6. **Performance Optimization**: Automated recommendations based on historical analysis and pattern recognition

## What's Next

In Lesson 5, we'll explore enterprise deployment strategies and operational patterns for production-ready context engineering systems.

---

**Practice Exercise**: Build a complete monitoring system tracking 20+ metrics with real-time anomaly detection. Demonstrate predictive capacity planning, intelligent alerting with <1% false positive rate, and executive reporting capabilities across multiple time horizons.
