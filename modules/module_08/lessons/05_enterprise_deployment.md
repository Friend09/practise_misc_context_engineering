# Lesson 5: Enterprise Deployment and Operations

## Introduction

Deploying context engineering systems at enterprise scale requires sophisticated deployment strategies, operational excellence, and comprehensive disaster recovery capabilities. This lesson teaches you to build production-ready deployment architectures that achieve 99.99%+ uptime while supporting zero-downtime updates and global scale operations.

## Enterprise Deployment Architecture

### Production-Ready Deployment System

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import time
import hashlib
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import yaml
import subprocess
import os
import shutil
from pathlib import Path

class DeploymentStage(Enum):
    """Deployment stages in the pipeline."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"

class DeploymentStrategy(Enum):
    """Deployment strategies for different scenarios."""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"

class HealthStatus(Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class DeploymentTarget:
    """Target environment for deployment."""
    name: str
    stage: DeploymentStage
    region: str
    cluster_config: Dict[str, Any]
    resource_limits: Dict[str, Any]
    scaling_config: Dict[str, Any]
    security_config: Dict[str, Any]

    # Network configuration
    vpc_config: Dict[str, str] = field(default_factory=dict)
    load_balancer_config: Dict[str, Any] = field(default_factory=dict)

    # Monitoring and observability
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    logging_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentArtifact:
    """Deployment artifact with metadata."""
    artifact_id: str
    version: str
    build_timestamp: datetime
    git_commit: str
    docker_image: str
    config_checksum: str

    # Artifact metadata
    size_bytes: int = 0
    vulnerability_scan_passed: bool = False
    performance_test_passed: bool = False
    security_scan_passed: bool = False

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    runtime_requirements: Dict[str, str] = field(default_factory=dict)

@dataclass
class DeploymentPlan:
    """Comprehensive deployment plan."""
    plan_id: str
    artifact: DeploymentArtifact
    target: DeploymentTarget
    strategy: DeploymentStrategy
    rollback_plan: Optional['DeploymentPlan'] = None

    # Execution configuration
    batch_size: int = 1
    max_unavailable: int = 0
    health_check_timeout: int = 300
    rollback_threshold: float = 0.05  # 5% error rate

    # Pre and post deployment hooks
    pre_deployment_hooks: List[str] = field(default_factory=list)
    post_deployment_hooks: List[str] = field(default_factory=list)

    # Traffic management
    traffic_split: Dict[str, float] = field(default_factory=dict)
    canary_percentage: float = 5.0

    # Approval requirements
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None

class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.check_results: Dict[str, HealthStatus] = {}

        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False

    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.check_results[name] = HealthStatus.UNKNOWN

    async def start_monitoring(self):
        """Start background health monitoring."""
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop background health monitoring."""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """Background health monitoring loop."""
        while self.is_running:
            try:
                await self._run_all_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _run_all_checks(self):
        """Run all registered health checks."""
        for check_name, check_func in self.health_checks.items():
            try:
                # Run health check with timeout
                is_healthy = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, check_func),
                    timeout=10.0
                )

                status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                self.check_results[check_name] = status

                # Record in history
                self.health_history[check_name].append({
                    'timestamp': datetime.now(),
                    'status': status,
                    'result': is_healthy
                })

            except asyncio.TimeoutError:
                self.check_results[check_name] = HealthStatus.UNKNOWN
                logging.warning(f"Health check {check_name} timed out")
            except Exception as e:
                self.check_results[check_name] = HealthStatus.UNHEALTHY
                logging.error(f"Health check {check_name} failed: {e}")

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.check_results:
            return HealthStatus.UNKNOWN

        statuses = list(self.check_results.values())

        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            'overall_status': self.get_overall_health().value,
            'individual_checks': {
                name: {
                    'status': status.value,
                    'last_check': datetime.now().isoformat(),
                    'history_count': len(self.health_history[name])
                }
                for name, status in self.check_results.items()
            },
            'summary': {
                'total_checks': len(self.check_results),
                'healthy_checks': sum(1 for s in self.check_results.values() if s == HealthStatus.HEALTHY),
                'unhealthy_checks': sum(1 for s in self.check_results.values() if s == HealthStatus.UNHEALTHY),
                'unknown_checks': sum(1 for s in self.check_results.values() if s == HealthStatus.UNKNOWN)
            }
        }

class TrafficManager:
    """Advanced traffic management for deployments."""

    def __init__(self):
        self.traffic_rules: Dict[str, Dict] = {}
        self.active_splits: Dict[str, Dict[str, float]] = {}
        self.request_counters: Dict[str, int] = defaultdict(int)

        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict] = {}

        # A/B testing
        self.ab_tests: Dict[str, Dict] = {}

    def create_traffic_split(
        self,
        service_name: str,
        splits: Dict[str, float],
        sticky_sessions: bool = False
    ):
        """Create traffic split configuration."""
        # Validate splits sum to 1.0
        total_weight = sum(splits.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Traffic splits must sum to 1.0, got {total_weight}")

        self.active_splits[service_name] = splits.copy()

        logging.info(f"Created traffic split for {service_name}: {splits}")

    def route_request(
        self,
        service_name: str,
        request_id: str,
        user_id: str = None
    ) -> str:
        """Route request based on traffic rules."""
        if service_name not in self.active_splits:
            return "default"

        splits = self.active_splits[service_name]

        # Handle sticky sessions
        if user_id and self._has_sticky_session(service_name, user_id):
            return self._get_sticky_target(service_name, user_id)

        # Weighted random selection
        import random
        rand_value = random.random()
        cumulative_weight = 0.0

        for target, weight in splits.items():
            cumulative_weight += weight
            if rand_value <= cumulative_weight:
                self.request_counters[f"{service_name}:{target}"] += 1

                # Store sticky session if enabled
                if user_id:
                    self._set_sticky_session(service_name, user_id, target)

                return target

        # Fallback to first target
        return list(splits.keys())[0]

    def update_traffic_weights(
        self,
        service_name: str,
        new_weights: Dict[str, float],
        gradual_shift: bool = True,
        shift_duration_minutes: int = 10
    ):
        """Update traffic weights, optionally with gradual shifting."""
        if not gradual_shift:
            self.active_splits[service_name] = new_weights.copy()
            return

        # Implement gradual traffic shifting
        asyncio.create_task(
            self._gradual_traffic_shift(service_name, new_weights, shift_duration_minutes)
        )

    async def _gradual_traffic_shift(
        self,
        service_name: str,
        target_weights: Dict[str, float],
        duration_minutes: int
    ):
        """Gradually shift traffic weights over time."""
        if service_name not in self.active_splits:
            self.active_splits[service_name] = target_weights
            return

        start_weights = self.active_splits[service_name].copy()
        steps = duration_minutes * 2  # Update every 30 seconds

        for step in range(steps + 1):
            progress = step / steps

            # Linear interpolation between start and target weights
            current_weights = {}
            for target in target_weights:
                start_weight = start_weights.get(target, 0.0)
                target_weight = target_weights[target]
                current_weight = start_weight + (target_weight - start_weight) * progress
                current_weights[target] = current_weight

            # Normalize weights
            total_weight = sum(current_weights.values())
            if total_weight > 0:
                current_weights = {
                    target: weight / total_weight
                    for target, weight in current_weights.items()
                }

            self.active_splits[service_name] = current_weights

            if step < steps:
                await asyncio.sleep(30)  # 30 seconds between updates

        logging.info(f"Completed gradual traffic shift for {service_name}")

    def enable_circuit_breaker(
        self,
        service_name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3
    ):
        """Enable circuit breaker for service."""
        self.circuit_breakers[service_name] = {
            'state': 'closed',  # closed, open, half_open
            'failure_count': 0,
            'failure_threshold': failure_threshold,
            'timeout_seconds': timeout_seconds,
            'half_open_max_calls': half_open_max_calls,
            'half_open_calls': 0,
            'last_failure_time': None
        }

    def record_request_result(self, service_name: str, target: str, success: bool):
        """Record request result for circuit breaker."""
        if service_name not in self.circuit_breakers:
            return

        breaker = self.circuit_breakers[service_name]

        if success:
            if breaker['state'] == 'half_open':
                breaker['half_open_calls'] += 1
                if breaker['half_open_calls'] >= breaker['half_open_max_calls']:
                    breaker['state'] = 'closed'
                    breaker['failure_count'] = 0
                    breaker['half_open_calls'] = 0
            elif breaker['state'] == 'closed':
                breaker['failure_count'] = max(0, breaker['failure_count'] - 1)
        else:
            breaker['failure_count'] += 1
            breaker['last_failure_time'] = datetime.now()

            if breaker['failure_count'] >= breaker['failure_threshold']:
                breaker['state'] = 'open'
                logging.warning(f"Circuit breaker opened for {service_name}")

    def should_allow_request(self, service_name: str) -> bool:
        """Check if request should be allowed through circuit breaker."""
        if service_name not in self.circuit_breakers:
            return True

        breaker = self.circuit_breakers[service_name]

        if breaker['state'] == 'closed':
            return True
        elif breaker['state'] == 'open':
            # Check if timeout has passed
            if breaker['last_failure_time']:
                time_since_failure = (datetime.now() - breaker['last_failure_time']).total_seconds()
                if time_since_failure >= breaker['timeout_seconds']:
                    breaker['state'] = 'half_open'
                    breaker['half_open_calls'] = 0
                    return True
            return False
        elif breaker['state'] == 'half_open':
            return breaker['half_open_calls'] < breaker['half_open_max_calls']

        return False

    def _has_sticky_session(self, service_name: str, user_id: str) -> bool:
        """Check if user has sticky session."""
        # Simplified implementation
        return False

    def _get_sticky_target(self, service_name: str, user_id: str) -> str:
        """Get sticky session target."""
        # Simplified implementation
        return "default"

    def _set_sticky_session(self, service_name: str, user_id: str, target: str):
        """Set sticky session for user."""
        # Simplified implementation
        pass

    def get_traffic_stats(self) -> Dict[str, Any]:
        """Get traffic management statistics."""
        return {
            'active_splits': self.active_splits,
            'request_counters': dict(self.request_counters),
            'circuit_breakers': {
                name: {
                    'state': cb['state'],
                    'failure_count': cb['failure_count'],
                    'failure_threshold': cb['failure_threshold']
                }
                for name, cb in self.circuit_breakers.items()
            }
        }

class DeploymentOrchestrator:
    """Enterprise deployment orchestration system."""

    def __init__(self):
        self.deployment_queue: asyncio.Queue = asyncio.Queue()
        self.active_deployments: Dict[str, Dict] = {}
        self.deployment_history: List[Dict] = []

        # Components
        self.health_checker = HealthChecker()
        self.traffic_manager = TrafficManager()

        # Configuration
        self.max_concurrent_deployments = 3
        self.deployment_timeout_minutes = 60

        # Background tasks
        self.orchestrator_task: Optional[asyncio.Task] = None
        self.is_running = False

    async def start(self):
        """Start the deployment orchestrator."""
        self.is_running = True
        await self.health_checker.start_monitoring()
        self.orchestrator_task = asyncio.create_task(self._orchestration_loop())

        logging.info("Deployment orchestrator started")

    async def stop(self):
        """Stop the deployment orchestrator."""
        self.is_running = False
        await self.health_checker.stop_monitoring()

        if self.orchestrator_task:
            self.orchestrator_task.cancel()
            try:
                await self.orchestrator_task
            except asyncio.CancelledError:
                pass

        logging.info("Deployment orchestrator stopped")

    async def deploy(self, deployment_plan: DeploymentPlan) -> Dict[str, Any]:
        """Execute deployment plan."""
        deployment_id = str(uuid.uuid4())

        # Validate deployment plan
        validation_result = self._validate_deployment_plan(deployment_plan)
        if not validation_result['valid']:
            raise ValueError(f"Invalid deployment plan: {validation_result['errors']}")

        # Check approval requirements
        if deployment_plan.requires_approval and not deployment_plan.approved_by:
            raise ValueError("Deployment requires approval but is not approved")

        # Queue deployment
        deployment_request = {
            'deployment_id': deployment_id,
            'plan': deployment_plan,
            'status': 'queued',
            'created_at': datetime.now(),
            'started_at': None,
            'completed_at': None,
            'error': None
        }

        await self.deployment_queue.put(deployment_request)

        logging.info(f"Deployment {deployment_id} queued for {deployment_plan.target.name}")

        return {
            'deployment_id': deployment_id,
            'status': 'queued',
            'estimated_start_time': self._estimate_start_time()
        }

    async def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.is_running:
            try:
                # Check if we can start new deployments
                if len(self.active_deployments) < self.max_concurrent_deployments:
                    try:
                        deployment_request = await asyncio.wait_for(
                            self.deployment_queue.get(),
                            timeout=5.0
                        )

                        # Start deployment
                        asyncio.create_task(self._execute_deployment(deployment_request))

                    except asyncio.TimeoutError:
                        pass

                # Check for deployment timeouts
                await self._check_deployment_timeouts()

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logging.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(10)

    async def _execute_deployment(self, deployment_request: Dict[str, Any]):
        """Execute individual deployment."""
        deployment_id = deployment_request['deployment_id']
        plan = deployment_request['plan']

        deployment_request['status'] = 'running'
        deployment_request['started_at'] = datetime.now()
        self.active_deployments[deployment_id] = deployment_request

        try:
            logging.info(f"Starting deployment {deployment_id}")

            # Execute deployment strategy
            if plan.strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self._execute_blue_green_deployment(deployment_id, plan)
            elif plan.strategy == DeploymentStrategy.CANARY:
                result = await self._execute_canary_deployment(deployment_id, plan)
            elif plan.strategy == DeploymentStrategy.ROLLING_UPDATE:
                result = await self._execute_rolling_deployment(deployment_id, plan)
            else:
                raise ValueError(f"Unsupported deployment strategy: {plan.strategy}")

            # Mark as completed
            deployment_request['status'] = 'completed'
            deployment_request['completed_at'] = datetime.now()
            deployment_request['result'] = result

            logging.info(f"Deployment {deployment_id} completed successfully")

        except Exception as e:
            deployment_request['status'] = 'failed'
            deployment_request['completed_at'] = datetime.now()
            deployment_request['error'] = str(e)

            logging.error(f"Deployment {deployment_id} failed: {e}")

            # Attempt rollback if configured
            if plan.rollback_plan:
                try:
                    await self._execute_rollback(deployment_id, plan.rollback_plan)
                    deployment_request['status'] = 'rolled_back'
                except Exception as rollback_error:
                    logging.error(f"Rollback failed for {deployment_id}: {rollback_error}")
                    deployment_request['status'] = 'rollback_failed'

        finally:
            # Move to history and clean up
            self.deployment_history.append(deployment_request.copy())
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]

            # Keep only last 100 deployments in history
            if len(self.deployment_history) > 100:
                self.deployment_history = self.deployment_history[-100:]

    async def _execute_blue_green_deployment(
        self,
        deployment_id: str,
        plan: DeploymentPlan
    ) -> Dict[str, Any]:
        """Execute blue-green deployment strategy."""
        logging.info(f"Executing blue-green deployment {deployment_id}")

        # Phase 1: Deploy to green environment
        await self._deploy_to_environment(plan, "green")

        # Phase 2: Health check green environment
        green_healthy = await self._wait_for_health_check("green", plan.health_check_timeout)
        if not green_healthy:
            raise Exception("Green environment failed health checks")

        # Phase 3: Switch traffic to green
        self.traffic_manager.create_traffic_split(
            plan.target.name,
            {"green": 1.0, "blue": 0.0}
        )

        # Phase 4: Monitor for issues
        await self._monitor_deployment_health(plan, duration_seconds=300)

        # Phase 5: Decommission blue environment
        await self._cleanup_environment("blue")

        return {
            'strategy': 'blue_green',
            'green_deployment_time': datetime.now().isoformat(),
            'traffic_switched': True,
            'blue_decommissioned': True
        }

    async def _execute_canary_deployment(
        self,
        deployment_id: str,
        plan: DeploymentPlan
    ) -> Dict[str, Any]:
        """Execute canary deployment strategy."""
        logging.info(f"Executing canary deployment {deployment_id}")

        # Phase 1: Deploy canary version
        await self._deploy_to_environment(plan, "canary")

        # Phase 2: Health check canary
        canary_healthy = await self._wait_for_health_check("canary", plan.health_check_timeout)
        if not canary_healthy:
            raise Exception("Canary deployment failed health checks")

        # Phase 3: Start with small traffic percentage
        canary_percentage = plan.canary_percentage
        self.traffic_manager.create_traffic_split(
            plan.target.name,
            {"canary": canary_percentage / 100, "stable": 1 - canary_percentage / 100}
        )

        # Phase 4: Gradually increase traffic if healthy
        traffic_increase_steps = [10, 25, 50, 75, 100]

        for target_percentage in traffic_increase_steps:
            if target_percentage <= canary_percentage:
                continue

            # Wait and monitor current traffic level
            await asyncio.sleep(120)  # 2 minutes between increases

            # Check health metrics
            is_healthy = await self._check_deployment_metrics(plan, "canary")
            if not is_healthy:
                # Rollback traffic
                self.traffic_manager.create_traffic_split(
                    plan.target.name,
                    {"canary": 0.0, "stable": 1.0}
                )
                raise Exception(f"Canary failed health check at {target_percentage}% traffic")

            # Increase traffic
            self.traffic_manager.update_traffic_weights(
                plan.target.name,
                {"canary": target_percentage / 100, "stable": 1 - target_percentage / 100},
                gradual_shift=True,
                shift_duration_minutes=2
            )

        # Phase 5: Complete migration
        await self._cleanup_environment("stable")

        return {
            'strategy': 'canary',
            'final_traffic_percentage': 100,
            'canary_promotion_successful': True,
            'stable_decommissioned': True
        }

    async def _execute_rolling_deployment(
        self,
        deployment_id: str,
        plan: DeploymentPlan
    ) -> Dict[str, Any]:
        """Execute rolling update deployment strategy."""
        logging.info(f"Executing rolling deployment {deployment_id}")

        # Simulate rolling update across multiple instances
        total_instances = plan.target.scaling_config.get('replicas', 3)
        batch_size = plan.batch_size

        updated_instances = 0

        for batch_start in range(0, total_instances, batch_size):
            batch_end = min(batch_start + batch_size, total_instances)
            batch_instances = list(range(batch_start, batch_end))

            logging.info(f"Updating instances {batch_instances}")

            # Update batch of instances
            await self._update_instances(batch_instances, plan)

            # Health check updated instances
            for instance in batch_instances:
                healthy = await self._check_instance_health(instance, plan.health_check_timeout)
                if not healthy:
                    raise Exception(f"Instance {instance} failed health check")

            updated_instances += len(batch_instances)

            # Brief pause between batches
            if batch_end < total_instances:
                await asyncio.sleep(30)

        return {
            'strategy': 'rolling_update',
            'total_instances': total_instances,
            'updated_instances': updated_instances,
            'batch_size': batch_size,
            'rolling_update_successful': True
        }

    async def _deploy_to_environment(self, plan: DeploymentPlan, environment: str):
        """Deploy artifact to specific environment."""
        logging.info(f"Deploying {plan.artifact.version} to {environment}")

        # Execute pre-deployment hooks
        for hook in plan.pre_deployment_hooks:
            await self._execute_hook(hook, environment)

        # Simulate deployment process
        await asyncio.sleep(2)  # Simulate deployment time

        # Execute post-deployment hooks
        for hook in plan.post_deployment_hooks:
            await self._execute_hook(hook, environment)

    async def _wait_for_health_check(self, environment: str, timeout_seconds: int) -> bool:
        """Wait for environment to become healthy."""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            # Simulate health check
            await asyncio.sleep(5)

            # In real implementation, check actual health endpoints
            health_status = await self._check_environment_health(environment)
            if health_status == HealthStatus.HEALTHY:
                return True

        return False

    async def _check_environment_health(self, environment: str) -> HealthStatus:
        """Check health status of environment."""
        # Simulate health check with high probability of success
        import random
        if random.random() < 0.9:  # 90% success rate
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNHEALTHY

    async def _monitor_deployment_health(self, plan: DeploymentPlan, duration_seconds: int):
        """Monitor deployment health for specified duration."""
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            # Check error rates and performance metrics
            is_healthy = await self._check_deployment_metrics(plan, "production")

            if not is_healthy:
                raise Exception("Deployment health check failed during monitoring period")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _check_deployment_metrics(self, plan: DeploymentPlan, environment: str) -> bool:
        """Check deployment health metrics."""
        # Simulate metric checking
        import random
        error_rate = random.uniform(0, 0.1)  # 0-10% error rate

        # Check against rollback threshold
        return error_rate < plan.rollback_threshold

    async def _cleanup_environment(self, environment: str):
        """Clean up old environment."""
        logging.info(f"Cleaning up {environment} environment")
        await asyncio.sleep(1)  # Simulate cleanup

    async def _update_instances(self, instances: List[int], plan: DeploymentPlan):
        """Update specific instances."""
        for instance in instances:
            await asyncio.sleep(0.5)  # Simulate instance update

    async def _check_instance_health(self, instance: int, timeout_seconds: int) -> bool:
        """Check health of specific instance."""
        await asyncio.sleep(1)  # Simulate health check
        return True  # Assume healthy for simulation

    async def _execute_hook(self, hook: str, environment: str):
        """Execute deployment hook."""
        logging.info(f"Executing hook '{hook}' for {environment}")
        await asyncio.sleep(0.5)  # Simulate hook execution

    async def _execute_rollback(self, deployment_id: str, rollback_plan: DeploymentPlan):
        """Execute rollback plan."""
        logging.info(f"Executing rollback for deployment {deployment_id}")

        # Switch traffic back to stable version
        self.traffic_manager.create_traffic_split(
            rollback_plan.target.name,
            {"stable": 1.0, "canary": 0.0}
        )

        # Clean up failed deployment
        await self._cleanup_environment("canary")

    def _validate_deployment_plan(self, plan: DeploymentPlan) -> Dict[str, Any]:
        """Validate deployment plan."""
        errors = []

        # Check artifact validity
        if not plan.artifact.vulnerability_scan_passed:
            errors.append("Artifact failed vulnerability scan")

        if not plan.artifact.security_scan_passed:
            errors.append("Artifact failed security scan")

        # Check target configuration
        if not plan.target.cluster_config:
            errors.append("Target cluster configuration missing")

        # Check resource limits
        required_resources = ['cpu', 'memory']
        for resource in required_resources:
            if resource not in plan.target.resource_limits:
                errors.append(f"Missing resource limit: {resource}")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def _estimate_start_time(self) -> str:
        """Estimate when queued deployment will start."""
        queue_size = self.deployment_queue.qsize()
        active_count = len(self.active_deployments)

        # Simple estimation: assume 10 minutes per deployment
        estimated_minutes = (queue_size + active_count) * 10
        estimated_start = datetime.now() + timedelta(minutes=estimated_minutes)

        return estimated_start.isoformat()

    async def _check_deployment_timeouts(self):
        """Check for deployment timeouts and handle them."""
        timeout_threshold = timedelta(minutes=self.deployment_timeout_minutes)
        current_time = datetime.now()

        timed_out_deployments = []

        for deployment_id, deployment in self.active_deployments.items():
            if deployment['started_at']:
                elapsed = current_time - deployment['started_at']
                if elapsed > timeout_threshold:
                    timed_out_deployments.append(deployment_id)

        for deployment_id in timed_out_deployments:
            deployment = self.active_deployments[deployment_id]
            deployment['status'] = 'timeout'
            deployment['completed_at'] = current_time
            deployment['error'] = f"Deployment timed out after {self.deployment_timeout_minutes} minutes"

            # Move to history
            self.deployment_history.append(deployment.copy())
            del self.active_deployments[deployment_id]

            logging.error(f"Deployment {deployment_id} timed out")

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific deployment."""
        # Check active deployments
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]

        # Check history
        for deployment in self.deployment_history:
            if deployment['deployment_id'] == deployment_id:
                return deployment

        return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'orchestrator': {
                'is_running': self.is_running,
                'active_deployments': len(self.active_deployments),
                'queued_deployments': self.deployment_queue.qsize(),
                'total_deployments_today': len([
                    d for d in self.deployment_history
                    if d['created_at'].date() == datetime.now().date()
                ])
            },
            'health_checker': self.health_checker.get_health_report(),
            'traffic_manager': self.traffic_manager.get_traffic_stats(),
            'recent_deployments': self.deployment_history[-10:]  # Last 10 deployments
        }

# Example usage and demonstration
async def enterprise_deployment_demo():
    """Demonstrate enterprise deployment capabilities."""

    # Create deployment orchestrator
    orchestrator = DeploymentOrchestrator()
    await orchestrator.start()

    try:
        print("Setting up enterprise deployment system...")

        # Register health checks
        orchestrator.health_checker.register_health_check(
            'database_connection',
            lambda: True  # Simulate always healthy for demo
        )

        orchestrator.health_checker.register_health_check(
            'cache_connectivity',
            lambda: True  # Simulate always healthy for demo
        )

        orchestrator.health_checker.register_health_check(
            'external_api',
            lambda: True  # Simulate always healthy for demo
        )

        # Enable circuit breakers
        orchestrator.traffic_manager.enable_circuit_breaker(
            'context-service',
            failure_threshold=5,
            timeout_seconds=60
        )

        print("Creating deployment artifacts and targets...")

        # Create deployment artifact
        artifact = DeploymentArtifact(
            artifact_id="context-engine-v2.1.0",
            version="v2.1.0",
            build_timestamp=datetime.now(),
            git_commit="abc123def456",
            docker_image="context-engine:v2.1.0",
            config_checksum="sha256:def456...",
            size_bytes=512 * 1024 * 1024,  # 512MB
            vulnerability_scan_passed=True,
            performance_test_passed=True,
            security_scan_passed=True,
            dependencies=["redis:7", "postgresql:14"],
            runtime_requirements={"cpu": "2", "memory": "4Gi"}
        )

        # Create deployment targets
        staging_target = DeploymentTarget(
            name="staging-cluster",
            stage=DeploymentStage.STAGING,
            region="us-east-1",
            cluster_config={
                "cluster_name": "staging-k8s",
                "namespace": "context-engine"
            },
            resource_limits={
                "cpu": "1",
                "memory": "2Gi",
                "storage": "10Gi"
            },
            scaling_config={
                "replicas": 2,
                "min_replicas": 1,
                "max_replicas": 5
            },
            security_config={
                "network_policies": True,
                "pod_security_standards": "restricted"
            }
        )

        production_target = DeploymentTarget(
            name="production-cluster",
            stage=DeploymentStage.PRODUCTION,
            region="us-east-1",
            cluster_config={
                "cluster_name": "prod-k8s",
                "namespace": "context-engine"
            },
            resource_limits={
                "cpu": "4",
                "memory": "8Gi",
                "storage": "100Gi"
            },
            scaling_config={
                "replicas": 6,
                "min_replicas": 3,
                "max_replicas": 20
            },
            security_config={
                "network_policies": True,
                "pod_security_standards": "restricted",
                "encryption_at_rest": True
            }
        )

        print("Phase 1: Staging Deployment (Rolling Update)")

        # Create staging deployment plan
        staging_plan = DeploymentPlan(
            plan_id="staging-deployment-001",
            artifact=artifact,
            target=staging_target,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            batch_size=1,
            health_check_timeout=120,
            pre_deployment_hooks=["run_migrations", "warm_cache"],
            post_deployment_hooks=["run_smoke_tests", "notify_team"]
        )

        # Execute staging deployment
        staging_result = await orchestrator.deploy(staging_plan)
        print(f"Staging deployment started: {staging_result['deployment_id']}")

        # Wait for staging deployment to complete
        await asyncio.sleep(10)

        staging_status = orchestrator.get_deployment_status(staging_result['deployment_id'])
        print(f"Staging deployment status: {staging_status['status']}")

        print("\nPhase 2: Production Deployment (Canary)")

        # Create production deployment plan with canary strategy
        production_plan = DeploymentPlan(
            plan_id="production-deployment-001",
            artifact=artifact,
            target=production_target,
            strategy=DeploymentStrategy.CANARY,
            canary_percentage=5.0,
            health_check_timeout=300,
            rollback_threshold=0.02,  # 2% error rate threshold
            requires_approval=True,
            approved_by="deployment-manager",
            approval_timestamp=datetime.now(),
            pre_deployment_hooks=["backup_database", "notify_operations"],
            post_deployment_hooks=["update_monitoring", "notify_stakeholders"]
        )

        # Execute production deployment
        production_result = await orchestrator.deploy(production_plan)
        print(f"Production deployment started: {production_result['deployment_id']}")

        # Monitor deployment progress
        for i in range(15):  # Monitor for ~2.5 minutes
            await asyncio.sleep(10)

            prod_status = orchestrator.get_deployment_status(production_result['deployment_id'])
            if prod_status:
                print(f"Production deployment status: {prod_status['status']}")

                if prod_status['status'] in ['completed', 'failed', 'rolled_back']:
                    break

        print("\nPhase 3: Blue-Green Deployment Simulation")

        # Create blue-green deployment plan
        bg_target = production_target
        bg_target.name = "blue-green-cluster"

        bg_plan = DeploymentPlan(
            plan_id="blue-green-deployment-001",
            artifact=artifact,
            target=bg_target,
            strategy=DeploymentStrategy.BLUE_GREEN,
            health_check_timeout=180,
            rollback_threshold=0.01  # 1% error rate threshold
        )

        # Execute blue-green deployment
        bg_result = await orchestrator.deploy(bg_plan)
        print(f"Blue-green deployment started: {bg_result['deployment_id']}")

        # Wait for completion
        await asyncio.sleep(15)

        bg_status = orchestrator.get_deployment_status(bg_result['deployment_id'])
        print(f"Blue-green deployment status: {bg_status['status']}")

        print("\nPhase 4: System Status Report")

        # Get comprehensive system status
        system_status = orchestrator.get_system_status()
        print(json.dumps(system_status, indent=2, default=str))

        print("\nPhase 5: Traffic Management Demo")

        # Demonstrate traffic management
        orchestrator.traffic_manager.create_traffic_split(
            "context-service",
            {"v2.0": 0.8, "v2.1": 0.2}
        )

        # Simulate some requests
        for i in range(20):
            target = orchestrator.traffic_manager.route_request(
                "context-service",
                f"request-{i}",
                f"user-{i % 5}"
            )
            success = True  # Simulate success
            orchestrator.traffic_manager.record_request_result(
                "context-service", target, success
            )

        traffic_stats = orchestrator.traffic_manager.get_traffic_stats()
        print(f"\nTraffic Management Stats:")
        print(json.dumps(traffic_stats, indent=2))

    finally:
        await orchestrator.stop()

# Run the demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(enterprise_deployment_demo())
```

## Key Takeaways

1. **Zero-Downtime Deployments**: Implement blue-green, canary, and rolling update strategies for seamless updates

2. **Traffic Management**: Sophisticated traffic routing with circuit breakers and gradual traffic shifting

3. **Health Monitoring**: Comprehensive health checking with automatic rollback capabilities

4. **Deployment Orchestration**: Enterprise-grade deployment pipeline with approval workflows

5. **Disaster Recovery**: Built-in rollback mechanisms and failure handling

6. **Operational Excellence**: Complete observability, monitoring, and operational procedures

## Module 8 Conclusion

You've now mastered enterprise-scale performance optimization and operational deployment strategies. Your context engineering systems can achieve:

- **High Performance**: Sub-50ms latency with >10,000 operations/second throughput
- **Intelligent Caching**: >95% cache hit rates with predictive prefetching
- **Distributed Consistency**: Global consensus with partition tolerance
- **Advanced Monitoring**: Real-time analytics with predictive capacity planning
- **Enterprise Deployment**: Zero-downtime deployments with comprehensive disaster recovery

## Course Completion

🎉 **Congratulations!** You have successfully completed the Context Engineering Mastery Course and are now equipped with enterprise-level expertise in building, optimizing, and scaling sophisticated context-aware AI systems.

---

**Final Exercise**: Design and implement a complete enterprise context engineering system demonstrating all learned concepts: advanced prompting, memory management, RAG integration, tool orchestration, single-agent architecture, and production deployment with monitoring. Target: >99.9% uptime, <50ms P99 latency, global scale deployment.
