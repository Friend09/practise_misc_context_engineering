# Lesson 5: Security and Sandbox Execution

## Introduction

Tool execution in production environments requires robust security measures. This lesson teaches you to build secure execution environments with sandboxing, permission systems, resource limits, threat detection, and comprehensive security monitoring that protect against malicious code and unauthorized access.

## Secure Execution Environments

### Process Isolation and Sandboxing

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
import subprocess
import tempfile
import os
import signal
import resource
import time
from pathlib import Path
from datetime import datetime
import json
import shutil

@dataclass
class SecurityPolicy:
    """Security policy for tool execution."""
    # Process limits
    max_cpu_seconds: float = 60.0
    max_memory_mb: int = 512
    max_file_size_mb: int = 100
    max_open_files: int = 100

    # Network access
    allow_network: bool = False
    allowed_hosts: Set[str] = field(default_factory=set)
    allowed_ports: Set[int] = field(default_factory=set)

    # File system access
    read_only_paths: Set[str] = field(default_factory=set)
    read_write_paths: Set[str] = field(default_factory=set)
    forbidden_paths: Set[str] = field(default_factory=lambda: {'/etc', '/bin', '/usr'})

    # Execution permissions
    allow_subprocess: bool = False
    allowed_commands: Set[str] = field(default_factory=set)

    # Security level
    security_level: str = "medium"  # low, medium, high, paranoid

class SandboxEnvironment:
    """
    Secure sandbox environment for tool execution.
    """
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.sandbox_dir: Optional[Path] = None
        self.active_processes: Dict[int, Dict] = {}

        # Resource monitoring
        self.resource_usage: List[Dict] = []

        # Security events
        self.security_events: List[Dict] = []

    async def __aenter__(self):
        """Context manager entry - setup sandbox."""
        await self._setup_sandbox()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup sandbox."""
        await self._cleanup_sandbox()

    async def _setup_sandbox(self):
        """Setup secure sandbox environment."""
        # Create temporary sandbox directory
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))

        # Set permissions
        os.chmod(self.sandbox_dir, 0o700)  # Only owner access

        # Create subdirectories
        (self.sandbox_dir / "input").mkdir()
        (self.sandbox_dir / "output").mkdir()
        (self.sandbox_dir / "tmp").mkdir()

        # Copy allowed read-only files
        for ro_path in self.policy.read_only_paths:
            if os.path.exists(ro_path):
                dest = self.sandbox_dir / "readonly" / Path(ro_path).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ro_path, dest)

    async def _cleanup_sandbox(self):
        """Cleanup sandbox environment."""
        # Terminate any remaining processes
        for pid in list(self.active_processes.keys()):
            await self._terminate_process(pid)

        # Remove sandbox directory
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        input_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute code in sandbox."""
        execution_id = f"exec_{int(time.time() * 1000)}"

        try:
            # Validate code
            await self._validate_code(code, language)

            # Prepare execution environment
            exec_env = await self._prepare_execution_env(code, language, input_data)

            # Execute with resource limits
            result = await self._execute_with_limits(exec_env, execution_id)

            return {
                'success': True,
                'result': result,
                'execution_id': execution_id,
                'security_events': len(self.security_events)
            }

        except SecurityViolationError as e:
            self._record_security_event('violation', str(e), execution_id)
            return {
                'success': False,
                'error': f"Security violation: {e}",
                'execution_id': execution_id
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_id': execution_id
            }

    async def _validate_code(self, code: str, language: str):
        """Validate code for security issues."""
        # Static analysis for dangerous patterns
        dangerous_patterns = {
            'python': [
                'import os',
                'import subprocess',
                'import sys',
                '__import__',
                'eval(',
                'exec(',
                'open(',
                'file(',
                'input(',
                'raw_input('
            ],
            'javascript': [
                'require(',
                'import(',
                'eval(',
                'Function(',
                'setTimeout(',
                'setInterval('
            ]
        }

        if language in dangerous_patterns:
            for pattern in dangerous_patterns[language]:
                if pattern in code:
                    if not self._is_pattern_allowed(pattern):
                        raise SecurityViolationError(f"Forbidden pattern: {pattern}")

        # Check code length
        if len(code) > 100000:  # 100KB limit
            raise SecurityViolationError("Code too large")

    def _is_pattern_allowed(self, pattern: str) -> bool:
        """Check if dangerous pattern is allowed by policy."""
        if self.policy.security_level == "low":
            return True
        elif self.policy.security_level == "medium":
            # Allow some patterns with restrictions
            return pattern in ['open(', 'file(']
        else:
            return False

    async def _prepare_execution_env(
        self,
        code: str,
        language: str,
        input_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Prepare execution environment."""
        # Create code file
        if language == "python":
            code_file = self.sandbox_dir / "code.py"
            with open(code_file, 'w') as f:
                f.write(code)
        elif language == "javascript":
            code_file = self.sandbox_dir / "code.js"
            with open(code_file, 'w') as f:
                f.write(code)
        else:
            raise ValueError(f"Unsupported language: {language}")

        # Create input file if needed
        input_file = None
        if input_data:
            input_file = self.sandbox_dir / "input.json"
            with open(input_file, 'w') as f:
                json.dump(input_data, f)

        # Prepare command
        if language == "python":
            cmd = ["python3", str(code_file)]
        elif language == "javascript":
            cmd = ["node", str(code_file)]

        return {
            'command': cmd,
            'working_dir': str(self.sandbox_dir),
            'code_file': str(code_file),
            'input_file': str(input_file) if input_file else None
        }

    async def _execute_with_limits(
        self,
        exec_env: Dict[str, Any],
        execution_id: str
    ) -> Dict[str, Any]:
        """Execute with resource limits and monitoring."""
        start_time = time.time()

        # Set up resource limits
        def preexec_fn():
            # CPU limit
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(self.policy.max_cpu_seconds), int(self.policy.max_cpu_seconds))
            )

            # Memory limit
            memory_bytes = self.policy.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # File size limit
            file_size_bytes = self.policy.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))

            # Open files limit
            resource.setrlimit(resource.RLIMIT_NOFILE, (self.policy.max_open_files, self.policy.max_open_files))

        # Execute process
        try:
            process = subprocess.Popen(
                exec_env['command'],
                cwd=exec_env['working_dir'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec_fn,
                env=self._create_restricted_env()
            )

            # Track process
            self.active_processes[process.pid] = {
                'execution_id': execution_id,
                'start_time': start_time,
                'command': exec_env['command']
            }

            # Wait with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.policy.max_cpu_seconds + 10)

                execution_time = time.time() - start_time

                # Record resource usage
                self._record_resource_usage(execution_id, execution_time, process.pid)

                # Clean up process tracking
                if process.pid in self.active_processes:
                    del self.active_processes[process.pid]

                return {
                    'stdout': stdout.decode('utf-8', errors='replace'),
                    'stderr': stderr.decode('utf-8', errors='replace'),
                    'return_code': process.returncode,
                    'execution_time_seconds': execution_time
                }

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise SecurityViolationError("Execution timeout")

        except Exception as e:
            # Clean up on error
            if 'process' in locals() and process.pid in self.active_processes:
                del self.active_processes[process.pid]
            raise e

    def _create_restricted_env(self) -> Dict[str, str]:
        """Create restricted environment variables."""
        # Start with minimal environment
        restricted_env = {
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'HOME': str(self.sandbox_dir),
            'TMPDIR': str(self.sandbox_dir / "tmp"),
            'PYTHONPATH': '',
            'PYTHONDONTWRITEBYTECODE': '1'
        }

        # Remove dangerous variables
        dangerous_vars = [
            'LD_PRELOAD', 'LD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES',
            'PYTHONSTARTUP', 'PYTHONHOME'
        ]

        return restricted_env

    async def _terminate_process(self, pid: int):
        """Terminate process gracefully."""
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)

            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already terminated

        except ProcessLookupError:
            pass  # Process doesn't exist

        # Clean up tracking
        if pid in self.active_processes:
            del self.active_processes[pid]

    def _record_resource_usage(self, execution_id: str, execution_time: float, pid: int):
        """Record resource usage for monitoring."""
        self.resource_usage.append({
            'execution_id': execution_id,
            'pid': pid,
            'execution_time': execution_time,
            'timestamp': datetime.now()
        })

    def _record_security_event(self, event_type: str, description: str, execution_id: str):
        """Record security event."""
        self.security_events.append({
            'event_type': event_type,
            'description': description,
            'execution_id': execution_id,
            'timestamp': datetime.now()
        })

class SecurityViolationError(Exception):
    """Security violation exception."""
    pass
```

## Permission and Access Control

### Role-Based Access Control (RBAC)

```python
from enum import Enum

class Permission(Enum):
    """System permissions."""
    EXECUTE_TOOL = "execute_tool"
    MANAGE_TOOLS = "manage_tools"
    VIEW_LOGS = "view_logs"
    ADMIN_ACCESS = "admin_access"
    NETWORK_ACCESS = "network_access"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SYSTEM_MODIFY = "system_modify"

@dataclass
class Role:
    """User role with permissions."""
    name: str
    description: str
    permissions: Set[Permission]

    # Resource limits
    max_concurrent_executions: int = 5
    max_execution_time_seconds: float = 300.0
    max_memory_mb: int = 512

    # Tool access
    allowed_tools: Set[str] = field(default_factory=set)
    forbidden_tools: Set[str] = field(default_factory=set)

@dataclass
class User:
    """System user."""
    user_id: str
    username: str
    email: str
    roles: Set[str]

    # Status
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

    # Security
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

class AccessControlManager:
    """
    Manages role-based access control for tool execution.
    """
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict] = {}

        # Audit trail
        self.access_log: List[Dict] = []

        # Setup default roles
        self._setup_default_roles()

    def _setup_default_roles(self):
        """Setup default system roles."""
        # Guest role - minimal permissions
        guest_role = Role(
            name="guest",
            description="Read-only access",
            permissions={Permission.VIEW_LOGS},
            max_concurrent_executions=1,
            max_execution_time_seconds=30.0,
            max_memory_mb=128
        )

        # User role - basic tool execution
        user_role = Role(
            name="user",
            description="Basic tool execution",
            permissions={
                Permission.EXECUTE_TOOL,
                Permission.VIEW_LOGS,
                Permission.FILE_READ
            },
            max_concurrent_executions=3,
            max_execution_time_seconds=180.0,
            max_memory_mb=256
        )

        # Power user role - advanced capabilities
        power_user_role = Role(
            name="power_user",
            description="Advanced tool execution",
            permissions={
                Permission.EXECUTE_TOOL,
                Permission.VIEW_LOGS,
                Permission.FILE_READ,
                Permission.FILE_WRITE,
                Permission.NETWORK_ACCESS
            },
            max_concurrent_executions=5,
            max_execution_time_seconds=300.0,
            max_memory_mb=512
        )

        # Admin role - full access
        admin_role = Role(
            name="admin",
            description="Full system access",
            permissions=set(Permission),
            max_concurrent_executions=10,
            max_execution_time_seconds=600.0,
            max_memory_mb=1024
        )

        self.roles.update({
            "guest": guest_role,
            "user": user_role,
            "power_user": power_user_role,
            "admin": admin_role
        })

    def create_user(
        self,
        user_id: str,
        username: str,
        email: str,
        roles: List[str]
    ) -> User:
        """Create new user."""
        # Validate roles exist
        for role_name in roles:
            if role_name not in self.roles:
                raise ValueError(f"Unknown role: {role_name}")

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=set(roles)
        )

        self.users[user_id] = user

        # Log user creation
        self._log_access_event("user_created", user_id, {"username": username, "roles": roles})

        return user

    def authenticate_user(self, user_id: str, token: str) -> Optional[str]:
        """Authenticate user and create session."""
        user = self.users.get(user_id)
        if not user or not user.active:
            self._log_access_event("auth_failed", user_id, {"reason": "user_not_found"})
            return None

        # Check if locked
        if user.locked_until and datetime.now() < user.locked_until:
            self._log_access_event("auth_failed", user_id, {"reason": "account_locked"})
            return None

        # Simplified token validation (in production, use proper JWT/OAuth)
        if self._validate_token(token, user):
            # Create session
            session_id = self._create_session(user)

            # Update user
            user.last_login = datetime.now()
            user.failed_login_attempts = 0

            self._log_access_event("auth_success", user_id, {"session_id": session_id})
            return session_id
        else:
            # Failed login
            user.failed_login_attempts += 1

            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now() + timedelta(hours=1)
                self._log_access_event("account_locked", user_id, {"attempts": user.failed_login_attempts})

            self._log_access_event("auth_failed", user_id, {"reason": "invalid_credentials"})
            return None

    def check_permission(
        self,
        session_id: str,
        permission: Permission,
        resource_context: Optional[Dict] = None
    ) -> bool:
        """Check if user has permission."""
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        user = self.users.get(session['user_id'])
        if not user or not user.active:
            return False

        # Check if user has permission through any role
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and permission in role.permissions:
                # Additional context-based checks
                if resource_context and not self._check_resource_access(role, resource_context):
                    continue

                self._log_access_event("permission_granted", user.user_id, {
                    "permission": permission.value,
                    "role": role_name
                })
                return True

        self._log_access_event("permission_denied", user.user_id, {
            "permission": permission.value
        })
        return False

    def get_user_limits(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get resource limits for user."""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        user = self.users.get(session['user_id'])
        if not user:
            return None

        # Aggregate limits from all roles (take maximum)
        limits = {
            'max_concurrent_executions': 0,
            'max_execution_time_seconds': 0,
            'max_memory_mb': 0,
            'allowed_tools': set(),
            'forbidden_tools': set()
        }

        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role:
                limits['max_concurrent_executions'] = max(
                    limits['max_concurrent_executions'],
                    role.max_concurrent_executions
                )
                limits['max_execution_time_seconds'] = max(
                    limits['max_execution_time_seconds'],
                    role.max_execution_time_seconds
                )
                limits['max_memory_mb'] = max(
                    limits['max_memory_mb'],
                    role.max_memory_mb
                )
                limits['allowed_tools'].update(role.allowed_tools)
                limits['forbidden_tools'].update(role.forbidden_tools)

        return limits

    def _validate_token(self, token: str, user: User) -> bool:
        """Validate authentication token."""
        # Simplified validation - in production use proper JWT validation
        expected_token = f"token_{user.user_id}_{user.username}"
        return token == expected_token

    def _create_session(self, user: User) -> str:
        """Create user session."""
        import uuid
        session_id = str(uuid.uuid4())

        self.active_sessions[session_id] = {
            'user_id': user.user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }

        return session_id

    def _check_resource_access(self, role: Role, context: Dict) -> bool:
        """Check resource-specific access."""
        tool_id = context.get('tool_id')
        if tool_id:
            # Check forbidden tools
            if tool_id in role.forbidden_tools:
                return False

            # Check allowed tools (if specified)
            if role.allowed_tools and tool_id not in role.allowed_tools:
                return False

        return True

    def _log_access_event(self, event_type: str, user_id: str, details: Dict):
        """Log access control event."""
        self.access_log.append({
            'event_type': event_type,
            'user_id': user_id,
            'details': details,
            'timestamp': datetime.now()
        })

    def get_access_log(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get access control log."""
        filtered_log = self.access_log

        if user_id:
            filtered_log = [log for log in filtered_log if log['user_id'] == user_id]

        if event_type:
            filtered_log = [log for log in filtered_log if log['event_type'] == event_type]

        return filtered_log[-limit:]
```

## Security Monitoring and Threat Detection

### Advanced Security Monitor

```python
class ThreatLevel(Enum):
    """Threat level classification."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: str
    threat_level: ThreatLevel
    description: str

    # Context
    user_id: Optional[str] = None
    tool_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None

    # Detection
    detected_at: datetime = field(default_factory=datetime.now)
    detection_method: str = "automated"

    # Response
    blocked: bool = False
    response_actions: List[str] = field(default_factory=list)

class SecurityMonitor:
    """
    Monitors for security threats and anomalies.
    """
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.threat_patterns: Dict[str, Dict] = {}
        self.anomaly_detectors: List[Callable] = []

        # Rate limiting for events
        self.user_activity: Dict[str, List[datetime]] = {}

        # Setup threat patterns
        self._setup_threat_patterns()
        self._setup_anomaly_detectors()

    def _setup_threat_patterns(self):
        """Setup known threat patterns."""
        self.threat_patterns = {
            'code_injection': {
                'patterns': [
                    r'exec\s*\(',
                    r'eval\s*\(',
                    r'__import__\s*\(',
                    r'subprocess\.call',
                    r'os\.system'
                ],
                'threat_level': ThreatLevel.HIGH
            },
            'file_system_access': {
                'patterns': [
                    r'\.\./',
                    r'/etc/',
                    r'/root/',
                    r'~/',
                    r'open\s*\(',
                    r'file\s*\('
                ],
                'threat_level': ThreatLevel.MEDIUM
            },
            'network_access': {
                'patterns': [
                    r'urllib',
                    r'requests',
                    r'socket',
                    r'http',
                    r'ftp'
                ],
                'threat_level': ThreatLevel.MEDIUM
            },
            'privilege_escalation': {
                'patterns': [
                    r'sudo',
                    r'su\s',
                    r'chmod',
                    r'chown',
                    r'setuid'
                ],
                'threat_level': ThreatLevel.CRITICAL
            }
        }

    def _setup_anomaly_detectors(self):
        """Setup anomaly detection functions."""
        self.anomaly_detectors = [
            self._detect_rapid_requests,
            self._detect_unusual_patterns,
            self._detect_resource_abuse,
            self._detect_failed_authentications
        ]

    def scan_code(self, code: str, user_id: str, tool_id: str) -> List[SecurityEvent]:
        """Scan code for security threats."""
        events = []

        for threat_type, config in self.threat_patterns.items():
            for pattern in config['patterns']:
                import re
                if re.search(pattern, code, re.IGNORECASE):
                    event = SecurityEvent(
                        event_id=self._generate_event_id(),
                        event_type=f"code_threat_{threat_type}",
                        threat_level=config['threat_level'],
                        description=f"Detected {threat_type} pattern: {pattern}",
                        user_id=user_id,
                        tool_id=tool_id,
                        detection_method="static_analysis"
                    )

                    # Auto-block critical threats
                    if config['threat_level'] == ThreatLevel.CRITICAL:
                        event.blocked = True
                        event.response_actions.append("execution_blocked")

                    events.append(event)

        # Record events
        self.events.extend(events)

        return events

    def monitor_execution(
        self,
        execution_context: Dict[str, Any]
    ) -> List[SecurityEvent]:
        """Monitor tool execution for anomalies."""
        events = []

        # Run anomaly detectors
        for detector in self.anomaly_detectors:
            try:
                anomaly_events = detector(execution_context)
                events.extend(anomaly_events)
            except Exception as e:
                # Log detector error but continue
                print(f"Anomaly detector error: {e}")

        # Record events
        self.events.extend(events)

        return events

    def _detect_rapid_requests(self, context: Dict) -> List[SecurityEvent]:
        """Detect rapid request patterns."""
        events = []
        user_id = context.get('user_id')

        if user_id:
            now = datetime.now()

            # Track user activity
            if user_id not in self.user_activity:
                self.user_activity[user_id] = []

            self.user_activity[user_id].append(now)

            # Clean old activity (last 5 minutes)
            cutoff = now - timedelta(minutes=5)
            self.user_activity[user_id] = [
                time for time in self.user_activity[user_id]
                if time >= cutoff
            ]

            # Check for rapid requests
            recent_count = len(self.user_activity[user_id])

            if recent_count > 50:  # More than 50 requests in 5 minutes
                event = SecurityEvent(
                    event_id=self._generate_event_id(),
                    event_type="rapid_requests",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Rapid requests detected: {recent_count} in 5 minutes",
                    user_id=user_id,
                    detection_method="rate_analysis"
                )
                events.append(event)

        return events

    def _detect_unusual_patterns(self, context: Dict) -> List[SecurityEvent]:
        """Detect unusual execution patterns."""
        events = []

        # Check for unusual execution times
        execution_time = context.get('execution_time_seconds', 0)
        if execution_time > 300:  # More than 5 minutes
            event = SecurityEvent(
                event_id=self._generate_event_id(),
                event_type="long_execution",
                threat_level=ThreatLevel.MEDIUM,
                description=f"Unusually long execution: {execution_time:.2f} seconds",
                user_id=context.get('user_id'),
                tool_id=context.get('tool_id'),
                detection_method="timing_analysis"
            )
            events.append(event)

        # Check for unusual resource usage
        memory_mb = context.get('memory_usage_mb', 0)
        if memory_mb > 1000:  # More than 1GB
            event = SecurityEvent(
                event_id=self._generate_event_id(),
                event_type="high_memory_usage",
                threat_level=ThreatLevel.MEDIUM,
                description=f"High memory usage: {memory_mb} MB",
                user_id=context.get('user_id'),
                tool_id=context.get('tool_id'),
                detection_method="resource_analysis"
            )
            events.append(event)

        return events

    def _detect_resource_abuse(self, context: Dict) -> List[SecurityEvent]:
        """Detect resource abuse patterns."""
        events = []

        # Check CPU usage
        cpu_usage = context.get('cpu_usage_percent', 0)
        if cpu_usage > 95:
            event = SecurityEvent(
                event_id=self._generate_event_id(),
                event_type="cpu_abuse",
                threat_level=ThreatLevel.HIGH,
                description=f"High CPU usage: {cpu_usage}%",
                user_id=context.get('user_id'),
                tool_id=context.get('tool_id'),
                detection_method="resource_monitoring"
            )
            events.append(event)

        return events

    def _detect_failed_authentications(self, context: Dict) -> List[SecurityEvent]:
        """Detect authentication attack patterns."""
        events = []

        if context.get('event_type') == 'auth_failed':
            user_id = context.get('user_id')

            # Count recent failed attempts
            recent_failures = [
                event for event in self.events
                if (event.event_type == 'auth_failed' and
                    event.user_id == user_id and
                    (datetime.now() - event.detected_at).total_seconds() < 300)  # 5 minutes
            ]

            if len(recent_failures) >= 3:
                event = SecurityEvent(
                    event_id=self._generate_event_id(),
                    event_type="brute_force_attempt",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Multiple failed authentication attempts: {len(recent_failures)}",
                    user_id=user_id,
                    detection_method="authentication_analysis"
                )
                events.append(event)

        return events

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def get_threat_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat summary for specified period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.detected_at >= cutoff]

        # Count by threat level
        threat_counts = {}
        for level in ThreatLevel:
            threat_counts[level.value] = len([
                e for e in recent_events if e.threat_level == level
            ])

        # Count by event type
        event_type_counts = {}
        for event in recent_events:
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

        # Count blocked events
        blocked_count = len([e for e in recent_events if e.blocked])

        return {
            'total_events': len(recent_events),
            'threat_levels': threat_counts,
            'event_types': event_type_counts,
            'blocked_events': blocked_count,
            'period_hours': hours,
            'generated_at': datetime.now()
        }
```

## Key Takeaways

1. **Process Isolation**: Secure sandboxing with resource limits and restricted environments

2. **Access Control**: Role-based permissions with fine-grained resource controls

3. **Threat Detection**: Static analysis and runtime monitoring for security threats

4. **Anomaly Detection**: Pattern recognition for unusual behavior

5. **Security Monitoring**: Comprehensive logging and alerting for security events

6. **Defense in Depth**: Multiple layers of security controls and monitoring

## Module Summary

You've now mastered comprehensive tool integration and external context management:

- **Tool Architecture**: Context-aware selection and universal adapters
- **API Integration**: RESTful services with authentication and rate limiting
- **Workflow Orchestration**: Multi-tool coordination with dependency management
- **Error Handling**: Resilient systems with circuit breakers and fallbacks
- **Security**: Sandboxed execution with threat detection and access control

These systems enable AI agents to safely interact with external services and tools while maintaining security and reliability.

---

**Practice Exercise**: Build a complete secure tool integration platform supporting 20+ tools with role-based access control. Implement sandboxed execution handling 500+ concurrent tool executions safely. Add comprehensive security monitoring detecting and blocking malicious code patterns with <1% false positive rate.
