# Lesson 2: External API and Service Integration

## Introduction

Modern AI agents need to interact with external services: weather APIs, database systems, cloud services, and third-party platforms. This lesson teaches robust patterns for integrating external APIs with authentication, rate limiting, retry logic, and comprehensive error handling.

## RESTful API Integration

### Universal REST Client

```python
from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum
import aiohttp
import asyncio
from datetime import datetime, timedelta
import hashlib
import json

class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"

@dataclass
class APIEndpoint:
    """API endpoint specification."""
    url: str
    method: HTTPMethod
    description: str
    required_params: List[str]
    optional_params: List[str]
    auth_required: bool = True
    rate_limit_per_minute: Optional[int] = None
    timeout_seconds: float = 30.0

class RESTAPIClient:
    """
    Universal REST API client with connection pooling and retry logic.
    """
    def __init__(
        self,
        base_url: str,
        default_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 30.0
    ):
        self.base_url = base_url.rstrip('/')
        self.default_headers = default_headers or {}
        self.timeout_seconds = timeout_seconds
        self.session: Optional[aiohttp.ClientSession] = None

        # Request tracking
        self.request_count = 0
        self.failed_requests = 0
        self.total_response_time_ms = 0.0

    async def __aenter__(self):
        """Context manager entry."""
        self.session = aiohttp.ClientSession(
            headers=self.default_headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.session:
            await self.session.close()

    async def request(
        self,
        method: HTTPMethod,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        auth: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute HTTP request."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use as context manager.")

        # Build full URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Merge headers
        request_headers = {**self.default_headers}
        if headers:
            request_headers.update(headers)

        # Add authentication if provided
        if auth:
            if 'api_key' in auth:
                request_headers['Authorization'] = f"Bearer {auth['api_key']}"

        start_time = datetime.now()
        self.request_count += 1

        try:
            async with self.session.request(
                method=method.value,
                url=url,
                params=params,
                json=data,
                headers=request_headers
            ) as response:
                # Calculate response time
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self.total_response_time_ms += response_time

                # Parse response
                try:
                    response_data = await response.json()
                except:
                    response_data = await response.text()

                return {
                    'success': response.status < 400,
                    'status_code': response.status,
                    'data': response_data,
                    'response_time_ms': response_time,
                    'headers': dict(response.headers)
                }

        except asyncio.TimeoutError:
            self.failed_requests += 1
            return {
                'success': False,
                'error': 'Request timeout',
                'error_type': 'timeout'
            }
        except Exception as e:
            self.failed_requests += 1
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    async def get(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> Dict:
        """GET request."""
        return await self.request(HTTPMethod.GET, endpoint, params=params, **kwargs)

    async def post(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> Dict:
        """POST request."""
        return await self.request(HTTPMethod.POST, endpoint, data=data, **kwargs)

    async def put(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> Dict:
        """PUT request."""
        return await self.request(HTTPMethod.PUT, endpoint, data=data, **kwargs)

    async def delete(self, endpoint: str, **kwargs) -> Dict:
        """DELETE request."""
        return await self.request(HTTPMethod.DELETE, endpoint, **kwargs)

    def get_stats(self) -> Dict:
        """Get client statistics."""
        avg_response_time = 0.0
        if self.request_count > 0:
            avg_response_time = self.total_response_time_ms / self.request_count

        success_rate = 0.0
        if self.request_count > 0:
            success_rate = (self.request_count - self.failed_requests) / self.request_count

        return {
            'total_requests': self.request_count,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'avg_response_time_ms': avg_response_time
        }
```

## Authentication and Authorization

### Multi-Strategy Authentication Manager

```python
from abc import ABC, abstractmethod
import time
import jwt
from cryptography.fernet import Fernet

class AuthenticationStrategy(ABC):
    """Base class for authentication strategies."""

    @abstractmethod
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        pass

    @abstractmethod
    def is_expired(self) -> bool:
        """Check if authentication is expired."""
        pass

    @abstractmethod
    async def refresh(self):
        """Refresh authentication if needed."""
        pass

class APIKeyAuth(AuthenticationStrategy):
    """API key authentication."""

    def __init__(self, api_key: str, header_name: str = "Authorization"):
        self.api_key = api_key
        self.header_name = header_name

    def get_auth_headers(self) -> Dict[str, str]:
        """Get API key headers."""
        return {self.header_name: f"Bearer {self.api_key}"}

    def is_expired(self) -> bool:
        """API keys don't expire."""
        return False

    async def refresh(self):
        """No refresh needed."""
        pass

class OAuthTokenAuth(AuthenticationStrategy):
    """OAuth 2.0 token authentication."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: Optional[str] = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope

        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None

    async def get_token(self):
        """Get OAuth token."""
        async with aiohttp.ClientSession() as session:
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }

            if self.scope:
                data['scope'] = self.scope

            async with session.post(self.token_url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data['access_token']

                    # Calculate expiration
                    expires_in = token_data.get('expires_in', 3600)
                    self.expires_at = datetime.now() + timedelta(seconds=expires_in)

                    if 'refresh_token' in token_data:
                        self.refresh_token = token_data['refresh_token']
                else:
                    raise Exception(f"Failed to get token: {response.status}")

    def get_auth_headers(self) -> Dict[str, str]:
        """Get OAuth headers."""
        if not self.access_token:
            raise Exception("No access token available")

        return {'Authorization': f"Bearer {self.access_token}"}

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return True

        # Consider expired 5 minutes before actual expiration
        return datetime.now() >= (self.expires_at - timedelta(minutes=5))

    async def refresh(self):
        """Refresh token if needed."""
        if self.is_expired():
            await self.get_token()

class JWTAuth(AuthenticationStrategy):
    """JWT token authentication."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        expiration_seconds: int = 3600
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiration_seconds = expiration_seconds

        self.token: Optional[str] = None
        self.expires_at: Optional[datetime] = None

    def generate_token(self, payload: Dict) -> str:
        """Generate JWT token."""
        # Add expiration
        exp = datetime.now() + timedelta(seconds=self.expiration_seconds)
        payload['exp'] = exp.timestamp()

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        self.token = token
        self.expires_at = exp

        return token

    def get_auth_headers(self) -> Dict[str, str]:
        """Get JWT headers."""
        if not self.token or self.is_expired():
            # Generate new token with default payload
            self.generate_token({'sub': 'api_client'})

        return {'Authorization': f"Bearer {self.token}"}

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return True

        return datetime.now() >= (self.expires_at - timedelta(minutes=5))

    async def refresh(self):
        """Refresh token."""
        if self.is_expired():
            self.generate_token({'sub': 'api_client'})

class AuthenticationManager:
    """
    Manages authentication for multiple services.
    """
    def __init__(self):
        self.strategies: Dict[str, AuthenticationStrategy] = {}
        self.encrypted_credentials: Dict[str, bytes] = {}
        self.cipher: Optional[Fernet] = None

    def initialize_encryption(self, encryption_key: Optional[bytes] = None):
        """Initialize credential encryption."""
        if encryption_key is None:
            encryption_key = Fernet.generate_key()

        self.cipher = Fernet(encryption_key)
        return encryption_key

    def register_service(
        self,
        service_name: str,
        strategy: AuthenticationStrategy
    ):
        """Register authentication strategy for service."""
        self.strategies[service_name] = strategy

    async def get_auth_headers(self, service_name: str) -> Dict[str, str]:
        """Get authentication headers for service."""
        if service_name not in self.strategies:
            return {}

        strategy = self.strategies[service_name]

        # Refresh if needed
        if strategy.is_expired():
            await strategy.refresh()

        return strategy.get_auth_headers()

    def store_credentials(
        self,
        service_name: str,
        credentials: Dict[str, str]
    ):
        """Securely store credentials."""
        if not self.cipher:
            raise Exception("Encryption not initialized")

        # Encrypt credentials
        cred_bytes = json.dumps(credentials).encode()
        encrypted = self.cipher.encrypt(cred_bytes)

        self.encrypted_credentials[service_name] = encrypted

    def retrieve_credentials(self, service_name: str) -> Optional[Dict[str, str]]:
        """Retrieve and decrypt credentials."""
        if not self.cipher or service_name not in self.encrypted_credentials:
            return None

        # Decrypt credentials
        encrypted = self.encrypted_credentials[service_name]
        decrypted = self.cipher.decrypt(encrypted)

        return json.loads(decrypted.decode())
```

## Rate Limiting and Quota Management

### Token Bucket Rate Limiter

```python
import time
from collections import deque

class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    """
    def __init__(
        self,
        requests_per_second: float,
        burst_size: Optional[int] = None
    ):
        self.rate = requests_per_second
        self.burst_size = burst_size or int(requests_per_second * 2)

        self.tokens = self.burst_size
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens, waiting if necessary."""
        async with self.lock:
            # Refill tokens based on time passed
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + time_passed * self.rate
            )
            self.last_update = now

            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait_time)

            self.tokens = 0
            self.last_update = time.time()
            return True

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.
    """
    def __init__(
        self,
        max_requests: int,
        window_seconds: int
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire permission to make request."""
        async with self.lock:
            now = time.time()

            # Remove old requests outside window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()

            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True

            # Calculate wait time
            oldest_request = self.requests[0]
            wait_time = (oldest_request + self.window_seconds) - now

            await asyncio.sleep(wait_time)

            # Remove oldest and add new
            self.requests.popleft()
            self.requests.append(time.time())
            return True

class QuotaManager:
    """
    Manages API quotas across multiple services.
    """
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.quotas: Dict[str, Dict] = {}

    def register_service(
        self,
        service_name: str,
        requests_per_second: float,
        daily_quota: Optional[int] = None
    ):
        """Register service with rate limits."""
        self.limiters[service_name] = RateLimiter(requests_per_second)

        if daily_quota:
            self.quotas[service_name] = {
                'daily_limit': daily_quota,
                'used_today': 0,
                'reset_time': datetime.now() + timedelta(days=1)
            }

    async def acquire(self, service_name: str) -> Dict[str, Any]:
        """Acquire permission to make request."""
        if service_name not in self.limiters:
            return {'allowed': True}

        # Check daily quota
        if service_name in self.quotas:
            quota = self.quotas[service_name]

            # Reset if needed
            if datetime.now() >= quota['reset_time']:
                quota['used_today'] = 0
                quota['reset_time'] = datetime.now() + timedelta(days=1)

            # Check quota
            if quota['used_today'] >= quota['daily_limit']:
                return {
                    'allowed': False,
                    'reason': 'daily_quota_exceeded',
                    'reset_time': quota['reset_time']
                }

            quota['used_today'] += 1

        # Apply rate limiting
        limiter = self.limiters[service_name]
        await limiter.acquire()

        return {'allowed': True}

    def get_quota_status(self, service_name: str) -> Dict:
        """Get current quota status."""
        if service_name not in self.quotas:
            return {'status': 'no_quota'}

        quota = self.quotas[service_name]
        remaining = quota['daily_limit'] - quota['used_today']

        return {
            'daily_limit': quota['daily_limit'],
            'used': quota['used_today'],
            'remaining': remaining,
            'reset_time': quota['reset_time']
        }
```

## API Versioning and Compatibility

### Version-Aware API Client

```python
@dataclass
class APIVersion:
    """API version specification."""
    version: str
    base_url: str
    deprecated: bool = False
    deprecation_date: Optional[datetime] = None
    migration_guide_url: Optional[str] = None

class VersionedAPIClient:
    """
    API client with version management.
    """
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.versions: Dict[str, APIVersion] = {}
        self.current_version: Optional[str] = None
        self.clients: Dict[str, RESTAPIClient] = {}

    def register_version(
        self,
        version: str,
        base_url: str,
        deprecated: bool = False
    ):
        """Register API version."""
        api_version = APIVersion(
            version=version,
            base_url=base_url,
            deprecated=deprecated
        )

        self.versions[version] = api_version

        # Set as current if first or not deprecated
        if not self.current_version or not deprecated:
            self.current_version = version

    async def get_client(self, version: Optional[str] = None) -> RESTAPIClient:
        """Get client for specific version."""
        target_version = version or self.current_version

        if target_version not in self.versions:
            raise ValueError(f"Unknown version: {target_version}")

        # Check if deprecated
        api_version = self.versions[target_version]
        if api_version.deprecated:
            print(f"Warning: API version {target_version} is deprecated")

        # Get or create client
        if target_version not in self.clients:
            self.clients[target_version] = RESTAPIClient(
                base_url=api_version.base_url
            )

        return self.clients[target_version]
```

## Key Takeaways

1. **Universal REST Client**: Handle all HTTP methods with connection pooling

2. **Multi-Strategy Auth**: Support API keys, OAuth, JWT transparently

3. **Rate Limiting**: Token bucket and sliding window algorithms prevent throttling

4. **Quota Management**: Track daily limits across services

5. **Version Management**: Handle multiple API versions gracefully

6. **Secure Credentials**: Encrypted storage for sensitive data

## What's Next

In Lesson 3, we'll explore tool orchestration and workflow management for coordinating multiple tools.

---

**Practice Exercise**: Build a complete API integration system supporting 5+ external services with different authentication methods. Implement rate limiting maintaining <5% error rate under load. Add quota management and version handling with automatic client selection.
