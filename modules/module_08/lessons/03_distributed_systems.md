# Lesson 3: Distributed Systems and Consistency

## Introduction

Enterprise context engineering requires distributed architectures that maintain consistency while achieving horizontal scale. This lesson teaches you to build distributed context systems with advanced consensus algorithms, partition tolerance, and global state management that handle millions of operations across multiple data centers.

## Distributed Context Architecture

### Distributed Consensus Engine

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import time
import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import uuid
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ConsensusAlgorithm(Enum):
    """Consensus algorithms for distributed coordination."""
    RAFT = "raft"
    PBFT = "pbft"          # Practical Byzantine Fault Tolerance
    POW = "proof_of_work"   # For high-security scenarios
    CUSTOM = "custom"       # Custom consensus for specific needs

class NodeState(Enum):
    """States for distributed nodes."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    RECOVERING = "recovering"
    OFFLINE = "offline"

class ConsistencyLevel(Enum):
    """Consistency levels for distributed operations."""
    EVENTUAL = "eventual"
    STRONG = "strong"
    BOUNDED_STALENESS = "bounded_staleness"
    SESSION = "session"
    CONSISTENT_PREFIX = "consistent_prefix"

@dataclass
class LogEntry:
    """Entry in the distributed log for consensus."""
    term: int
    index: int
    command: str
    data: Dict[str, Any]
    timestamp: datetime
    client_id: str
    request_id: str
    checksum: str = field(default_factory=str)

    def __post_init__(self):
        """Calculate checksum for integrity verification."""
        content = f"{self.term}_{self.index}_{self.command}_{json.dumps(self.data, sort_keys=True)}"
        self.checksum = hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify entry integrity using checksum."""
        content = f"{self.term}_{self.index}_{self.command}_{json.dumps(self.data, sort_keys=True)}"
        expected_checksum = hashlib.sha256(content.encode()).hexdigest()
        return self.checksum == expected_checksum

@dataclass
class NodeInfo:
    """Information about a distributed node."""
    node_id: str
    host: str
    port: int
    region: str
    datacenter: str
    capabilities: Set[str]
    last_heartbeat: datetime
    state: NodeState = NodeState.FOLLOWER
    current_term: int = 0
    voted_for: Optional[str] = None

    @property
    def is_alive(self) -> bool:
        """Check if node is considered alive based on heartbeat."""
        return (datetime.now() - self.last_heartbeat).total_seconds() < 30

@dataclass
class DistributedOperation:
    """Operation to be executed across distributed system."""
    operation_id: str
    operation_type: str
    target_keys: List[str]
    data: Dict[str, Any]
    consistency_level: ConsistencyLevel
    timeout_seconds: int
    client_id: str
    created_at: datetime = field(default_factory=datetime.now)

    # Execution tracking
    required_nodes: Set[str] = field(default_factory=set)
    confirmed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    result: Any = None
    completed_at: Optional[datetime] = None

class DistributedContextNode:
    """Individual node in distributed context system."""

    def __init__(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 8000,
        region: str = "us-east-1",
        datacenter: str = "dc1",
        consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.RAFT
    ):
        self.node_info = NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            region=region,
            datacenter=datacenter,
            capabilities={'context_storage', 'consensus', 'replication'},
            last_heartbeat=datetime.now()
        )

        self.consensus_algorithm = consensus_algorithm

        # Raft-specific state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0

        # Leader state (valid only when leader)
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        # Cluster management
        self.cluster_nodes: Dict[str, NodeInfo] = {}
        self.leader_id: Optional[str] = None

        # Context storage
        self.context_data: Dict[str, Any] = {}
        self.pending_operations: Dict[str, DistributedOperation] = {}

        # Network simulation (in real implementation, use actual network)
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.network_latency_ms = 10  # Simulated network latency

        # Performance tracking
        self.performance_metrics = {
            'operations_committed': 0,
            'leadership_changes': 0,
            'network_errors': 0,
            'consensus_time_ms': [],
            'replication_lag_ms': []
        }

        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.election_task: Optional[asyncio.Task] = None
        self.replication_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the distributed node."""
        self.node_info.last_heartbeat = datetime.now()

        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        if self.consensus_algorithm == ConsensusAlgorithm.RAFT:
            self.election_task = asyncio.create_task(self._raft_election_loop())
            self.replication_task = asyncio.create_task(self._raft_replication_loop())

        logging.info(f"Node {self.node_info.node_id} started in {self.node_info.region}")

    async def stop(self):
        """Stop the distributed node gracefully."""
        # Cancel background tasks
        for task in [self.heartbeat_task, self.election_task, self.replication_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.node_info.state = NodeState.OFFLINE
        logging.info(f"Node {self.node_info.node_id} stopped")

    async def join_cluster(self, existing_nodes: List[NodeInfo]):
        """Join an existing cluster."""
        for node in existing_nodes:
            self.cluster_nodes[node.node_id] = node

        # Start in follower state
        self.node_info.state = NodeState.FOLLOWER

        # Request current state from leader
        await self._request_cluster_state()

    async def execute_distributed_operation(
        self,
        operation: DistributedOperation
    ) -> Any:
        """Execute operation across distributed system."""
        if self.node_info.state != NodeState.LEADER:
            # Forward to leader
            if self.leader_id and self.leader_id in self.cluster_nodes:
                return await self._forward_to_leader(operation)
            else:
                raise Exception("No leader available for operation")

        # Execute as leader
        return await self._execute_as_leader(operation)

    async def _execute_as_leader(self, operation: DistributedOperation) -> Any:
        """Execute operation as cluster leader."""
        start_time = time.time()

        try:
            # Create log entry
            log_entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                command=operation.operation_type,
                data=operation.data,
                timestamp=datetime.now(),
                client_id=operation.client_id,
                request_id=operation.operation_id
            )

            # Append to local log
            self.log.append(log_entry)

            # Determine required consensus
            required_nodes = self._calculate_required_nodes(operation.consistency_level)
            operation.required_nodes = required_nodes

            # Replicate to followers
            replication_results = await self._replicate_log_entry(log_entry, required_nodes)

            # Check if we have consensus
            confirmed_nodes = set(replication_results.keys())
            confirmed_nodes.add(self.node_info.node_id)  # Include self

            if len(confirmed_nodes) >= len(required_nodes):
                # Commit the operation
                self.commit_index = log_entry.index
                result = await self._apply_operation(log_entry)

                # Update operation status
                operation.confirmed_nodes = confirmed_nodes
                operation.result = result
                operation.completed_at = datetime.now()

                # Track performance
                consensus_time = (time.time() - start_time) * 1000
                self.performance_metrics['consensus_time_ms'].append(consensus_time)
                self.performance_metrics['operations_committed'] += 1

                return result
            else:
                # Not enough nodes confirmed
                failed_nodes = required_nodes - confirmed_nodes
                operation.failed_nodes = failed_nodes
                raise Exception(f"Consensus failed - confirmed: {len(confirmed_nodes)}, required: {len(required_nodes)}")

        except Exception as e:
            logging.error(f"Operation execution failed: {e}")
            raise

    def _calculate_required_nodes(self, consistency_level: ConsistencyLevel) -> Set[str]:
        """Calculate which nodes are required for the given consistency level."""
        all_nodes = set(self.cluster_nodes.keys())
        alive_nodes = {
            node_id for node_id, node in self.cluster_nodes.items()
            if node.is_alive
        }

        if consistency_level == ConsistencyLevel.STRONG:
            # Require majority of all nodes
            required_count = (len(all_nodes) + 1) // 2
            return set(list(alive_nodes)[:required_count])

        elif consistency_level == ConsistencyLevel.EVENTUAL:
            # Require only one additional node
            return set(list(alive_nodes)[:1])

        elif consistency_level == ConsistencyLevel.BOUNDED_STALENESS:
            # Require majority of alive nodes
            required_count = (len(alive_nodes) + 1) // 2
            return set(list(alive_nodes)[:required_count])

        elif consistency_level == ConsistencyLevel.SESSION:
            # Require nodes in same datacenter
            same_dc_nodes = {
                node_id for node_id, node in self.cluster_nodes.items()
                if node.datacenter == self.node_info.datacenter and node.is_alive
            }
            required_count = (len(same_dc_nodes) + 1) // 2
            return set(list(same_dc_nodes)[:required_count])

        else:  # CONSISTENT_PREFIX
            # Require at least 2 nodes for prefix consistency
            return set(list(alive_nodes)[:2])

    async def _replicate_log_entry(
        self,
        log_entry: LogEntry,
        target_nodes: Set[str]
    ) -> Dict[str, bool]:
        """Replicate log entry to target nodes."""
        replication_tasks = []

        for node_id in target_nodes:
            if node_id != self.node_info.node_id:  # Don't replicate to self
                task = asyncio.create_task(
                    self._send_append_entries(node_id, log_entry)
                )
                replication_tasks.append((node_id, task))

        # Wait for replication results
        results = {}
        for node_id, task in replication_tasks:
            try:
                success = await asyncio.wait_for(task, timeout=5.0)
                results[node_id] = success
            except asyncio.TimeoutError:
                results[node_id] = False
                logging.warning(f"Replication timeout for node {node_id}")
            except Exception as e:
                results[node_id] = False
                logging.error(f"Replication failed for node {node_id}: {e}")

        return results

    async def _send_append_entries(self, target_node_id: str, log_entry: LogEntry) -> bool:
        """Send append entries RPC to target node."""
        # Simulate network communication
        await asyncio.sleep(self.network_latency_ms / 1000)

        # In real implementation, this would be an actual network call
        # For simulation, we'll assume success with some probability
        success_probability = 0.95  # 95% success rate

        if random.random() < success_probability:
            # Simulate successful replication
            return True
        else:
            # Simulate network error
            self.performance_metrics['network_errors'] += 1
            return False

    async def _apply_operation(self, log_entry: LogEntry) -> Any:
        """Apply committed operation to local state."""
        command = log_entry.command
        data = log_entry.data

        if command == "put_context":
            key = data['key']
            value = data['value']
            self.context_data[key] = value
            return True

        elif command == "get_context":
            key = data['key']
            return self.context_data.get(key)

        elif command == "delete_context":
            key = data['key']
            if key in self.context_data:
                del self.context_data[key]
                return True
            return False

        elif command == "batch_update":
            updates = data['updates']
            for key, value in updates.items():
                self.context_data[key] = value
            return len(updates)

        else:
            raise Exception(f"Unknown command: {command}")

    # Raft consensus implementation
    async def _raft_election_loop(self):
        """Raft leader election loop."""
        while True:
            try:
                if self.node_info.state == NodeState.FOLLOWER:
                    # Wait for heartbeat or election timeout
                    election_timeout = random.uniform(150, 300) / 1000  # 150-300ms
                    await asyncio.sleep(election_timeout)

                    # Check if we received heartbeat recently
                    if self._should_start_election():
                        await self._start_election()

                elif self.node_info.state == NodeState.CANDIDATE:
                    # Continue election process
                    await self._continue_election()

                elif self.node_info.state == NodeState.LEADER:
                    # Send heartbeats
                    await self._send_heartbeats()
                    await asyncio.sleep(0.05)  # 50ms heartbeat interval

                else:
                    await asyncio.sleep(0.1)  # Other states

            except Exception as e:
                logging.error(f"Election loop error: {e}")
                await asyncio.sleep(0.1)

    def _should_start_election(self) -> bool:
        """Check if node should start election."""
        # Simple logic: start election if no recent heartbeat from leader
        if not self.leader_id:
            return True

        if self.leader_id not in self.cluster_nodes:
            return True

        leader_node = self.cluster_nodes[self.leader_id]
        time_since_heartbeat = (datetime.now() - leader_node.last_heartbeat).total_seconds()

        return time_since_heartbeat > 0.2  # 200ms timeout

    async def _start_election(self):
        """Start leader election."""
        self.current_term += 1
        self.node_info.state = NodeState.CANDIDATE
        self.voted_for = self.node_info.node_id
        self.leader_id = None

        # Vote for self
        votes_received = 1

        # Request votes from other nodes
        vote_tasks = []
        for node_id in self.cluster_nodes:
            if node_id != self.node_info.node_id:
                task = asyncio.create_task(self._request_vote(node_id))
                vote_tasks.append(task)

        # Wait for vote responses
        for task in vote_tasks:
            try:
                vote_granted = await asyncio.wait_for(task, timeout=0.1)
                if vote_granted:
                    votes_received += 1
            except asyncio.TimeoutError:
                pass  # Vote request timeout

        # Check if won election
        total_nodes = len(self.cluster_nodes) + 1  # Include self
        majority = (total_nodes + 1) // 2

        if votes_received >= majority:
            await self._become_leader()
        else:
            # Election failed, become follower
            self.node_info.state = NodeState.FOLLOWER
            self.voted_for = None

    async def _request_vote(self, target_node_id: str) -> bool:
        """Request vote from target node."""
        # Simulate vote request
        await asyncio.sleep(self.network_latency_ms / 1000)

        # In real implementation, this would be actual RPC
        # For simulation, assume nodes vote based on term and log consistency
        return random.random() < 0.8  # 80% chance of vote

    async def _become_leader(self):
        """Become cluster leader."""
        self.node_info.state = NodeState.LEADER
        self.leader_id = self.node_info.node_id

        # Initialize leader state
        for node_id in self.cluster_nodes:
            self.next_index[node_id] = len(self.log)
            self.match_index[node_id] = 0

        # Track leadership change
        self.performance_metrics['leadership_changes'] += 1

        logging.info(f"Node {self.node_info.node_id} became leader for term {self.current_term}")

    async def _send_heartbeats(self):
        """Send heartbeat messages to all followers."""
        heartbeat_tasks = []

        for node_id in self.cluster_nodes:
            if node_id != self.node_info.node_id:
                task = asyncio.create_task(self._send_heartbeat(node_id))
                heartbeat_tasks.append(task)

        # Wait for heartbeat responses
        await asyncio.gather(*heartbeat_tasks, return_exceptions=True)

    async def _send_heartbeat(self, target_node_id: str):
        """Send heartbeat to specific node."""
        # Simulate heartbeat
        await asyncio.sleep(self.network_latency_ms / 1000)

        # In real implementation, this would be append entries RPC with no entries
        # For simulation, assume success
        return True

    async def _continue_election(self):
        """Continue election process as candidate."""
        # Wait for election timeout
        await asyncio.sleep(random.uniform(150, 300) / 1000)

        # If still candidate, start new election
        if self.node_info.state == NodeState.CANDIDATE:
            await self._start_election()

    async def _raft_replication_loop(self):
        """Raft log replication loop."""
        while True:
            try:
                if self.node_info.state == NodeState.LEADER:
                    # Replicate uncommitted entries
                    await self._replicate_uncommitted_entries()

                await asyncio.sleep(0.01)  # 10ms replication check interval

            except Exception as e:
                logging.error(f"Replication loop error: {e}")
                await asyncio.sleep(0.1)

    async def _replicate_uncommitted_entries(self):
        """Replicate uncommitted log entries to followers."""
        for node_id in self.cluster_nodes:
            if node_id != self.node_info.node_id:
                next_index = self.next_index.get(node_id, len(self.log))

                if next_index < len(self.log):
                    # There are entries to replicate
                    entries_to_send = self.log[next_index:]

                    for entry in entries_to_send[:10]:  # Batch size limit
                        success = await self._send_append_entries(node_id, entry)
                        if success:
                            self.next_index[node_id] = entry.index + 1
                            self.match_index[node_id] = entry.index
                        else:
                            # Replication failed, will retry next iteration
                            break

    async def _heartbeat_loop(self):
        """Update heartbeat timestamp."""
        while self.node_info.state != NodeState.OFFLINE:
            self.node_info.last_heartbeat = datetime.now()
            await asyncio.sleep(1.0)  # Update every second

    async def _request_cluster_state(self):
        """Request current cluster state when joining."""
        # In real implementation, this would request state from existing nodes
        # For simulation, we'll assume successful join
        logging.info(f"Node {self.node_info.node_id} requested cluster state")

    async def _forward_to_leader(self, operation: DistributedOperation) -> Any:
        """Forward operation to current leader."""
        if not self.leader_id or self.leader_id not in self.cluster_nodes:
            raise Exception("No available leader to forward operation")

        # In real implementation, this would be an actual network call
        # For simulation, assume the leader processes it
        logging.info(f"Forwarding operation {operation.operation_id} to leader {self.leader_id}")

        # Simulate network delay
        await asyncio.sleep(self.network_latency_ms / 1000)

        # Return simulated result
        return f"Forwarded to leader {self.leader_id}"

    def get_node_status(self) -> Dict[str, Any]:
        """Get comprehensive node status."""
        return {
            'node_info': {
                'node_id': self.node_info.node_id,
                'state': self.node_info.state.value,
                'current_term': self.current_term,
                'leader_id': self.leader_id,
                'region': self.node_info.region,
                'datacenter': self.node_info.datacenter
            },
            'consensus_state': {
                'log_length': len(self.log),
                'commit_index': self.commit_index,
                'last_applied': self.last_applied,
                'voted_for': self.voted_for
            },
            'cluster_info': {
                'cluster_size': len(self.cluster_nodes) + 1,
                'alive_nodes': sum(1 for node in self.cluster_nodes.values() if node.is_alive),
                'known_nodes': list(self.cluster_nodes.keys())
            },
            'storage': {
                'context_entries': len(self.context_data),
                'pending_operations': len(self.pending_operations)
            },
            'performance': {
                'operations_committed': self.performance_metrics['operations_committed'],
                'leadership_changes': self.performance_metrics['leadership_changes'],
                'network_errors': self.performance_metrics['network_errors'],
                'avg_consensus_time_ms': (
                    np.mean(self.performance_metrics['consensus_time_ms'])
                    if self.performance_metrics['consensus_time_ms'] else 0
                )
            }
        }

class DistributedContextCluster:
    """Manages a cluster of distributed context nodes."""

    def __init__(
        self,
        cluster_id: str,
        consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.RAFT
    ):
        self.cluster_id = cluster_id
        self.consensus_algorithm = consensus_algorithm
        self.nodes: Dict[str, DistributedContextNode] = {}
        self.client_connections: Dict[str, str] = {}  # client_id -> preferred_node_id

        # Cluster-wide metrics
        self.cluster_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'leader_elections': 0,
            'network_partitions': 0
        }

    async def add_node(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 8000,
        region: str = "us-east-1",
        datacenter: str = "dc1"
    ) -> DistributedContextNode:
        """Add a new node to the cluster."""
        node = DistributedContextNode(
            node_id=node_id,
            host=host,
            port=port,
            region=region,
            datacenter=datacenter,
            consensus_algorithm=self.consensus_algorithm
        )

        # Add references to existing nodes
        existing_nodes = [n.node_info for n in self.nodes.values()]
        if existing_nodes:
            await node.join_cluster(existing_nodes)

        # Update existing nodes with new node info
        for existing_node in self.nodes.values():
            existing_node.cluster_nodes[node_id] = node.node_info

        # Add to cluster
        self.nodes[node_id] = node
        await node.start()

        logging.info(f"Added node {node_id} to cluster {self.cluster_id}")
        return node

    async def remove_node(self, node_id: str):
        """Remove node from cluster."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            await node.stop()

            # Remove from other nodes' cluster info
            for other_node in self.nodes.values():
                if node_id in other_node.cluster_nodes:
                    del other_node.cluster_nodes[node_id]

            del self.nodes[node_id]
            logging.info(f"Removed node {node_id} from cluster")

    async def execute_operation(
        self,
        operation_type: str,
        data: Dict[str, Any],
        consistency_level: ConsistencyLevel = ConsistencyLevel.STRONG,
        client_id: str = None,
        preferred_region: str = None
    ) -> Any:
        """Execute operation on the cluster."""
        client_id = client_id or str(uuid.uuid4())

        # Select appropriate node for operation
        target_node = self._select_node_for_operation(
            client_id, preferred_region, consistency_level
        )

        if not target_node:
            raise Exception("No available nodes for operation")

        # Create operation
        operation = DistributedOperation(
            operation_id=str(uuid.uuid4()),
            operation_type=operation_type,
            target_keys=data.get('keys', []),
            data=data,
            consistency_level=consistency_level,
            timeout_seconds=30,
            client_id=client_id
        )

        try:
            # Execute operation
            result = await target_node.execute_distributed_operation(operation)

            self.cluster_metrics['total_operations'] += 1
            self.cluster_metrics['successful_operations'] += 1

            return result

        except Exception as e:
            self.cluster_metrics['total_operations'] += 1
            self.cluster_metrics['failed_operations'] += 1
            logging.error(f"Operation failed: {e}")
            raise

    def _select_node_for_operation(
        self,
        client_id: str,
        preferred_region: str = None,
        consistency_level: ConsistencyLevel = ConsistencyLevel.STRONG
    ) -> Optional[DistributedContextNode]:
        """Select best node for executing operation."""
        available_nodes = [
            node for node in self.nodes.values()
            if node.node_info.state in [NodeState.LEADER, NodeState.FOLLOWER]
        ]

        if not available_nodes:
            return None

        # For strong consistency, prefer leader
        if consistency_level == ConsistencyLevel.STRONG:
            leaders = [node for node in available_nodes if node.node_info.state == NodeState.LEADER]
            if leaders:
                return leaders[0]

        # For other consistency levels, consider region preference
        if preferred_region:
            region_nodes = [
                node for node in available_nodes
                if node.node_info.region == preferred_region
            ]
            if region_nodes:
                return region_nodes[0]

        # Return any available node
        return available_nodes[0]

    async def simulate_network_partition(
        self,
        partition_groups: List[List[str]],
        duration_seconds: int = 30
    ):
        """Simulate network partition for testing."""
        logging.info(f"Simulating network partition for {duration_seconds} seconds")
        self.cluster_metrics['network_partitions'] += 1

        # Increase network latency between partition groups
        original_latencies = {}
        for node in self.nodes.values():
            original_latencies[node.node_info.node_id] = node.network_latency_ms
            node.network_latency_ms = 1000  # 1 second delay

        # Wait for partition duration
        await asyncio.sleep(duration_seconds)

        # Restore normal network conditions
        for node in self.nodes.values():
            node.network_latency_ms = original_latencies[node.node_info.node_id]

        logging.info("Network partition simulation ended")

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        node_statuses = {}
        for node_id, node in self.nodes.items():
            node_statuses[node_id] = node.get_node_status()

        # Find current leader
        leaders = [
            node_id for node_id, status in node_statuses.items()
            if status['node_info']['state'] == 'leader'
        ]

        return {
            'cluster_info': {
                'cluster_id': self.cluster_id,
                'consensus_algorithm': self.consensus_algorithm.value,
                'total_nodes': len(self.nodes),
                'active_nodes': len([n for n in self.nodes.values() if n.node_info.is_alive]),
                'current_leaders': leaders
            },
            'cluster_metrics': self.cluster_metrics,
            'node_statuses': node_statuses,
            'health_summary': {
                'healthy_nodes': len([
                    n for n in self.nodes.values()
                    if n.node_info.is_alive and n.node_info.state != NodeState.OFFLINE
                ]),
                'consensus_active': len(leaders) == 1,
                'operations_success_rate': (
                    self.cluster_metrics['successful_operations'] /
                    max(1, self.cluster_metrics['total_operations'])
                )
            }
        }

# Example usage and testing
async def distributed_system_demo():
    """Demonstrate distributed context system capabilities."""

    # Create distributed cluster
    cluster = DistributedContextCluster(
        cluster_id="context-cluster-1",
        consensus_algorithm=ConsensusAlgorithm.RAFT
    )

    try:
        # Add nodes across multiple regions
        await cluster.add_node("node-1", region="us-east-1", datacenter="dc1")
        await cluster.add_node("node-2", region="us-east-1", datacenter="dc2")
        await cluster.add_node("node-3", region="us-west-2", datacenter="dc1")
        await cluster.add_node("node-4", region="eu-west-1", datacenter="dc1")
        await cluster.add_node("node-5", region="ap-southeast-1", datacenter="dc1")

        # Wait for cluster formation
        await asyncio.sleep(2)

        print("Phase 1: Testing basic operations")

        # Test basic operations
        operations = [
            ("put_context", {"key": "user_123", "value": {"name": "Alice", "preferences": {"theme": "dark"}}}),
            ("put_context", {"key": "session_456", "value": {"user_id": "123", "expires": "2025-01-01"}}),
            ("get_context", {"key": "user_123"}),
            ("batch_update", {"updates": {
                "user_124": {"name": "Bob", "preferences": {"theme": "light"}},
                "user_125": {"name": "Charlie", "preferences": {"theme": "auto"}}
            }}),
            ("get_context", {"key": "user_124"}),
        ]

        for op_type, data in operations:
            try:
                result = await cluster.execute_operation(
                    operation_type=op_type,
                    data=data,
                    consistency_level=ConsistencyLevel.STRONG
                )
                print(f"  {op_type}: {result}")
            except Exception as e:
                print(f"  {op_type} failed: {e}")

        print("\nPhase 2: Testing consistency levels")

        # Test different consistency levels
        consistency_tests = [
            ConsistencyLevel.STRONG,
            ConsistencyLevel.EVENTUAL,
            ConsistencyLevel.BOUNDED_STALENESS,
            ConsistencyLevel.SESSION
        ]

        for consistency in consistency_tests:
            try:
                result = await cluster.execute_operation(
                    operation_type="put_context",
                    data={"key": f"test_{consistency.value}", "value": {"level": consistency.value}},
                    consistency_level=consistency
                )
                print(f"  {consistency.value}: {result}")
            except Exception as e:
                print(f"  {consistency.value} failed: {e}")

        print("\nPhase 3: Testing fault tolerance")

        # Remove a node to test fault tolerance
        await cluster.remove_node("node-5")

        # Continue operations with reduced cluster
        for i in range(5):
            try:
                result = await cluster.execute_operation(
                    operation_type="put_context",
                    data={"key": f"fault_test_{i}", "value": {"test": "fault_tolerance"}},
                    consistency_level=ConsistencyLevel.STRONG
                )
                print(f"  Fault test {i}: Success")
            except Exception as e:
                print(f"  Fault test {i}: Failed - {e}")

        print("\nPhase 4: Network partition simulation")

        # Simulate network partition
        partition_task = asyncio.create_task(
            cluster.simulate_network_partition(
                partition_groups=[["node-1", "node-2"], ["node-3", "node-4"]],
                duration_seconds=10
            )
        )

        # Try operations during partition
        await asyncio.sleep(2)  # Let partition take effect

        for i in range(3):
            try:
                result = await cluster.execute_operation(
                    operation_type="put_context",
                    data={"key": f"partition_test_{i}", "value": {"test": "partition"}},
                    consistency_level=ConsistencyLevel.EVENTUAL
                )
                print(f"  Partition test {i}: Success")
            except Exception as e:
                print(f"  Partition test {i}: Failed - {e}")

        # Wait for partition to end
        await partition_task

        print("\nPhase 5: Final cluster status")

        # Get final cluster status
        status = cluster.get_cluster_status()
        print(json.dumps(status, indent=2, default=str))

    finally:
        # Cleanup
        for node_id in list(cluster.nodes.keys()):
            await cluster.remove_node(node_id)

# Run the demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(distributed_system_demo())
```

## Key Takeaways

1. **Consensus Algorithms**: Implement Raft consensus for distributed coordination and fault tolerance

2. **Consistency Levels**: Support multiple consistency models for different application requirements

3. **Partition Tolerance**: Handle network partitions gracefully while maintaining availability

4. **Leader Election**: Automatic leader election and failover for high availability

5. **Global State Management**: Maintain consistent state across geographically distributed nodes

6. **Performance Monitoring**: Track consensus performance and network health across the cluster

## What's Next

In Lesson 4, we'll explore performance monitoring and analytics systems for understanding system behavior and optimizing performance.

---

**Practice Exercise**: Build a distributed context system with 5+ nodes across multiple regions. Demonstrate consensus under network partitions, achieve <100ms consensus time for 95% of operations, and maintain >99.9% availability during single node failures.
