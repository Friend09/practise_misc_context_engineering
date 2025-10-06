# Lesson 2: Persistent Context Storage Systems

## Introduction

Persistent storage enables AI agents to maintain memory across sessions, survive system restarts, and learn from historical interactions. This lesson teaches you to design robust storage systems with serialization, backup, recovery, and distributed synchronization capabilities.

## Database Design for AI Agent Memory

### SQL-Based Episodic Memory

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import sqlite3
import json
import pickle
from contextlib import contextmanager

class EpisodicMemoryStore:
    """
    SQL-based persistent storage for episodic memories.
    Optimized for sequential access and temporal queries.
    """
    def __init__(self, db_path: str = "episodic_memory.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Main memory table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    content_data BLOB NOT NULL,
                    importance REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    INDEX idx_agent_time (agent_id, created_at),
                    INDEX idx_importance (importance),
                    INDEX idx_access (last_accessed)
                )
            ''')

            # Tags table for flexible querying
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_tags (
                    memory_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (memory_id, tag),
                    FOREIGN KEY (memory_id) REFERENCES episodic_memory(id)
                        ON DELETE CASCADE
                )
            ''')

            # Associations table for memory relationships
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_associations (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    association_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    created_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (source_id, target_id, association_type),
                    FOREIGN KEY (source_id) REFERENCES episodic_memory(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES episodic_memory(id)
                        ON DELETE CASCADE
                )
            ''')

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def store(self, memory_item: 'MemoryItem') -> bool:
        """Store memory item persistently."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Serialize content
                content_data = pickle.dumps(memory_item.content)
                metadata_json = json.dumps(memory_item.metadata)

                # Insert or replace memory
                cursor.execute('''
                    INSERT OR REPLACE INTO episodic_memory
                    (id, agent_id, content_type, content_data, importance,
                     created_at, last_accessed, access_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory_item.id,
                    memory_item.metadata.get('agent_id', 'default'),
                    type(memory_item.content).__name__,
                    content_data,
                    memory_item.importance,
                    memory_item.created_at.isoformat(),
                    memory_item.last_accessed.isoformat(),
                    memory_item.access_count,
                    metadata_json
                ))

                # Store tags
                cursor.execute('DELETE FROM memory_tags WHERE memory_id = ?',
                             (memory_item.id,))

                for tag in memory_item.tags:
                    cursor.execute('''
                        INSERT INTO memory_tags (memory_id, tag)
                        VALUES (?, ?)
                    ''', (memory_item.id, tag))

                conn.commit()
                return True

        except Exception as e:
            print(f"Error storing memory: {e}")
            return False

    def retrieve(self, memory_id: str) -> Optional['MemoryItem']:
        """Retrieve memory item by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM episodic_memory WHERE id = ?
            ''', (memory_id,))

            row = cursor.fetchone()
            if not row:
                return None

            # Fetch tags
            cursor.execute('''
                SELECT tag FROM memory_tags WHERE memory_id = ?
            ''', (memory_id,))
            tags = {row['tag'] for row in cursor.fetchall()}

            # Reconstruct memory item
            from modules.module_05.lessons import MemoryItem, MemoryTier

            memory_item = MemoryItem(
                id=row['id'],
                content=pickle.loads(row['content_data']),
                tier=MemoryTier.EPISODIC,
                created_at=datetime.fromisoformat(row['created_at']),
                last_accessed=datetime.fromisoformat(row['last_accessed']),
                access_count=row['access_count'],
                importance=row['importance'],
                tags=tags,
                metadata=json.loads(row['metadata'])
            )

            # Update access statistics
            cursor.execute('''
                UPDATE episodic_memory
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            ''', (datetime.now().isoformat(), memory_id))

            conn.commit()

            return memory_item

    def query(
        self,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_importance: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List['MemoryItem']:
        """Query memories with filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT DISTINCT m.* FROM episodic_memory m"
            conditions = []
            params = []

            # Tag filtering requires join
            if tags:
                query += " INNER JOIN memory_tags t ON m.id = t.memory_id"
                conditions.append(f"t.tag IN ({','.join(['?'] * len(tags))})")
                params.extend(tags)

            # Other filters
            if agent_id:
                conditions.append("m.agent_id = ?")
                params.append(agent_id)

            if min_importance is not None:
                conditions.append("m.importance >= ?")
                params.append(min_importance)

            if start_time:
                conditions.append("m.created_at >= ?")
                params.append(start_time.isoformat())

            if end_time:
                conditions.append("m.created_at <= ?")
                params.append(end_time.isoformat())

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY m.created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                # Fetch tags for each memory
                cursor.execute('''
                    SELECT tag FROM memory_tags WHERE memory_id = ?
                ''', (row['id'],))
                tags_set = {tag_row['tag'] for tag_row in cursor.fetchall()}

                from modules.module_05.lessons import MemoryItem, MemoryTier

                memory_item = MemoryItem(
                    id=row['id'],
                    content=pickle.loads(row['content_data']),
                    tier=MemoryTier.EPISODIC,
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_accessed=datetime.fromisoformat(row['last_accessed']),
                    access_count=row['access_count'],
                    importance=row['importance'],
                    tags=tags_set,
                    metadata=json.loads(row['metadata'])
                )
                results.append(memory_item)

            return results

    def add_association(
        self,
        source_id: str,
        target_id: str,
        association_type: str,
        strength: float = 1.0
    ) -> bool:
        """Create association between memories."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO memory_associations
                    (source_id, target_id, association_type, strength, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (source_id, target_id, association_type, strength,
                     datetime.now().isoformat()))

                conn.commit()
                return True

        except Exception as e:
            print(f"Error creating association: {e}")
            return False

    def get_associated_memories(
        self,
        memory_id: str,
        association_type: Optional[str] = None,
        min_strength: float = 0.5
    ) -> List['MemoryItem']:
        """Get memories associated with given memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = '''
                SELECT m.*, a.strength
                FROM episodic_memory m
                INNER JOIN memory_associations a ON m.id = a.target_id
                WHERE a.source_id = ? AND a.strength >= ?
            '''
            params = [memory_id, min_strength]

            if association_type:
                query += " AND a.association_type = ?"
                params.append(association_type)

            query += " ORDER BY a.strength DESC"

            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                from modules.module_05.lessons import MemoryItem, MemoryTier

                memory_item = MemoryItem(
                    id=row['id'],
                    content=pickle.loads(row['content_data']),
                    tier=MemoryTier.EPISODIC,
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_accessed=datetime.fromisoformat(row['last_accessed']),
                    access_count=row['access_count'],
                    importance=row['importance'],
                    tags=set(),
                    metadata=json.loads(row['metadata'])
                )
                results.append(memory_item)

            return results

    def cleanup_old_memories(
        self,
        days_old: int = 90,
        min_importance: float = 0.3
    ) -> int:
        """Remove old, low-importance memories."""
        cutoff_date = datetime.now() - timedelta(days=days_old)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                DELETE FROM episodic_memory
                WHERE created_at < ? AND importance < ?
            ''', (cutoff_date.isoformat(), min_importance))

            deleted_count = cursor.rowcount
            conn.commit()

            return deleted_count
```

## Context Serialization Strategies

### Advanced Serialization System

```python
import gzip
from abc import ABC, abstractmethod

class SerializationStrategy(ABC):
    """Base class for serialization strategies."""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to object."""
        pass

class PickleSerializer(SerializationStrategy):
    """Pickle-based serialization."""

    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

class CompressedSerializer(SerializationStrategy):
    """Compressed serialization for large objects."""

    def __init__(self, base_serializer: SerializationStrategy):
        self.base_serializer = base_serializer

    def serialize(self, obj: Any) -> bytes:
        serialized = self.base_serializer.serialize(obj)
        return gzip.compress(serialized)

    def deserialize(self, data: bytes) -> Any:
        decompressed = gzip.decompress(data)
        return self.base_serializer.deserialize(decompressed)

class JSONSerializer(SerializationStrategy):
    """JSON serialization for simple objects."""

    def serialize(self, obj: Any) -> bytes:
        return json.dumps(obj, default=str).encode('utf-8')

    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))

class AdaptiveSerializer:
    """
    Automatically selects best serialization strategy.
    """
    def __init__(self):
        self.strategies = {
            'pickle': PickleSerializer(),
            'compressed': CompressedSerializer(PickleSerializer()),
            'json': JSONSerializer()
        }

    def serialize(self, obj: Any) -> Tuple[bytes, str]:
        """Serialize with best strategy."""
        # Try simple types with JSON
        if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
            try:
                data = self.strategies['json'].serialize(obj)
                return data, 'json'
            except:
                pass

        # Try pickle
        try:
            data = self.strategies['pickle'].serialize(obj)

            # Use compression if data is large
            if len(data) > 1024:  # > 1KB
                compressed = self.strategies['compressed'].serialize(obj)
                if len(compressed) < len(data) * 0.9:  # 10% savings
                    return compressed, 'compressed'

            return data, 'pickle'

        except Exception as e:
            raise ValueError(f"Could not serialize object: {e}")

    def deserialize(self, data: bytes, strategy: str) -> Any:
        """Deserialize with specified strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self.strategies[strategy].deserialize(data)
```

## Backup and Recovery Systems

### Automated Backup System

```python
import shutil
from pathlib import Path

class MemoryBackupSystem:
    """
    Automated backup system for memory databases.
    """
    def __init__(
        self,
        primary_db_path: str,
        backup_dir: str = "backups"
    ):
        self.primary_db_path = Path(primary_db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.backup_history: List[Dict] = []

    def create_backup(self, backup_type: str = "full") -> str:
        """Create database backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{backup_type}_{timestamp}.db"
        backup_path = self.backup_dir / backup_name

        try:
            # Copy database file
            shutil.copy2(self.primary_db_path, backup_path)

            # Compress backup
            compressed_path = self._compress_backup(backup_path)

            # Record backup
            self.backup_history.append({
                'type': backup_type,
                'path': str(compressed_path),
                'timestamp': datetime.now(),
                'size_bytes': compressed_path.stat().st_size
            })

            # Cleanup old backups
            self._cleanup_old_backups()

            return str(compressed_path)

        except Exception as e:
            print(f"Backup failed: {e}")
            raise

    def _compress_backup(self, backup_path: Path) -> Path:
        """Compress backup file."""
        import gzip

        compressed_path = backup_path.with_suffix('.db.gz')

        with open(backup_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove uncompressed backup
        backup_path.unlink()

        return compressed_path

    def restore_backup(self, backup_path: str) -> bool:
        """Restore from backup."""
        backup_path = Path(backup_path)

        if not backup_path.exists():
            print(f"Backup not found: {backup_path}")
            return False

        try:
            # Decompress if needed
            if backup_path.suffix == '.gz':
                temp_path = backup_path.with_suffix('')
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                restore_source = temp_path
            else:
                restore_source = backup_path

            # Backup current database
            if self.primary_db_path.exists():
                emergency_backup = self.primary_db_path.with_suffix('.emergency.db')
                shutil.copy2(self.primary_db_path, emergency_backup)

            # Restore backup
            shutil.copy2(restore_source, self.primary_db_path)

            # Cleanup temp file
            if restore_source != backup_path:
                restore_source.unlink()

            return True

        except Exception as e:
            print(f"Restore failed: {e}")
            return False

    def _cleanup_old_backups(self, keep_count: int = 10):
        """Remove old backups, keeping most recent."""
        if len(self.backup_history) <= keep_count:
            return

        # Sort by timestamp
        self.backup_history.sort(key=lambda x: x['timestamp'], reverse=True)

        # Remove old backups
        for backup in self.backup_history[keep_count:]:
            backup_path = Path(backup['path'])
            if backup_path.exists():
                backup_path.unlink()

        # Update history
        self.backup_history = self.backup_history[:keep_count]

    def schedule_automatic_backups(self, interval_hours: int = 24):
        """Schedule automatic backups."""
        # In production, use proper scheduling system (cron, celery, etc.)
        # This is a simplified example
        import threading
        import time

        def backup_loop():
            while True:
                time.sleep(interval_hours * 3600)
                try:
                    self.create_backup(backup_type="automatic")
                except Exception as e:
                    print(f"Automatic backup failed: {e}")

        backup_thread = threading.Thread(target=backup_loop, daemon=True)
        backup_thread.start()
```

## Key Takeaways

1. **Persistent Storage**: SQL databases provide reliable, queryable memory storage

2. **Flexible Querying**: Tag-based and temporal queries enable efficient retrieval

3. **Serialization**: Multiple strategies optimize for different data types

4. **Associations**: Memory relationships enable sophisticated retrieval

5. **Backup Systems**: Automated backups prevent data loss

6. **Recovery**: Robust restoration capabilities ensure reliability

## What's Next

In Lesson 3, we'll explore hierarchical memory management with automatic consolidation and optimization.

---

**Practice Exercise**: Build a persistent memory system with backup/recovery. Test with 100,000+ memories, verify <100ms query times, and demonstrate successful recovery from backup.
