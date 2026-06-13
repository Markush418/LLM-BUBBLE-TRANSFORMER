"""
SQLite Storage Layer for LLM-SNAP
==================================

Provides persistent storage for agent execution snapshots using SQLite.
This enables audit trails, timeline views, and recovery between sessions.
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

class SQLiteSnapshotStore:
    """
    Persistent snapshot storage using SQLite.
    
    Database Schema:
    - runs: Agent run metadata
    - snapshots: Individual step snapshots with pre/post state hashes
    """
    
    def __init__(self, db_path: str = ".sisyphus/data/snapshots.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                description TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT DEFAULT 'running',
                total_steps INTEGER DEFAULT 0
            )
        """)
        
        # Create snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                step_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                messages_json TEXT,
                tools_json TEXT,
                metadata_json TEXT,
                pre_state_hash TEXT NOT NULL,
                post_state_hash TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                UNIQUE(run_id, step_index)
            )
        """)
        
        # Create index for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_run 
            ON snapshots(run_id, step_index)
        """)
        
        conn.commit()
        conn.close()
    
    def create_run(self, run_id: str, description: str = "") -> bool:
        """Create a new agent run entry."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO runs (run_id, description, start_time, status)
                VALUES (?, ?, ?, 'running')
            """, (run_id, description, datetime.now().isoformat()))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Run already exists
        finally:
            conn.close()
    
    def complete_run(self, run_id: str, total_steps: int) -> bool:
        """Mark a run as completed."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE runs 
            SET end_time = ?, status = 'completed', total_steps = ?
            WHERE run_id = ?
        """, (datetime.now().isoformat(), total_steps, run_id))
        
        conn.commit()
        conn.close()
        return True
    
    def save_snapshot(self, run_id: str, step_index: int, step_type: str,
                      messages: List[Dict], tools: List[Dict],
                      metadata: Dict, pre_hash: str, post_hash: str) -> bool:
        """Save a single snapshot to the database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO snapshots 
                (run_id, step_index, step_type, timestamp, messages_json, 
                 tools_json, metadata_json, pre_state_hash, post_state_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, step_index, step_type, datetime.now().isoformat(),
                json.dumps(messages), json.dumps(tools), json.dumps(metadata),
                pre_hash, post_hash
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Snapshot already exists
        finally:
            conn.close()
    
    def get_snapshots(self, run_id: str) -> List[Dict]:
        """Retrieve all snapshots for a run, ordered by step_index."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM snapshots 
            WHERE run_id = ? 
            ORDER BY step_index
        """, (run_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        snapshots = []
        for row in rows:
            snapshots.append({
                'run_id': row['run_id'],
                'step_index': row['step_index'],
                'step_type': row['step_type'],
                'timestamp': row['timestamp'],
                'messages': json.loads(row['messages_json']),
                'tools': json.loads(row['tools_json']),
                'metadata': json.loads(row['metadata_json']),
                'pre_state_hash': row['pre_state_hash'],
                'post_state_hash': row['post_state_hash']
            })
        
        return snapshots
    
    def get_runs(self, limit: int = 50) -> List[Dict]:
        """Retrieve recent runs."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM runs 
            ORDER BY start_time DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_run_timeline(self, run_id: str) -> List[Dict]:
        """Get a timeline view of a run (summary of each step)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT step_index, step_type, timestamp, 
                   length(messages_json) as msg_size,
                   pre_state_hash, post_state_hash
            FROM snapshots 
            WHERE run_id = ?
            ORDER BY step_index
        """, (run_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def run_exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        
        return exists
    
    def get_snapshot_count(self, run_id: str) -> int:
        """Get total snapshot count for a run."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM snapshots WHERE run_id = ?", (run_id,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def compute_state_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of state data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]