"""
Snapshot Interceptor for LLM-SNAP
=================================

Core interceptor class that captures agent execution snapshots.
This is a Python adaptation of the LLM-SNAP TypeScript interceptor.
"""

import json
import hashlib
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Import SQLite store
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_snap.storage.sqlite_store import SQLiteSnapshotStore


class StepType(Enum):
    PERCEIVE = "perceive"
    REASON = "reason"
    ACT = "act"
    OBSERVE = "observe"
    FINAL = "final"


@dataclass
class Snapshot:
    """Represents a single snapshot of agent state."""
    step_index: int
    step_type: str
    timestamp: str
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    pre_state_hash: str
    post_state_hash: str
    run_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SnapshotInterceptor:
    """
    Intercepts agent-LLM communications to create snapshots.
    
    This is a Python adaptation designed to work with LLM-BUBBLE's
    architecture while maintaining compatibility with LLM-SNAP concepts.
    Uses SQLite for persistent storage.
    """
    
    def __init__(self, storage_path: str = ".sisyphus/data/snapshots.db"):
        # Use SQLite store for persistence
        self.store = SQLiteSnapshotStore(str(storage_path))
        self.current_run_id: Optional[str] = None
        self.step_counter = 0
        self.snapshots: List[Snapshot] = []
        
        # Legacy compatibility - keep memory store for current session
        self._memory_store: Dict[str, List[Snapshot]] = {}
    
    def intercept(self, run_id: str, step_index: int, 
                  params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intercept an agent execution step and create snapshots.
        
        Args:
            run_id: Unique identifier for this agent run
            step_index: Index of this step in the run
            params: Parameters containing messages, tools, model, etc.
            
        Returns:
            The original params (unchanged) - this is a pass-through interceptor
        """
        # Initialize run if needed
        if self.current_run_id != run_id:
            self._start_new_run(run_id)
        
        # Create pre-state snapshot
        pre_snapshot = self._create_snapshot(
            run_id=run_id,
            step_index=step_index,
            step_type=self._classify_step(params, is_pre=True),
            params=params,
            is_pre_state=True
        )
        
        # Store pre-state
        self._store_snapshot(pre_snapshot)
        
        # In a real implementation, we would:
        # 1. Send params to LLM
        # 2. Get response
        # 3. Create post-state snapshot
        # 4. Return the response
        
        # For this adaptation, we simulate by creating both snapshots immediately
        # In practice, this would be split into pre-intercept and post-intercept calls
        post_snapshot = self._create_snapshot(
            run_id=run_id,
            step_index=step_index,
            step_type=self._classify_step(params, is_pre=False),
            params=params,
            is_pre_state=False
        )
        
        # Store post-state
        self._store_snapshot(post_snapshot)
        
        # Return original params (pass-through)
        return params
    
    def _start_new_run(self, run_id: str, description: str = ""):
        """Start tracking a new agent run."""
        self.current_run_id = run_id
        self.step_counter = 0
        self.snapshots = []
        self._memory_store[run_id] = []
        
        # Create run in SQLite for persistence
        self.store.create_run(run_id, description)
    
    def _classify_step(self, params: Dict[str, Any], is_pre: bool) -> str:
        """
        Classify the step type based on parameters.
        
        This is a simplified classifier - in practice would be more sophisticated.
        """
        messages = params.get("messages", [])
        tools = params.get("tools", [])
        
        if not messages:
            return StepType.PERCEIVE.value
        
        last_message = messages[-1] if messages else None
        if last_message and last_message.get("role") == "user":
            return StepType.PERCEIVE.value
        elif tools:
            return StepType.ACT.value
        elif is_pre:
            return StepType.REASON.value
        else:
            return StepType.OBSERVE.value
    
    def _create_snapshot(self, run_id: str, step_index: int, step_type: str,
                        params: Dict[str, Any], is_pre_state: bool) -> Snapshot:
        """Create a snapshot from the given parameters."""
        messages = params.get("messages", [])
        tools = params.get("tools", [])
        model = params.get("model", "unknown")
        
        # Create a string representation for hashing
        state_str = json.dumps({
            "messages": messages,
            "tools": tools,
            "model": model,
            "timestamp": time.time(),
            "is_pre": is_pre_state
        }, sort_keys=True)
        
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()
        
        metadata = {
            "model": model,
            "is_pre_state": is_pre_state,
            "interceptor_version": "0.1.0"
        }
        
        return Snapshot(
            step_index=step_index,
            step_type=step_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            messages=messages,
            tools=tools,
            metadata=metadata,
            pre_state_hash=state_hash if is_pre_state else "",
            post_state_hash=state_hash if not is_pre_state else "",
            run_id=run_id
        )
    
    def _store_snapshot(self, snapshot: Snapshot):
        """Store a snapshot in memory and persist to SQLite."""
        if self.current_run_id not in self._memory_store:
            self._memory_store[self.current_run_id] = []
        
        self._memory_store[self.current_run_id].append(snapshot)
        self.snapshots.append(snapshot)
        
        # Persist to SQLite for persistent storage
        self.store.save_snapshot(
            run_id=snapshot.run_id,
            step_index=snapshot.step_index,
            step_type=snapshot.step_type,
            messages=snapshot.messages,
            tools=snapshot.tools,
            metadata=snapshot.metadata,
            pre_hash=snapshot.pre_state_hash,
            post_hash=snapshot.post_state_hash
        )
        
        # Also keep JSON for debugging/inspection
        self._persist_snapshot_json(snapshot)
    
    def get_timeline(self, run_id: str) -> List[Dict[str, Any]]:
        """Get timeline of snapshots for a run."""
        if run_id not in self._memory_store:
            return []
        
        return [snapshot.to_dict() for snapshot in self._memory_store[run_id]]
    
    def get_snapshot_at_step(self, run_id: str, step_index: int) -> Optional[Dict[str, Any]]:
        """Get a specific snapshot by step index."""
        if run_id not in self._memory_store:
            return None
        
        for snapshot in self._memory_store[run_id]:
            if snapshot.step_index == step_index:
                return snapshot.to_dict()
        return None
    
    def finish_run(self, run_id: str):
        """Mark a run as finished."""
        # In a full implementation, we would finalize the run in storage
        pass


# Example usage function for testing
def example_usage():
    """Example of how to use the SnapshotInterceptor."""
    interceptor = SnapshotInterceptor()
    
    # Simulate an agent run
    run_id = f"run-{int(time.time())}"
    
    # Step 1: Perceive user request
    params1 = {
        "messages": [{"role": "user", "content": "Create a hello world program"}],
        "tools": [],
        "model": "claude-sonnet-4-6"
    }
    result1 = interceptor.intercept(run_id, 0, params1)
    
    # Step 2: Reason about the request
    params2 = {
        "messages": [
            {"role": "user", "content": "Create a hello world program"},
            {"role": "assistant", "content": "I'll create a simple Python program that prints 'Hello, World!'"}
        ],
        "tools": [],
        "model": "claude-sonnet-4-6"
    }
    result2 = interceptor.intercept(run_id, 1, params2)
    
    # Step 3: Act - create the file
    params3 = {
        "messages": [
            {"role": "user", "content": "Create a hello world program"},
            {"role": "assistant", "content": "I'll create a simple Python program that prints 'Hello, World!'"},
            {"role": "assistant", "content": "Creating file: hello.py"}
        ],
        "tools": [{"type": "function", "function": {"name": "write_file", "arguments": {"path": "hello.py", "content": "print('Hello, World!')"}}}],
        "model": "claude-sonnet-4-6"
    }
    result3 = interceptor.intercept(run_id, 2, params3)
    
    # Get timeline
    timeline = interceptor.get_timeline(run_id)
    print(f"Timeline for run {run_id}:")
    for step in timeline:
        print(f"  [{step['step_index']}] {step['step_type']} - {step['timestamp']}")
    
    return interceptor


if __name__ == "__main__":
    example_usage()