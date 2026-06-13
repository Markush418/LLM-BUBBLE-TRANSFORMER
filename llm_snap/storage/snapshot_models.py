"""
Snapshot Models for LLM-SNAP Storage
====================================

Data models representing snapshots in the storage layer.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import json


@dataclass
class Snapshot:
    """
    Represents a single snapshot of agent execution state.
    
    This matches the structure used in the interceptor for consistency.
    """
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
        """Convert snapshot to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Snapshot':
        """Create snapshot from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert snapshot to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Snapshot':
        """Create snapshot from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)