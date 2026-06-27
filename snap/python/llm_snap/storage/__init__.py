"""
LLM-SNAP Storage Module
=======================

Storage backend for snapshots. In this simplified implementation,
we use JSON files for demonstration. A full implementation would
use SQLite via SQLAlchemy or similar.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from .snapshot_models import Snapshot


class SnapshotStorage:
    """
    Storage backend for snapshots.
    
    This is a simplified version that uses JSON files.
    A production implementation would use a proper database.
    """
    
    def __init__(self, storage_path: str = ".sisyphus/snapshots"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(self, snapshot: Snapshot) -> str:
        """
        Save a snapshot and return its ID.
        
        Args:
            snapshot: The snapshot to save
            
        Returns:
            String ID of the saved snapshot
        """
        # Create run directory
        run_dir = self.storage_path / snapshot.run_id
        run_dir.mkdir(exist_ok=True)
        
        # Create snapshot file
        snapshot_file = run_dir / f"step_{snapshot.step_index:04d}_{snapshot.step_type}.json"
        
        # Save snapshot
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        
        return str(snapshot_file)
    
    def load_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        """
        Load a snapshot by its ID.
        
        Args:
            snapshot_id: File path ID of the snapshot
            
        Returns:
            Snapshot object or None if not found
        """
        try:
            with open(snapshot_id, 'r') as f:
                data = json.load(f)
            return Snapshot.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def get_snapshots_for_run(self, run_id: str) -> List[Snapshot]:
        """
        Get all snapshots for a specific run.
        
        Args:
            run_id: Identifier of the run
            
        Returns:
            List of snapshots sorted by step index
        """
        run_dir = self.storage_path / run_id
        if not run_dir.exists():
            return []
        
        snapshots = []
        for snapshot_file in run_dir.glob("step_*_*.json"):
            snapshot = self.load_snapshot(str(snapshot_file))
            if snapshot:
                snapshots.append(snapshot)
        
        # Sort by step index
        snapshots.sort(key=lambda s: s.step_index)
        return snapshots
    
    def delete_run(self, run_id: str) -> bool:
        """
        Delete all snapshots for a run.
        
        Args:
            run_id: Identifier of the run to delete
            
        Returns:
            True if run was deleted, False if not found
        """
        run_dir = self.storage_path / run_id
        if not run_dir.exists():
            return False
        
        import shutil
        shutil.rmtree(run_dir)
        return True