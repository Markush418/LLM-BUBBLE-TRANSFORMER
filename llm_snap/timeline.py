"""
Timeline Viewing Functionality for LLM-SNAP
===========================================

Provides timeline viewing capabilities for agent execution snapshots.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from .storage import SnapshotStorage


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%H:%M:%S")
    except:
        return timestamp_str


def format_step_type(step_type: str) -> str:
    """Format step type for display."""
    step_icons = {
        "perceive": "👁️",
        "reason": "💭", 
        "act": "🎬",
        "observe": "👀",
        "final": "🏁"
    }
    icon = step_icons.get(step_type, "❓")
    return f"{icon} {step_type.upper()}"


def view_timeline(run_id: str, storage_path: str = ".sisyphus/snapshots") -> None:
    """
    View timeline of snapshots for a run.
    
    Args:
        run_id: Identifier of the run to view
        storage_path: Path to snapshot storage
    """
    storage = SnapshotStorage(storage_path)
    snapshots = storage.get_snapshots_for_run(run_id)
    
    if not snapshots:
        print(f"No snapshots found for run: {run_id}")
        return
    
    print(f"\n{'='*60}")
    print(f"LLM-SNAP Timeline - Run: {run_id}")
    print(f"{'='*60}")
    print(f"Total steps: {len(snapshots)}")
    print()
    
    for snapshot in snapshots:
        time_str = format_timestamp(snapshot.timestamp)
        step_str = format_step_type(snapshot.step_type)
        
        # Show message preview
        msg_preview = ""
        if snapshot.messages:
            last_msg = snapshot.messages[-1]
            if last_msg.get("content"):
                content = last_msg["content"]
                # Truncate long messages
                if len(content) > 50:
                    content = content[:47] + "..."
                msg_preview = f" | {content}"
        
        print(f"[{snapshot.step_index:03d}] {time_str} {step_str}{msg_preview}")
    
    print(f"{'='*60}\n")


def view_timeline_detailed(run_id: str, storage_path: str = ".sisyphus/snapshots") -> None:
    """
    View detailed timeline of snapshots for a run.
    
    Args:
        run_id: Identifier of the run to view
        storage_path: Path to snapshot storage
    """
    storage = SnapshotStorage(storage_path)
    snapshots = storage.get_snapshots_for_run(run_id)
    
    if not snapshots:
        print(f"No snapshots found for run: {run_id}")
        return
    
    print(f"\n{'='*80}")
    print(f"LLM-SNAP Detailed Timeline - Run: {run_id}")
    print(f"{'='*80}")
    
    for snapshot in snapshots:
        time_str = format_timestamp(snapshot.timestamp)
        step_str = format_step_type(snapshot.step_type)
        
        print(f"\n[{snapshot.step_index:03d}] {time_str} {step_str}")
        print(f"    Run ID: {snapshot.run_id}")
        print(f"    Timestamp: {snapshot.timestamp}")
        
        # Show tools if any
        if snapshot.tools:
            print(f"    Tools: {len(snapshot.tools)} tool(s)")
            for i, tool in enumerate(snapshot.tools[:3]):  # Show first 3 tools
                tool_name = tool.get('function', {}).get('name', 'unknown') if isinstance(tool, dict) else str(tool)
                print(f"      {i+1}. {tool_name}")
            if len(snapshot.tools) > 3:
                print(f"      ... and {len(snapshot.tools) - 3} more")
        
        # Show message count
        print(f"    Messages: {len(snapshot.messages)}")
        
        # Show metadata highlights
        if snapshot.metadata:
            model = snapshot.metadata.get('model', 'unknown')
            print(f"    Model: {model}")
        
        # Show state info
        if snapshot.pre_state_hash:
            print(f"    Pre-state hash: {snapshot.pre_state_hash[:16]}...")
        if snapshot.post_state_hash:
            print(f"    Post-state hash: {snapshot.post_state_hash[:16]}...")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
        view_timeline(run_id)
        view_timeline_detailed(run_id)
    else:
        print("Usage: python timeline.py <run_id>")