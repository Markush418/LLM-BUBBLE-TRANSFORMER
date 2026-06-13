"""
Rewind/Restore Capability for LLM-SNAP
======================================

Provides ability to rewind agent state to previous steps for debugging and analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from .storage import SnapshotStorage
from .storage.snapshot_models import Snapshot


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def view_snapshot_details(snapshot: Snapshot) -> None:
    """Display detailed information about a snapshot."""
    print(f"Step Index: {snapshot.step_index}")
    print(f"Step Type: {snapshot.step_type.upper()}")
    print(f"Timestamp: {format_timestamp(snapshot.timestamp)}")
    print(f"Run ID: {snapshot.run_id}")
    
    # Show message count and preview
    print(f"Messages: {len(snapshot.messages)}")
    if snapshot.messages:
        last_msg = snapshot.messages[-1]
        if last_msg.get("content"):
            content = last_msg["content"]
            if len(content) > 100:
                content = content[:97] + "..."
            print(f"Last Message: {content}")
    
    # Show tools if any
    if snapshot.tools:
        print(f"Tools: {len(snapshot.tools)}")
        for i, tool in enumerate(snapshot.tools[:3]):  # Show first 3 tools
            if isinstance(tool, dict):
                tool_name = tool.get('function', {}).get('name', 'unknown')
                print(f"  {i+1}. {tool_name}")
            else:
                print(f"  {i+1}. {str(tool)}")
        if len(snapshot.tools) > 3:
            print(f"  ... and {len(snapshot.tools) - 3} more")
    
    # Show metadata
    if snapshot.metadata:
        print("Metadata:")
        for key, value in snapshot.metadata.items():
            if key not in ['pre_state_hash', 'post_state_hash']:  # Skip hashes for brevity
                print(f"  {key}: {value}")
    
    # Show state hashes (truncated)
    if snapshot.pre_state_hash:
        print(f"Pre-state Hash: {snapshot.pre_state_hash[:32]}...")
    if snapshot.post_state_hash:
        print(f"Post-state Hash: {snapshot.post_state_hash[:32]}...")


def rewind_to_step(run_id: str, step_index: int, 
                   storage_path: str = ".sisyphus/snapshots") -> Optional[Dict[str, Any]]:
    """
    Rewind agent state to a specific step.
    
    Args:
        run_id: Identifier of the run to rewind
        step_index: Step index to rewind to
        storage_path: Path to snapshot storage
        
    Returns:
        Dictionary representing the agent state at that step, or None if not found
    """
    storage = SnapshotStorage(storage_path)
    snapshots = storage.get_snapshots_for_run(run_id)
    
    if not snapshots:
        print(f"No snapshots found for run: {run_id}")
        return None
    
    # Find snapshot at or before the requested step
    target_snapshot = None
    for snapshot in snapshots:
        if snapshot.step_index <= step_index:
            target_snapshot = snapshot
        else:
            break
    
    if target_snapshot is None:
        print(f"No snapshot found at or before step {step_index}")
        return None
    
    # In a full implementation, this would reconstruct the actual agent state
    # For this demonstration, we return the snapshot data which represents
    # what the agent knew at that point
    print(f"Rewound to step {target_snapshot.step_index} (requested step {step_index})")
    print(f"Step type: {target_snapshot.step_type}")
    print(f"Timestamp: {format_timestamp(target_snapshot.timestamp)}")
    
    # Return a representation of the agent state at this point
    agent_state = {
        "run_id": target_snapshot.run_id,
        "step_index": target_snapshot.step_index,
        "step_type": target_snapshot.step_type,
        "timestamp": target_snapshot.timestamp,
        "messages": target_snapshot.messages.copy(),
        "tools": target_snapshot.tools.copy(),
        "metadata": target_snapshot.metadata.copy(),
        "rewind_info": {
            "requested_step": step_index,
            "actual_step": target_snapshot.step_index,
            "rewound_by": step_index - target_snapshot.step_index if target_snapshot.step_index < step_index else 0
        }
    }
    
    return agent_state


def list_available_steps(run_id: str, 
                        storage_path: str = ".sisyphus/snapshots") -> List[int]:
    """
    List all available step indices for a run.
    
    Args:
        run_id: Identifier of the run
        storage_path: Path to snapshot storage
        
    Returns:
        List of step indices available for rewinding
    """
    storage = SnapshotStorage(storage_path)
    snapshots = storage.get_snapshots_for_run(run_id)
    
    if not snapshots:
        return []
    
    step_indices = [snapshot.step_index for snapshot in snapshots]
    return sorted(step_indices)


def interactive_rewind(run_id: str, 
                      storage_path: str = ".sisyphus/snapshots") -> None:
    """
    Interactive rewind session for exploring agent execution history.
    
    Args:
        run_id: Identifier of the run to explore
        storage_path: Path to snapshot storage
    """
    storage = SnapshotStorage(storage_path)
    snapshots = storage.get_snapshots_for_run(run_id)
    
    if not snapshots:
        print(f"No snapshots found for run: {run_id}")
        return
    
    print(f"\n{'='*60}")
    print(f"LLM-SNAP Interactive Rewind - Run: {run_id}")
    print(f"{'='*60}")
    print(f"Available steps: {[s.step_index for s in snapshots]}")
    print("Commands: list, show <step>, rewind <step>, quit, help")
    print(f"{'='*60}\n")
    
    while True:
        try:
            command = input("rewind> ").strip().lower()
            
            if command == "quit" or command == "exit":
                print("Exiting rewind session.")
                break
                
            elif command == "help":
                print("Available commands:")
                print("  list - List all available steps")
                print("  show <step> - Show details of a specific step")
                print("  rewind <step> - Rewind to and show agent state at step")
                print("  quit - Exit the rewind session")
                print("  help - Show this help")
                
            elif command == "list":
                step_indices = [s.step_index for s in snapshots]
                print(f"Available steps: {step_indices}")
                
            elif command.startswith("show "):
                try:
                    step_index = int(command.split()[1])
                    snapshot = next((s for s in snapshots if s.step_index == step_index), None)
                    if snapshot:
                        print(f"\nDetails for step {step_index}:")
                        view_snapshot_details(snapshot)
                    else:
                        print(f"No snapshot found at step {step_index}")
                except (IndexError, ValueError):
                    print("Please specify a valid step number: show <step>")
                    
            elif command.startswith("rewind "):
                try:
                    step_index = int(command.split()[1])
                    state = rewind_to_step(run_id, step_index, storage_path)
                    if state:
                        print(f"\nAgent state at step {state['step_index']}:")
                        print(f"  Run ID: {state['run_id']}")
                        print(f"  Step Type: {state['step_type']}")
                        print(f"  Timestamp: {format_timestamp(state['timestamp'])}")
                        print(f"  Message Count: {len(state['messages'])}")
                        print(f"  Tool Count: {len(state['tools'])}")
                        if state['rewind_info']['rewound_by'] > 0:
                            print(f"  Note: Rewound by {state['rewind_info']['rewound_by']} steps "
                                  f"(exact step {state['rewind_info']['actual_step']} used)")
                except (IndexError, ValueError):
                    print("Please specify a valid step number: rewind <step>")
                    
            elif command == "":
                continue
                
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting rewind session.")
            break
        except EOFError:
            print("\nExiting rewind session.")
            break
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
        if len(sys.argv) > 2:
            try:
                step_index = int(sys.argv[2])
                state = rewind_to_step(run_id, step_index)
                if state:
                    print(f"Rewound to step {state['step_index']}")
                    view_snapshot_details(Snapshot(**{k: v for k, v in state.items() 
                                                    if k not in ['rewind_info']}))
            except ValueError:
                print("Step index must be an integer")
        else:
            interactive_rewind(run_id)
    else:
        print("Usage: python rewind.py <run_id> [step_index]")