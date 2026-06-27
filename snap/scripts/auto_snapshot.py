#!/usr/bin/env python3
"""
Auto-Snapshot Watcher for LLM-BUBBLE
===================================
Monitors file changes in the project and automatically captures snapshots.
Uses watchdog for file system events.

Usage:
    python auto_snapshot.py                    # Watch experiments/ and scripts/
    python auto_snapshot.py --stop            # Stop any running watcher
    python auto_snapshot.py --status        # Check watcher status
"""

import argparse
import os
import sys
import time
import signal
import threading
from pathlib import Path

# Add parent to path
_PARENT_DIR = str(Path(__file__).parent.parent / 'python')
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Try to import watchdog, install if needed
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Installing watchdog...")
    os.system(f"{sys.executable} -m pip install watchdog -q")
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler


class ChangeHandler(FileSystemEventHandler):
    """Handle file system events and capture snapshots."""
    
    def __init__(self, watch_paths, snapshot_func=None):
        self.watch_paths = watch_paths
        self.snapshot_func = snapshot_func
        self.last_snapshot = {}
        self.cooldown = 2.0  # seconds between snapshots
        
    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_change(event.src_path, "modified")
        
    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_change(event.src_path, "created")
        
    def on_deleted(self, event):
        if event.is_directory:
            return
        self._handle_change(event.src_path, "deleted")
    
    def _handle_change(self, filepath, event_type):
        """Process a file change event."""
        # Check if in watched paths
        filepath = Path(filepath)
        in_watch = any(str(filepath).startswith(str(p)) for p in self.watch_paths)
        
        if not in_watch:
            return
            
        # Skip non-code files
        if filepath.suffix not in ['.py', '.md', '.yaml', '.json', '.toml']:
            return
            
        # Skip __pycache__ and other temp files
        if '__pycache__' in str(filepath) or filepath.name.startswith('.'):
            return
            
        # Cooldown check
        now = time.time()
        last = self.last_snapshot.get(filepath, 0)
        if now - last < self.cooldown:
            return
            
        self.last_snapshot[filepath] = now
        
        # Print colored output
        icon = {"modified": "📝", "created": "✨", "deleted": "🗑️"}[event_type]
        print(f"{icon} {event_type}: {filepath.name}")
        
        # Capture snapshot if function provided
        if self.snapshot_func:
            self.snapshot_func(str(filepath), event_type)


def capture_snapshot(filepath, event_type):
    """Capture a snapshot of the change."""
    try:
        from llm_snap import SnapshotInterceptor
        import uuid
        
        # Generate run ID if not exists
        run_id = f"auto-{int(time.time())}-{str(uuid.uuid4())[:8]}"
        
        # Create interceptor
        storage_path = ".sisyphus/snapshots"
        interceptor = SnapshotInterceptor(storage_path)
        
        # Capture the change event
        interceptor.intercept(run_id, 0, {
            'messages': [
                {'role': 'system', 'content': f'File {event_type}'},
                {'role': 'user', 'content': str(filepath)}
            ],
            'tools': [],
            'metadata': {
                'event_type': event_type,
                'filepath': str(filepath),
                'timestamp': time.time()
            }
        })
        
        # Also save to a persistent auto-snapshots directory
        auto_dir = Path(".sisyphus/auto-snapshots")
        auto_dir.mkdir(exist_ok=True)
        
        import json
        snapshot_file = auto_dir / f"{run_id}.json"
        with open(snapshot_file, 'w') as f:
            json.dump({
                'run_id': run_id,
                'filepath': str(filepath),
                'event_type': event_type,
                'timestamp': time.time()
            }, f, indent=2)
            
        print(f"   ✓ Snapshot saved: {run_id}")
        
    except Exception as e:
        print(f"   ✗ Snapshot failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Auto-Snapshot Watcher for LLM-BUBBLE")
    parser.add_argument("--stop", action="store_true", help="Stop running watcher")
    parser.add_argument("--status", action="store_true", help="Check watcher status")
    parser.add_argument("--watch", nargs="+", default=["experiments/", "scripts/", "models/"],
                       help="Paths to watch")
    args = parser.parse_args()
    
    # Check for existing observer file
    observer_file = Path(".sisyphus/watcher.pid")
    
    if args.stop:
        if observer_file.exists():
            with open(observer_file) as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print("✓ Watcher stopped")
            except ProcessLookupError:
                print("Watcher process not found, cleaning up...")
            observer_file.unlink()
        else:
            print("No watcher running")
        return
        
    if args.status:
        if observer_file.exists():
            with open(observer_file) as f:
                pid = f.read().strip()
            print(f"Watcher running (PID: {pid})")
        else:
            print("Watcher not running")
        return
    
    # Start watcher
    print("=" * 60)
    print("  LLM-BUBBLE Auto-Snapshot Watcher")
    print("=" * 60)
    print(f"Watching: {', '.join(args.watch)}")
    print("Press Ctrl+C to stop")
    print()
    
    # Verify paths exist
    watch_paths = [Path(p) for p in args.watch]
    for p in watch_paths:
        if not p.exists():
            print(f"Warning: {p} does not exist, creating...")
            p.mkdir(parents=True, exist_ok=True)
    
    # Create event handler
    handler = ChangeHandler(watch_paths, capture_snapshot)
    
    # Create and start observer
    observer = Observer()
    for path in watch_paths:
        observer.schedule(handler, str(path), recursive=True)
    
    observer.start()
    
    # Save PID
    with open(observer_file, 'w') as f:
        f.write(str(os.getpid()))
    
    print("🚀 Watcher started!")
    print()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping watcher...")
        observer.stop()
        observer_file.unlink()
        print("✓ Watcher stopped")


if __name__ == "__main__":
    main()
