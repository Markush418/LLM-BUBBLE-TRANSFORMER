#!/usr/bin/env python3
"""
LLM-SNAP Web Dashboard Backend
===============================
Flask server providing API and UI for LLM-SNAP management.

Usage:
    python llm_snap_server.py
    # Opens at http://localhost:5000
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from flask import Flask, send_file, jsonify, request

app = Flask(__name__)

# Storage paths
_PARENT_DIR = Path(__file__).parent
_PYTHON_DIR = _PARENT_DIR.parent / 'python'
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

SNAPSHOTS_DIR = ".sisyphus/snapshots"
AUTO_DIR = ".sisyphus/auto-snapshots"


def get_runs():
    """Get all runs with their snapshot counts."""
    runs = []
    snapshots_path = Path(SNAPSHOTS_DIR)
    
    if snapshots_path.exists():
        for run_dir in sorted(snapshots_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if run_dir.is_dir():
                run_id = run_dir.name
                # Count snapshots in run
                snaps = list(run_dir.glob("*.json"))
                runs.append({
                    'id': run_id,
                    'timestamp': datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                    'stepCount': len(snaps),
                    'status': 'active' if len(snaps) < 3 else 'completed'
                })
    
    return runs


def get_auto_captures():
    """Get auto-captured snapshots count."""
    auto_path = Path(AUTO_DIR)
    if auto_path.exists():
        return len(list(auto_path.glob("*.json")))
    return 0


def get_last_activity():
    """Get the most recent activity timestamp."""
    runs = get_runs()
    if runs:
        return runs[0]['timestamp']
    
    # Check auto captures
    auto_path = Path(AUTO_DIR)
    if auto_path.exists():
        auto_files = sorted(auto_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if auto_files:
            return datetime.fromtimestamp(auto_files[0].stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    
    return None


def load_snapshot(file_path):
    """Load a snapshot from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None


@app.route('/')
def index():
    """Serve the main UI."""
    return send_file(__file__.replace('_server.py', '_ui.html'))


@app.route('/api/snapshots')
def api_snapshots():
    """Get all snapshots data."""
    runs = get_runs()
    total_snapshots = sum(r['stepCount'] for r in runs)
    auto_captures = get_auto_captures()
    last_activity = get_last_activity()
    
    return jsonify({
        'runs': runs,
        'totalSnapshots': total_snapshots,
        'autoCaptures': auto_captures,
        'lastActivity': last_activity
    })


@app.route('/api/timeline/<run_id>')
def api_timeline(run_id):
    """Get timeline for a specific run."""
    run_path = Path(SNAPSHOTS_DIR) / run_id
    
    if not run_path.exists():
        return jsonify([])
    
    snapshots = []
    for snap_file in sorted(run_path.glob("*.json")):
        data = load_snapshot(snap_file)
        if data:
            step_idx = data.get('step_index', 0)
            step_type = data.get('step_type', 'unknown')
            timestamp = data.get('timestamp', '')
            messages = data.get('messages', [])
            
            snapshots.append({
                'step_index': step_idx,
                'step_type': step_type,
                'timestamp': timestamp,
                'messages': messages
            })
    
    # Sort by step index
    snapshots.sort(key=lambda x: x['step_index'])
    
    return jsonify(snapshots)


@app.route('/api/rewind/<run_id>/<int:step>')
def api_rewind(run_id, step):
    """Rewind to a specific step."""
    try:
        # Import and use rewind module
        from llm_snap.rewind import rewind_to_step
        state = rewind_to_step(run_id, step, SNAPSHOTS_DIR)
        return jsonify({'success': True, 'state': state})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/watcher/start', methods=['POST'])
def start_watcher():
    """Start the file watcher."""
    try:
        subprocess.Popen([sys.executable, str(_parent_dir / 'auto_snapshot.py')], 
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/watcher/stop', methods=['POST'])
def stop_watcher():
    """Stop the file watcher."""
    try:
        subprocess.run([sys.executable, str(_parent_dir / 'auto_snapshot.py'), '--stop'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/watcher/status')
def watcher_status():
    """Get watcher status."""
    try:
        result = subprocess.run([sys.executable, str(_parent_dir / 'auto_snapshot.py'), '--status'],
                         capture_output=True, text=True)
        return jsonify({'success': True, 'status': result.stdout.strip()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/terminal/open', methods=['POST'])
def open_terminal():
    """Open PowerShell terminal."""
    try:
        subprocess.Popen(['start', 'powershell', '-WorkingDirectory', str(_parent_dir)],
                      shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/experiment/run')
def run_experiment():
    """Trigger an experiment run."""
    try:
        subprocess.Popen([sys.executable, '-m', 'experiments.run_experiment', '--snapshot', '--mode', 'mock'],
                     cwd=str(_parent_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return jsonify({'success': True, 'message': 'Experiment started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("=" * 60)
    print("  LLM-SNAP Dashboard")
    print("=" * 60)
    print()
    print("  Opening: http://localhost:5000")
    print()
    
    # Ensure directories exist
    Path(SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(AUTO_DIR).mkdir(parents=True, exist_ok=True)
    
    app.run(debug=True, port=5000)