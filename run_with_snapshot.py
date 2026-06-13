#!/usr/bin/env python3
"""
Wrapper Script for Running LLM-BUBBLE Experiments with Snapshotting
==================================================================

This script provides an easy way to run LLM-BUBBLE experiments with
LLM-SNAP snapshotting enabled for audit trails, debugging, and recovery.

Usage:
    python run_with_snapshot.py "Experiment description" [experiment arguments]

Example:
    python run_with_snapshot.py "Testing epsilon sweep" --mode mock --epsilon-values 0.001 0.01 0.1
"""

import sys
import os
import subprocess
import argparse
import time
import uuid
from pathlib import Path

# Add the LLM-BUBBLE directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_snap.core.interceptor import SnapshotInterceptor


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-BUBBLE experiments with LLM-SNAP snapshotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_with_snapshot.py "Mock experiment test" --mode mock
  python run_with_snapshot.py "Epsilon sweep analysis" --mode mock --epsilon-values 0.001 0.01 0.1
  python run_with_snapshot.py "Real experiment" --mode real --skip-visualization
        """
    )
    
    parser.add_argument(
        "description",
        help="Description of the experiment for snapshot tracking"
    )
    
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Run experiment without snapshotting (for comparison)"
    )
    
    parser.add_argument(
        "--snapshot-storage",
        default=".sisyphus/snapshots",
        help="Path to snapshot storage directory (default: .sisyphus/snapshots)"
    )
    
    # Add all the standard run_experiment.py arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "mock", "real", "tension", "layer-selection"],
        help="Experiment mode: auto (detect), mock, real, tension (dual-head sweep), layer-selection"
    )
    parser.add_argument(
        "--d-model", type=int, default=None, help="Model hidden dimension (mock only)"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Number of attention heads (mock only)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=24,
        help="Number of layers to simulate (mock only)"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--seq-len", type=int, default=64, help="Sequence length (mock only)"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="embeddings",
        help="Directory for embeddings"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory for results"
    )
    parser.add_argument(
        "--skip-generation", action="store_true", help="Skip embedding generation"
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--epsilon-values",
        type=float,
        nargs="+",
        default=None,
        help="Custom epsilon values"
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="Target layers for analysis"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with reduced parameters"
    )
    
    args = parser.parse_args()
    
    # All code below executes when script is run directly
    # Generate a unique run ID
    run_id = f"exp-{int(time.time())}-{str(uuid.uuid4())[:8]}"
    
    print(f"LLM-BUBBLE Experiment with LLM-SNAP Snapshotting")
    print(f"{'='*60}")
    print(f"Experiment: {args.description}")
    print(f"Run ID: {run_id}")
    print(f"Mode: {args.mode}")
    print(f"Snapshot Storage: {args.snapshot_storage}")
    print(f"{'='*60}\n")
    
    if args.no_snapshot:
        # Run experiment without snapshotting
        print("Running experiment WITHOUT snapshotting...")
        return run_experiment_directly(args)
    else:
        # Run experiment with snapshotting
        print("Running experiment WITH snapshotting...")
        return run_experiment_with_snapshot(args, run_id, args.snapshot_storage)


def run_experiment_directly(args):
    """Run the experiment directly without snapshotting."""
    # Build command to run experiments/run_experiment.py
    cmd = [sys.executable, "experiments/run_experiment.py"]
    
    # Add all arguments
    if args.mode != "auto":
        cmd.extend(["--mode", args.mode])
    if args.d_model is not None:
        cmd.extend(["--d-model", str(args.d_model)])
    if args.num_heads is not None:
        cmd.extend(["--num-heads", str(args.num_heads)])
    if args.num_layers != 24:
        cmd.extend(["--num-layers", str(args.num_layers)])
    if args.batch_size != 4:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.seq_len != 64:
        cmd.extend(["--seq-len", str(args.seq_len)])
    if args.embeddings_dir != "embeddings":
        cmd.extend(["--embeddings-dir", args.embeddings_dir])
    if args.output_dir != "results":
        cmd.extend(["--output-dir", args.output_dir])
    if args.skip_generation:
        cmd.append("--skip-generation")
    if args.skip_visualization:
        cmd.append("--skip-visualization")
    if args.epsilon_values is not None:
        cmd.extend(["--epsilon-values"] + [str(v) for v in args.epsilon_values])
    if args.target_layers is not None:
        cmd.extend(["--target-layers"] + [str(v) for v in args.target_layers])
    if args.quick_test:
        cmd.append("--quick-test")
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the experiment
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    print("-" * 60)
    if result.returncode == 0:
        print("Experiment completed successfully!")
    else:
        print(f"Experiment failed with exit code {result.returncode}")
    
    return result.returncode


def run_experiment_with_snapshot(args, run_id, storage_path):
    """Run the experiment with snapshotting enabled."""
    # Import here to avoid circular imports
    from llm_snap.core.interceptor import SnapshotInterceptor
    
    # Initialize snapshot interceptor
    interceptor = SnapshotInterceptor(storage_path)
    
print(f"Snapshotting enabled - Run ID: {run_id}")
print(f"Storage: {storage_path}")
    print("-" * 60)
    
    # We'll simulate the interception by wrapping the experiment execution
    # In a real implementation, we would intercept actual LLM calls
    # For this wrapper, we'll create snapshots at key points in the experiment
    
    try:
        # Step 1: Perceive - Experiment configuration
        perceive_params = {
            "messages": [
                {"role": "user", "content": f"Run experiment: {args.description}"},
                {"role": "assistant", "content": f"Configuring experiment with mode={args.mode}"}
            ],
            "tools": [],
            "model": "experiment-configurator"
        }
        interceptor.intercept(run_id, 0, perceive_params)
        
        # Step 2: Reason - Setting up experiment
        reason_params = {
            "messages": [
                {"role": "user", "content": f"Run experiment: {args.description}"},
                {"role": "assistant", "content": f"Configuring experiment with mode={args.mode}"},
                {"role": "assistant", "content": "Setting up experiment parameters and loading embeddings..."}
            ],
            "tools": [],
            "model": "experiment-configurator"
        }
        interceptor.intercept(run_id, 1, reason_params)
        
        # Step 3: Act - Running experiment
        # Build command to run experiments/run_experiment.py
        cmd = [sys.executable, "experiments/run_experiment.py"]
        
        # Add all arguments
        if args.mode != "auto":
            cmd.extend(["--mode", args.mode])
        if args.d_model is not None:
            cmd.extend(["--d-model", str(args.d_model)])
        if args.num_heads is not None:
            cmd.extend(["--num-heads", str(args.num_heads)])
        if args.num_layers != 24:
            cmd.extend(["--num-layers", str(args.num_layers)])
        if args.batch_size != 4:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.seq_len != 64:
            cmd.extend(["--seq-len", str(args.seq_len)])
        if args.embeddings_dir != "embeddings":
            cmd.extend(["--embeddings-dir", args.embeddings_dir])
        if args.output_dir != "results":
            cmd.extend(["--output-dir", args.output_dir])
        if args.skip_generation:
            cmd.append("--skip-generation")
        if args.skip_visualization:
            cmd.append("--skip-visualization")
        if args.epsilon_values is not None:
            cmd.extend(["--epsilon-values"] + [str(v) for v in args.epsilon_values])
        if args.target_layers is not None:
            cmd.extend(["--target-layers"] + [str(v) for v in args.target_layers])
        if args.quick_test:
            cmd.append("--quick-test")
        
        act_params = {
            "messages": [
                {"role": "user", "content": f"Run experiment: {args.description}"},
                {"role": "assistant", "content": f"Configuring experiment with mode={args.mode}"},
                {"role": "assistant", "content": "Setting up experiment parameters and loading embeddings..."},
                {"role": "assistant", "content": f"Running experiment: {' '.join(cmd[1:])}"}
            ],
            "tools": [
                {"type": "function", "function": {
                    "name": "run_experiment",
                    "arguments": {
                        "command": " ".join(cmd),
                        "working_directory": os.path.dirname(os.path.abspath(__file__))
                    }
                }}
            ],
            "model": "experiment-runner"
        }
        interceptor.intercept(run_id, 2, act_params)
        
        # Actually run the experiment
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)
        
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Step 4: Observe - Experiment completion
        observe_params = {
            "messages": [
                {"role": "user", "content": f"Run experiment: {args.description}"},
                {"role": "assistant", "content": f"Configuring experiment with mode={args.mode}"},
                {"role": "assistant", "content": "Setting up experiment parameters and loading embeddings..."},
                {"role": "assistant", "content": f"Running experiment: {' '.join(cmd[1:])}"},
                {"role": "assistant", "content": f"Experiment completed with exit code {result.returncode}"}
            ],
            "tools": [],
            "model": "experiment-observer"
        }
        interceptor.intercept(run_id, 3, observe_params)
        
        # Step 5: Final - Summary
        final_params = {
            "messages": [
                {"role": "user", "content": f"Run experiment: {args.description}"},
                {"role": "assistant", "content": f"Configuring experiment with mode={args.mode}"},
                {"role": "assistant", "content": "Setting up experiment parameters and loading embeddings..."},
                {"role": "assistant", "content": f"Running experiment: {' '.join(cmd[1:])}"},
                {"role": "assistant", "content": f"Experiment completed with exit code {result.returncode}"},
                {"role": "assistant", "content": f"Snapshots stored in: {storage_path}"}
            ],
            "tools": [],
            "model": "experiment-summarizer"
        }
        interceptor.intercept(run_id, 4, final_params)
        
        print("-" * 60)
        if result.returncode == 0:
            print("✅ Experiment completed successfully!")
            print(f"📸 Snapshots stored in: {storage_path}")
            print(f"🔍 To view timeline: python -m llm_snap timeline {run_id}")
            print(f"⏪ To rewind to step: python -m llm_snap rewind {run_id} <step>")
        else:
            print(f"❌ Experiment failed with exit code {result.returncode}")
            print(f"📸 Snapshots stored in: {storage_path} (for debugging)")
            print(f"🔍 To view timeline: python -m llm_snap timeline {run_id}")
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running experiment with snapshotting: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())