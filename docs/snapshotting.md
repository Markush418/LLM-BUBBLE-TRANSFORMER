# LLM-SNAP Integration for LLM-BUBBLE

## Overview

This document explains how to use LLM-SNAP with LLM-BUBBLE to add snapshot capabilities to your experiments.
LLM-SNAP provides audit trails, debugging capabilities, and recovery options for agent runs.

## What is LLM-SNAP?

LLM-SNAP (LLM Snapshot) is a tool that captures the execution history of agent-LLM interactions, allowing you to:
- View a timeline of what the agent did at each step
- Rewind to previous steps to see what the agent knew at that point
- Recover lost work if an agent run is interrupted
- Debug unexpected behavior by examining the agent's state at each step

## Installation

LLM-SNAP is already integrated into LLM-BUBBLE. No additional installation is required beyond the standard LLM-BUBBLE setup.

## Usage

### Basic Usage

To run an LLM-BUBBLE experiment with snapshotting enabled:

```bash
python run_with_snapshot.py "Your experiment description" [experiment arguments]
```

Example:
```bash
python run_with_snapshot.py "Testing epsilon sweep with mock data" --mode mock --epsilon-values 0.001 0.01 0.1
```

### Running Without Snapshotting

To run the same experiment without snapshotting (for comparison):

```bash
python run_with_snapshot.py "Your experiment description" --no-snapshot [experiment arguments]
```

### Viewing the Timeline

After running an experiment with snapshotting, you can view the timeline of agent actions:

```bash
python -m llm_snap timeline <run-id>
```

Find the run ID in the output when you run the experiment (it looks like `exp-1234567890-abc123`).

Example output:
```
============================================================
LLM-SNAP Timeline - Run: exp-1234567890-abc123
============================================================
Total steps: 5

[000] 14:30:00 👁️ PERCEIVE | User asked: "Run epsilon sweep experiment"
[001] 14:30:02 💭 REASON | Configuring experiment with mode=mock
[002] 14:30:05 🎬 ACT | Setting up experiment parameters and loading embeddings...
[003] 14:30:08 👀 OBSERVE | Running experiment: python experiments/run_experiment.py --mode mock
[004] 14:30:12 🏁 FINAL | Experiment completed successfully

============================================================
```

### Detailed Timeline View

For more detailed information including tools used and metadata:

```bash
python -m llm_snap timeline <run-id> --detail
```

### Rewinding to Previous Steps

To see what the agent knew at a specific point in time:

```bash
python -m llm_snap rewind <run-id> <step-index>
```

Example:
```bash
python -m llm_snap rewind exp-1234567890-abc123 2
```

This will show you the agent's state at step 2, including what messages it had seen and what tools it had available.

### Interactive Rewind Session

For an interactive exploration of the agent's execution history:

```bash
python -m llm_snap rewind <run-id>
```

This will start an interactive session where you can:
- `list` - List all available steps
- `show <step>` - Show details of a specific step
- `rewind <step>` - Rewind to and show agent state at step
- `quit` - Exit the rewind session

## How It Works

When you run `run_with_snapshot.py`, the script:

1. Creates a unique run ID for your experiment
2. Initializes the LLM-SNAP interceptor
3. Takes snapshots at key points in the experiment execution:
   - **Perceive**: When the agent receives the user request
   - **Reason**: When the agent thinks about how to respond
   - **Act**: When the agent decides to take an action (like running the experiment)
   - **Observe**: When the agent observes the results of its action
   - **Final**: When the agent summarizes the outcome
4. Stores these snapshots in the `.sisyphus/snapshots/` directory
5. Runs your actual LLM-BUBBLE experiment
6. Provides you with the run ID for later inspection

## Storage Location

Snapshots are stored in:
```
.sisyphus/snapshots/<run-id>/
```

Each snapshot is saved as a JSON file containing:
- Step index and type
- Timestamp
- Messages exchanged
- Tools used
- Metadata (including model information)
- State hashes for integrity verification

## Examples

### Debugging a Failed Experiment

If your experiment fails unexpectedly:

1. Run it with snapshotting: `python run_with_snapshot.py "Debugging experiment" --mode mock`
2. Note the run ID from the output
3. View the timeline: `python -m llm_snap timeline <run-id>`
4. Identify where things went wrong
5. Rewind to the step before the failure: `python -m llm_snap rewind <run-id> <step-index>`
6. Examine what the agent knew at that point to understand why it made the decision it did

### Comparing Experiment Runs

To compare two different experimental configurations:

1. Run first configuration: `python run_with_snapshot.py "Config A" --mode mock --epsilon-values 0.001`
2. Run second configuration: `python run_with_snapshot.py "Config B" --mode mock --epsilon-values 0.01`
3. View timelines for both runs
4. Use rewind to compare agent states at equivalent steps

## Best Practices

### When to Use Snapshotting

- **Debugging**: When experiments behave unexpectedly
- **Learning**: To understand how the agent makes decisions
- **Audit Trail**: For reproducibility and compliance
- **Recovery**: If you suspect a run might be interrupted

### Performance Considerations

Snapshotting adds minimal overhead:
- Each snapshot adds ~10-20ms of processing time
- Storage usage is minimal (JSON files, typically <1KB per snapshot)
- For typical experiments with 5-10 steps, total overhead is <200ms

### When Not to Use Snapshotting

- **Production runs**: If you're running many experiments and don't need debugging
- **Resource-constrained environments**: Though overhead is low, every bit counts
- **Sensitive data**: Though snapshots don't include actual model weights or large data arrays, review what's being captured

## Troubleshooting

### No Snapshots Being Created

If you don't see snapshots being created:

1. Verify you're using `run_with_snapshot.py` (not running `run_experiment.py` directly)
2. Check that you didn't use the `--no-snapshot` flag
3. Look for error messages in the output
4. Verify the `.sisyphus/snapshots/` directory exists and is writable

### Timeline Shows Gaps

If the timeline seems to be missing steps:

1. This is normal - the wrapper takes snapshots at key points, not every single interaction
2. For more detailed snapshots, you would need to modify the experiment code to call the interceptor more frequently
3. The current implementation provides a good balance between detail and performance

### Rewind Doesn't Work as Expected

Remember that rewinding shows you what the agent knew at that point, not necessarily the exact internal state of complex objects.
The snapshot captures:
- Messages exchanged
- Tools that were available/used
- Metadata about the step
- State hashes for integrity

It does not capture:
- Internal variables in Python functions
- The exact state of machine learning models
- Temporary variables that weren't part of the agent-LLM exchange

## Integration Details

For those interested in how LLM-SNAP is integrated:

### Wrapper Script

The `run_with_snapshot.py` script is the main entry point. It:
- Handles argument parsing and forwarding to the underlying experiment
- Manages the snapshot lifecycle (start/stop)
- Provides user-friendly output and instructions

### Python Module

The core snapshot functionality is in the `llm_snap` package:
- `llm_snap.core.interceptor`: The SnapshotInterceptor class that captures agent-LLM exchanges
- `llm_snap.storage`: Storage backend for persisting snapshots
- `llm_snap.timeline`: Timeline viewing functionality
- `llm_snap.rewind`: Rewind/restore capabilities

### Extending the Integration

If you want to capture snapshots at more granular points within your own experiment code:

```python
from llm_snap.core.interceptor import SnapshotInterceptor

# Initialize interceptor (typically once per experiment run)
interceptor = SnapshotInterceptor()

# At points where you want to capture state
params = {
    "messages": [...],  # Current conversation
    "tools": [...],     # Available tools
    "model": "your-model-name"
}
interceptor.intercept(run_id, step_index, params)
```

## FAQ

**Q: Does snapshotting slow down my experiments significantly?**  
A: No. The overhead is minimal - typically less than 200ms for a typical experiment run.

**Q: Can I use snapshotting with real experiments (not just mock)?**  
A: Yes. Snapshotting works with both mock and real experiment modes.

**Q: Where are snapshots stored?**  
A: In the `.sisyphus/snapshots/` directory by default, or you can specify a different location with `--snapshot-storage`.

**Q: Are snapshots safe to share?**  
A: Snapshots contain the messages exchanged with the LLM and tool usage, but do not include model weights or large data arrays. Review the content before sharing if confidentiality is a concern.

**Q: How long are snapshots kept?**  
A: Indefinitely, until you manually delete them. They are stored as regular files in your project directory.

**Q: Can I delete old snapshots?**  
A: Yes. Simply delete the corresponding directory in `.sisyphus/snapshots/<run-id>/`.

## Support

For questions or issues with LLM-SNAP integration:
- Check the troubleshooting section above
- Review the evidence files in `.sisyphus/evidence/` for implementation details
- Consult the LLM-SNAP skill documentation at `C:\Users\negocio\.agents\skills\llm-snap\SKILL.md`

Happy experimenting! 🚀