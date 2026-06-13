"""
LLM-SNAP Integration for LLM-BUBBLE
====================================

Provides snapshot capabilities for LLM-BUBBLE experiments to enable
audit trails, debugging, and recovery of agent runs.

This module wraps the LLM-SNAP Node.js tool to provide snapshot
functionality without modifying core experiment code.
"""

__version__ = "0.1.0"
__author__ = "LLM-BUBBLE Team"

from .core.interceptor import SnapshotInterceptor
from .storage import SnapshotStorage

__all__ = [
    "SnapshotInterceptor",
    "SnapshotStorage",
]