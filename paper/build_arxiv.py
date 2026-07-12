#!/usr/bin/env python3
"""
arXiv Package Builder
=====================
Creates the submission package for arXiv.
"""
import os
import shutil
import subprocess
from pathlib import Path

def build_arxiv_package():
    """Build the arXiv submission package."""
    root = Path(__file__).parent.parent
    paper_dir = root / "paper"
    arxiv_dir = root / "arxiv_package"
    
    # Clean
    if arxiv_dir.exists():
        shutil.rmtree(arxiv_dir)
    arxiv_dir.mkdir()
    
    # Copy paper files
    for f in paper_dir.glob("*.tex"):
        shutil.copy(f, arxiv_dir)
    for f in paper_dir.glob("*.bib"):
        shutil.copy(f, arxiv_dir)
    for f in paper_dir.glob("*.png"):
        shutil.copy(f, arxiv_dir)
    for f in paper_dir.glob("*.pdf"):
        shutil.copy(f, arxiv_dir)
    
    # Create submission.tar.gz
    subprocess.run(
        ["tar", "-czf", "submission.tar.gz", "-C", str(arxiv_dir), "."],
        cwd=str(root),
        check=True
    )
    
    print(f"arXiv package built: {root / 'submission.tar.gz'}")
    print(f"Contents: {list(arxiv_dir.iterdir())}")

if __name__ == "__main__":
    build_arxiv_package()
