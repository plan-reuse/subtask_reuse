"""File operations utilities."""

import os
from pathlib import Path
from datetime import datetime
from typing import Tuple


def create_output_directory(output_base: Path) -> Tuple[Path, Path]:
    """Create a timestamped output directory and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"run_{timestamp}"
    subtask_problem_dir = output_dir / "subtask_problem"
    
    # Create the directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    subtask_problem_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    print(f"Created subtask problem directory: {subtask_problem_dir}")
    
    return output_dir, subtask_problem_dir


def read_file(file_path: Path) -> str:
    """Read and return file contents."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    return content


def write_file(file_path: Path, content: str) -> None:
    """Write content to file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(content)


def copy_file(source: Path, destination: Path) -> None:
    """Copy file from source to destination."""
    content = read_file(source)
    write_file(destination, content)