from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def bundled_task_dir() -> Path:
    """Return the path to the bundled policy-sentiment example task."""
    task_dir = Path(__file__).resolve().parent.parent / "codebook_lab" / "tasks" / "policy-sentiment"
    assert task_dir.exists(), f"Bundled task directory not found: {task_dir}"
    return task_dir
