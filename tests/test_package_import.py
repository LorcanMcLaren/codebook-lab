from __future__ import annotations

import subprocess
import sys


def test_top_level_import_stays_lightweight(tmp_path):
    script = """
from pathlib import Path
import sys

from codebook_lab import copy_example_task, list_example_tasks

target = Path(sys.argv[1])
tasks = list_example_tasks()
assert "policy-sentiment" in tasks
copied = copy_example_task("policy-sentiment", target, overwrite=True)
assert copied.exists()
assert copied.name == "policy-sentiment"
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
