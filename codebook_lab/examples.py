from __future__ import annotations

from pathlib import Path
import shutil


def _examples_root() -> Path:
    """Return the directory that stores bundled example tasks."""
    return Path(__file__).resolve().parent / "tasks"


def list_example_tasks() -> list[str]:
    """List bundled example task names shipped with the package.

    Returns:
        Sorted list of task directory names available under the package's
        bundled ``tasks`` data folder.
    """
    root = _examples_root()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def get_example_task_dir(task_name: str) -> Path:
    """Return the filesystem path to a bundled example task.

    Args:
        task_name: Bundled task directory name such as ``"policy-sentiment"``.

    Returns:
        Filesystem path to the bundled example task directory.
    """
    task_dir = _examples_root() / task_name
    if not task_dir.exists():
        available = ", ".join(list_example_tasks())
        raise FileNotFoundError(
            f"Bundled example task '{task_name}' was not found. "
            f"Available example tasks: {available or 'none'}."
        )
    return task_dir


def get_example_task_files(task_name: str) -> dict[str, Path]:
    """Return the standard file paths for a bundled example task.

    Args:
        task_name: Bundled task directory name.

    Returns:
        Dictionary containing ``task_dir``, ``codebook_path``, and
        ``ground_truth_csv`` paths.
    """
    task_dir = get_example_task_dir(task_name)
    return {
        "task_dir": task_dir,
        "codebook_path": task_dir / "codebook.json",
        "ground_truth_csv": task_dir / "ground-truth.csv",
    }


def copy_example_task(task_name: str, destination: str | Path, overwrite: bool = False) -> Path:
    """Copy a bundled example task to a user-controlled directory.

    Args:
        task_name: Bundled task directory name.
        destination: Directory where the example task folder should be copied.
        overwrite: If ``True``, replace an existing destination task folder.

    Returns:
        Path to the copied task directory inside ``destination``.
    """
    source_dir = get_example_task_dir(task_name)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target_dir = destination / task_name

    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination task directory already exists: {target_dir}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)
    return target_dir
