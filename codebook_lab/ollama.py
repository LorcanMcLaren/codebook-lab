from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib import error as urllib_error
from urllib import request as urllib_request


def get_ollama_base_url() -> str:
    """Return the Ollama base URL used for connectivity checks.

    Returns:
        Base URL string, defaulting to ``http://127.0.0.1:11434`` when
        ``OLLAMA_HOST`` is not set.
    """
    base_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


def _can_auto_start_local_ollama(base_url: str) -> bool:
    """Return whether CodeBook Lab can reasonably auto-start Ollama locally."""
    return base_url in {
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://0.0.0.0:11434",
    }


def ensure_ollama_available(
    timeout: float = 2.0,
    start_if_needed: bool = False,
    startup_timeout: float = 10.0,
) -> str:
    """Check that the Ollama server is reachable, optionally starting it locally.

    Args:
        timeout: Timeout in seconds for each connectivity probe.
        start_if_needed: If ``True``, try to start ``ollama serve`` when the
            default local server is not reachable.
        startup_timeout: Maximum seconds to wait after auto-starting the server.

    Returns:
        Base URL string for the reachable Ollama server.

    Raises:
        RuntimeError: If Ollama is not reachable and cannot be started.
    """
    base_url = get_ollama_base_url()
    tags_url = f"{base_url}/api/tags"

    def _probe() -> bool:
        try:
            with urllib_request.urlopen(tags_url, timeout=timeout) as response:
                return response.status < 400
        except urllib_error.URLError:
            return False

    if _probe():
        return base_url

    if not start_if_needed:
        raise RuntimeError(
            "Ollama is not reachable. Start the local server with `ollama serve` "
            f"and make sure it is available at {base_url}."
        )

    if not _can_auto_start_local_ollama(base_url):
        raise RuntimeError(
            "Ollama is not reachable, and CodeBook Lab only auto-starts local "
            f"servers on the default host. Current host: {base_url}"
        )

    ollama_executable = shutil.which("ollama")
    if ollama_executable is None:
        raise RuntimeError(
            "Ollama is not installed or not on PATH. Install Ollama first, then "
            "either start it manually with `ollama serve` or rerun the script."
        )

    subprocess.Popen(
        [ollama_executable, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=Path.cwd(),
    )

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _probe():
            return base_url
        time.sleep(0.25)

    raise RuntimeError(
        "Tried to auto-start Ollama, but it still was not reachable after "
        f"{startup_timeout:.1f} seconds at {base_url}."
    )


def ensure_ollama_model(model: str) -> None:
    """Pull an Ollama model so it is available locally before a run.

    Args:
        model: Ollama model identifier such as ``"gemma3:270m"``.
    """
    ollama_executable = shutil.which("ollama")
    if ollama_executable is None:
        raise RuntimeError(
            "Ollama is not installed or not on PATH, so the model cannot be pulled."
        )
    subprocess.run([ollama_executable, "pull", model], check=True)
