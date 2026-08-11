"""Operating-system-specific benchmark result paths."""

from __future__ import annotations

import platform
from pathlib import Path


MAC_RESULTS_DIR = Path(
    "/Users/harmankhunguradevagent/python-projects/llama.cpp-benchmarks/results/m4-mac-mini"
)


def result_directory() -> Path:
    """Return the result directory appropriate for the current OS."""

    system = platform.system()
    if system == "Darwin":
        return MAC_RESULTS_DIR
    if system == "Windows":
        # Preserve the existing Windows layout: results/<device>/...
        return Path.cwd() / "results"

    raise RuntimeError(f"Unsupported operating system: {system}")


def results_root() -> Path:
    """Return the common ``results`` root used for relative path mapping."""

    result_dir = result_directory()
    return result_dir.parent if platform.system() == "Darwin" else result_dir


def device_name(default_windows_device: str = "nvidia-rtx-3080") -> str:
    """Return the selected device folder name for the current OS."""

    if platform.system() == "Darwin":
        return result_directory().name
    return default_windows_device

