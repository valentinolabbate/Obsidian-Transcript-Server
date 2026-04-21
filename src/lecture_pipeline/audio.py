from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# macOS GUI apps (like Obsidian) launch child processes without the user's
# full shell PATH, so /opt/homebrew/bin is often missing. We augment the
# lookup with the most common install locations before falling back to PATH.
_FFMPEG_FALLBACK_DIRS = [
    "/opt/homebrew/bin",   # Apple Silicon Homebrew
    "/usr/local/bin",      # Intel Homebrew / manual installs
    "/opt/local/bin",      # MacPorts
]


def _find_ffmpeg() -> str | None:
    """Return the absolute path to ffmpeg, or None if not found."""
    # First try the standard PATH lookup.
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Fall back to well-known directories that GUI apps typically miss.
    for directory in _FFMPEG_FALLBACK_DIRS:
        candidate = Path(directory) / "ffmpeg"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def ffmpeg_available() -> bool:
    return _find_ffmpeg() is not None


def preprocess_audio(source: Path, destination: Path) -> Path:
    ffmpeg_bin = _find_ffmpeg()
    if not ffmpeg_bin:
        return source

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return destination
