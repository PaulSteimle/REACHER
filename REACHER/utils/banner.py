"""Run-start banner with version + git metadata.

The banner is logged at INFO level so it lands in both the console and
the per-run log file initialized by :func:`REACHER.setup_logging`. It
fires exactly once per process; subsequent calls are no-ops unless
``force=True`` is passed.

Auto-fires from ``setup_logging``; users can also call
:func:`print_banner` manually from a script that does not invoke
``setup_logging``.
"""
from __future__ import annotations

import logging
import multiprocessing
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


_REACHER_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PRINTED = False


def _git(*args: str) -> str:
    """Run a git command rooted at the package's directory; '' on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_REACHER_PACKAGE_ROOT)] + list(args),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _git_state() -> Optional[str]:
    """Return a compact 'branch@sha [dirty]' string, or None if no git."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    sha = _git("rev-parse", "--short", "HEAD")
    if not branch and not sha:
        return None
    label = branch or "?"
    if sha:
        label += f"@{sha}"
    dirty = bool(_git("status", "--porcelain"))
    if dirty:
        label += " (dirty)"
    return label


def _reacher_version() -> str:
    try:
        from REACHER import __version__
        return __version__
    except Exception:
        return "unknown"


def print_banner(force: bool = False) -> None:
    """Log a 5-line run-start banner. No-op after first call unless forced.

    Also a no-op in multiprocessing.Pool worker processes — the parent
    process already emitted the banner once, and ``spawn``-based workers
    would otherwise re-print it because they re-import the entry script.
    """
    if multiprocessing.current_process().name != "MainProcess":
        return
    global _PRINTED
    if _PRINTED and not force:
        return
    _PRINTED = True

    log = logging.getLogger("REACHER")
    bar = "=" * 72
    version = _reacher_version()
    git = _git_state()

    log.info(bar)
    if git is not None:
        log.info("REACHER v%s | git: %s", version, git)
    else:
        log.info("REACHER v%s", version)
    log.info(
        "Run start: %s | host: %s | python: %s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        socket.gethostname(),
        platform.python_version(),
    )
    log.info(bar)
