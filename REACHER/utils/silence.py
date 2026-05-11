"""Context managers for suppressing third-party console output.

The VIP-HCI routines used by several pipeline steps emit progress
messages directly to stdout/stderr (via ``print()`` and tqdm). These
helpers let callers wrap a single VIP call without permanently
silencing the rest of the process.

Two redirection layers are required:

* :func:`suppress_stdout_stderr` redirects Python-level
  ``sys.stdout`` / ``sys.stderr`` AND the underlying OS file
  descriptors 1 and 2. This catches both pure-Python ``print()`` calls
  and any output written by compiled extensions.

Caveat: while the context manager is active, log handlers writing to
``sys.stderr`` (e.g. ``StreamHandler`` defaults) are also silenced.
Keep the with-block tight around the noisy VIP call.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def suppress_stdout_stderr() -> Iterator[None]:
    """Silence stdout + stderr (Python streams + OS file descriptors)."""
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
