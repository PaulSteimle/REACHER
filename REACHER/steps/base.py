"""Base pipeline step definition."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import ReacherConfig


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    def __init__(self, config: ReacherConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        # Set by Pipeline / ParallelStepGroup before run(). When True,
        # per-step Start/Finish + resume-skip lines are suppressed
        # because the outer Pipeline already logs them. False when the
        # step is called standalone.
        self._quiet: bool = False

    @abstractmethod
    def run(self) -> None:
        """Execute the pipeline step."""

    @abstractmethod
    def _check_inputs(self) -> None:
        """Validate inputs before running the step."""

    def _skip_if_exists(self, filepath: Path) -> bool:
        """Return True if output exists and overwrite is False."""
        if filepath.exists() and not self.config.processing.overwrite:
            log = self.logger.debug if self._quiet else self.logger.info
            log("%s exists; skipping", filepath.name)
            return True
        return False

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("logger", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.logger = logging.getLogger(self.__class__.__name__)
