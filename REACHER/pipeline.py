"""Pipeline orchestration for REACHER.

This module wires the nine refactored ``PipelineStep`` subclasses
(produced by Phase 2) into a single, resumable, top-level
:class:`Pipeline`. It is the only orchestration layer in the package
post-Phase 3 -- the legacy ``master_routines.py`` / ``wrappers.py``
chain is retired.

Expected step execution order (matches the legacy flow):

    Stage 1 -- calibration products (per obs_night):
        1. BadPixelMaskCreator
        2. MasterDarkCreator       (per obs_night, exp_time)
        3. MasterFlatCreator       (per obs_night, filter)
    Stage 2 -- per-science-sequence pre-calibration:
        4. FileCalibrator          (per raw science file)
        5. ChannelSplitter         (per calibrated file)
        6. CenteringPreparer       (CENTER files only)

    Steps 4 and 5 are independent across files, so the batch helper
    :func:`pre_calibration_and_extraction` wraps their per-file
    instances in :class:`ParallelStepGroup`. That step runs its inner
    steps in a :class:`multiprocessing.Pool` of size
    ``config.processing.nproc``, restoring the legacy
    ``calibrate_files_multiprocessing_wrapper`` /
    ``split_files_multiprocessing_wrapper`` behavior.
        [HUMAN-IN-THE-LOOP: user verifies/corrects centers.csv]
    Stage 3 -- centering & cube assembly (per LEFT/RIGHT channel):
        7. MasterSkyCreator        (optional, when sky files exist)
        8. FluxCalibrator          (per flux file)
        9. CubeAssembler           (final master cube + angles)

Pipeline ownership / control flow:

* The :class:`Pipeline` class owns an ordered sequence of
  :class:`PipelineStep` instances built externally (either by hand or
  by a builder routine). On :meth:`Pipeline.run` it iterates the steps
  and calls ``step.run()``.
* Resumability: each step already implements ``_skip_if_exists`` and
  honours ``config.processing.overwrite``. The Pipeline does not
  duplicate that logic.
* Error handling: any exception raised by a step propagates after
  being logged. ``StepExecutionError`` is raised wrapping non-Reacher
  exceptions so callers always see a Reacher-typed error.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import ReacherConfig
from .exceptions import (
    InvalidConfigError,
    ReacherError,
    StepExecutionError,
)
from .steps.base import PipelineStep
from .utils.formatting import format_duration
from .utils.logging import log_blank_line, setup_logging


# ----------------------------------------------------------------------
# TODO: POST-PROCESSING STEPS (ADI/SDI) WILL BE INSERTED HERE
# ----------------------------------------------------------------------
# Expected interface for future Phase 4+ post-processing steps:
#     class PostProcessingStep(PipelineStep):
#         '''Consumes:
#             input: master_cube.fits + parallactic_angles.txt
#             produced by CubeAssembler.
#         '''
# Plug them into a Pipeline instance the same way the Stage 1-3 steps
# are plugged in today (see ``add_step`` / ``Pipeline(steps=...)``).
# ----------------------------------------------------------------------


class Pipeline:
    """Orchestrate an ordered sequence of :class:`PipelineStep` instances.

    Parameters
    ----------
    config
        Reacher configuration object. Initializing the pipeline also
        initializes the per-run log file under
        ``config.paths.reduced_output_dir/logs/`` unless
        ``configure_logging=False``.
    steps
        Optional iterable of pre-built :class:`PipelineStep` instances.
        Steps may also be appended later via :meth:`add_step`.
    configure_logging
        When ``True`` (default) the pipeline calls
        :func:`REACHER.utils.logging.setup_logging` so a per-run log
        file is created. Tests pass ``configure_logging=False`` to keep
        pytest's stdout/stderr capture in place.

    Notes
    -----
    The Pipeline does not dictate which steps must be present or how
    they are constructed -- step construction is the caller's
    responsibility (see ``run_REACHER.py`` for a reference builder).
    The Pipeline only enforces the invariants documented above:
    sequential execution, exception wrapping, and a concise log entry
    per step.
    """

    def __init__(
        self,
        config: ReacherConfig,
        steps: Iterable[PipelineStep] | None = None,
        *,
        configure_logging: bool = True,
    ) -> None:
        if config is None:
            raise InvalidConfigError("Pipeline requires a ReacherConfig")
        self.config = config
        self._steps: List[PipelineStep] = []
        if steps is not None:
            for step in steps:
                self.add_step(step)

        if configure_logging:
            log_dir = Path(config.paths.reduced_output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / time.strftime("reacher_%Y%m%d_%H%M%S.log")
            setup_logging(log_file=str(log_file))

        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_step(self, step: PipelineStep) -> None:
        """Append a step to the pipeline."""
        if not isinstance(step, PipelineStep):
            raise InvalidConfigError(
                f"Pipeline.add_step expects a PipelineStep instance; "
                f"got {type(step).__name__}"
            )
        self._steps.append(step)

    def add_steps(self, steps: Sequence[PipelineStep]) -> None:
        """Append multiple steps preserving order."""
        for step in steps:
            self.add_step(step)

    @property
    def steps(self) -> List[PipelineStep]:
        """Read-only view of the configured step sequence."""
        return list(self._steps)

    def run(self) -> None:
        """Execute every configured step in order.

        Each step's ``run`` method is called once. ``_skip_if_exists``
        on the step itself drives resumability (the Pipeline does not
        re-implement that check). Any exception is logged and re-raised:
        non-Reacher exceptions are wrapped in
        :class:`StepExecutionError` so callers always observe a
        Reacher-typed error.
        """
        if not self._steps:
            self.logger.warning(
                "Pipeline has no steps configured; nothing to do"
            )
            return

        self.logger.info(
            "Starting Pipeline with %d step(s): %s",
            len(self._steps),
            ", ".join(type(s).__name__ for s in self._steps),
        )

        log_blank_line()

        total_t0 = time.perf_counter()
        for i, step in enumerate(self._steps, start=1):
            name = type(step).__name__
            # Per-step running/finished markers are demoted to DEBUG --
            # each step that produces an artifact now logs its own
            # "Created X in Ys" line, which is the user-facing signal.
            self.logger.debug(
                "[%d/%d] Running %s", i, len(self._steps), name
            )
            t0 = time.perf_counter()
            step._quiet = True
            try:
                step.run()
            except ReacherError:
                self.logger.exception(
                    "[%d/%d] %s raised a ReacherError; halting pipeline",
                    i, len(self._steps), name,
                )
                raise
            except Exception as exc:  # pragma: no cover - exhaustive guard
                self.logger.exception(
                    "[%d/%d] %s raised %s; halting pipeline",
                    i, len(self._steps), name, type(exc).__name__,
                )
                raise StepExecutionError(
                    f"{name} failed: {exc}"
                ) from exc

            elapsed = time.perf_counter() - t0
            self.logger.debug(
                "[%d/%d] %s finished in %s", i, len(self._steps),
                name, format_duration(elapsed),
            )
            # Visual separator between step output blocks (console + log file).
            log_blank_line()

        total_elapsed = time.perf_counter() - total_t0
        self.logger.info(
            "Pipeline finished %d step(s) in %s",
            len(self._steps), format_duration(total_elapsed),
        )
        
        log_blank_line()


def _run_step_worker(step: PipelineStep) -> None:
    """Module-scope worker that runs a single PipelineStep.

    Defined at module scope so ``multiprocessing.Pool`` (which requires
    a picklable target on macOS spawn semantics) can serialize it.
    """
    step.run()


class ParallelStepGroup(PipelineStep):
    """Run a collection of independent :class:`PipelineStep` instances in parallel.

    Restores the legacy
    ``calibrate_files_multiprocessing_wrapper`` /
    ``split_files_multiprocessing_wrapper`` parallelism on top of the
    new PipelineStep contract: callers build N independent step
    instances (one per input file) and wrap them in a
    ``ParallelStepGroup``. The group is itself a ``PipelineStep`` and
    therefore drops cleanly into an outer :class:`Pipeline`.

    Parameters
    ----------
    config
        Reacher configuration. Defaults to the value of
        ``config.processing.nproc`` for the pool size, unless ``nproc``
        is passed explicitly.
    steps
        Independent ``PipelineStep`` instances to run. Must be
        picklable (the base class implements ``__getstate__`` /
        ``__setstate__`` to drop the logger across the pickle boundary).
    nproc
        Pool size. Defaults to ``config.processing.nproc``. Clamped to
        ``len(steps)``; falls back to a sequential run when
        ``nproc <= 1``.
    label
        Optional human-readable label used in log messages.

    Notes
    -----
    * Each inner step's ``_skip_if_exists`` still drives resumability;
      the group does not duplicate that check.
    * If any inner step raises, the exception propagates out of
      ``Pool.map`` and the outer :class:`Pipeline` catches and logs it.
    """

    def __init__(
        self,
        config: ReacherConfig,
        steps: Sequence[PipelineStep],
        *,
        nproc: Optional[int] = None,
        label: str = "ParallelStepGroup",
    ) -> None:
        super().__init__(config)
        if not steps:
            raise InvalidConfigError(
                "ParallelStepGroup requires at least one inner step"
            )
        for s in steps:
            if not isinstance(s, PipelineStep):
                raise InvalidConfigError(
                    f"ParallelStepGroup expects PipelineStep instances; "
                    f"got {type(s).__name__}"
                )
        self._inner_steps: List[PipelineStep] = list(steps)
        self._nproc = nproc if nproc is not None else config.processing.nproc
        self._label = label

    def _check_inputs(self) -> None:
        # Inner steps validate their own inputs at run time.
        return

    def run(self) -> None:
        n = len(self._inner_steps)
        nproc = max(1, min(self._nproc or 1, n))

        # Inside a Pipeline the outer "[i/N] Running ParallelStepGroup"
        # line already names this group; the per-worker step start/end
        # lines would then triple-log the same fact. Demote to DEBUG.
        log = self.logger.debug if self._quiet else self.logger.info
        log(
            "%s: running %d step(s) with nproc=%d",
            self._label, n, nproc,
        )

        for step in self._inner_steps:
            step._quiet = True

        if nproc <= 1:
            for step in self._inner_steps:
                step.run()
            return

        with mp.Pool(processes=nproc) as pool:
            pool.map(_run_step_worker, self._inner_steps)
