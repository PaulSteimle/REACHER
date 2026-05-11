"""Tests for :class:`REACHER.pipeline.Pipeline` orchestration."""
import logging

import numpy as np
import pytest
from astropy.io import fits

from REACHER.config import (
    CalibrationConfig,
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from REACHER.exceptions import (
    InvalidConfigError,
    ReacherError,
    StepExecutionError,
)
from REACHER.pipeline import ParallelStepGroup, Pipeline
from REACHER.steps.base import PipelineStep
from REACHER.steps.bad_pixel import BadPixelMaskCreator
from REACHER.steps.calibrate import FileCalibrator
from REACHER.steps.dark import MasterDarkCreator
from REACHER.steps.flat import MasterFlatCreator


def _make_config(tmp_path, overwrite=True):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(
        raw_input_dir=str(raw_dir),
        reduced_output_dir=str(reduced_dir),
    )
    processing = ProcessingConfig(overwrite=overwrite, make_plots=False)
    calibration = CalibrationConfig()
    return ReacherConfig(
        paths=paths, processing=processing, calibration=calibration,
    )


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------
class _SpyStep(PipelineStep):
    """A test double that records ``run`` invocations to a shared list."""

    def __init__(self, config, log, name, output_path=None,
                 raise_exc=None):
        super().__init__(config)
        self._log = log
        self._name = name
        self._output_path = output_path
        self._raise = raise_exc

    def _check_inputs(self):
        # nothing to check for the spy
        return

    def run(self):
        self._log.append(self._name)
        if self._raise is not None:
            raise self._raise
        if self._output_path is not None:
            if self._skip_if_exists(self._output_path):
                return
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_path.write_text("done")


class _FileStep(PipelineStep):
    """Picklable step that writes a sentinel file when run.

    Used by ``ParallelStepGroup`` tests: workers in a separate process
    can't share an in-memory list, so the spy records its execution by
    writing a file on disk that the parent process inspects afterwards.
    """

    def __init__(self, config, output_path, payload="done"):
        super().__init__(config)
        self._output_path = output_path
        self._payload = payload

    def _check_inputs(self):
        return

    def run(self):
        if self._skip_if_exists(self._output_path):
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(self._payload)


class _ExplodingStep(PipelineStep):
    """Picklable step that raises a ReacherError when run."""

    def __init__(self, config, message):
        super().__init__(config)
        self._message = message

    def _check_inputs(self):
        return

    def run(self):
        raise StepExecutionError(self._message)


# ----------------------------------------------------------------------
# 1. End-to-end happy path on synthetic IRDIS-like data
# ----------------------------------------------------------------------
def test_pipeline_end_to_end_synthetic(
    tmp_path, synthetic_dark_cube, minimal_sphere_header,
):
    """Run the calibration-side of the pipeline on synthetic data.

    Steps wired up: BadPixelMaskCreator -> MasterDarkCreator ->
    MasterFlatCreator -> FileCalibrator. The flux/sky/cube assembly
    steps require VIP-HCI numerical kernels that go beyond a smoke
    test; they are exercised by their dedicated test files.
    """
    config = _make_config(tmp_path, overwrite=True)
    dark_cube, _, _ = synthetic_dark_cube

    # --- Build inputs ---
    dark_path = tmp_path / "dark_20200101.fits"
    fits.writeto(
        dark_path, dark_cube, header=minimal_sphere_header, overwrite=True,
    )

    # Synthetic flat: positive gradient (no negatives -> no zero division)
    flat_data = np.ones((1024, 2048), dtype=np.float32) * 2.0
    flat_path = tmp_path / "flat_20200101.fits"
    fits.writeto(
        flat_path, flat_data, header=minimal_sphere_header, overwrite=True,
    )

    science_data = np.full((1024, 2048), 5.0, dtype=np.float32)
    science_path = tmp_path / "science_raw.fits"
    fits.writeto(
        science_path, science_data, header=minimal_sphere_header,
        overwrite=True,
    )

    # --- Output paths ---
    bad_pix_mask_path = tmp_path / "out" / "bad_pix_mask_20200101.fits"
    clump_mask_path = tmp_path / "out" / "IRDIS_clump_mask.fits"
    master_dark_path = tmp_path / "out" / "master_dark_20200101.fits"
    master_flat_path = tmp_path / "out" / "master_flat_20200101.fits"
    calibrated_path = tmp_path / "out" / "CALIB_science.fits"

    # --- Build the pipeline ---
    pipeline = Pipeline(config, configure_logging=False)
    pipeline.add_step(
        BadPixelMaskCreator(
            config=config,
            dark_files=[dark_path],
            bad_pix_mask_path=bad_pix_mask_path,
            clump_mask_path=clump_mask_path,
        )
    )
    pipeline.add_step(
        MasterDarkCreator(
            config=config,
            dark_files=[dark_path],
            master_dark_path=master_dark_path,
        )
    )
    pipeline.add_step(
        MasterFlatCreator(
            config=config,
            flat_files=[flat_path],
            master_dark_files=[master_dark_path],
            master_flat_path=master_flat_path,
        )
    )
    pipeline.add_step(
        FileCalibrator(
            config=config,
            input_file=science_path,
            output_file=calibrated_path,
            master_dark_path=master_dark_path,
            master_flat_path=master_flat_path,
        )
    )

    pipeline.run()

    # All four outputs should exist.
    assert bad_pix_mask_path.exists()
    assert clump_mask_path.exists()
    assert master_dark_path.exists()
    assert master_flat_path.exists()
    assert calibrated_path.exists()

    # Calibrated science should be (science - dark) / flat. The synthetic
    # dark has hot pixels but at 5.0 mean signal we still expect a
    # finite result with shape (1024, 2048).
    cal = fits.getdata(calibrated_path)
    assert cal.shape == (1024, 2048)
    # Master flat is normalised to ~1.0 by the LEFT/RIGHT channel
    # mean; therefore calibrated = (5 - 0) / 1.0 = 5.0 (excluding the
    # hot-pixel locations injected into the dark).
    median = np.nanmedian(cal)
    assert np.isclose(median, 5.0, atol=1e-3)


# ----------------------------------------------------------------------
# 2. Resumability: re-run skips completed outputs
# ----------------------------------------------------------------------
def test_pipeline_resumability(tmp_path):
    """If outputs exist + overwrite=False, steps short-circuit."""
    config = _make_config(tmp_path, overwrite=False)

    log: list = []
    out_path = tmp_path / "out" / "spy_step.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("pre-existing")
    original = out_path.read_text()

    spy = _SpyStep(config, log, "spy_step", output_path=out_path)
    pipeline = Pipeline(config, [spy], configure_logging=False)
    pipeline.run()

    # The step was invoked but its body short-circuited via
    # _skip_if_exists, leaving the pre-existing content intact.
    assert log == ["spy_step"]
    assert out_path.read_text() == original


def test_pipeline_resumability_writes_when_overwrite(tmp_path):
    """When overwrite=True, the step should overwrite existing outputs."""
    config = _make_config(tmp_path, overwrite=True)

    log: list = []
    out_path = tmp_path / "out" / "spy_step.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("stale")

    spy = _SpyStep(config, log, "spy_step", output_path=out_path)
    pipeline = Pipeline(config, [spy], configure_logging=False)
    pipeline.run()

    assert log == ["spy_step"]
    assert out_path.read_text() == "done"


# ----------------------------------------------------------------------
# 3. Failure propagation: a step raising halts the pipeline
# ----------------------------------------------------------------------
def test_pipeline_failure_propagates_reacher_error(tmp_path):
    """A ReacherError raised by a step is propagated unchanged."""
    config = _make_config(tmp_path)
    log: list = []

    step_a = _SpyStep(config, log, "a")
    step_b = _SpyStep(
        config, log, "b",
        raise_exc=StepExecutionError("boom from step b"),
    )
    step_c = _SpyStep(config, log, "c")

    pipeline = Pipeline(
        config, [step_a, step_b, step_c], configure_logging=False,
    )

    with pytest.raises(StepExecutionError, match="boom from step b"):
        pipeline.run()

    # Step c must NOT have run.
    assert log == ["a", "b"]


def test_pipeline_failure_wraps_unexpected_exception(tmp_path):
    """A non-ReacherError exception is wrapped in StepExecutionError."""
    config = _make_config(tmp_path)
    log: list = []

    step_a = _SpyStep(config, log, "a")
    step_b = _SpyStep(
        config, log, "b", raise_exc=RuntimeError("unexpected!"),
    )

    pipeline = Pipeline(config, [step_a, step_b], configure_logging=False)

    with pytest.raises(StepExecutionError) as exc_info:
        pipeline.run()

    # The original RuntimeError should be the __cause__.
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "unexpected!" in str(exc_info.value.__cause__)
    assert log == ["a", "b"]


# ----------------------------------------------------------------------
# 4. Config & API validation
# ----------------------------------------------------------------------
def test_pipeline_requires_config():
    with pytest.raises(InvalidConfigError):
        Pipeline(config=None, configure_logging=False)


def test_pipeline_add_step_type_check(tmp_path):
    config = _make_config(tmp_path)
    pipeline = Pipeline(config, configure_logging=False)

    with pytest.raises(InvalidConfigError):
        pipeline.add_step("not a step")


def test_pipeline_empty_warns_and_returns(tmp_path, caplog):
    """An empty pipeline logs a warning and returns without raising."""
    config = _make_config(tmp_path)
    pipeline = Pipeline(config, configure_logging=False)

    with caplog.at_level(logging.WARNING):
        pipeline.run()

    assert any(
        "no steps configured" in record.message.lower()
        for record in caplog.records
    )


def test_pipeline_step_order_preserved(tmp_path):
    """Steps execute in the order they were added."""
    config = _make_config(tmp_path)
    log: list = []

    pipeline = Pipeline(config, configure_logging=False)
    pipeline.add_steps([
        _SpyStep(config, log, f"step_{i}") for i in range(5)
    ])
    pipeline.run()

    assert log == [f"step_{i}" for i in range(5)]


def test_pipeline_steps_property_is_a_copy(tmp_path):
    """``Pipeline.steps`` should not allow external mutation."""
    config = _make_config(tmp_path)
    log: list = []
    pipeline = Pipeline(config, configure_logging=False)
    pipeline.add_step(_SpyStep(config, log, "a"))

    snapshot = pipeline.steps
    snapshot.clear()

    # Internal state is unchanged.
    assert len(pipeline.steps) == 1


# ----------------------------------------------------------------------
# ParallelStepGroup
# ----------------------------------------------------------------------
def test_parallel_step_group_runs_all_inner_steps(tmp_path):
    config = _make_config(tmp_path)
    config.processing.nproc = 3

    outputs = [tmp_path / f"out_{i}.txt" for i in range(6)]
    steps = [_FileStep(config, p) for p in outputs]

    group = ParallelStepGroup(config, steps, label="batch")
    group.run()

    assert all(p.exists() for p in outputs)
    assert all(p.read_text() == "done" for p in outputs)


def test_parallel_step_group_falls_back_to_sequential_when_nproc_is_one(tmp_path):
    config = _make_config(tmp_path)
    outputs = [tmp_path / f"out_{i}.txt" for i in range(3)]
    steps = [_FileStep(config, p) for p in outputs]

    group = ParallelStepGroup(config, steps, nproc=1)
    group.run()

    assert all(p.exists() for p in outputs)


def test_parallel_step_group_propagates_inner_exception(tmp_path):
    config = _make_config(tmp_path)
    config.processing.nproc = 2
    steps = [_ExplodingStep(config, "worker failed")]

    group = ParallelStepGroup(config, steps)
    with pytest.raises(StepExecutionError, match="worker failed"):
        group.run()


def test_parallel_step_group_rejects_empty(tmp_path):
    config = _make_config(tmp_path)
    with pytest.raises(InvalidConfigError):
        ParallelStepGroup(config, [])


def test_parallel_step_group_rejects_non_pipeline_step(tmp_path):
    config = _make_config(tmp_path)
    with pytest.raises(InvalidConfigError):
        ParallelStepGroup(config, [object()])


def test_parallel_step_group_inside_pipeline(tmp_path):
    """`ParallelStepGroup` should drop cleanly into an outer Pipeline."""
    config = _make_config(tmp_path)
    config.processing.nproc = 2
    outputs = [tmp_path / f"out_{i}.txt" for i in range(4)]
    steps = [_FileStep(config, p) for p in outputs]

    pipeline = Pipeline(config, configure_logging=False)
    pipeline.add_step(ParallelStepGroup(config, steps, label="batch"))
    pipeline.run()

    assert all(p.exists() for p in outputs)


def test_pipeline_silences_inner_step_start_finish_logs(tmp_path, caplog):
    """Pipeline mode: per-step [i/N] markers are DEBUG; Pipeline-level
    Starting/Finished lines stay at INFO so each batch helper still shows
    a one-line summary of what it's about to run."""
    config = _make_config(tmp_path)
    out = tmp_path / "out.txt"
    step = _FileStep(config, out)

    pipeline = Pipeline(config, configure_logging=False)
    pipeline.add_step(step)

    with caplog.at_level(logging.INFO):
        pipeline.run()

    pipeline_msgs = [
        r.message for r in caplog.records if r.name == "Pipeline"
    ]
    # Pipeline-level start/finish at INFO.
    assert any("Starting Pipeline" in m for m in pipeline_msgs)
    assert any("Pipeline finished" in m for m in pipeline_msgs)
    # Per-step [i/N] markers are DEBUG-only.
    assert not any("[1/1] Running" in m for m in pipeline_msgs)
    assert not any("[1/1] _FileStep finished" in m for m in pipeline_msgs)
    # The step itself does not log Start/Finish either.
    step_msgs = [
        r.message for r in caplog.records if r.name == "_FileStep"
    ]
    assert not any("Starting" in m or "Finished" in m for m in step_msgs)


def test_pipeline_per_step_markers_visible_at_debug(tmp_path, caplog):
    """The demoted Pipeline + [i/N] markers surface when DEBUG is on."""
    config = _make_config(tmp_path)
    step = _FileStep(config, tmp_path / "out.txt")
    pipeline = Pipeline(config, configure_logging=False)
    pipeline.add_step(step)

    with caplog.at_level(logging.DEBUG, logger="Pipeline"):
        pipeline.run()

    msgs = [r.message for r in caplog.records if r.name == "Pipeline"]
    assert any("Starting Pipeline" in m for m in msgs)
    assert any("Pipeline finished" in m for m in msgs)
    assert any("[1/1] Running" in m for m in msgs)
    assert any("[1/1] _FileStep finished" in m for m in msgs)


def test_step_logs_start_finish_when_called_standalone(tmp_path, caplog):
    """Outside a Pipeline, a step still logs Start/Finish at INFO."""
    from REACHER.steps.bad_pixel import BadPixelMaskCreator

    config = _make_config(tmp_path)
    # Provide a dummy dark file so _check_inputs passes.
    dark = tmp_path / "dark.fits"
    fits.writeto(dark, np.zeros((1024, 2048), dtype=np.float32), overwrite=True)

    step = BadPixelMaskCreator(
        config=config,
        dark_files=[dark],
        bad_pix_mask_path=tmp_path / "bp.fits",
        clump_mask_path=tmp_path / "clump.fits",
    )
    assert step._quiet is False  # default

    with caplog.at_level(logging.INFO, logger="BadPixelMaskCreator"):
        try:
            step.run()
        except Exception:
            # We don't care if the body crashes on the synthetic dark;
            # we only need to see the Starting log line fire.
            pass

    msgs = [r.message for r in caplog.records if r.name == "BadPixelMaskCreator"]
    assert any("Starting BadPixelMaskCreator" in m for m in msgs)


def test_parallel_step_group_silences_inner_steps_logs(tmp_path, caplog):
    """ParallelStepGroup sets _quiet=True on inner steps."""
    config = _make_config(tmp_path)
    config.processing.nproc = 1  # sequential branch keeps things in-process
    outputs = [tmp_path / f"out_{i}.txt" for i in range(3)]
    steps = [_FileStep(config, p) for p in outputs]

    group = ParallelStepGroup(config, steps, nproc=1, label="batch")

    with caplog.at_level(logging.INFO):
        group.run()

    # The group itself still logs its summary (it's not inside a Pipeline).
    group_msgs = [
        r.message for r in caplog.records
        if r.name == "ParallelStepGroup"
    ]
    assert any("running 3 step" in m for m in group_msgs)
    # But inner steps were marked quiet for the worker calls.
    for s in steps:
        assert s._quiet is True


def test_parallel_step_group_respects_skip_if_exists(tmp_path):
    """Resumability: pre-existing outputs are not re-written when overwrite=False."""
    config = _make_config(tmp_path, overwrite=False)
    config.processing.nproc = 2
    outputs = [tmp_path / f"out_{i}.txt" for i in range(3)]
    for p in outputs:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("pre-existing")

    steps = [_FileStep(config, p, payload="overwritten") for p in outputs]
    group = ParallelStepGroup(config, steps, nproc=2)
    group.run()

    assert all(p.read_text() == "pre-existing" for p in outputs)
