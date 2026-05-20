"""Pre-calibration + extraction batch helper.

Phase 4 replacement for the legacy
:func:`master_routines.pre_calibration_and_extraction`. For one
:class:`StarTarget` it:

1. Discovers the science / flux / center / sky files on disk.
2. Resolves bad-pix mask + master dark + master flat via
   :func:`REACHER.discovery.resolve_star_calibration`.
3. Builds a :class:`Pipeline` of
   :class:`FileCalibrator` -> :class:`ChannelSplitter` ->
   :class:`CenteringPreparer` instances.
4. Runs the pipeline.

The output is the standard ``stage1`` (calibrated raw files) +
``stage2`` (LEFT/RIGHT splits + ``centers.csv``) directory layout
expected by the centering / cube-assembly batch helper.
"""
from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import ReacherConfig
from ..exceptions import InvalidConfigError, StepExecutionError
from ..pipeline import ParallelStepGroup, Pipeline
from ..steps.calibrate import FileCalibrator
from ..steps.centering_prep import CenteringPreparer
from ..steps.splitting import ChannelSplitter
from ..discovery.calib_resolver import resolve_star_calibration
from ..discovery.dataframes import make_star_DF
from ..discovery.layout import DataLayout
from ..discovery.target import StarTarget


logger = logging.getLogger(__name__)


def pre_calibration_and_extraction(
    config: ReacherConfig,
    target: StarTarget,
    *,
    star_df: Optional[pd.DataFrame] = None,
    configure_logging: bool = False,
) -> Pipeline:
    """Run the calibrate -> split -> centering-prep chain for one target.

    Parameters
    ----------
    config
        :class:`ReacherConfig`.
    target
        :class:`StarTarget` identifying the
        ``(star_name, obs_night, filter_name)`` triple to process.
    star_df
        Optional pre-built per-star DataFrame (output of
        :func:`make_star_DF`). When ``None`` a fresh DataFrame is
        built.
    configure_logging
        Forwarded to :class:`Pipeline`.

    Returns
    -------
    Pipeline
        The executed pipeline.

    Raises
    ------
    InvalidConfigError
        If multiple bad-pix masks are resolved for the same
        ``(obs_night, filter)`` pair (legacy guard).
    StepExecutionError
        If the per-star raw input folder does not exist or contains no
        ``*SPHER*.fits`` files.
    """
    if config is None:
        raise InvalidConfigError("config must not be None")
    if target is None:
        raise InvalidConfigError("target must not be None")

    layout = DataLayout(config.paths)
    star_input_folder = layout.star_input_folder(
        target.star_name, target.obs_night, target.filter_name
    )
    if not star_input_folder.exists():
        raise StepExecutionError(
            f"star input folder does not exist: {star_input_folder}"
        )

    logger.info(
        "working on %s, %s, %s",
        target.star_name, target.obs_night, target.filter_name,
    )

    if star_df is None:
        star_df = make_star_DF(
            config.paths.raw_input_dir,
            target.star_name,
            target.obs_night,
            target.filter_name,
            instrument=config.instrument,
        )
    if star_df.empty:
        raise StepExecutionError(
            f"no science files found in {star_input_folder}"
        )

    star_df = resolve_star_calibration(
        star_df, config, layout=layout, calib_type="science"
    )

    unique_masks = star_df["bad_pix_mask"].unique()
    if len(unique_masks) > 1:
        raise InvalidConfigError(
            f"more than one bad pixel mask resolved for "
            f"{target.star_name} {target.obs_night} {target.filter_name}; "
            f"got {list(unique_masks)}"
        )
    bad_pix_arg = unique_masks[0] if unique_masks[0] else None
    clump_mask_path = layout.clump_mask_filepath
    clump_arg = clump_mask_path if clump_mask_path.exists() else None

    stage1 = layout.stage1_output_folder(
        target.star_name, target.obs_night, target.filter_name
    )
    stage2 = layout.stage2_output_folder(
        target.star_name, target.obs_night, target.filter_name
    )
    stage1.mkdir(parents=True, exist_ok=True)
    stage2.mkdir(parents=True, exist_ok=True)

    # Cache the channel-split + centering inputs for the build phase.
    raw_files: List[Path] = [Path(p) for p in star_df["filepath"].tolist()]
    calibrated_paths = [stage1 / ("CALIB_" + p.name) for p in raw_files]
    left_paths = [stage2 / ("LEFT_" + p.name) for p in raw_files]
    right_paths = [stage2 / ("RIGHT_" + p.name) for p in raw_files]

    pipeline = Pipeline(config, configure_logging=configure_logging)

    # ---- Stage 1: per-file dark/flat/badpix calibration (parallel) ----
    calibrate_steps = [
        FileCalibrator(
            config=config,
            input_file=sci_path,
            output_file=cal_path,
            master_dark_path=master_dark,
            master_flat_path=master_flat,
            bad_pix_mask_path=bad_pix_arg,
            clump_mask_path=clump_arg,
        )
        for sci_path, cal_path, master_dark, master_flat in zip(
            raw_files,
            calibrated_paths,
            star_df["master_dark"].tolist(),
            star_df["master_flat"].tolist(),
        )
    ]
    pipeline.add_step(ParallelStepGroup(
        config=config, steps=calibrate_steps, label="FileCalibrator",
    ))

    # ---- Stage 2a: LEFT/RIGHT split (parallel) ----
    split_steps = [
        ChannelSplitter(
            config=config,
            input_file=cal_path,
            left_output_file=left_path,
            right_output_file=right_path,
        )
        for cal_path, left_path, right_path in zip(
            calibrated_paths, left_paths, right_paths,
        )
    ]
    pipeline.add_step(ParallelStepGroup(
        config=config, steps=split_steps, label="ChannelSplitter",
    ))

    # ---- Stage 2b: centering prep over CENTER files only ----
    center_inputs = [
        p for p in (left_paths + right_paths) if "CENTER" in p.name
    ]
    if center_inputs:
        pipeline.add_step(CenteringPreparer(
            config=config,
            input_files=center_inputs,
            output_dir=stage2,
        ))
    else:
        logger.info(
            "no CENTER files found for %s %s %s; skipping CenteringPreparer",
            target.star_name, target.obs_night, target.filter_name,
        )

    pipeline.run()
    return pipeline
