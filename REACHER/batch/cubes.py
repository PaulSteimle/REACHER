"""Centering + bad-pixel + cube assembly batch helper.

Phase 4 replacement for the legacy
:func:`master_routines.centering_and_bad_pix_correction`. For one
:class:`StarTarget` it:

1. Discovers ``stage2`` LEFT / RIGHT science / flux / center / sky
   files on disk.
2. (Optional) Builds a master sky per channel (skipped when the target
   carries a ``fixed_user_mastersky_*`` override or when no sky files
   exist).
3. Builds flux files via :class:`FluxCalibrator` per channel.
4. Reads the satellite-spot recentering shift from a CENTER file via
   :func:`REACHER.utils.centering.get_center`.
5. Assembles the master cube + parallactic angles via
   :class:`CubeAssembler` for the science target and (optionally) the
   reference stars listed in ``ref_cubes``.

The reference-star branch reuses the science target's satellite-spot
recentering shift and CENTER file, mirroring the legacy semantics
(reference stars share their science target's center).
"""
from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import ReacherConfig
from ..exceptions import InvalidConfigError, StepExecutionError
from ..pipeline import Pipeline
from ..steps.cube_assembly import CubeAssembler
from ..steps.flux import FluxCalibrator
from ..steps.sky import MasterSkyCreator
from ..discovery.calib_resolver import resolve_star_calibration
from ..discovery.dataframes import make_star_DF
from ..discovery.layout import DataLayout
from ..discovery.target import StarTarget
from ..utils.centering import get_center


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _resolve_ref_stars(
    ref_cubes: Optional[Dict],
    target: StarTarget,
) -> List[str]:
    """Resolve the reference-star list for ``target`` from ``ref_cubes``.

    Mirrors the legacy logic exactly:

    * ``ref_cubes`` is the ``ref_cubes_dict`` passed by the caller.
    * If ``ref_cubes[target.star_name]`` is a dict it must be keyed on
      ``obs_night``.
    * If it is a list / str it applies to every obs_night.
    * Anything else -> empty list.
    """
    if not isinstance(ref_cubes, dict):
        return []
    if target.star_name not in ref_cubes:
        return []
    value = ref_cubes[target.star_name]
    if isinstance(value, dict):
        ref_stars = value.get(target.obs_night, [])
    elif isinstance(value, (list, str)):
        ref_stars = value
    else:
        ref_stars = []
    if isinstance(ref_stars, str):
        ref_stars = [ref_stars]
    elif ref_stars is None:
        ref_stars = []
    return list(ref_stars)


def _filter_files_with_prefix(
    folder: Path, prefix: str, keyword: str,
) -> List[Path]:
    """Return ``folder/<prefix>_*<keyword>*.fits`` paths in sorted order."""
    pattern = f"{prefix}_*{keyword}*.fits"
    return [Path(p) for p in sorted(glob(str(folder / pattern)))]


def _all_files_with_prefix(folder: Path, prefix: str) -> List[Path]:
    """Return ``folder/<prefix>_*.fits`` paths in sorted order."""
    pattern = f"{prefix}_*.fits"
    return [Path(p) for p in sorted(glob(str(folder / pattern)))]


def _resolve_science_exp_time(
    star_df: pd.DataFrame, config: ReacherConfig,
) -> float:
    """Extract the unique science exposure time.

    The DPR-type vocabulary that classifies a row as a science frame
    is read from ``config.instrument.values.science_types``; set
    membership rather than equality so that multi-token vocabularies
    (e.g. NACO's ``{SCIENCE, IMAGE,OBJECT}``) work transparently.
    """
    science_types = config.instrument.values.science_types
    object_df = star_df[star_df.get("type").isin(science_types)]
    exp_times = object_df["exp_time"].unique()
    if len(exp_times) != 1:
        raise StepExecutionError(
            f"science files have different exposure times: "
            f"{list(exp_times)}; check the STAR_DF csv"
        )
    return float(exp_times[0])


# ----------------------------------------------------------------------
# Public batch helper
# ----------------------------------------------------------------------
def centering_and_bad_pix_correction(
    config: ReacherConfig,
    target: StarTarget,
    *,
    ref_cubes: Optional[Dict] = None,
    star_df: Optional[pd.DataFrame] = None,
    configure_logging: bool = False,
) -> Pipeline:
    """Run sky + flux + cube assembly for one target.

    Parameters
    ----------
    config
        :class:`ReacherConfig`.
    target
        :class:`StarTarget` for the ``(star_name, obs_night,
        filter_name)`` to process.
    ref_cubes
        Optional ``{star_name: ref_star_or_list_or_dict}`` mapping.
        See :func:`_resolve_ref_stars` for the resolution rules.
    star_df
        Optional pre-built STAR DataFrame for ``target``.
    configure_logging
        Forwarded to :class:`Pipeline`.

    Returns
    -------
    Pipeline
        The executed pipeline.

    Notes
    -----
    Reference-star handling preserves the legacy behaviour: reference
    stars without their own CENTER files are processed alongside their
    science target, reusing the science target's satellite-spot center.
    The reference-star CubeAssembler instance is configured to read
    LEFT / RIGHT splits from the reference star's own ``stage2``
    directory.
    """
    if config is None:
        raise InvalidConfigError("config must not be None")
    if target is None:
        raise InvalidConfigError("target must not be None")

    logger.info(
        "working on %s, %s, %s",
        target.star_name, target.obs_night, target.filter_name,
    )

    layout = DataLayout(config.paths)
    stage2 = layout.stage2_output_folder(
        target.star_name, target.obs_night, target.filter_name
    )
    if not stage2.exists():
        raise StepExecutionError(
            f"stage2 folder does not exist: {stage2}; run "
            "pre_calibration_and_extraction first"
        )

    # Decide do_flux / do_cubes / do_angles based on the only_* flags,
    # mirroring the legacy semantics. Preserved as local booleans rather
    # than mutating the config.
    do_flux = config.processing.do_flux
    do_cubes = config.processing.do_cubes
    do_angles = config.processing.do_angles
    if config.processing.only_flux:
        do_flux = True
        do_cubes = False
        do_angles = False
    if config.processing.only_angles:
        do_flux = False
        do_cubes = False
        do_angles = True

    ref_stars = _resolve_ref_stars(ref_cubes, target)
    if ref_stars:
        logger.info("star has ref stars: %s", ref_stars)
    else:
        logger.info("star has no ref stars!")

    # Discover CENTER files (LEFT prefix only -- they all come in pairs).
    center_files_left = _filter_files_with_prefix(stage2, "LEFT", "CENTER")
    if not center_files_left:
        logger.info(
            "no starcenter files for %s %s %s; this is probably a ref "
            "star (ref stars are processed via their science target)",
            target.star_name, target.obs_night, target.filter_name,
        )
        return Pipeline(config, configure_logging=configure_logging)

    # Resolve per-target STAR DataFrame for the science exposure time.
    if star_df is None:
        star_df = make_star_DF(
            config.paths.raw_input_dir,
            target.star_name,
            target.obs_night,
            target.filter_name,
            instrument=config.instrument,
        )

    science_exp_time = _resolve_science_exp_time(star_df, config)

    pipeline = Pipeline(config, configure_logging=configure_logging)

    # ---- Stage 3a: master sky per channel (optional) ----
    sky_files_exist = _add_master_sky_steps(
        pipeline, config, layout, target, stage2,
    )

    # ---- Stage 3b: flux files per channel (optional) ----
    if do_flux:
        _add_flux_steps(
            pipeline, config, layout, target, stage2,
            science_exp_time=science_exp_time,
            sky_files_exist=sky_files_exist,
        )

    if config.processing.only_flux:
        logger.info(
            "only_flux=True: skipping cube assembly for %s",
            target.star_name,
        )
        pipeline.run()
        return pipeline

    if not do_cubes:
        logger.info(
            "do_cubes=False: skipping cube assembly for %s",
            target.star_name,
        )
        pipeline.run()
        return pipeline

    # ---- Stage 3c: master cube assembly per channel + ref stars ----
    _add_cube_steps(
        pipeline, config, layout, target, stage2,
        center_files_left=center_files_left,
        ref_stars=ref_stars,
        sky_files_exist=sky_files_exist,
    )

    pipeline.run()
    return pipeline


# ----------------------------------------------------------------------
# Step builders
# ----------------------------------------------------------------------
def _add_master_sky_steps(
    pipeline: Pipeline,
    config: ReacherConfig,
    layout: DataLayout,
    target: StarTarget,
    stage2: Path,
) -> Dict[str, bool]:
    """Append :class:`MasterSkyCreator` steps; return per-channel exist map.

    Resolution priority (mirrors the legacy semantics):

    1. ``target.fixed_user_mastersky_<channel>``: skip the step (the
       caller has supplied a master sky); mark the channel as
       sky-exists.
    2. Auto-detect ``<channel>_*SKY*.fits`` files in ``stage2/``: when
       present add a :class:`MasterSkyCreator` step.
    3. No sky files: leave the channel unmarked.
    """
    sky_exists = {"LEFT": False, "RIGHT": False}

    # Honour an explicit override on the target first.
    if target.sky_files_exist is not None:
        sky_exists = {"LEFT": target.sky_files_exist, "RIGHT": target.sky_files_exist}

    for channel in ("LEFT", "RIGHT"):
        user_path = target.fixed_user_mastersky_for(channel)
        if user_path is not None:
            logger.info(
                "%s: using user-supplied master sky %s; skipping "
                "MasterSkyCreator",
                channel, user_path,
            )
            sky_exists[channel] = True
            continue

        sky_files = _filter_files_with_prefix(stage2, channel, "SKY")
        if not sky_files:
            logger.warning(
                "no sky files found for %s %s",
                target.star_name, channel,
            )
            # Only update if no explicit override is set.
            if target.sky_files_exist is None:
                sky_exists[channel] = False
            continue

        master_sky_path = layout.master_sky_filepath(
            target.star_name, target.obs_night, target.filter_name,
            channel=channel,
        )
        pipeline.add_step(MasterSkyCreator(
            config=config,
            sky_files=sky_files,
            master_sky_path=master_sky_path,
        ))
        sky_exists[channel] = True

    return sky_exists


def _add_flux_steps(
    pipeline: Pipeline,
    config: ReacherConfig,
    layout: DataLayout,
    target: StarTarget,
    stage2: Path,
    *,
    science_exp_time: float,
    sky_files_exist: Dict[str, bool],
) -> None:
    """Append per-channel :class:`FluxCalibrator` steps."""
    for channel in ("LEFT", "RIGHT"):
        out_folder = layout.channel_output_folder(
            target.star_name, target.obs_night, target.filter_name,
            channel=channel,
        )
        out_folder.mkdir(parents=True, exist_ok=True)

        flux_inputs = _filter_files_with_prefix(stage2, channel, "FLUX")
        if not flux_inputs:
            logger.warning(
                "no flux files for %s %s; skipping flux calibration",
                target.star_name, channel,
            )
            continue

        master_sky_path = None
        if sky_files_exist.get(channel):
            master_sky_path = layout.master_sky_filepath(
                target.star_name, target.obs_night, target.filter_name,
                channel=channel,
                fixed_user_mastersky=target.fixed_user_mastersky_for(channel),
            )

        if len(flux_inputs) == 1:
            output_paths = [out_folder / f"{channel.lower()}_flux.fits"]
        else:
            output_paths = [
                out_folder / f"{channel.lower()}_flux_{i + 1}.fits"
                for i in range(len(flux_inputs))
            ]

        for inp, out in zip(flux_inputs, output_paths):
            pipeline.add_step(FluxCalibrator(
                config=config,
                input_file=inp,
                output_file=out,
                science_exp_time=science_exp_time,
                channel=channel,
                master_sky_path=master_sky_path,
            ))


def _add_cube_steps(
    pipeline: Pipeline,
    config: ReacherConfig,
    layout: DataLayout,
    target: StarTarget,
    stage2: Path,
    *,
    center_files_left: List[Path],
    ref_stars: List[str],
    sky_files_exist: Dict[str, bool],
) -> None:
    """Append per-channel :class:`CubeAssembler` steps + ref-star cubes.

    The recentering shift is computed eagerly via
    :func:`REACHER.utils.centering.get_center` because
    :class:`CubeAssembler` takes a precomputed ``shift`` argument. This
    mirrors the legacy ``get_center_wrapper -> make_the_cube_wrapper``
    chain.
    """
    # Pick the CENTER file using target.star_center_index (with bounds fallback).
    idx = target.star_center_index
    if idx >= len(center_files_left):
        logger.warning(
            "star_center_index %s is out of bounds for %s CENTER files; "
            "falling back to index 0",
            idx, len(center_files_left),
        )
        idx = 0
    star_center_left = center_files_left[idx]
    star_center_right = stage2 / star_center_left.name.replace("LEFT_", "RIGHT_")

    # Compute the recentering shift eagerly (LEFT-side; the CENTER file
    # carries the satspot positions for both channels in centers.csv).
    try:
        shift_yx = get_center(
            star_center_left,
            make_plots=config.processing.make_plots,
        )
    except Exception as exc:
        raise StepExecutionError(
            f"get_center failed for {star_center_left.name}: {exc}"
        ) from exc

    for channel in ("LEFT", "RIGHT"):
        out_folder = layout.channel_output_folder(
            target.star_name, target.obs_night, target.filter_name,
            channel=channel,
        )
        out_folder.mkdir(parents=True, exist_ok=True)

        science_inputs = _filter_files_with_prefix(stage2, channel, "SCIENCE")
        if not science_inputs:
            logger.warning(
                "no science files for %s %s; skipping cube assembly",
                target.star_name, channel,
            )
            continue

        center_file = (
            star_center_left if channel == "LEFT" else star_center_right
        )

        master_sky_path = None
        if sky_files_exist.get(channel):
            master_sky_path = layout.master_sky_filepath(
                target.star_name, target.obs_night, target.filter_name,
                channel=channel,
                fixed_user_mastersky=target.fixed_user_mastersky_for(channel),
            )

        pipeline.add_step(CubeAssembler(
            config=config,
            science_files=science_inputs,
            output_file_cube=out_folder / f"{channel.lower()}_master_cube.fits",
            output_file_angs=out_folder
            / f"{channel.lower()}_parallactic_angles.txt",
            shift=shift_yx,
            star_center_file=center_file,
            star_name=target.star_name,
            obs_night=target.obs_night,
            filter_name=target.filter_name,
            master_sky_path=master_sky_path,
        ))

        # Ref stars: reuse the same shift + CENTER file.
        for ref_star in ref_stars:
            ref_target = StarTarget(
                star_name=ref_star,
                obs_night=target.obs_night,
                filter_name=target.filter_name,
            )
            ref_stage2 = layout.stage2_output_folder(
                ref_target.star_name,
                ref_target.obs_night,
                ref_target.filter_name,
            )
            if not ref_stage2.exists():
                logger.warning(
                    "ref star stage2 folder missing: %s; "
                    "skipping ref cube",
                    ref_stage2,
                )
                continue

            ref_inputs = _filter_files_with_prefix(
                ref_stage2, channel, "SCIENCE",
            )
            if not ref_inputs:
                logger.warning(
                    "no ref-star science files in %s; skipping",
                    ref_stage2,
                )
                continue

            ref_out_folder = layout.channel_output_folder(
                ref_target.star_name, ref_target.obs_night,
                ref_target.filter_name, channel=channel,
            )
            ref_out_folder.mkdir(parents=True, exist_ok=True)

            pipeline.add_step(CubeAssembler(
                config=config,
                science_files=ref_inputs,
                output_file_cube=(
                    ref_out_folder / f"{channel.lower()}_master_cube.fits"
                ),
                output_file_angs=(
                    ref_out_folder / f"{channel.lower()}_parallactic_angles.txt"
                ),
                shift=shift_yx,
                star_center_file=center_file,
                star_name=ref_target.star_name,
                obs_night=ref_target.obs_night,
                filter_name=ref_target.filter_name,
                master_sky_path=master_sky_path,
            ))
