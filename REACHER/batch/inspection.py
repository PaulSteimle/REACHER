"""Inspection + cleanup batch helpers.

Phase 4 replacement for the legacy
:func:`misc.grab_all_files` /
:func:`misc.return_csv_files` /
:func:`master_routines.cleanup_all_folders` /
:func:`misc.list_directory` helpers. Architecture Rule 5 forbids
``os.system()``: cleanup uses :func:`shutil.rmtree` exclusively.
"""
from __future__ import annotations

import logging
import shutil
from glob import glob
from pathlib import Path
from typing import Iterable, List

from ..config import ReacherConfig
from ..exceptions import InvalidConfigError
from ..discovery.dataframes import iter_star_targets
from ..discovery.layout import DataLayout, list_directory


logger = logging.getLogger(__name__)


def _check_config(config: ReacherConfig) -> None:
    if config is None:
        raise InvalidConfigError("config must not be None")


# ----------------------------------------------------------------------
# centers.csv distribution
# ----------------------------------------------------------------------
def grab_all_files(
    config: ReacherConfig,
    stars: Iterable[str],
) -> List[Path]:
    """Copy ``centers.csv`` + recentering plots into the recentering folder.

    For each ``(star, obs_night, filter)`` combination that has a
    ``stage2/centers.csv`` file, the file (and its associated
    ``recentering_pngs/*.png`` quick-looks) are copied into a flat
    inspection-friendly directory at
    ``reduced_output_dir/recentering/<star>_<obs_night>_<filter>/``.

    Parameters
    ----------
    config
        :class:`ReacherConfig`.
    stars
        Iterable of star names to grab.

    Returns
    -------
    list of Path
        The destination directories created (one per
        ``(star, obs_night, filter)``).
    """
    _check_config(config)
    layout = DataLayout(config.paths)
    layout.recentering_folder.mkdir(parents=True, exist_ok=True)
    logger.info("Grabbing all files for stars: %s", list(stars))
    logger.info(
        "Output folder for recentering files: %s",
        layout.recentering_folder,
    )

    created: List[Path] = []
    for star in stars:
        star_parent = layout.reduced_output_dir / star / "IRDIS"
        if not star_parent.exists():
            logger.warning("no reduced output for %s; skipping", star)
            continue
        for obs_night in sorted(list_directory(star_parent)):
            obs_dir = star_parent / obs_night
            if not obs_dir.is_dir():
                continue
            for filter_name in sorted(list_directory(obs_dir)):
                filt_dir = obs_dir / filter_name
                if not filt_dir.is_dir():
                    continue

                stage2 = filt_dir / "stage2"
                center_csv = stage2 / "centers.csv"
                if not center_csv.exists():
                    continue

                dest_dir = (
                    layout.recentering_folder
                    / f"{star}_{obs_night}_{filter_name}"
                )
                dest_dir.mkdir(parents=True, exist_ok=True)
                created.append(dest_dir)

                shutil.copy2(center_csv, dest_dir / "centers.csv")
                logger.info("\t -> copied %s centers.csv", star)

                pngs = sorted(
                    glob(str(stage2 / "recentering_pngs" / "*.png"))
                ) + sorted(
                    glob(str(stage2 / "recentering_pngs" / "*.pdf"))
                )
                for png in pngs:
                    shutil.copy2(png, dest_dir / Path(png).name)
                    logger.info(
                        "\t -> copied %s to %s",
                        Path(png).name, dest_dir.name,
                    )

    return created


def return_csv_files(
    config: ReacherConfig,
    stars: Iterable[str],
) -> List[Path]:
    """Re-distribute hand-edited ``centers.csv`` files back into ``stage2``.

    Inverse of :func:`grab_all_files`. After the operator has hand-edited
    the satellite-spot rows in
    ``recentering/<star>_<obs_night>_<filter>/centers.csv`` this helper
    copies the edited file back to
    ``<star>/IRDIS/<obs_night>/<filter>/stage2/centers.csv`` so the
    cube-assembly step picks up the updated centers.

    Parameters
    ----------
    config
        :class:`ReacherConfig`.
    stars
        Iterable of star names to return.

    Returns
    -------
    list of Path
        The ``stage2/centers.csv`` files that were written.
    """
    _check_config(config)
    layout = DataLayout(config.paths)
    logger.info("Re-distributing centers.csv for stars: %s", list(stars))

    written: List[Path] = []
    for star in stars:
        star_parent = layout.reduced_output_dir / star / "IRDIS"
        if not star_parent.exists():
            logger.warning("no reduced output for %s; skipping", star)
            continue
        for obs_night in sorted(list_directory(star_parent)):
            obs_dir = star_parent / obs_night
            if not obs_dir.is_dir():
                continue
            for filter_name in sorted(list_directory(obs_dir)):
                filt_dir = obs_dir / filter_name
                if not filt_dir.is_dir():
                    continue

                src = (
                    layout.recentering_folder
                    / f"{star}_{obs_night}_{filter_name}"
                    / "centers.csv"
                )
                if not src.exists():
                    logger.warning(
                        "no edited centers.csv at %s; skipping", src
                    )
                    continue

                stage2 = filt_dir / "stage2"
                stage2.mkdir(parents=True, exist_ok=True)
                dst = stage2 / "centers.csv"
                shutil.copy2(src, dst)
                written.append(dst)
                logger.info("\t -> re-copied %s centers.csv", star)

    return written


# ----------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------
def cleanup_all_folders(config: ReacherConfig) -> None:
    """Remove calibration / stage1 / stage2 directories per the config flags.

    Two cleanup flags drive this function (preserved from the legacy
    behaviour):

    * ``config.processing.clean_calibrations`` -- removes
      ``master_dark/``, ``master_flat/``, ``master_background/``,
      ``bad_pix_masks/``, ``recentering/`` if present.
    * ``config.processing.clean_stage_1_and_2`` -- removes per-target
      ``stage1/`` and ``stage2/`` directories for every
      ``(star, obs_night, filter)`` triple under ``raw_input_dir``.

    Both are off by default. ``shutil.rmtree`` (NOT ``os.system('rm -r')``)
    is used per Architecture Rule 5.

    Parameters
    ----------
    config
        :class:`ReacherConfig`. Must expose
        ``processing.clean_calibrations`` and
        ``processing.clean_stage_1_and_2`` attributes; missing flags
        default to ``False``.
    """
    _check_config(config)
    layout = DataLayout(config.paths)

    clean_calibrations = getattr(
        config.processing, "clean_calibrations", False
    )
    clean_stage_1_and_2 = getattr(
        config.processing, "clean_stage_1_and_2", False
    )

    if clean_calibrations:
        logger.info("cleaning up all calibration folders")
        for folder in (
            layout.recentering_folder,
            layout.master_dark_folder,
            layout.master_flat_folder,
            layout.bad_pix_mask_folder,
            layout.master_background_folder,
        ):
            if folder.exists():
                logger.info("\t -> removing %s", folder)
                shutil.rmtree(folder)

    if clean_stage_1_and_2:
        logger.info("cleaning up all stage 1 and 2 folders")
        # Need the full list of stars under the *reduced* tree (the
        # legacy used the *raw* tree but stage1/stage2 only exist under
        # the reduced tree, so we walk the reduced tree here).
        for star_dir in layout.reduced_output_dir.glob("*/IRDIS/*/*"):
            stage1 = star_dir / "stage1"
            stage2 = star_dir / "stage2"
            if stage1.exists():
                logger.info("\t -> removing %s", stage1)
                shutil.rmtree(stage1)
            if stage2.exists():
                logger.info("\t -> removing %s", stage2)
                shutil.rmtree(stage2)
