"""Pre-pipeline utility for organizing raw SPHERE FITS downloads.

The ESO archive delivers all SPHERE FITS files into a single flat
directory. This utility reads each file's headers, derives the
``<object>/IRDIS/<obs_night>/<filter>/<type>_<basename>.fits`` layout
that :mod:`REACHER.discovery` expects, and copies each file into place.

Phase 4 home of the legacy ``REACHER/download_data.py`` module retired
in Phase 3. The algorithm is preserved verbatim; only variable names
were cleaned up to avoid shadowing ``type`` / ``object``.
"""
from __future__ import annotations

import logging
import os
import shutil
from glob import glob
from typing import Mapping, Optional, Sequence, Union

import pandas as pd
from astropy.io import fits


logger = logging.getLogger(__name__)


def _build_layout_dataframe(
    files: Union[str, Sequence[str]],
    rename_dict: Mapping[str, str],
) -> Optional[pd.DataFrame]:
    """Return a DataFrame of `(orig_file, dest_filepath)` rows.

    Reads each FITS header to extract object, exposure time, observation
    night, filter, ND filter, and DPR type, then derives the destination
    path under the layout the discovery layer reads.
    """
    if isinstance(files, str):
        files = [files]

    rows = []

    for file in files:
        header = fits.getheader(file)

        try:
            obj = header["OBJECT"]
            if obj.strip() == "":
                continue

            exp_time = header["EXPTIME"]
            obs_date = header["DATE-OBS"].split("T")[0]
            obs_time = header["DATE-OBS"].split("T")[1]

            # Observations before 14:00 local belong to the previous night.
            if int(obs_time[:2]) < 14:
                obs_night = obs_date[:8] + str(int(obs_date[8:11]) - 1).zfill(2)
            else:
                obs_night = obs_date

            filter_name = header["HIERARCH ESO INS COMB IFLT"]
            nd_filter = header["HIERARCH ESO INS4 FILT2 NAME"]
            dpr_type = header["HIERARCH ESO DPR TYPE"]
        except KeyError as exc:
            logger.warning("KeyError %s in file %s; skipping", exc, file)
            continue

        if dpr_type == "OBJECT":
            dpr_type = "SCIENCE"
        elif "OBJECT" in dpr_type:
            dpr_type = dpr_type.replace("OBJECT,", "")
        if "DARK,BACKGROUND" in dpr_type:
            dpr_type = dpr_type.replace("DARK,BACKGROUND", "DARK")
        if "DARK,BACKGROUND" in obj:
            obj = obj.replace("DARK,BACKGROUND", "DARK")

        for token in ("DARK,", ",LAMP", "LAMP,"):
            if token in dpr_type:
                dpr_type = dpr_type.replace(token, "")
            if token in obj:
                obj = obj.replace(token, "")

        obj = obj.replace(" ", "")

        if obj in rename_dict:
            obj = rename_dict[obj]

        folderpath = os.path.join(obj, "IRDIS", obs_night, filter_name)
        filepath = os.path.join(folderpath, f"{dpr_type}_{os.path.basename(file)}")

        rows.append({
            "orig_file": file,
            "object": obj,
            "exp_time": exp_time,
            "obs_night": obs_night,
            "filter_name": filter_name,
            "nd_filter": nd_filter,
            "type": dpr_type,
            "filepath": filepath,
            "folderpath": folderpath,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["object", "obs_night", "filter_name", "exp_time", "orig_file"],
        ascending=[True, True, True, True, True],
    )
    return df


def move_files_from_download_dir(
    download_directory: str,
    new_directory: str,
    rename_dict: Mapping[str, str],
) -> None:
    """Copy raw SPHERE FITS files into the REACHER on-disk layout.

    Globs ``download_directory/SPHER*.fits``, reads each header to derive
    ``<object>/IRDIS/<obs_night>/<filter>/<type>_<basename>.fits``, and
    copies each file to that location under ``new_directory``. Object
    names matching a key in ``rename_dict`` are renamed before the path
    is built. Files whose ``OBJECT`` header resolves to ``'OBJECT'`` are
    skipped (matches the legacy filter).

    Parameters
    ----------
    download_directory
        Flat directory holding ESO archive ``SPHER*.fits`` downloads.
    new_directory
        Root under which the structured layout is created.
    rename_dict
        Mapping from raw header object name to canonical star name.
    """
    all_files = glob(os.path.join(download_directory, "SPHER*.fits"))
    df = _build_layout_dataframe(all_files, rename_dict)
    if df is None:
        logger.warning("no FITS files found under %s", download_directory)
        return

    df = df[df["object"] != "OBJECT"]

    for directory in df["folderpath"].unique():
        os.makedirs(os.path.join(new_directory, directory), exist_ok=True)

    for i, (orig_file, filepath) in enumerate(
        zip(df["orig_file"], df["filepath"].to_list())
    ):
        if i % 100 == 0:
            logger.info("copied %d / %d", i, len(df))
        shutil.copy2(orig_file, os.path.join(new_directory, filepath))

    logger.info("copied %d files into %s", len(df), new_directory)
