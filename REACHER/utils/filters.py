"""ND filter / IRDIS filter transmission utilities.

Refactor of :func:`REACHER.routines.get_ratio_for_nd_filter`. The ratio is
the fraction of the IRDIS filter integral that survives the neutral
density (ND) filter. It is used by :class:`FluxCalibrator` to rescale flux
calibration frames.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..exceptions import InvalidConfigError
from ..irdis_filters import get_filter_path


def _resolve_filter_name(wave_filter: str) -> str:
    """Translate an IRDIS combined filter name to the .dat file stem.

    The legacy code maps ``DB_*`` -> ``D_*`` and ``BB_*`` -> ``B_*`` so
    that ``SPHERE_IRDIS_<filter>.dat`` is the on-disk filename.
    """
    if "DB" in wave_filter:
        wave_filter = wave_filter.replace("DB", "D")
    if "BB" in wave_filter:
        wave_filter = wave_filter.replace("BB", "B")
    return wave_filter


def _load_irdis_transmission(
    wave_filter: str, channel: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the IRDIS transmission profile for the requested channel.

    Returns
    -------
    trans_wave, trans : np.ndarray
        Wavelength grid (Angstroms) and transmission for the given
        channel.
    """
    wave_filter = _resolve_filter_name(wave_filter)
    path = get_filter_path(f"SPHERE_IRDIS_{wave_filter}.dat")
    raw = np.loadtxt(path).transpose()

    if raw.shape[0] == 3:
        trans_wave = raw[0]
        trans1 = raw[1]
        trans2 = raw[2]
        if channel == "LEFT":
            trans = trans1
        elif channel == "RIGHT":
            trans = trans2
        else:
            raise InvalidConfigError(
                f"channel must be 'LEFT' or 'RIGHT'; got {channel!r}"
            )
    elif raw.shape[0] == 2:
        trans_wave, trans = raw[0], raw[1]
    else:
        raise InvalidConfigError(
            f"Unexpected SPHERE_IRDIS filter file shape {raw.shape}"
        )

    return trans_wave, trans


def get_ratio_for_nd_filter(
    wave_filter: str, nd_filter: str, channel: str
) -> float:
    """Compute the ND filter transmission ratio for the IRDIS filter.

    Parameters
    ----------
    wave_filter
        Combined filter name (read from
        ``InstrumentConfig.keys.filter_name`` -- the SPHERE default
        is ``HIERARCH ESO INS COMB IFLT``), e.g. ``BB_H``, ``DB_H23``.
    nd_filter
        ND filter ID (read from
        ``InstrumentConfig.keys.nd_filter_id`` -- the SPHERE default
        is ``HIERARCH ESO INS4 FILT2 ID``), e.g. ``ND_1.0``,
        ``OPEN`` or ``ND``. The numeric suffix after the last
        underscore selects the column from ``SPHERE_CPI_ND.dat``
        (one of ``0.0`` / ``1.0`` / ``2.0`` / ``3.5``).
    channel
        ``LEFT`` or ``RIGHT`` IRDIS detector channel. Only used when
        the filter file has dual transmission columns (DB filters).

    Returns
    -------
    float
        ``integral(filter * ND_profile) / integral(filter)`` over the
        wavelength range common to both.

    Notes
    -----
    The legacy implementation samples both profiles at 1 Angstrom so
    the integral is the simple sum of the transmission values. We
    preserve that exactly — :func:`np.nansum` over the masked, sampled
    arrays.
    """
    trans_wave, trans = _load_irdis_transmission(wave_filter, channel)

    nd_path = get_filter_path("SPHERE_CPI_ND.dat")
    nd_df = pd.DataFrame(
        np.loadtxt(nd_path),
        columns=["wavelength", "ND0.0", "ND1.0", "ND2.0", "ND3.5"],
    )

    nd_wave = nd_df["wavelength"].values
    nd_column = "ND" + nd_filter.split("_")[-1]
    if nd_column not in nd_df.columns:
        raise InvalidConfigError(
            f"Unknown ND filter column {nd_column!r} for ND filter "
            f"id {nd_filter!r}"
        )
    nd_trans = nd_df[nd_column].values

    start = max(nd_wave.min(), trans_wave.min())
    end = min(nd_wave.max(), trans_wave.max())

    mask_a = (nd_wave >= start) & (nd_wave <= end)
    mask_b = (trans_wave >= start) & (trans_wave <= end)

    nd_trans_cut = nd_trans[mask_a]
    trans_cut = trans[mask_b]

    convoluted_trans = nd_trans_cut * trans_cut

    trans_integrate = np.nansum(trans_cut)
    convoluted_trans_integrate = np.nansum(convoluted_trans)

    if trans_integrate == 0:
        raise InvalidConfigError(
            f"IRDIS filter integral is zero for {wave_filter!r} ({channel})"
        )

    return float(convoluted_trans_integrate / trans_integrate)
