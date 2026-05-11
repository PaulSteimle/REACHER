"""Satellite-spot based recentering utility.

This is the Phase-3 home of :func:`REACHER.routines.get_center`. It is a
**utility**, not a :class:`PipelineStep`, because:

* it has no FITS file output (returns a tuple of floats consumed by the
  caller), so the ``_skip_if_exists`` resumability contract has nothing
  to gate on;
* it has no side effects beyond the optional plotting branch, which is
  already gated on ``config.processing.make_plots``;
* it is a one-shot compute called once per LEFT/RIGHT channel from the
  cube-assembly orchestration. Wrapping it in a step would force a
  contrived per-call file output that downstream code immediately
  reloads.

The decision is documented in ``STATUS.md`` (Phase 3 section).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits

from ..exceptions import StepExecutionError


_CENTERS_FILENAME = "centers.csv"


def _load_satspots_from_csv(input_file: Path) -> Optional[list]:
    """Load the four ``(x, y)`` satellite-spot positions for ``input_file``.

    Mirrors :func:`REACHER.routines.get_satspots`. Returns ``None`` when
    the CSV row is missing or contains the placeholder zeros that
    indicate the operator has not yet hand-edited the row.
    """
    csv_path = input_file.parent / _CENTERS_FILENAME
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    matches = df[df["frames"] == input_file.name]
    if matches.empty:
        return None

    centers = [
        (
            float(matches[f"x{f + 1}"].values[0]),
            float(matches[f"y{f + 1}"].values[0]),
        )
        for f in range(4)
    ]
    return centers


def get_center(
    input_file: str | Path,
    *,
    make_plots: bool = False,
    center: Optional[Tuple[float, float]] = None,
    cut_size: int = 20,
) -> Tuple[float, float]:
    """Compute the (y, x) shift that recenters a science cube.

    Parameters
    ----------
    input_file
        Path to a 3D FITS cube (typically a CENTER frame). The matching
        ``centers.csv`` (in the same directory) is consulted for the
        four satellite-spot ``(x, y)`` positions.
    make_plots
        Render the legacy quick-look figures. Off by default.
    center
        Optional initial (x, y) center; if ``None`` a 1D-Gaussian
        centroid is computed from the ``sigma=10`` blurred median image.
    cut_size
        Number of edge pixels to crop before centroiding (legacy default
        = 20).

    Returns
    -------
    (center_shifty, center_shiftx)
        The mean ``(y, x)`` shifts produced by VIP's
        :func:`cube_recenter_satspots`.

    Raises
    ------
    StepExecutionError
        If the input cube is missing, unreadable, or not 3D.

    Notes
    -----
    The legacy implementation lives in ``routines.get_center``; the
    algorithm is preserved verbatim. The numerical heart
    (cube_recenter_satspots) and centroid call sites import VIP / scipy
    / photutils lazily so this module can be imported in a stripped
    test environment.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise StepExecutionError(f"get_center input missing: {input_path}")

    try:
        data = np.nan_to_num(fits.getdata(input_path))
    except Exception as exc:
        raise StepExecutionError(
            f"get_center could not read {input_path.name}: {exc}"
        ) from exc

    if data.ndim != 3:
        raise StepExecutionError(
            f"get_center expects a 3D cube; got ndim={data.ndim} "
            f"(shape={data.shape}) for {input_path.name}"
        )

    sat_spots = _load_satspots_from_csv(input_path)

    # Crop edges to remove bad pixels (legacy behaviour).
    data = data[:, cut_size:-cut_size, cut_size:-cut_size]
    image = np.nanmedian(data, axis=0)

    # Resolve the initial center (lazy imports).
    if center is None:
        from photutils.centroids import centroid_1dg
        from scipy import ndimage

        data_blurred = ndimage.gaussian_filter(image.astype(float), sigma=10)
        center = centroid_1dg(data_blurred)

    if sat_spots is None or sat_spots == [(0, 0), (0, 0), (0, 0), (0, 0)]:
        # No hand-edited spots in centers.csv -> fall back to defaults
        # relative to the centroid.
        sat_spots = (
            np.array([[-35, 35], [35, 35], [-35, -35], [35, -35]])
            + np.asarray(center)
        )
        sat_spots_input = [(float(f[0]), float(f[1])) for f in sat_spots]
    else:
        # Account for the cut_size crop that we applied above.
        sat_spots_input = [
            (f[0] - cut_size, f[1] - cut_size) for f in sat_spots
        ]

    logging.info(
        "get_center: using satellite spots %s for %s",
        sat_spots_input, input_path.name,
    )

    from vip_hci.preproc import cube_recenter_satspots

    res = cube_recenter_satspots(
        data,
        xy=sat_spots_input,
        subi_size=21,
        verbose=False,
        debug=False,
        full_output=True,
        fit_type="gaus",
    )
    center_shifty = float(np.mean(res[1]))
    center_shiftx = float(np.mean(res[2]))
    true_center = (float(np.mean(res[3])), float(np.mean(res[4])))

    logging.info(
        "get_center: %s -> shift y=%s x=%s; recovered center=%s",
        input_path.name, center_shifty, center_shiftx, true_center,
    )

    if make_plots:
        _make_plots(image, res, sat_spots_input, center)

    return center_shifty, center_shiftx


def _make_plots(image, res, sat_spots_input, center):  # pragma: no cover
    """Render the legacy diagnostic figures (matplotlib lazy import)."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from vip_hci.var import frame_center

    from ..plots import show_img

    shifted_image = np.nanmedian(res[0], axis=0)
    cy, cx = frame_center(image)
    true_center = (float(np.mean(res[3])), float(np.mean(res[4])))

    show_img(
        image, minper=0.1, maxper=99.9,
        xlim=[center[0] - 100, center[0] + 100],
        ylim=[center[1] - 100, center[1] + 100], size=2,
    )
    plt.scatter(center[0], center[1], color="black", s=10,
                label="input center", marker="+")
    plt.scatter(true_center[1], true_center[0], color="black", s=10,
                label="true center", marker="x")
    plt.scatter(cx, cy, color="black", s=10, label="center frame", marker="*")
    for j, xy in enumerate(sat_spots_input):
        plt.scatter(xy[0], xy[1], color=f"C{j}", s=10, label=f"xy {j}")
    plt.legend()
    plt.close()

    show_img(
        shifted_image, minper=0.1, maxper=99.9,
        xlim=[cx - 100, cx + 100], ylim=[cy - 100, cy + 100], size=2,
    )
    plt.scatter(cx, cy, color="black", s=10, label="center frame", marker="+")
    plt.legend()
    plt.close()
