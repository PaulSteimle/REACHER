"""Bad-pixel interpolation utilities.

The ``interpolate_bad_pixels`` function lives here (rather than under
``REACHER/steps/``) because it is consumed by multiple step classes
(BadPixelMaskCreator, MasterDarkCreator, MasterFlatCreator,
FileCalibrator). Promoting it to ``utils`` removes the cross-step import
chain established during Phase 2.
"""
from __future__ import annotations

import numpy as np

from ..exceptions import InvalidConfigError


def interpolate_bad_pixels(
    image: np.ndarray, bad_pixel_mask: np.ndarray
) -> np.ndarray:
    """Interpolate bad pixels using the median of 3x3 neighbors.

    Parameters
    ----------
    image
        2D image to correct.
    bad_pixel_mask
        2D mask with 1 for bad pixels.

    Returns
    -------
    np.ndarray
        Corrected image.

    Raises
    ------
    InvalidConfigError
        If the inputs are not 2D or if their shapes do not match.
    """

    if image.ndim != 2 or bad_pixel_mask.ndim != 2:
        raise InvalidConfigError(
            "interpolate_bad_pixels expects 2D arrays"
        )
    if image.shape != bad_pixel_mask.shape:
        raise InvalidConfigError(
            "Image and bad pixel mask shapes must match"
        )

    corrected = image.copy()
    bad_indices = np.argwhere(bad_pixel_mask > 0)
    for y, x in bad_indices:
        y0 = max(0, y - 1)
        y1 = min(image.shape[0], y + 2)
        x0 = max(0, x - 1)
        x1 = min(image.shape[1], x + 2)
        window = image[y0:y1, x0:x1]
        corrected[y, x] = np.nanmedian(window)
    return corrected
