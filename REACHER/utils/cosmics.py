"""Cosmic-ray detection and correction utilities.

Refactor of the legacy :func:`REACHER.routines.cosmic_ray_detect`,
:func:`clean_bp` and :func:`correct_cosmic_rays_multiprocessing`. The
legacy ``clean_bp`` imported the optional ``bottleneck`` package; we
replace those calls with :func:`numpy.nanmedian` per Architecture Rule
7 (bottleneck dependency removed).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

try:
    from bottleneck import median
    from bottleneck import nanmedian
except ImportError:
    logging.warning(
        "Bottleneck not found; falling back to numpy.nanmedian (this may be slower)"
    )
    from numpy import nanmedian, median


####
# These are new claude routines which I don't trust

####
# def cosmic_ray_detect(
#     image: np.ndarray, n_sigma: float, box_radius: int
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Detect cosmic-ray hits via local MAD-based sigma clipping.

#     Preserves the algorithm of :func:`routines.cosmic_ray_detect`. A
#     pixel is flagged when its deviation from the local median exceeds
#     ``n_sigma * 1.4826 * MAD``.

#     Returns
#     -------
#     tuple of np.ndarray
#         ``(y_indices, x_indices)`` of the cosmic-ray pixels (the legacy
#         ``np.where`` output shape).
#     """
#     box_radius = int(box_radius)
#     box_size = int(2 * box_radius + 1)

#     median_vals = ndimage.median_filter(image, size=box_size)

#     n_stdev_vals = box_size ** 2 - 1
#     stdev_array = np.zeros((image.shape[0], image.shape[1], n_stdev_vals))
#     shift_index = 0
#     for yshift in np.arange(-box_radius, box_radius + 1):
#         for xshift in np.arange(-box_radius, box_radius + 1):
#             if xshift == 0 and yshift == 0:
#                 continue
#             shifted_image = np.roll(image, (yshift, xshift), axis=(0, 1))
#             stdev_array[:, :, shift_index] = shifted_image - median_vals
#             shift_index += 1

#     med_abs_dev = np.nanmedian(np.abs(stdev_array), axis=2)
#     med_abs_dev[med_abs_dev == 0] = 1e-10
#     inv_std_dev = 1.0 / (med_abs_dev * 1.4826)
#     n_sig_array = (image - median_vals) * inv_std_dev
#     bad_array = np.abs(n_sig_array) > n_sigma

#     cosmic_rays = np.where(bad_array)
#     n_bad = int(np.sum(bad_array))
#     logging.info(
#         "%s cosmic rays detected using n_sigma=%s", n_bad, n_sigma
#     )
#     return cosmic_rays


# def clean_bad_pixels(
#     badpix: Tuple[np.ndarray, np.ndarray], cube: np.ndarray
# ) -> np.ndarray:
#     """Replace each bad pixel with the median of its 3x3 neighborhood.

#     This is a numpy-only replacement for the legacy ``clean_bp`` which
#     relied on the optional ``bottleneck`` package (per Architecture
#     Rule 7 we removed that dependency). Behavior matches the legacy:

#     * ``cube`` is expected to be 3D (``n_frames, h, w``).
#     * The bad pixel coordinates apply to every frame; each frame is
#       treated independently.
#     * Boundary pixels use the available neighbors only.
#     """
#     if cube.ndim != 3:
#         raise ValueError(
#             f"clean_bad_pixels expects a 3D cube; got ndim={cube.ndim}"
#         )
#     cube_out = cube.astype(float, copy=True)
#     bad_y, bad_x = badpix

#     if len(bad_y) == 0:
#         return cube_out

#     # NaN out the bad pixels first so they don't poison the median.
#     cube_out[:, bad_y, bad_x] = np.nan

#     h, w = cube_out.shape[1], cube_out.shape[2]
#     for f in range(cube_out.shape[0]):
#         for j in range(len(bad_y)):
#             y, x = int(bad_y[j]), int(bad_x[j])
#             y0 = max(0, y - 1)
#             y1 = min(h, y + 2)
#             x0 = max(0, x - 1)
#             x1 = min(w, x + 2)
#             window = cube_out[f, y0:y1, x0:x1]
#             cube_out[f, y, x] = np.nanmedian(window)

#     return cube_out

def cosmic_ray_detect(image: np.ndarray, n_sigma: float, box_radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Cosmic ray detection function. This routine is taken from GRAPHIC (Holmberg+2016)
    Works by comparing each pixel to the local median absolute deviation.
    If the pixel value is more than n_sigma * MAD from the local median, it is called bad
    """

    n_sigma = n_sigma
    box_radius = int(box_radius)
    box_size = int(2 * box_radius + 1) #make sure width is odd so the target pixel is centred

    # Get the median value in a box around each pixel (to use to calculate the MAD)
    median_vals = ndimage.median_filter(image, size=box_size)

    # Rather than loop through pixels, loop through the shifts and calculate
    #  the deviation for all pixels in the image at the same time
    n_stdev_vals = box_size**2 - 1  # We will ignore the centre pixel
    stdev_array = np.zeros((image.shape[0], image.shape[1], n_stdev_vals))
    shift_index = 0
    for yshift in np.arange(-box_radius, box_radius + 1):
        for xshift in np.arange(-box_radius, box_radius + 1):

            # Don't include the pixel in the MAD calculation
            if xshift == 0 and yshift == 0:
                continue

            shifted_image = np.roll(image, (yshift, xshift), axis=(0, 1))
            stdev_array[:, :, shift_index] = (shifted_image - median_vals)

            shift_index += 1

    med_abs_dev = np.nanmedian(np.abs(stdev_array), axis=2)

    med_abs_dev[med_abs_dev == 0] = 1e-10  # avoid division by zero warnings

    inv_std_dev = 1.0 / (med_abs_dev * 1.4826)  # this number is to convert MAD to std. deviation
    n_sig_array = (image - median_vals) * inv_std_dev
    bad_array = np.abs(n_sig_array) > n_sigma

    
    # In case we want to check the bad pixels that we detected:
    # pyfits.writeto('cosmic_ray_array.fits',np.abs(n_sig_array),overwrite=True)

    cosmic_rays = np.where(bad_array)
    n_bad = np.sum(bad_array)
    logging.info('\t -> {} cosmic rays detected using n_sigma= {}'.format(n_bad, n_sigma))
    
    comsic_ray_mask = np.zeros(image.shape)
    for y,x in zip(cosmic_rays[0],cosmic_rays[1]):
        comsic_ray_mask[y,x] = 1

    return cosmic_rays


def clean_bad_pixels(badpix: Tuple[np.ndarray, np.ndarray], cube: np.ndarray) -> np.ndarray:
    ''' This routine is taken from GRAPHIC (Holmberg+2016)
    Function to clean bad pixels in a cube using the median of surrounding pixels .
    '''

    cube[:, badpix[0], badpix[1]] = np.NaN
    for f in range(cube.shape[0]):

        for j in range(len(badpix[0])):
            y = badpix[0][j]
            x = badpix[1][j]

            if y == cube.shape[1] - 1:  # At the image edge !!!
                if x == cube.shape[2] - 1:  # In a corner
                    cube[f, y, x] = nanmedian([
                            cube[f, y - 1, x - 1], cube[f, y - 1, x],
                            cube[f, y, x - 1]
                    ])
                elif x == 0:
                    cube[f, y, x] = nanmedian([
                            cube[f, y - 1, x], cube[f, y - 1, x + 1],
                            cube[f, y, x + 1]
                    ])
                else:  # Along the edge
                    cube[f, y, x] = nanmedian([
                            cube[f, y - 1, x - 1], cube[f, y - 1, x],
                            cube[f, y - 1, x + 1], cube[f, y, x - 1],
                            cube[f, y, x + 1]
                    ])
            elif y == 0:  # At the image edge !!!
                if x == cube.shape[2] - 1:  # In a corner
                    cube[f, y, x] = nanmedian([
                            cube[f, y, x - 1], cube[f, y + 1, x - 1],
                            cube[f, y + 1, x]
                    ])
                elif x == 0:  # In a corner
                    cube[f, y, x] = nanmedian([
                            cube[f, y, x + 1], cube[f, y + 1, x],
                            cube[f, y + 1, x + 1]
                    ])
                else:  # Along the edge
                    cube[f, y, x] = nanmedian([
                            cube[f, y, x - 1], cube[f, y, x + 1],
                            cube[f, y + 1, x - 1], cube[f, y + 1, x],
                            cube[f, y + 1, x + 1]
                    ])
            elif x == 0:  # Along the edge
                cube[f, y, x] = nanmedian([
                        cube[f, y - 1, x], cube[f, y - 1, x + 1],
                        cube[f, y, x + 1], cube[f, y + 1, x], cube[f, y + 1, x + 1]
                ])
            elif x == cube.shape[2] - 1:  # Along the edge
                cube[f, y, x] = nanmedian([
                        cube[f, y - 1, x - 1], cube[f, y - 1, x],
                        cube[f, y, x - 1], cube[f, y + 1, x - 1], cube[f, y + 1, x]
                ])
            else:  # Usual case, not on an edge
                cube[f, y, x] = nanmedian([
                        cube[f, y - 1, x - 1], cube[f, y - 1, x],
                        cube[f, y - 1, x + 1], cube[f, y, x - 1],
                        cube[f, y, x + 1], cube[f, y + 1, x - 1],
                        cube[f, y + 1, x], cube[f, y + 1, x + 1]
                ])
                
    return cube


def correct_cosmic_rays_frame(
    image: np.ndarray, n_sigma: float, box_radius: int
) -> np.ndarray:
    """Detect and replace cosmic-ray hits in a single 2D image."""
    cosmic_mask = cosmic_ray_detect(image, n_sigma, box_radius)
    cleaned = clean_bad_pixels(cosmic_mask, np.array([image]))[0]
    return cleaned


def correct_cosmic_rays_cube(
    cube: np.ndarray,
    n_sigma: float,
    box_radius: int,
    nproc: Optional[int] = None,
) -> np.ndarray:
    """Multi-process wrapper to correct cosmic rays frame by frame."""
    if nproc is None:
        nproc = mp.cpu_count() // 2 or 1

    if nproc <= 1:
        return np.array([
            correct_cosmic_rays_frame(frame, n_sigma, box_radius)
            for frame in cube
        ])

    args = list(zip(
        list(cube),
        [n_sigma] * len(cube),
        [box_radius] * len(cube),
    ))
    with mp.Pool(processes=nproc) as pool:
        results = pool.starmap(correct_cosmic_rays_frame, args)
    return np.array(results)
