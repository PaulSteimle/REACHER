import os
import multiprocessing
import time
import re
import logging
import pandas as pd
import numpy as np 
import shutil

import matplotlib.pyplot as plt
from photutils.centroids import centroid_1dg, centroid_quadratic, centroid_2dg
from os.path import join, basename

from scipy import ndimage
from astropy.io import fits 
from glob import glob 

from vip_hci.preproc import cube_fix_badpix_isolated, cube_recenter_satspots, frame_center_satspots, frame_center_radon, frame_shift, cube_shift, cube_fix_badpix_interp, cube_recenter_dft_upsampling
from vip_hci.var import fit_2dgaussian, frame_center 

from .misc import sanitize_header
from .irdis_filters import get_filter_path 
from .plots import show_img, circular_aperture


def make_bad_pix_mask(dark_file_list, bad_pix_mask_filepath, overwrite, coef=5):
    """Create a bad pixel mask from the dark files. 

    """

    if not overwrite and os.path.exists(bad_pix_mask_filepath):
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(bad_pix_mask_filepath)))
        return

    logging.info('making bad pixel mask '.format(basename(bad_pix_mask_filepath)))

    all_dark_file_data = []
    
    for file in dark_file_list:
        hdul = fits.open(file)
        cube = hdul[0].data
        for image in cube:
            all_dark_file_data.append(image)
        hdul.close()

    all_dark_file_data = np.array(all_dark_file_data)

    sky = np.nanmedian(all_dark_file_data, axis = 0)

    med = np.nanmedian(sky)

    sigma = 1.4826 * np.nanmedian(np.abs(sky - med))
    # the 1.4826 converts the Median Absolute Deviation
    # to the standard deviation for a Normal distribution

    print("Sigma: " + str(sigma) + ", median: " + str(med))

    # Creates a tuple with the x-y positions of the dead pixels
    deadpix = np.where(sky < med - sigma * coef)

    # Creates a tuple with the x-y positions of the dead pixels
    hotpix = np.where(sky > med + sigma * coef)

    badpix = tuple(np.append(np.array(deadpix), np.array(hotpix), axis=1))

    bpx_mask = np.zeros(sky.shape)
    for y,x in zip(badpix[0], badpix[1]):
        bpx_mask[y,x] = 1

    hdu = fits.PrimaryHDU(data=bpx_mask)
    hdul = fits.HDUList([hdu])

    hdul.writeto(bad_pix_mask_filepath, overwrite=True)

    return


def make_IRDIS_clump_mask(clump_mask_filepath):

    excl_mask = np.ones((1024, 2048))
    excl_mask[25:1020, 50:925] = 0
    excl_mask[15:1010, 1080:1955] = 0

    circ_mask1 = circular_aperture(excl_mask, 10, (368, 394))
    circ_mask2 = circular_aperture(excl_mask, 10, (362, 394))
    circ_mask3 = circular_aperture(excl_mask, 10, (356, 394))
    circ_mask4 = circular_aperture(excl_mask, 10, (311, 1400))

    clump_mask = np.zeros(excl_mask.shape)

    clump_mask[circ_mask1 == 1] = 1
    clump_mask[circ_mask2 == 1] = 1
    clump_mask[circ_mask3 == 1] = 1
    clump_mask[circ_mask4 == 1] = 1

    hdu = fits.PrimaryHDU(data=clump_mask)
    hdul = fits.HDUList([hdu])
    hdul.writeto(clump_mask_filepath, overwrite=True)

    return


def correct_bad_pix(input_cube, bad_pix_mask_filepath, clump_mask_filepath, correct_indvidual_badpix=False):
    """Correct bad pixels in a cube using the bad pixel mask and clump mask.
    
    input_cube: 2D or 3D numpy array
        The input image or cube to correct.
    bad_pix_mask_filepath: str
        The file path to the bad pixel mask.
    clump_mask_filepath: str
        The file path to the clump mask.
    correct_indvidual_badpix: bool
        Whether to correct individual bad pixels (True) or only clumps (False).
    
    """


    # check that the input data is a 3d cube, if not make it one
    d2 = False
    if len(np.shape(input_cube)) == 2:
        input_cube = np.array([input_cube])
        d2 = True

    # get the bad_pixel mask
    
    bpm_mask_hdul = fits.open(bad_pix_mask_filepath)
    bpm_mask = bpm_mask_hdul[0].data
    bpm_mask_hdul.close()

    # and the clump mask 
    clump_mask_hdul = fits.open(clump_mask_filepath)
    clump_mask = clump_mask_hdul[0].data
    clump_mask_hdul.close()

    nan_mask = np.ones_like(clump_mask)
    nan_mask[clump_mask == 1] = np.nan
    # correct all for the bad pixels
    # first the indicual bad pixels
    badpix_corr_data = cube_fix_badpix_isolated(input_cube, bpm_mask=bpm_mask, correct_only=True,
                                            frame_by_frame=False, mad=True, verbose=True, ignore_nan=False)

    if correct_indvidual_badpix:
        badpix_corr_data = cube_fix_badpix_isolated(badpix_corr_data, correct_only=False,
                                        frame_by_frame=False, mad=True, verbose=True, ignore_nan=True)

    # then the clumps, and first set the areas to nans to make sure they do not contemiante
    badpix_corr_data *= nan_mask
    clump_corr_data = cube_fix_badpix_interp(badpix_corr_data, bpm_mask=clump_mask, mode='gauss', kernel_sz=35)

    if d2: 
        return clump_corr_data[0] 
    else:
        return clump_corr_data


def make_mastersky(sky_files_list, master_sky_filepath, owerwrite):
    """Create a master sky from a list of sky files.
    
    sky_files_list: list of str
        List of file paths to the sky files.
    master_sky_filepath: str
        The file path to save the master sky.
    overwrite: bool     
        Whether to overwrite the master sky file if it exists.
        
    """ 

    if not owerwrite and os.path.exists(master_sky_filepath):
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(master_sky_filepath)))
        return
    
    #check for sky data 
    sky_data = []
    sky_temps = []
    for file in sky_files_list:
        hdul_sky = fits.open(file)
        sky_temps.append(hdul_sky[0].header['HIERARCH ESO INS4 TEMP449 VAL'])

        if hdul_sky[0].data.ndim == 2:
            sky_data.append(hdul_sky[0].data)

        elif hdul_sky[0].data.ndim == 3:
            logging.info('\t\t -> sky data exists of multiple exposures, combining to median file')
            sky_data.append(np.nanmedian(hdul_sky[0].data, axis=0))
        
        hdul_sky.close()

    sky_data = np.array(sky_data)
    median_sky = np.nanmedian(sky_data, axis=0)

    hdu = fits.PrimaryHDU(data=median_sky)
    hdul = fits.HDUList([hdu])
    hdul[0].header = hdul[0].header

    hdul[0].header['MEANTEMP'] = np.mean(sky_temps)
    for temps in sky_temps:
        hdul[0].header['SKYTEMP'] = temps
    hdul[0].header['HISTORY'] = 'MASTER SKY'

    hdul.writeto(master_sky_filepath, overwrite=True)
    hdul.close()

    logging.info('created master sky {}'.format(os.path.basename(master_sky_filepath)))

    return 


