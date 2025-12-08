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
from . import routines


def make_bad_pix_mask_wrapper(obs_run, params):
    """ Wrapper function to create a bad pixel mask from the dark files. It is important that the parameters object contains the correct DARK_DF dataframe.
    
    obs_run : str
        The observation run to create the bad pixel mask for.
    params : Parameters 
        The parameters object containing all settings. 

    """
    params.bad_pix_mask_filepath = os.path.join(params.bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_run))
    darks = params.DARK_DF[(params.DARK_DF['obs_run']== obs_run)]['filepath']
    
    routines.make_bad_pix_mask(darks.tolist(), params.bad_pix_mask_filepath, params.overwrite, coef=5)
    return


def make_IRDIS_clump_mask_wrapper(params):
    """ Wrapper function to create the IRDIS clump mask. It is important that the parameters object contains the correct bad_pix_mask_filepath."""
    routines.make_IRDIS_clump_mask(params.clump_mask_filepath)
    return


def correct_bad_pix_wrapper(input_cube, params, correct_indvidual_badpix=False):
    """ Wrapper function to correct bad pixels in a cube. It is important that the parameters object contains the correct bad_pix_mask_filepath and clump_mask_filepath."""
    return routines.correct_bad_pix(input_cube, params.bad_pix_mask_filepath, params.clump_mask_filepath, correct_indvidual_badpix) 


def make_master_dark_wrapper(dark_filepaths, params):
    """Create a master flat by combining all flat files."""
    routines.make_master_dark(dark_filepaths, params.masterdark_filepath, params.bad_pix_mask_filepath, params.clump_mask_filepath, params.overwrite, params.make_plots)
    return


def correct_for_dark_wrapper(input_file, output_file, params):
    """Correct the science data by subtracting the dark data."""
    routines.correct_for_dark(input_file, output_file, params.masterdark_filepath, params.overwrite)
    return


def make_master_flat_wrapper(flats_list, master_darks_list, params):
    """Wrapper to create a master flat by combining all flat files. """
    routines.make_master_flat(flats_list, master_darks_list, params.masterflat_filepath, 
                              params.bad_pix_mask_filepath, params.clump_mask_filepath, params.overwrite, params.make_plots)
    return


def make_master_bkg_wrapper(bkg_filepaths, params, plot=False):
    """Create a master background by combining all background files."""
    raise NotImplementedError("This function has not been updated yet")
    routines.make_master_bkg(bkg_filepaths, params.master_background_filepath, params.overwrite, plot=plot)
    return


def calibrate_files_wrapper(input_file, output_file, masterdark_filepath, masterflat_filepath, params):
    """Calibrate the raw files with Masterdark, Masterflat and Masterbackground, if applicable."""
    routines.calibrate_files(input_file, output_file, masterdark_filepath, masterflat_filepath, 
                             params.master_background_filepath, params.bad_pix_mask_filepath, params.clump_mask_filepath,
                             params.overwrite)
    return


def calibrate_files_multiprocessing_wrapper(input_files, output_files, masterdark_list, masterflat_list, params):
    """Calibrate files using multiprocessing."""
    if params.nproc is None:
        params.nproc = multiprocessing.cpu_count()//2

    """Correct flats using multiprocessing."""
    with multiprocessing.Pool(processes=params.nproc) as pool:
        _ = pool.starmap(calibrate_files_wrapper, zip(input_files,
                                                 output_files, 
                                                 masterdark_list, 
                                                 masterflat_list, 
                                                 [params]*len(input_files)))
        pool.close()  
    return


def split_file_subtract_sky_wrapper(input_file, left_output_file, right_output_file, params):
    """Wrapper for splitting files and subtracting sky."""
    routines.split_file_subtract_sky(input_file, left_output_file, right_output_file, params.overwrite)
    return


def split_files_multiprocessing_wrapper(input_files, output_files_left, output_files_right, params):
    """Split files and subtract sky using multiprocessing."""
    if params.nproc is None:
        params.nproc = multiprocessing.cpu_count()//2

    """Correct bad pixels using multiprocessing."""
    with multiprocessing.Pool(processes=params.nproc) as pool:
        _ = pool.starmap(split_file_subtract_sky_wrapper, zip(input_files, output_files_left, output_files_right, [params] * len(input_files) ) )
        pool.close()  
    return


def make_mastersky_wrapper(sky_files_list, params):
    """ Wrapper function to create a master sky from a list of sky files. It is important that the parameters object contains the correct master_sky_path."""
    routines.make_mastersky(sky_files_list, params.master_sky_filepath, params.overwrite)
    return 


def prepare_centering_wrapper(input_files, output_dir):
    """ Wrapper function to prepare the centering of the files. """
    routines.prepare_centering(input_files, output_dir)
    return


def get_science_exp_time_wrapper(input_files):
    """ Get the exposure time of the science files. Assumes all science files have the same exposure time."""
    return routines.get_science_exp_time(input_files)


def create_flux_file_wrapper(input_file, output_file, science_exp_time, L_or_R, params, center = (401, 466)):
    """ Wrapper function to create a flux file from a science file. """
    routines.create_flux_file(input_file, output_file, science_exp_time, L_or_R, 
                              params.sky_files_exist, params.master_sky_filepath,
                              params.overwrite, params.image_shift_library, params.image_shift_interpolation, 
                              params.make_plots, params.flux_cut_out_size, params.star_loc, params.save_fullsize_flux)
    return


def get_center_wrapper(input_file, params, center = None):
    """ Wrapper function to get the center of the star in a file"""
    return routines.get_center(input_file, params.make_plots, center = None)


def make_the_cube_wrapper(all_science, output_file_cube, output_file_angs, shift, star_center_file, params):
    """ Wrapper function to create the final cube from all science files. """
    routines.make_the_cube(all_science, output_file_cube, output_file_angs, shift, star_center_file,
                            params.star_name, params.obs_night, params.filter_name, params.subtract_median_science_sky_bkg,
                            params.sphere_offset, params.sphere_true_north_correction, params.sky_files_exist, params.master_sky_filepath,
                            params.image_shift_library, params.image_shift_interpolation,
                            params.correct_anamorphism, params.sphere_anamorphic_ratio,
                            params.skip_dither_correction, params.nproc, params.n_sigma, params.box_radius,
                            params.make_plots, params.overwrite, params.full_output)
    return 



