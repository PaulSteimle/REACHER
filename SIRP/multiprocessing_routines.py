import os
import multiprocessing
import time
import re
import logging
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
from scipy import ndimage
from photutils.centroids import centroid_1dg, centroid_quadratic, centroid_2dg
from HCIutils import show_img
from astropy.io import fits 
from glob import glob 

from vip_hci.preproc import cube_fix_badpix_isolated, cube_recenter_satspots, frame_center_satspots, frame_center_radon, frame_shift, cube_shift
from vip_hci.var import fit_2dgaussian, frame_center 
from hciplot import plot_frames
from vip_hci.preproc import frame_fix_badpix_isolated


from. import wrappers as rt 

def correct_darks_multiprocessing(input_files, output_files, params):

    if params['nproc'] is None:
        params['nproc'] = multiprocessing.cpu_count()//2

    """Correct darks using multiprocessing."""
    with multiprocessing.Pool(processes=params['nproc']) as pool:
        _ = pool.starmap(rt.correct_for_dark, zip(input_files, output_files, [params]*len(input_files)))
        pool.close()  
    return


def correct_darks_ispy_multiprocessing(input_files, output_files, params):

    if params['nproc'] is None:
        params['nproc'] = multiprocessing.cpu_count()//2

    """Correct darks using multiprocessing."""
    with multiprocessing.Pool(processes=params['nproc']) as pool:
        _ = pool.starmap(rt.correct_for_dark_ispy, zip(input_files, output_files, [params]*len(input_files)))
        pool.close()  
    return


def correct_darks_ispy_multiprocessing_new(input_files_list, output_files_list, params_list):

    if params_list[0]['nproc'] is None:
        params_list[0]['nproc'] = multiprocessing.cpu_count()//2

    #params_list[0]['nproc'] = 1
    """Correct darks using multiprocessing."""
    with multiprocessing.Pool(processes=params_list[0]['nproc']) as pool:
        _ = pool.starmap(rt.correct_for_dark_ispy, zip(input_files_list, output_files_list, params_list))
        pool.close()  
    return


def correct_badpix_multiprocessing(input_files, output_files, params):

    if params['nproc'] is None:
        params['nproc'] = multiprocessing.cpu_count()//2

    """Correct bad pixels using multiprocessing."""
    with multiprocessing.Pool(processes=params['nproc']) as pool:
        _ = pool.starmap(rt.bad_pixel_correction_mp, zip(input_files, output_files, [params] * len(input_files) ))
        pool.close()  
    return


def split_files_multiprocessing(input_files, output_files_left, output_files_right, params):
    
    if params.nproc is None:
        params.nproc = multiprocessing.cpu_count()//2

    """Correct bad pixels using multiprocessing."""
    with multiprocessing.Pool(processes=params.nproc) as pool:
        _ = pool.starmap(rt.split_file_subtract_sky, zip(input_files, output_files_left, output_files_right, [params] * len(input_files) ) )
        pool.close()  
    return


def make_cube_quadratic_multiprocessing(input_files, output_files, params):

    if params['nproc'] is None:
        params['nproc'] = multiprocessing.cpu_count()//2

    """Correct darks using multiprocessing."""
    with multiprocessing.Pool(processes=params['nproc']) as pool:
        _ = pool.starmap(rt.make_cube_quadratic, zip(input_files, output_files, [params]*len(input_files)))
        pool.close()  
        
    return

def calibrate_files_multiprocessing(input_files, output_files, masterdark_list, masterflat_list, params):

    if params.nproc is None:
        params.nproc = multiprocessing.cpu_count()//2

    """Correct flats using multiprocessing."""
    with multiprocessing.Pool(processes=params.nproc) as pool:
        _ = pool.starmap(rt.calibrate_files, zip(input_files,
                                                 output_files, 
                                                 masterdark_list, 
                                                 masterflat_list, 
                                                 [params]*len(input_files)))
        pool.close()  
    return

