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


def make_master_dark(dark_filepaths, params):
    """Create a master flat by combining all flat files."""
    
    # first get the filepath and check if the file exits - if so also check the overwite param
    if os.path.exists(params.master_dark) and not params.overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(params.master_dark)))
        return 

    dark_data = []
    for dark in dark_filepaths:

        if not os.path.exists(dark):
            logging.warning('\t -> {} does not exist, However this file should not appear here, check the dark updating code'.format(dark))
            continue

        logging.info('\t -> working on {}'.format(os.path.basename(dark)))
        hdul = fits.open(dark)
        header_master_dark = hdul[0].header

        if hdul[0].data.ndim == 2:
            dark_data.append(hdul[0].data)

        elif hdul[0].data.ndim == 3:
            logging.info('\t\t -> Dark {}s exists of multiple exposures, combining to median file'.format(hdul[0].header['EXPTIME']))
            dark_data.append(np.nanmedian(hdul[0].data, axis=0))

        hdul.close()

    # combine all the individual files into a median file
    dark_data = np.array(dark_data)
    median_dark = np.nanmedian(dark_data, axis=0)

    # then correct for all bad pixels
    master_dark_bdpix_corr = correct_bad_pix_wrapper(median_dark, params)

    # lastly save everyting and make a plot 
    if params.make_plots:
        show_img(master_dark_bdpix_corr, title='{}'.format(os.path.basename(params.master_dark)))
        plt.savefig(params.master_dark.replace('fits','pdf'), dpi=120, format='pdf')
        plt.close()

    hdu = fits.PrimaryHDU(data=master_dark_bdpix_corr)
    hdul_masterdark = fits.HDUList([hdu])
    hdul_masterdark[0].header = sanitize_header(header_master_dark)
    hdul_masterdark[0].header['HISTORY'] = 'MASTER DARK'

    for i, file in enumerate([os.path.basename(f) for f in dark_filepaths]):
        hdul_masterdark[0].header['DARK{}'.format(i)] = file

    hdul_masterdark.writeto(params.master_dark, overwrite=True)
    hdul_masterdark.close()
    
    return


def correct_for_dark_ispy(input_file, output_file, params):
    """Correct the science data by subtracting the dark data.
    
    """

    # check if the file exists and wether to overwrite 
    if os.path.exists(output_file) and not params.overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file)) )
        return
    
    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))

    # get the master dark 
    master_dark_filepath = params.master_dark
    if not os.path.exists(master_dark_filepath):
        logging.warning("Warning: masterdark {} does not exist. \n " \
                        "Skipping dark subtraction for {}.".format(
                        master_dark_filepath, os.path.basename(input_file)))
        return
    master_dark = fits.getdata(master_dark_filepath)

    # open the file and correct for dark 
    hdul = fits.open(input_file)
    science_data = hdul[0].data
    science_header = hdul[0].header
    hdul.close()

    # define a new array 
    science_data_minus_dark = science_data - master_dark 

    # save all of this to the final file 
    hdu = fits.PrimaryHDU(data=science_data_minus_dark)
    hdul_minus_dark = fits.HDUList([hdu])

    hdul_minus_dark[0].header = sanitize_header(science_header)
    hdul_minus_dark[0].header['HISTORY'] = 'DARK SUBTRACTED'
    hdul_minus_dark[0].header[f'MASTDARK'] = os.path.basename(master_dark_filepath)

    hdul_minus_dark.writeto(output_file, overwrite=True)
    hdul_minus_dark.close()

    return


def correct_for_dark_ispy_new(input_file, output_file, params):
    """Correct the science data by subtracting the dark data.
    
    """

    overwrite = params['overwrite']
    if os.path.exists(output_file) and not overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file)) )
        return
    
    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))

    master_dark_filepath = params['master_dark']
    if not os.path.exists(master_dark_filepath):
        logging.info("Warning: masterdark {} does not exist.".format(master_dark_filepath))
        logging.info("Skipping dark subtraction for {}.".format(os.path.basename(input_file)))
        return

    hdul = fits.open(input_file)
    science_data = hdul[0].data
    science_header = hdul[0].header
    hdul.close()

    science_data_minus_dark = np.zeros(science_data.shape)
    master_dark = fits.getdata(master_dark_filepath)

    for i, image in enumerate(science_data):
        science_data_minus_dark[i] = science_data[i] - master_dark

    hdu = fits.PrimaryHDU(data=science_data_minus_dark)
    hdul_minus_dark = fits.HDUList([hdu])

    hdul_minus_dark[0].header = sanitize_header(science_header)
    hdul_minus_dark[0].header['HISTORY'] = 'DARK SUBTRACTED'
    hdul_minus_dark[0].header[f'MASTDARK'] = os.path.basename(master_dark_filepath)

    hdul_minus_dark.writeto(output_file, overwrite=True)
    hdul_minus_dark.close()

    return


def make_master_flat(flats_list, master_darks_list, params):
    
    """Create a master flat by combining all flat files.
    
    
    
    """
    
    logging.info("Creating master flat {}".format(basename(params.masterflat_filepath)))
    if os.path.exists(params.masterflat_filepath) and not params.overwrite:
        logging.info('\t -> {} exists skipping file'.format(basename(params.masterflat_filepath)))
        return 

    flat_data = []
    header_master_flat = None
    for flat_filepat, masterdark_filepath in zip(flats_list, master_darks_list):

        if not os.path.exists(masterdark_filepath):
            logging.info('\t -> {} does not exist skipping {}'.format(basename(masterdark_filepath), basename(flat_filepat)))
            continue
        logging.info('\t -> working on {}'.format(os.path.basename(flat_filepat)))
        
        hdul_masterdark = fits.open(masterdark_filepath)
        hdul_flat = fits.open(flat_filepat)

        if header_master_flat is None:
            header_master_flat = sanitize_header(hdul_flat[0].header)
        
        if hdul_flat[0].data.ndim == 2:
            flat_i =  hdul_flat[0].data - hdul_masterdark[0].data

        elif hdul_flat[0].data.ndim == 3:
            logging.info('\t -> Flat exists of multiple exposures, combining to median file')
            flat_i = np.nanmedian((hdul_flat[0].data - hdul_masterdark[0].data), axis=0)
            
        flat_data.append(flat_i/np.nanmedian(flat_i))
        hdul_flat.close()
        hdul_masterdark.close()

    flat_data = np.array(flat_data)   
    master_flat = np.nanmedian(flat_data, axis=0)
    master_flat = np.nan_to_num(master_flat)

    # correct for bad pixels turn on individual bad pixel correction
    master_flat_bdpix_corr = correct_bad_pix_wrapper(master_flat ,params, True)

    # calculate the median of each filter image
    median_of_masterflat_left = np.nanmedian(master_flat_bdpix_corr[25:1020, 50:925])
    median_of_masterflat_right = np.nanmedian(master_flat_bdpix_corr[15:1010, 1080:1955])

    # and get the mean of the two medians such that both filters are normed to 1 
    mean_masterflat = (median_of_masterflat_left + median_of_masterflat_right) / 2

    logging.info('median of master flat left {}, right {}'.format(median_of_masterflat_left, median_of_masterflat_right))
    logging.info('mean of master flat {}'.format(mean_masterflat))

    # devide the makes masterflat
    masked_master_flat_norm = master_flat_bdpix_corr / mean_masterflat

    if params.make_plots:
        show_img(masked_master_flat_norm, title='{}'.format(os.path.basename(params.masterflat_filepath)))
        plt.savefig(params.masterflat_filepath.replace('fits', 'pdf'), dpi=120, format='pdf')
        plt.close()

    hdu = fits.PrimaryHDU(data=masked_master_flat_norm)
    hdul_masterflat = fits.HDUList([hdu])
    hdul_masterflat[0].header = header_master_flat
    hdul_masterflat[0].header['HISTORY'] = 'MASTER FLAT'
    for i, file in enumerate([os.path.basename(f) for f in flats_list]):
        hdul_masterflat[0].header['FLAT{}'.format(i)] = file

    hdul_masterflat.writeto(params.masterflat_filepath, overwrite=True)
    hdul_masterflat.close()
    
    return


def make_master_bkg(bkg_filepaths, params, plot=False):
    """Create a master background by combining all background files."""

    logging.info("Creating master background...")
    logging.info("Files used for master background:")
    master_bkg_filepath = params['master_bkg_path']

    if os.path.exists(master_bkg_filepath) and not params['overwrite']:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(master_bkg_filepath)))
        return master_bkg_filepath

    # load calib files 
    bpm_mask_initial = fits.getdata(params['bad_pix_mask_filepath'])
    master_flat = fits.getdata(params['master_flat_path'])

    bkg_data = []

    for bkg in bkg_filepaths:

        if not os.path.exists(bkg):
            logging.info('\t -> {} does not exist, However this file should not appear here, check the flat uptaing code'.format(bkg))
            continue

        logging.info('\t -> working on {}'.format(os.path.basename(bkg)))
        hdul = fits.open(bkg)
        header_master_bkg = hdul[0].header

        if hdul[0].data.ndim == 2:
            bkg_data.append(hdul[0].data)

        elif hdul[0].data.ndim == 3:
            logging.info('\t -> Background {}s exists of multiple exposures, combining to median file'.format(hdul[0].header['EXPTIME']))
            median_bkg = np.nanmedian(hdul[0].data, axis=0)
            bkg_data.append(median_bkg)

        hdul.close()

    bkg_data = np.array(bkg_data)
    master_bkg = np.nanmedian(bkg_data, axis=0)
    master_bkg = np.nan_to_num(master_bkg)

    # create the clump masks 
    circ_mask1 = circular_aperture(master_bkg, 10, (368, 394))
    circ_mask2 = circular_aperture(master_bkg, 10, (362, 394))
    circ_mask3 = circular_aperture(master_bkg, 10, (356, 394))
    circ_mask4 = circular_aperture(master_bkg, 10, (311, 1400))
    clump_mask = np.zeros(master_bkg.shape)
    clump_mask[circ_mask1 == 1] = 1
    clump_mask[circ_mask2 == 1] = 1
    clump_mask[circ_mask3 == 1] = 1
    clump_mask[circ_mask4 == 1] = 1

    excl_mask = np.ones_like(master_bkg) 
    excl_mask[25:1020, 50:925] = 0
    excl_mask[15:1010, 1080:1955] = 0
    excl_mask[clump_mask == 1] = 1

    # first correct for bad pixels
    master_bkg_bdpix_corr = cube_fix_badpix_isolated(np.array([master_bkg]), bpm_mask=bpm_mask_initial, correct_only=True,
                                        frame_by_frame=False, mad=True, verbose=False, ignore_nan=True, excl_mask=excl_mask)[0] 
    
    master_bkg_bdpix_corr = cube_fix_badpix_isolated(np.array([master_bkg_bdpix_corr]), correct_only=False,
                                        frame_by_frame=False, mad=True, verbose=False, ignore_nan=True, excl_mask=excl_mask)[0] 

    # set all clumps to nan to reduce influence 
    master_bkg_bdpix_corr[clump_mask == 1] = np.nan
    master_bkg_bdpix_corr = cube_fix_badpix_interp(np.array([master_bkg_bdpix_corr]), bpm_mask=clump_mask, excl_mask=excl_mask, mode='gauss', kernel_sz=35)[0]

    masked_master_bkg = np.zeros(master_bkg.shape) * np.nan

    masked_master_bkg[25:1020, 50:925] = master_bkg_bdpix_corr[25:1020, 50:925]
    masked_master_bkg[15:1010, 1080:1955] = master_bkg_bdpix_corr[15:1010, 1080:1955]

    masked_master_bkg_flat_corr = masked_master_bkg/master_flat
    master_bkg_counts_sum = np.nansum(master_bkg_bdpix_corr[25:1010, 925:1080])
    master_bkg_counts_median = np.nanmedian(master_bkg_bdpix_corr[25:1010, 925:1080])

    if plot:
        show_img(masked_master_bkg_flat_corr, title='Master Background')
        plt.savefig('master_background.png', dpi=200, format='png')
        plt.close()

    hdu = fits.PrimaryHDU(data=masked_master_bkg_flat_corr)
    hdul_masterbkg = fits.HDUList([hdu])
    hdul_masterbkg[0].header = header_master_bkg
    hdul_masterbkg[0].header['HISTORY'] = 'MASTER BACKGROUND'
    hdul_masterbkg[0].header['BKG counts sum'] = master_bkg_counts_sum
    hdul_masterbkg[0].header['BKG counts median'] = master_bkg_counts_median
    for i, file in enumerate([os.path.basename(f) for f in bkg_filepaths]):
        hdul_masterbkg[0].header['BKG{}'.format(i)] = file

    hdul_masterbkg.writeto(master_bkg_filepath, overwrite=True)
    hdul_masterbkg.close()

    return


def calibrate_files(input_file, output_file, masterdark_filepath, masterflat_filepath, params):
    """Calibrate the raw files with Masterdark, Masterflat and Masterbackground, if applicable
    
    """

    if os.path.exists(output_file) and not params.overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file)))
        return

    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))
    
    # get all calibration files 
    master_dark = fits.getdata(masterdark_filepath)
    master_flat = fits.getdata(masterflat_filepath)

    # load the science file
    hdul = fits.open(input_file)
    science_data = hdul[0].data

    calibrated_science_data = (science_data - np.nan_to_num(master_dark))/np.nan_to_num(master_flat)
    # if master_background is not None:
    #     master_background = fits.getdata(master_background_filepath)
    #     calibrated_science_data = calibrated_science_data - np.nan_to_num(master_background) #/np.nanmedian(master_background)*np.nanmedian(calibrated_science_data))

    # correct for the individual bad pixels and for the clumps 
    science_data_clump_corr = correct_bad_pix_wrapper(calibrated_science_data, params)
    
    # save the intermediate calibrated data 
    hdu = fits.PrimaryHDU(data=science_data_clump_corr)
    hdul_flat_corr = fits.HDUList([hdu])
    hdul_flat_corr[0].header = sanitize_header(hdul[0].header)

    hdul_flat_corr[0].header['HISTORY1'] = 'DARK CORRECTED'
    hdul_flat_corr[0].header['HISTORY2'] = 'FLAT CORRECTED'
    hdul_flat_corr[0].header['HISTORY3'] = 'BAD PIXEL CORRECTED'
    hdul_flat_corr[0].header['MASTDARK'] = os.path.basename(masterdark_filepath)
    hdul_flat_corr[0].header['MASTFLAT'] = os.path.basename(masterflat_filepath)
    hdul_flat_corr[0].header['BADPIX_FILE'] = os.path.basename(params.bad_pix_mask_filepath)

    if params.master_background_filepath is not None:
        hdul_flat_corr[0].header['HISTORY4'] = 'INSTRUMENTAL BKG CORRECTED'
        hdul_flat_corr[0].header['MASTERBKGND'] = os.path.basename(params.master_background_filepath)

    hdul_flat_corr.writeto(output_file, overwrite=True)
    hdul_flat_corr.close()
    hdul.close()

    return


def get_science_exp_time(filepaths):

    all_exp_times = []
    for file in filepaths:

        exp_time = fits.getheader(file)['EXPTIME']

        if exp_time not in all_exp_times:
            all_exp_times.append(exp_time)

    if len(all_exp_times) != 1:
        raise ValueError('several exposure times in timeseries: {}'.format(all_exp_times) )

    return all_exp_times[0]


def bad_pixel_correction_mp(input_file, output_file, params):

    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))
    
    if os.path.exists(output_file) and not params.overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file)))
        return

    hdul = fits.open(input_file)
    data = hdul[0].data
    header = hdul[0].header
    hdul.close()

    cube_sigcorr, bpm_mask = cube_fix_badpix_isolated(data, sigma_clip=5, num_neig=29, size=9, 
                                                    frame_by_frame=False, mad=True, verbose=True, 
                                                    full_output=True, protect_mask= 100) 

    # cube_sigcorr, bpm_mask_tmp = cube_fix_badpix_isolated(cube_sigcorr, bpm_mask=bpm_mask, correct_only=False, 
    #                                                   sigma_clip=5, num_neig=29, size=9, frame_by_frame=True,
    #                                                   mad=True, verbose=True, full_output=True, protect_mask= 100)

    hdu = fits.PrimaryHDU(data=cube_sigcorr)
    hdul_badpix_corr = fits.HDUList([hdu])
    hdul_badpix_corr[0].header = header
    hdul_badpix_corr[0].header['HISTORY'] = 'BAD PIXEL CORRECTED'

    hdul_badpix_corr.writeto(output_file, overwrite=True)
    hdul_badpix_corr.close()
    hdul.close()

    return


def bad_pixel_correction(input_file, output_file, params):

    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))
    
    if os.path.exists(output_file) and not params.overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file)))
        return
    
    hdul = fits.open(input_file)

    cube_sigcorr = hdul[0].data

    cube_sigcorr, bpm_mask = cube_fix_badpix_isolated(cube_sigcorr, sigma_clip=5, num_neig=29, size=9, 
                                                    frame_by_frame=False, mad=True, verbose=True, 
                                                    full_output=True, protect_mask= 100) 
    
    cube_sigcorr, bpm_mask_tmp = cube_fix_badpix_isolated(cube_sigcorr, bpm_mask=bpm_mask, correct_only=False, 
                                                      sigma_clip=5, num_neig=29, size=9, frame_by_frame=True,
                                                      mad=True, verbose=True, full_output=True, protect_mask= 100, nproc=params['nproc'])

    hdul.close()

    hdu = fits.PrimaryHDU(data=cube_sigcorr)
    hdul_badpix_corr = fits.HDUList([hdu])
    hdul_badpix_corr[0].header = hdul[0].header
    hdul_badpix_corr[0].header['HISTORY'] = 'BAD PIXEL CORRECTED'

    hdul_badpix_corr.writeto(output_file, overwrite=True)
    hdul_badpix_corr.close()
    hdul.close()

    return


def split_file_subtract_sky(input_file, left_output_file, right_output_file, params):

    logging.info('\t -> working on {} splitting the frames, saves to {} & {}'.format(os.path.basename(input_file), os.path.basename(left_output_file), os.path.basename(right_output_file)))

    if os.path.exists(left_output_file) and os.path.exists(right_output_file) and not params.overwrite:
        logging.info('\t -> exists skipping file {}'.format(os.path.basename(input_file)))
        return

    # load the data 
    hdul = fits.open(input_file)
    data = hdul[0].data

    left_frame = data[:, 25:1020, 50:925]
    right_frame = data[:, 15:1010, 1080:1955]

    hdu_left = fits.PrimaryHDU(data=left_frame)
    hdul_left = fits.HDUList([hdu_left])
    hdul_left[0].header = hdul[0].header

    hdu_right = fits.PrimaryHDU(data=right_frame)
    hdul_right = fits.HDUList([hdu_right])
    hdul_right[0].header = hdul[0].header

    hdul_left.writeto(left_output_file, overwrite=True)
    hdul_right.writeto(right_output_file, overwrite=True)
    
    hdul_left.close()
    hdul_right.close()
    hdul.close()
    return


def make_mastersky_wrapper(sky_files_list, params):
    """ Wrapper function to create a master sky from a list of sky files. It is important that the parameters object contains the correct master_sky_path."""

    routines.make_master_sky(sky_files_list, params.master_sky_path, params.overwrite)
    return 


def prepare_centering(input_files, output_dir):

    logging.info('preparing centering of files:')
    for file in input_files:

        #open the file 
        hdul = fits.open(file)

        #get the data
        data = hdul[0].data
        hdul.close()

        cy, cx = frame_center(data[0])

        if len(data.shape) == 3:
            if len(data) > 1:
                image = data[1]
            else :
                image = data[0]
        elif len(data.shape) == 2:
            image = data
        else:
            raise ValueError('data shape not recognized: {}'.format(data.shape))        

        # find the centers of the frame
        center = centroid_1dg(image)
        cy, cx = frame_center(image)

        # and try to use the now centered image with the sattelite spots
        sat_spots = np.array([ [-35, 35], [35, 35], [-35,-35], [35, -35]]) + center

        show_img(image, minper=0.1, maxper=99.9, title=os.path.basename(file), xlim=(cx-70, cx+70), ylim=(cy-70, cy+70))

        plt.scatter(center[0], center[1], color='black', s=10, label='centroid 1dg', marker = '+')
        plt.scatter(cx, cy, color='black', s=10, label='center frame', marker = '*')

        for j, xy in enumerate(sat_spots):
            plt.scatter(xy[0], xy[1], color=f'C{j}', s=10, label=f'xy {j}')
        plt.legend()

        png_dir = os.path.join(output_dir ,'recentering_pngs')
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)
        plt.savefig(os.path.join(png_dir, os.path.basename(file)[:-5] +'.png'), dpi = 200, format='png')
        plt.close()


    csv_path = os.path.join(output_dir ,'centers.csv') 

    if os.path.exists(csv_path):
        DATAFRAME = pd.read_csv(csv_path)

        for file in [os.path.basename(f) for f in input_files]:
            
            if not file in DATAFRAME['frames']:

                DATAFRAME = pd.concat([DATAFRAME, pd.DataFrame({'frames': file, 'x1': [0], 'y1': [0],
                                                  'x2': [0], 'y2': [0],
                                                  'x3': [0], 'y3': [0],
                                                  'x4': [0], 'y4': [0], })], ignore_index=True)

    else:
        length = len(input_files)
        DATAFRAME = pd.DataFrame({'frames': [os.path.basename(f) for f in input_files],
                                'x1': [0] * length,
                                'y1': [0] * length,
                                'x2': [0] * length,   
                                'y2': [0] * length,
                                'x3': [0] * length,
                                'y3': [0] * length,
                                'x4': [0] * length,   
                                'y4': [0] * length,
                                })

    DATAFRAME.to_csv(csv_path, index=False)


    return


def get_ratio_for_nd_filter(wave_filter, nd_filter, L_or_R, plot = False):
    
    # get transmittsion profile
    if 'DB' in wave_filter:
        wave_filter = wave_filter.replace('DB', 'D')

    if 'BB' in wave_filter:
        wave_filter = wave_filter.replace('BB', 'B')

    trans_profile_file = get_filter_path('SPHERE_IRDIS_' + wave_filter + '.dat')

    if np.loadtxt(trans_profile_file).transpose().shape[0] == 3:
        trans_wave, trans1, trans2 = np.loadtxt(trans_profile_file).transpose()
        if L_or_R == 'LEFT':
            trans = trans1
        elif L_or_R == 'RIGHT':
            trans = trans2
        else:
            raise ValueError()

    elif np.loadtxt(trans_profile_file).transpose().shape[0] == 2:
        trans_wave, trans = np.loadtxt(trans_profile_file).transpose()

    else:
        raise ValueError()


    # get the ndfilter:
    nd_file = get_filter_path('SPHERE_CPI_ND.dat')
    ND_DF = pd.DataFrame(np.loadtxt(nd_file), columns=['wavelength', 'ND0.0','ND1.0','ND2.0','ND3.5'])

    nd_wave = ND_DF['wavelength'].values
    nd_trans = ND_DF['ND' + nd_filter.split('_')[-1]].values

    nd_wave_start, nd_wave_stop = np.nanmin(nd_wave), np.nanmax(nd_wave)

    start = max(nd_wave.min(), trans_wave.min())
    end = min(nd_wave.max(), trans_wave.max())

    # Create masks
    mask_A = (nd_wave >= start) & (nd_wave <= end)
    mask_B = (trans_wave >= start) & (trans_wave <= end)

    # Apply masks
    nd_wave_cut = nd_wave[mask_A]
    nd_trans_cut = nd_trans[mask_A]

    trans_wave_B_cut = trans_wave[mask_B]
    trans_cut = trans[mask_B]

    convoluted_trans = nd_trans_cut #trans_cut


    # as the profile is sampled by 1 Angstrom, the integral is simply the sum of the transmission
    trans_integrate = np.nansum(trans_cut)
    convoluted_trans_integrate = np.nansum(convoluted_trans)

    ratio = convoluted_trans_integrate/trans_integrate

    if plot:
        plt.figure(figsize = (7,7), dpi = 200)
        plt.plot(nd_wave, nd_trans, label = 'ND filter profile')
        plt.plot(trans_wave, trans, label = 'filter profile')
        plt.plot(nd_wave_cut, convoluted_trans, label = 'convolved profile')
        plt.plot(nd_wave_cut, np.ones(len(nd_wave_cut))*ratio, label = 'mean ratio = {:1.3f}'.format(ratio))
        plt.yscale('log')
        plt.legend()
        plt.show()


    return ratio


def create_flux_file(input_file, output_file, science_exp_time, L_or_R, params, center = (401, 466),):

    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))

    if os.path.exists(output_file) and not params.overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file)))
        return

    hdul = fits.open(input_file)
    data = hdul[0].data
    header = hdul[0].header
    exp_time = header['EXPTIME']
    nd_filter = header['HIERARCH ESO INS4 FILT2 ID']
    wave_filter = header['HIERARCH ESO INS COMB IFLT']
    hdul.close()

    # get the ratios by which the flux needs to be rescaled
    nd_ratio = get_ratio_for_nd_filter(wave_filter, nd_filter, L_or_R, plot=params.make_plots)
    logging.info('\t nd filter ratio for flux file {}: {}'.format(os.path.basename(output_file), nd_ratio))
    
    # and also the exp time ratio
    exp_time_ratio = science_exp_time/exp_time
    logging.info('\t exp time ratio for flux file {}: {}'.format(os.path.basename(output_file), exp_time_ratio))

    # now stack the exposures in a median file toegther to reduce noise
    stacked_flux_image = np.nanmedian(np.nan_to_num(data), axis = 0)

    if len(stacked_flux_image)%2 == 1:
        cropsize = 31
    elif len(stacked_flux_image)%2 == 0:
        cropsize = 30
    else: 
        cropsize = 30 

    # subtract the median sky background 
    # median_sky_bkg = np.median(stacked_flux_image)
    # logging.info('\t median sky backgorund in file {}: {}'.format(os.path.basename(output_file), median_sky_bkg))

    # stacked_flux_image = stacked_flux_image - median_sky_bkg

    # fit a gauss to find the center 
    med_y, med_x = fit_2dgaussian(stacked_flux_image, crop=True, cent=center, cropsize=cropsize, threshold=True, sigfactor=4, full_output=False, debug=False)
    
    #cut out the center of the flux image
    # divide the shift in integer and float
    int_center_x = int(np.floor(med_x))
    int_center_y = int(np.floor(med_y))

    float_center_x = med_x - int_center_x
    float_center_y = med_y - int_center_y

    # first shift the center to a full pixel
    float_centered_image = frame_shift(stacked_flux_image, -float_center_y, -float_center_x, imlib=params.image_shift_library, interpolation=params.image_shift_interpolation)

    # then cut out 20 pixels around the central one
    int_centered_image = float_centered_image[int_center_y-20:int_center_y+21, int_center_x-20:int_center_x+21]

    # and make a larger cutout for the background
    background = float_centered_image.copy()
    background = background[int_center_y-150:int_center_y+151, int_center_x-150:int_center_x+151]
    background[100:200,100:200] = np.nan

    # background = background[int_center_y-50:int_center_y+51, int_center_x-50:int_center_x+51]
    # background[30:70,30:70] = np.nan

    med_background = np.nanmedian(background)
    logging.info('\t median sky background in file {}: {}'.format(os.path.basename(output_file), med_background))

    int_centered_image = int_centered_image - med_background

    # lastly rescale the flux since this is the transmittion profile so smaller than 1 we need to divide by it
    rescaled_image = int_centered_image * exp_time_ratio / nd_ratio 

    # and plot the final image (optional)
    if params.make_plots: 
        plot_dir = os.path.join(os.path.dirname(output_file),  'flux_pngs')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        show_img(rescaled_image, minper=1, maxper=99, size =5)
        plt.scatter(20,20, marker='x', color='red', s=100)
        plt.savefig(os.path.join(plot_dir,os.path.basename(output_file).replace('.fits', '.pdf')), bbox_inches='tight', dpi=120)
        plt.close()

        show_img(stacked_flux_image * exp_time_ratio / nd_ratio, minper=1, maxper=99, size = 5)
        plt.scatter(med_x, med_y, marker='x', color='red', s=100)
        plt.xlim(med_x - 20, med_x  +20)
        plt.ylim(med_y - 20, med_y  +20)
        plt.savefig(os.path.join(plot_dir,os.path.basename(output_file).replace('.fits', '2.pdf')), bbox_inches='tight', dpi=120)
        plt.close()

    # lastly save the flux file 
    hdu_flux = fits.PrimaryHDU(data=rescaled_image)
    hdul_flux = fits.HDUList([hdu_flux])

    header['EXPTIMERATIO'] = exp_time_ratio
    header['ND_RATIO'] = 1/nd_ratio
    header['BACKGROUND'] = med_background

    hdul_flux[0].header = header
    hdul_flux.writeto(output_file, overwrite=True)
    hdul_flux.close()

    return


def get_satspots(input_file):

    logging.info('\t -> loading centers of {} from the CSV file'.format(os.path.basename(input_file)))
    csv_path = os.path.join(os.path.dirname(input_file) ,'centers.csv') 
    DF = pd.read_csv(csv_path)

    filename = os.path.basename(input_file)
    DF_Frame = DF[DF['frames'] == filename]
    centers = [ (DF_Frame['x{}'.format(f+1)].values[0], DF_Frame['y{}'.format(f+1)].values[0]) for f in range(4)]
        
    return centers


def get_center(input_file, center = None, plot = False):

    logging.info('\t -> getting center from file {}'.format(os.path.basename(input_file)))
    sat_spots = get_satspots(input_file)
    
    # first get the data 
    data = np.nan_to_num(fits.getdata(input_file))

    # for now cut away the edges to remove the bad pixels
    cut_size = 20 
    data = data[:, cut_size:-cut_size, cut_size:-cut_size]

    # creat a median image of the cube
    image = np.nanmedian(data, axis=0)

    # try to find the center using centroid_1dg
    if center is None:
        center = centroid_1dg(image)

    # find the center of the frame
    cy, cx = frame_center(image)

    # and try to use the now centered image with the sattelite spots
    if sat_spots is None or sat_spots == [(0,0), (0,0), (0,0), (0,0)]:

        sat_spots = np.array([ [-35, 35], [35, 35], [-35,-35], [35, -35]]) + center
        sat_spots_input = [(f[0], f[1]) for f in sat_spots]
    
    # if the sat_spots are given, we need to take of the same amount that we cut away at the beginning
    else:
         sat_spots_input = [(f[0] -cut_size, f[1] -cut_size) for f in sat_spots]


    string = ', '.join(['s{:1.0f} = ({:1.1f}, {:1.1f})'.format(i, f[0], f[1]) for i, f in enumerate(sat_spots_input)])
    logging.info('Using satellite spots at: {}'.format(string))

    # now comes the recentering, here we do it on the cube
    res = cube_recenter_satspots(data, xy=sat_spots_input, subi_size=21, verbose=False, debug=False, full_output=True, fit_type='gaus')
    center_shifty = np.mean(res[1])
    center_shiftx = np.mean(res[2])

    plot = False
    if plot: 
        shifted_image = np.nanmedian(res[0], axis = 0)
        shiftxy = (res[1], res[2])
        true_center = (np.mean(res[3]), np.mean(res[4]))


        # the unshifted image 
        show_img(image, minper=0.1, maxper=99.9, xlim=[center[0]-100,center[0]+100], ylim=[center[1]-100,center[1]+100], size = 2,)
        plt.scatter(center[0], center[1], color='black', s=10, label='input center', marker = '+')
        plt.scatter(true_center[1], true_center[0], color='black', s=10, label='true center', marker = 'x')
        plt.scatter(cx, cy, color='black', s=10, label='center frame', marker = '*')
        for j, xy in enumerate(sat_spots_input):
            plt.scatter(xy[0], xy[1], color=f'C{j}', s=10, label=f'xy {j}')
        plt.legend()
        plt.savefig('test.png')
        plt.close()

        # the shifted image with centers
        show_img(shifted_image, minper=0.1, maxper=99.9, xlim=[cx-100,cx+100], ylim=[cy-100,cy+100], size = 2)
        plt.scatter(cx, cy, color='black', s=10, label='center frame', marker = '+')
        plt.legend()
        plt.savefig('test.png')
        plt.close()

    

    return center_shifty, center_shiftx


def make_the_cube(all_science, output_file_cube, output_file_angs, shift, star_center_file, params, full_output=False):

    if os.path.exists(output_file_cube) and not params.overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file_cube)))
        return
    
    cube = []
    angs = []

    header_infos = pd.DataFrame(columns = ['file', 'start_angle', 'end_angle', 'dither_x', 'dither_y', 'n_frames'])

    logging.info('\t -> extracting the single exposures from the files into one cube')
    for file in all_science: 
        data = fits.getdata(file)

        try:
            # do the %360 as the angles range from -180 to 180 which gives issues when crossing the 180 line
            header = fits.getheader(file)
            angend = header['HIERARCH ESO TEL PARANG END'] %360
            angstart = header['HIERARCH ESO TEL PARANG START'] %360
            dith_x = header['HIERARCH ESO INS1 DITH POSX']
            dith_y = header['HIERARCH ESO INS1 DITH POSY']
            temp = header['HIERARCH ESO INS4 TEMP449 VAL']
            
        except KeyError as e:
            print(file)
            raise KeyError(e)

        header_infos = pd.concat([header_infos,pd.DataFrame({
                                                'file': [os.path.basename(file)],
                                                'start_angle': [angstart],
                                                'end_angle': [angend],
                                                'dither_x': [dith_x],
                                                'dither_y': [dith_y],
                                                'n_frames': [len(data)],
                                                'filter_temp': [temp]
                                                })],ignore_index=True)

        diff = (angstart - angend) / len(data) 
        angs_i = np.linspace(angstart + diff/2, angend - diff/2, len(data))

        # add sphere offset
        angs_i += params.sphere_offset

        for ang, image in zip(angs_i, data): 
            
            # subtract the median skybkg 
            median_sky_bkg = np.nanmedian(image)
            # image = image - median_sky_bkg

            cube.append(image)
            # int divide ang by 360 to only have angles between 0 and 360
            angs.append(ang)

    angs = np.array(angs)
    cube = np.array(cube)
    
    header_infos.to_csv(os.path.join(os.path.dirname(output_file_cube), 'header_infos.csv'), index=False)

    if params.make_plots: 
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi = 120)
        ax.plot(angs, marker = '.', ls = 'none')
        ax.set_ylabel('Parallactic angle [deg]')
        ax.set_xlabel('Frame number')
        ax.set_title(f"rotation angles for {params.star_name} on {params.obs_night} with filter {params.filter_name}")
        plt.savefig(os.path.join(os.path.dirname(output_file_cube), 'rotation_angles.pdf'), format='pdf')
        plt.close()

    # convert the infos given per file to infos per frame
    total_number_of_frames = len(cube)
    all_x_dith_shifts = np.zeros(total_number_of_frames) * np.nan
    all_y_dith_shifts = np.zeros(total_number_of_frames) * np.nan
    temps = np.zeros(len(cube)) * np.nan

    k = 0
    for i, n_exp in enumerate(header_infos['n_frames']):
        for j in range(int(n_exp)):
            all_x_dith_shifts[k] = header_infos['dither_x'][i]
            all_y_dith_shifts[k] = header_infos['dither_y'][i]
            temps[k] = header_infos['filter_temp'][i]
            k += 1

    if np.isnan(all_x_dith_shifts).any() or np.isnan(all_y_dith_shifts).any() or np.isnan(temps).any():
        raise ValueError('Dither shifts contain NaNs, please check the headers')
    
    if params.sky_files_exist:
        master_sky = fits.getdata(params.master_sky_filepath)
        mean_sky_filter_temp = fits.getheader(params.master_sky_filepath)['MEANTEMP']
        
        for i in range(len(cube)):
            cube[i] = cube[i] - master_sky/mean_sky_filter_temp * temps[i]

    for i, (xx,yy) in enumerate(zip(all_x_dith_shifts, all_y_dith_shifts)):
        cube[i] = np.roll(cube[i], (-int(yy), -int(xx)), axis=(0,1))

    if params.make_plots: 
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi = 120)

        ax.plot(all_y_dith_shifts, marker = '.', label='shifts y')
        ax.plot(all_x_dith_shifts, marker = '.', label='shifts x')

        ax.set_ylabel('Shift in pixels')
        ax.set_xlabel('Frame number')

        ylim = np.abs(ax.get_ylim()).max()
        ax.set_ylim(-ylim, ylim)

        ax.set_title(f"Alignments for {params.star_name} on {params.obs_night} with filter {params.filter_name}")
        ax.legend()
        plt.savefig(os.path.join(os.path.dirname(output_file_cube), 'dither_shifts.pdf'), format='pdf', dpi = 120)
        plt.close()

    # now shift the cube to the center
    center_shifty, center_shiftx = shift
    
    int_shift_x = int(np.floor(center_shiftx))
    int_shift_y = int(np.floor(center_shifty))

    float_center_x = center_shiftx - int_shift_x
    float_center_y = center_shifty - int_shift_y

    # first shift the center of the cube by the full integer number
    logging.info('\t -> centering the cube on a full pixel level')
    y_size_cube = cube.shape[1]
    x_size_cube = cube.shape[2]

    expanded_cube = np.zeros((len(cube),1025,1025))
    diff_y = (expanded_cube.shape[1] - y_size_cube)/2
    diff_x = (expanded_cube.shape[2] - x_size_cube)/2
    
    expanded_cube[:,
                  int(diff_y + int_shift_y) : int(-diff_y + int_shift_y),
                  int(diff_x + int_shift_x) : int(-diff_x + int_shift_x)] = cube

    nan_mask = np.zeros((1025,1025)) * np.nan
    nan_mask[int(diff_y + int_shift_y) : int(-diff_y + int_shift_y),
             int(diff_x + int_shift_x) : int(-diff_x + int_shift_x)] = 1  
    
    mask = circular_aperture(nan_mask, 520, inverted=False)
    nan_mask[mask == 0] = np.nan

    # make the cosmic ray correction
    logging.info('\t -> correcting the cube for cosmic rays')
    cube_sigcorr_indv = correct_cosmic_rays_multiprocessing(np.nan_to_num(expanded_cube), params)

    # now shift the center to a full pixel
    logging.info('\t -> now centering the cube on a sub pixel level')
    float_centered_cube = cube_shift(np.nan_to_num(cube_sigcorr_indv), float_center_y, float_center_x, imlib=params.image_shift_library, interpolation=params.image_shift_interpolation)

    if not params.skip_dither_correction:
        # now correct for the dithers:
        logging.info('\t -> finally correcting the cube for telescope dithers')
        dither_corr_cube, shy2a, shx2a = cube_recenter_dft_upsampling(np.nan_to_num(float_centered_cube), upsample_factor=100,
                                                                    subi_size=None, # set this to None so no over all recentering is done, this has been done already using satellite spots
                                                                    center_fr1=frame_center(cube[0]), 
                                                                    fwhm=4, negative=True,
                                                                    full_output=True, debug=False, nproc = params.nproc, 
                                                                    imlib=params.image_shift_library, interpolation=params.image_shift_interpolation)

        # save the shifts to a file
        np.savetxt(os.path.join(os.path.dirname(output_file_cube), 'dither_shifts.txt'), np.array([shy2a, shx2a]), fmt='%f')

        # and plot this
        if params.make_plots: 
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi = 120)

            ax.plot(shy2a, marker = '.', label='shifts y')
            ax.plot(shx2a, marker = '.', label='shifts x')

            ax.set_ylabel('Shift in pixels')
            ax.set_xlabel('Frame number')

            ylim = np.abs(ax.get_ylim()).max()
            ax.set_ylim(-ylim, ylim)

            ax.set_title(f"Alignments for {params.star_name} on {params.obs_night} with filter {params.filter_name}")
            ax.legend()
            plt.savefig(os.path.join(os.path.dirname(output_file_cube), '_subpixel_alignment_shifts.pdf'), format='pdf', dpi = 120)
            plt.close()
    else:
        dither_corr_cube = float_centered_cube
        
    # now masking out the center of the cube
    logging.info('\t -> masking out the center of the cube, setting the edges to NaN')
    masked_cube = dither_corr_cube * nan_mask
    center_mask = circular_aperture(masked_cube[0], 300, inverted=True)

    #subtract the median sky background from each frame
    if params.subtract_median_science_sky_bkg:
        logging.info('\t -> subtracting the median of each frame as sky background')
        
        medians = np.zeros(len(masked_cube)) * np.nan
        for i, image in enumerate(masked_cube):
            center_masked_image = image.copy()
            center_masked_image[center_mask == 0] = np.nan
            median_sky_bkg = np.nanmedian(center_masked_image)
            masked_cube[i] = masked_cube[i] - median_sky_bkg

            medians[i] = median_sky_bkg

        # saving the median sky background to a file
        np.savetxt(os.path.join(os.path.dirname(output_file_cube), 'median_sky_bkg.txt'), medians, fmt='%f')
        masked_cube -= median_sky_bkg


    # now save the cube
    logging.info('\t -> saving the cube to {}'.format(os.path.basename(output_file_cube)))
    hdu2 = fits.PrimaryHDU(data=masked_cube)
    hdul2 = fits.HDUList([hdu2])
    header2 = fits.getheader(all_science[0])

    for i, file in enumerate(all_science):
        header2['SCIENC{}'.format(i)] = os.path.basename(file)

    header2['CENTSTAR'] = os.path.basename(star_center_file)
    header2['SHIFTX'] = np.mean(center_shiftx)
    header2['SHIFTY'] = np.mean(center_shifty)

    if params.sky_files_exist:
        header2['MASTSKY'] = os.path.basename(params.master_sky_filepath)
        header2['MNSKYTMP'] = mean_sky_filter_temp

    hdul2[0].header = header2

    hdul2.writeto(output_file_cube, overwrite=True)
    hdul2.close()
    np.savetxt(output_file_angs, angs, fmt='%f')

    if full_output:
        cube_shifted = cube_shift(cube, center_shifty, center_shiftx, imlib='ndimage-interp', interpolation='biquintic')
        # now save the cube
        hdu = fits.PrimaryHDU(data=cube_shifted)
        hdul = fits.HDUList([hdu])
        header = fits.getheader(all_science[0])

        for i, file in enumerate(all_science):
            header['SCIEN{}'.format(i)] = os.path.basename(file)

        header['CENTSTAR'] = os.path.basename(star_center_file)
        header['SHIFTX'] = np.mean(center_shiftx)
        header['SHIFTY'] = np.mean(center_shifty)

        hdul[0].header = header
        hdul.writeto(output_file_cube.replace('cube', 'cube_basic_shift'), overwrite=True)
        hdul.close()

    logging.info('cube finished!')

    # optional to save the mask 
    # hdu_mask = fits.PrimaryHDU(data=nan_mask)
    # hdul_mask = fits.HDUList([hdu_mask])
    # hdul_mask.writeto(output_file_cube.replace('cube', 'nan_mask'), overwrite=True)
    # hdul_mask.close()

    return 


def make_cube_quadratic(input_file, output_file, params):
    """
    
    """

    hdul = fits.open(input_file)

    data = hdul[0].data
    header = hdul[0].header

    hdul.close()

    #find longest side
    long_side = int(np.max(data[0].shape))
    short_side = int(np.min(data[0].shape))

    nan_cube = np.zeros((len(data), long_side, long_side)) * np.nan
    diff = int((long_side-short_side)/2)

    if data.shape[1] == long_side:
        nan_cube[:,:,diff:-diff] = data

    elif data.shape[2] == long_side:
        nan_cube[:,diff:-diff,:] = data

    # apply_mask
    mask = circular_aperture(nan_cube[0], 520, inverted=False)
    mask[mask == 0] = np.nan

    masked_nan_cube = nan_cube * mask 

    hdu = fits.PrimaryHDU(data=masked_nan_cube)
    hdul = fits.HDUList([hdu])

    hdul[0].header = header
    hdul.writeto(output_file, overwrite=True)
    hdul.close()
    
    return


def cosmic_ray_detect(image, params):
    """ Cosmic ray detection function.
    Works by comparing each pixel to the local median absolute deviation.
    If the pixel value is more than n_sigma * MAD from the local median, it is called bad
    """

    n_sigma = params.n_sigma
    box_radius = int(params.box_radius)
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


def clean_bp(badpix, cub_in):
    ''' This routine is taken from GRAPHIC (Holmberg+2016)
    Function to clean bad pixels in a cube using the median of surrounding pixels .'''

    from bottleneck import median
    from bottleneck import nanmedian

    cub_in[:, badpix[0], badpix[1]] = np.NaN
    for f in range(cub_in.shape[0]):

        for j in range(len(badpix[0])):
            y = badpix[0][j]
            x = badpix[1][j]

            if y == cub_in.shape[1] - 1:  # At the image edge !!!
                if x == cub_in.shape[2] - 1:  # In a corner
                    cub_in[f, y, x] = nanmedian([
                            cub_in[f, y - 1, x - 1], cub_in[f, y - 1, x],
                            cub_in[f, y, x - 1]
                    ])
                elif x == 0:
                    cub_in[f, y, x] = nanmedian([
                            cub_in[f, y - 1, x], cub_in[f, y - 1, x + 1],
                            cub_in[f, y, x + 1]
                    ])
                else:  # Along the edge
                    cub_in[f, y, x] = nanmedian([
                            cub_in[f, y - 1, x - 1], cub_in[f, y - 1, x],
                            cub_in[f, y - 1, x + 1], cub_in[f, y, x - 1],
                            cub_in[f, y, x + 1]
                    ])
            elif y == 0:  # At the image edge !!!
                if x == cub_in.shape[2] - 1:  # In a corner
                    cub_in[f, y, x] = nanmedian([
                            cub_in[f, y, x - 1], cub_in[f, y + 1, x - 1],
                            cub_in[f, y + 1, x]
                    ])
                elif x == 0:  # In a corner
                    cub_in[f, y, x] = nanmedian([
                            cub_in[f, y, x + 1], cub_in[f, y + 1, x],
                            cub_in[f, y + 1, x + 1]
                    ])
                else:  # Along the edge
                    cub_in[f, y, x] = nanmedian([
                            cub_in[f, y, x - 1], cub_in[f, y, x + 1],
                            cub_in[f, y + 1, x - 1], cub_in[f, y + 1, x],
                            cub_in[f, y + 1, x + 1]
                    ])
            elif x == 0:  # Along the edge
                cub_in[f, y, x] = nanmedian([
                        cub_in[f, y - 1, x], cub_in[f, y - 1, x + 1],
                        cub_in[f, y, x + 1], cub_in[f, y + 1,
                                                    x], cub_in[f, y + 1, x + 1]
                ])
            elif x == cub_in.shape[2] - 1:  # Along the edge
                cub_in[f, y, x] = nanmedian([
                        cub_in[f, y - 1, x - 1], cub_in[f, y - 1, x],
                        cub_in[f, y, x - 1], cub_in[f, y + 1,
                                                    x - 1], cub_in[f, y + 1, x]
                ])
            else:  # Usual case, not on an edge
                cub_in[f, y, x] = nanmedian([
                        cub_in[f, y - 1, x - 1], cub_in[f, y - 1, x],
                        cub_in[f, y - 1, x + 1], cub_in[f, y, x - 1],
                        cub_in[f, y, x + 1], cub_in[f, y + 1, x - 1],
                        cub_in[f, y + 1, x], cub_in[f, y + 1, x + 1]
                ])
                
    return cub_in


def correct_cosmic_rays(image, params):

    cosmic_mask = cosmic_ray_detect(image,params)
    image_cleaned = clean_bp(cosmic_mask, np.array([image]))[0]

    return image_cleaned


def correct_cosmic_rays_multiprocessing(input_cube, params):

    if params.nproc is None:
        params.nproc = multiprocessing.cpu_count()//2

    """Correct darks using multiprocessing."""
    with multiprocessing.Pool(processes=params.nproc) as pool:
        results = pool.starmap(correct_cosmic_rays, zip(input_cube, [params]*len(input_cube)))
        pool.close()  
        
    return np.array(results)



