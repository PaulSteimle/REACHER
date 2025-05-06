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
from HCIutils import show_img, circular_aperture
from astropy.io import fits 
from glob import glob 

from vip_hci.preproc import cube_fix_badpix_isolated, cube_recenter_satspots, frame_center_satspots, frame_center_radon, frame_shift, cube_shift
from vip_hci.var import fit_2dgaussian, frame_center 
from hciplot import plot_frames

from .misc import sanitize_header, get_file_name
from .irdis_filters import get_filter_path 



def open_dark(exp_time, exp_times_dark):
    """Open the dark file corresponding to the given exposure time."""

    if exp_time in exp_times_dark:
        dark_file = exp_times_dark[exp_time]

        if isinstance(dark_file, str):
            
            hdul = fits.open(dark_file)
            dark_data = hdul[0].data
            hdul.close()
            return np.nanmedian(dark_data, axis=0)

        elif isinstance(dark_file, list):
            
            dark_data_cube = []

            for dark_f in dark_file:

                hdul = fits.open(dark_f)
                dark_data = hdul[0].data
                hdul.close()

                if len(dark_data.shape) == 2:
                    dark_data_cube.append(dark_data)                
                elif len(dark_data.shape) == 3:
                    dark_data_cube.append(np.nanmedian(dark_data, axis=0))
                else:
                    raise ValueError()

            return np.nanmedian(dark_data_cube, axis=0)
    else:
        raise KeyError(f"No matching dark file for exposure time {exp_time}.")


def correct_for_dark(input_file, output_file, params):
    """Correct the science data by subtracting the dark data.
    
    """

    exp_times_dark = params['exp_times_dark']
    overwrite = params['overwrite']
    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))


    # output_file = output_folder + os.path.basename(filename).replace(cat, cat + '_MINUS_DARK')

    if os.path.exists(output_file) and not overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file)) )
        return

    hdul = fits.open(input_file)
    exp_time = hdul[0].header['EXPTIME']

    if exp_time not in exp_times_dark.keys():
        logging.info("Warning: No matching dark file for exposure time {}.".format(exp_time) )
        logging.info("Skipping dark subtraction for {}.".format(os.path.basename(input_file)))
        return
    
    dark_data = open_dark(exp_time, exp_times_dark)

    science_data = hdul[0].data

    science_data_minus_dark = np.zeros(science_data.shape)

    for i, image in enumerate(science_data):
        science_data_minus_dark[i] = science_data[i] - dark_data

    hdu = fits.PrimaryHDU(data=science_data_minus_dark)
    hdul_minus_dark = fits.HDUList([hdu])

    hdul_minus_dark[0].header = sanitize_header(hdul[0].header)


    hdul_minus_dark[0].header['HISTORY'] = 'DARK SUBTRACTED'
    if isinstance(exp_times_dark[exp_time], str):
        hdul_minus_dark[0].header['darkfile'] = os.path.basename(exp_times_dark[exp_time])  
         
    elif isinstance(exp_times_dark[exp_time], (list, np.ndarray)):
        for i, file in enumerate([os.path.basename(f) for f in exp_times_dark[exp_time]]): 
            hdul_minus_dark[0].header[f'darkfile{i}'] = file
    
    else:
        raise TypeError(f"Unknown type for dark file: {type(exp_times_dark[exp_time])}")

    hdul_minus_dark.writeto(output_file, overwrite=True)
    hdul_minus_dark.close()

    return



def make_bad_pix_mask(output_file, obs_night, filter_name, params):

    DARK_DF = params['DARK_DF']
    obs_run = obs_night[:7]
    darks = DARK_DF[(DARK_DF['obs_run']== obs_run) & (DARK_DF['filter']== filter_name)]['filepath']

    all_dark_files = []
    
    for file in darks:
        hdul = fits.open(file)
        cube = hdul[0].data

        for image in cube:

            all_dark_files.append(image)
        hdul.close()

    all_dark_files = np.array(all_dark_files)

    coef = 5

    sky = np.nanmedian(all_dark_files, axis = 0)

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

    # the SPHERE IRDIS detector has some known badspots, also include them: 
    circ_mask1 = circular_aperture(bpx_mask, 12, (394, 364))
    circ_mask2 = circular_aperture(bpx_mask, 6, (311, 1400))
    
    bpx_mask[circ_mask1 == 1] = 1
    bpx_mask[circ_mask2 == 1] = 1

    hdu = fits.PrimaryHDU(data=bpx_mask)
    hdul = fits.HDUList([hdu])

    hdul.writeto(output_file, overwrite=True)

    return



def correct_for_dark_ispy(input_file, output_file, params):
    """Correct the science data by subtracting the dark data.
    
    """
    DARK_DF = params['DARK_DF']
    overwrite = params['overwrite']

    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))

    if os.path.exists(output_file) and not overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file)) )
        return

    hdul = fits.open(input_file)
    science_data = hdul[0].data
    science_header = hdul[0].header

    obs_run = hdul[0].header['DATE'][:7] 
    exp_time = hdul[0].header['EXPTIME']
    filter_name = hdul[0].header['HIERARCH ESO INS COMB IFLT']

    DARKS = DARK_DF[(DARK_DF['obs_run'] == obs_run) & (DARK_DF['exp_time'] == exp_time)]['filepath'].to_list()


    if len(DARKS) == 0:
        logging.info("Warning: No matching dark file for exposure time {} in obs_run {}.".format(exp_time, obs_run) )
        logging.info("Skipping dark subtraction for {}.".format(os.path.basename(input_file)))
        return
    

    dark_data_cube = []

    for dark_f in DARKS:

        hdul = fits.open(dark_f)
        dark_data = hdul[0].data
        hdul.close()

        if len(dark_data.shape) == 2:
            dark_data_cube.append(dark_data)                
        elif len(dark_data.shape) == 3:
            dark_data_cube.append(np.nanmedian(dark_data, axis=0))

    if len(dark_data) > 1:
        dark_data = np.nanmedian(dark_data_cube, axis=0)
    elif len(dark_data) == 1:
        dark_data = dark_data_cube[0]
    else:
        raise ValueError

    science_data_minus_dark = np.zeros(science_data.shape)

    for i, image in enumerate(science_data):
        science_data_minus_dark[i] = science_data[i] - dark_data

    hdu = fits.PrimaryHDU(data=science_data_minus_dark)
    hdul_minus_dark = fits.HDUList([hdu])

    hdul_minus_dark[0].header = sanitize_header(science_header)


    hdul_minus_dark[0].header['HISTORY'] = 'DARK SUBTRACTED'

    if isinstance(DARKS, str):
        hdul_minus_dark[0].header['darkfile'] = os.path.basename(DARKS)  

    if isinstance(DARKS, (list, np.ndarray)) and len(DARKS) == 1:
        hdul_minus_dark[0].header['darkfile'] = os.path.basename(DARKS[0])  
         
    elif isinstance(DARKS, (list, np.ndarray)):
        for i, file in enumerate([os.path.basename(f) for f in DARKS]): 
            hdul_minus_dark[0].header[f'darkfile{i}'] = file
    
    else:
        raise TypeError(f"Unknown type for dark file: {type(DARKS)}")

    hdul_minus_dark.writeto(output_file, overwrite=True)
    hdul_minus_dark.close()

    return


def make_master_flat(flat_filepaths, params, plot=False):
    """Create a master flat by combining all flat files."""
    
    logging.info("Creating master flat...")
    logging.info("Files used for masterflat:")

    master_flat_filepath = params['masterflat_path']

    if os.path.exists(master_flat_filepath) and not params['overwrite']:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(master_flat_filepath)))
        return master_flat_filepath
    
    flat_data = []

    for flat in flat_filepaths:

        if not os.path.exists(flat):
            logging.info('\t -> {} does not exist, However this file should not appear here, check the flat uptaing code'.format(flat))
            continue

        logging.info('\t -> working on {}'.format(os.path.basename(flat)))
        hdul = fits.open(flat)
        header_master_flat = hdul[0].header

        if hdul[0].data.ndim == 2:
            flat_data.append(hdul[0].data)

        elif hdul[0].data.ndim == 3:
            logging.info('\t -> Flat {}s exists of multiple exposures, combining to median file'.format(hdul[0].header['EXPTIME']))
            median_flat = np.nanmedian(hdul[0].data, axis=0)
            flat_data.append(median_flat/np.max(median_flat))

        hdul.close()

    flat_data = np.array(flat_data)   
    master_flat = np.nanmedian(flat_data, axis=0)

    median_of_masterflat_left = np.nanmedian(master_flat[25:1020, 50:935])
    median_of_masterflat_right = np.nanmedian(master_flat[15:1010, 1080:1965])

    logging.info('median of master flat left {}, right {}'.format(median_of_masterflat_left, median_of_masterflat_right))

    master_flat[25:1020, 50:935] = master_flat[25:1020, 50:935] / median_of_masterflat_left
    master_flat[15:1010, 1080:1965] = master_flat[15:1010, 1080:1965] / median_of_masterflat_right

    if plot:
        show_img(master_flat, title='Master Flat')
        plt.savefig('master_flat.png', dpi=200, format='png')

    hdu = fits.PrimaryHDU(data=master_flat)
    hdul_masterflat = fits.HDUList([hdu])
    hdul_masterflat[0].header = header_master_flat
    hdul_masterflat[0].header['HISTORY'] = 'MASTER FLAT'
    for i, file in enumerate([os.path.basename(f) for f in flat_filepaths]):
        hdul_masterflat[0].header['FLAT{}'.format(i)] = file

    hdul_masterflat.writeto(master_flat_filepath, overwrite=True)
    hdul_masterflat.close()
    
    return


def get_science_exp_time(filepaths):

    all_exp_times = []
    for file in filepaths:

        hdul = fits.open(file)
        exp_time = hdul[0].header['EXPTIME']
        if exp_time not in all_exp_times:
            all_exp_times.append(exp_time)

    if len(all_exp_times) != 1:
        raise ValueError('several exposure times in timeseries: {}'.format(all_exp_times) )

    return all_exp_times[0]


def correct_for_flat(input_file, output_file, params):
    """Correct the science data by dividing by the master flat.
    
    """
    master_flat_filepath = params['masterflat_path']
    overwrite = params['overwrite']
    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))

    if os.path.exists(output_file) and not overwrite:
        logging.info('\t -> {} exists skipping file'.format(os.path.basename(output_file)))
        return

    master_flat = fits.getdata(master_flat_filepath)

    hdul = fits.open(input_file)
    science_data = hdul[0].data

    science_data_flat_corr = np.zeros(science_data.shape)

    for i, image in enumerate(science_data):
        science_data_flat_corr[i] = science_data[i]/master_flat

    hdu = fits.PrimaryHDU(data=science_data_flat_corr)
    hdul_flat_corr = fits.HDUList([hdu])
    hdul_flat_corr[0].header = hdul[0].header

    hdul_flat_corr[0].header['HISTORY'] = 'FLAT CORRECTED'
    hdul_flat_corr[0].header['MASTFLAT'] = os.path.basename(master_flat_filepath)

    hdul_flat_corr.writeto(output_file, overwrite=True)
    hdul_flat_corr.close()

    return


def bad_pixel_correction_mp(input_file, output_file, params):

    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))
    
    if os.path.exists(output_file) and not params['overwrite']:
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

    return


def bad_pixel_correction(input_file, output_file, params):

    logging.info('\t -> working on {}, will be saved as {}'.format(os.path.basename(input_file), os.path.basename(output_file)))
    
    if os.path.exists(output_file) and not params['overwrite']:
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

    return


def split_file(input_file, left_output_file, right_output_file, params):

    logging.info('\t -> working on {} splitting the frames, saves to {} & {}'.format(os.path.basename(input_file), os.path.basename(left_output_file), os.path.basename(right_output_file)))

    if os.path.exists(left_output_file) and os.path.exists(right_output_file) and not params['overwrite']:
        logging.info('\t -> exists skipping file {}'.format(os.path.basename(input_file)))
        return

    hdul = fits.open(input_file)

    bpm_mask_initial = fits.getdata(params['bad_pix_mask_filepath'])

    data = hdul[0].data

    # cube_sigcorr, bpm_mask = cube_fix_badpix_isolated(data, bpm_mask=bpm_mask_initial, correct_only=True,
    #                                                 frame_by_frame=False, mad=True, verbose=True, 
    #                                                 full_output=True, protect_mask=100) 

    left_frame = data[:, 25:1020, 50:935]
    right_frame = data[:, 15:1010, 1080:1965]

    left_frame_sigcorr, bpm_mask = cube_fix_badpix_isolated(left_frame, bpm_mask=bpm_mask_initial[25:1020, 50:935], correct_only=True,
                                                frame_by_frame=False, mad=True, verbose=True, full_output=True) 
    
    right_frame_sigcorr, bpm_mask = cube_fix_badpix_isolated(right_frame, bpm_mask=bpm_mask_initial[15:1010, 1080:1965], correct_only=True,
                                            frame_by_frame=False, mad=True, verbose=True, full_output=True) 

    hdu_left = fits.PrimaryHDU(data=left_frame_sigcorr)
    hdul_left = fits.HDUList([hdu_left])
    hdul_left[0].header = hdul[0].header

    hdu_right = fits.PrimaryHDU(data=right_frame_sigcorr)
    hdul_right = fits.HDUList([hdu_right])
    hdul_right[0].header = hdul[0].header

    hdul_left.writeto(left_output_file, overwrite=True)
    hdul_right.writeto(right_output_file, overwrite=True)
    
    hdul_left.close()
    hdul_right.close()
    hdul.close()
    return


def prepare_centering(input_files, output_dir):

    logging.info('preparing centering of files:')
    for file in input_files:

        #open the file 
        hdul = fits.open(file)

        #get the data
        data = hdul[0].data

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

    #DATAFRAME.to_csv(csv_path, index=False)
    
    return


def get_ratio_for_nd_filter(wave_filter, nd_filter, L_or_R, plot = False):
    
    # get transmittsion profile
    if 'DB' in wave_filter:
        wave_filter = wave_filter.replace('DB', 'D')

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

    convoluted_trans = trans_cut * nd_trans_cut


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


def create_flux_file(input_file, output_file, science_exp_time, L_or_R, center = (401, 466), plot = False, overwrite=True):

    if os.path.exists(output_file) and not overwrite:
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

    # since this is the transmittion profile so smaller than 1 we need to divide by it
    nd_ratio = get_ratio_for_nd_filter(wave_filter, nd_filter, L_or_R, plot=plot)
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
    median_sky_bkg = np.median(stacked_flux_image)
    logging.info('\t median sky backgorund in file {}: {}'.format(os.path.basename(output_file), median_sky_bkg))

    stacked_flux_image = stacked_flux_image - median_sky_bkg

    # fit a gauss to find the center 
    med_y, med_x = fit_2dgaussian(stacked_flux_image, crop=True, cent=center, cropsize=cropsize, threshold=True, sigfactor=4, full_output=False, debug=False)
    
    #cut out the center of the flux image
    # divide the shift in integer and float
    int_center_x = int(np.floor(med_x))
    int_center_y = int(np.floor(med_y))

    float_center_x = med_x - int_center_x
    float_center_y = med_y - int_center_y

    # first shift the center to a full pixel
    float_centered_image = frame_shift(stacked_flux_image, -float_center_y, -float_center_x, imlib='ndimage-interp', interpolation='biquintic')

    # then cut out 20 pixels around the central one
    int_centered_image = float_centered_image[int_center_y-20:int_center_y+21, int_center_x-20:int_center_x+21]

    # and make a larger cutout for the background
    background = float_centered_image.copy()
    background = background[int_center_y-50:int_center_y+51, int_center_x-50:int_center_x+51]
    background[30:70,30:70] = np.nan

    med_background = np.nanmedian(background)

    int_centered_image = int_centered_image - med_background

    # lastly rescale the flux 
    rescaled_image = int_centered_image * exp_time_ratio / nd_ratio 

    # and plot the final image (optional)
    if plot: 
        plot_dir = os.path.join(os.path.dirname(output_file),  'flux_pngs')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        show_img(rescaled_image, minper=1, maxper=99, size =2)
        plt.scatter(20,20, marker='x', color='red', s=100)
        plt.savefig(os.path.join(plot_dir,os.path.basename(output_file).replace('.fits', '.png')))
        plt.close()

        show_img(stacked_flux_image * exp_time_ratio / nd_ratio, minper=1, maxper=99, size =2)
        plt.scatter(med_x, med_y, marker='x', color='red', s=100)
        plt.xlim(med_x - 20, med_x  +20)
        plt.ylim(med_y - 20, med_y  +20)
        plt.savefig(os.path.join(plot_dir,os.path.basename(output_file).replace('.fits', '2.png')))
        plt.close()

    # lastly save the flux file 
    hdu_flux = fits.PrimaryHDU(data=rescaled_image)
    hdul_flux = fits.HDUList([hdu_flux])

    header['EXPTIMERATIO'] = exp_time_ratio
    header['ND_RATIO'] = 1/nd_ratio

    hdul_flux[0].header = header
    hdul_flux.writeto(output_file, overwrite=True)

    return


def get_satspots(input_file):

    logging.info('\t -> working on {} getting the centers'.format(os.path.basename(input_file)))

    csv_path = os.path.join(os.path.dirname(input_file) ,'centers.csv') 
    DF = pd.read_csv(csv_path)

    filename = os.path.basename(input_file)
    DF_Frame = DF[DF['frames'] == filename]
    centers = [ (DF_Frame['x{}'.format(f+1)].values[0], DF_Frame['y{}'.format(f+1)].values[0]) for f in range(4)]
        
    return centers


def get_center(input_file, center = None, sat_spots = None, plot = False):

    logging.info('getting center from file {}'.format(os.path.basename(input_file)))
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

    logging.info('using satsplots at {}'.format(sat_spots))
    # now comes the recentering, here we do it on the cube
    res = cube_recenter_satspots(data, xy=sat_spots_input, subi_size=21, verbose=False, debug=False, full_output=True, fit_type='gaus')

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

    return res


def make_the_cube(all_science, star_center_file, output_file_cube, output_file_angs, params, plot = False):

    cube = []
    angs = []
    for file in all_science: 
        data = fits.getdata(file)

        try:
            angend = fits.getheader(file)['HIERARCH ESO TEL PARANG END']
            angstart = fits.getheader(file)['HIERARCH ESO TEL PARANG START']

        except KeyError as e:
            print(file)
            raise KeyError(e)

        diff = (angstart - angend) / len(data)

        angs_i = np.linspace(angstart + diff/2, angend - diff/2, len(data))

        # add sphere offset
        angs_i += 135.99

        for ang, image in zip(angs_i, data): 
            
            # subtract the mmedian skybkg 
            median_sky_bkg = np.nanmedian(image)
            image = image - median_sky_bkg

            cube.append(image)
            # int divide ang by 360 to only have angles between 0 and 360
            angs.append(ang % 360)

    angs = np.array(angs)
    cube = np.array(cube)

    # now shift the cube to the center

    satspots = get_satspots(star_center_file)

    res = get_center(star_center_file, sat_spots = satspots, plot = plot)

    center_shifty = np.mean(res[1])
    center_shiftx = np.mean(res[2])
    
    int_shift_x = int(np.floor(center_shiftx))
    int_shift_y = int(np.floor(center_shifty))

    float_center_x = center_shiftx - int_shift_x
    float_center_y = center_shifty - int_shift_y

    # first shift the center of the cube by the full integer number

    cube_center = frame_center(cube[0])

    y_size_cube = cube.shape[1]
    x_size_cube = cube.shape[2]

    expanded_cube = np.zeros((len(cube),1025,1025))
    center_expanded_cube = frame_center(expanded_cube[0])

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

    # make the bad pixel correction
    # correct the individual frames for bad pixels, e.g. cosmic rays, should be not needed for ADI 
    # cube_sigcorr_indv, bpm_mask_tmp = cube_fix_badpix_isolated(expanded_cube,
    #                                                   sigma_clip=5, num_neig=29, size=9, frame_by_frame=True,
    #                                                   mad=True, verbose=True, full_output=True, protect_mask=100, nproc=params['nproc'])

    cube_sigcorr_indv = expanded_cube.copy()

    # now shift the center to a full pixel
    float_centered_cube = cube_shift(np.nan_to_num(cube_sigcorr_indv), float_center_y, float_center_x, imlib='ndimage-interp', interpolation='biquintic')


    masked_cube = float_centered_cube * nan_mask
    
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

    # now save the cube
    hdu2 = fits.PrimaryHDU(data=masked_cube)
    hdul2 = fits.HDUList([hdu2])
    header2 = fits.getheader(all_science[0])

    for i, file in enumerate(all_science):
        header2['SCIENC{}'.format(i)] = os.path.basename(file)

    header2['CENTSTAR'] = os.path.basename(star_center_file)
    header2['SHIFTX'] = np.mean(center_shiftx)
    header2['SHIFTY'] = np.mean(center_shifty)

    hdul2[0].header = header2

    hdul2.writeto(output_file_cube, overwrite=True)
    hdul2.close()
    np.savetxt(output_file_angs, angs, fmt='%f')

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