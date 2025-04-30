import os
import time
import logging

from astropy.io import fits 
from glob import glob 

from .multiprocessing_routines import split_files_multiprocessing, correct_darks_multiprocessing, correct_flats_multiprocessing, correct_badpix_multiprocessing, make_cube_quadratic_multiprocessing
from . import routines
from .misc import FILEHANDLER, sanitize_header


def run_pre_processing(input_folder, output_folder=None, nproc = None, overwrite=True):

    if output_folder is None:
        output_folder = input_folder.replace('_raw_data', '_processed_data')

    stage_output_folder = os.path.join(output_folder, 'stage1')

    params = {'main_output_folder': output_folder,
              'output_folder': stage_output_folder,
              'input_folder': input_folder,
              'nproc': nproc,
              'overwrite': overwrite}

    file_handler = FILEHANDLER()
    file_handler.read_directory(input_folder)

    logging.info('all input files:')
    logging.info('############################################################')
    for filetype in ['flat','science','flux','dark','star_center']:
        logging.info('all {} files'.format(filetype))
        for file in getattr(file_handler, filetype):
            logging.info('\t ->  {}'.format(file))
        logging.info('')

    ### first find the dark file that matches the science frame
    exp_times_dark = {}

    for dark in file_handler.get_files('dark', None, input_folder):
        hdul = fits.open(dark)
        exp_time = hdul[0].header['EXPTIME']

        if exp_time in exp_times_dark:
            logging.info('Warning: Duplicate dark file for exposure time {}'.format(exp_time) )
            exp_times_dark[exp_time] = [dark, *exp_times_dark[exp_time]]
        else:
            exp_times_dark[exp_time] = dark
        hdul.close()


    params['exp_times_dark'] = exp_times_dark

    # check if the processed data directory exists, if not create it
    if not os.path.exists(stage_output_folder):
        os.makedirs(stage_output_folder)
    
    # Now correct for darks
    logging.info('Correcting for dark current. Files are processed in parallel.')

    # define input and output filepths
    # 
    raw_files = file_handler.get_files(['flat','flux','star_center','science'], None, input_folder)
    dark_corrected_files = file_handler.get_files(['flat','flux','star_center','science'], 'DARK_CORR', stage_output_folder)   

    _ = correct_darks_multiprocessing(raw_files, dark_corrected_files, params)

    # delete the flats that were not processed from the file handler
    dark_corr_flats = glob(stage_output_folder + '/FLAT_DARK_CORR*.fits')
    dark_corr_flats.sort()
    dark_corr_flats = [os.path.basename(f).replace('_DARK_CORR', '') for f in dark_corr_flats if '2023-07-24' in os.path.basename(f)]
    file_handler.flat = dark_corr_flats

    # Here comes the badpix correction
    logging.info('Correcting flats for bad pixels. Files are processed in parallel.')

    dark_corrected_flat_files = file_handler.get_files('flat', 'DARK_CORR', stage_output_folder)
    bad_pix_corrected_flat_files = file_handler.get_files('flat', 'BAD_PIX_CORR', stage_output_folder)

    _ = correct_badpix_multiprocessing(dark_corrected_flat_files, bad_pix_corrected_flat_files, params)


    params['masterflat_path'] = os.path.join(stage_output_folder, 'MASTERFLAT.fits')
    # Create the masterflat
    routines.make_master_flat(bad_pix_corrected_flat_files, params)
    logging.info('')
    logging.info('Master flat created: {}'.format(os.path.basename(params['masterflat_path'])))
    logging.info('')

    time.sleep(0.5)

    # And correct for the masterflat                        
    logging.info('Correcting for flat field. Files are processed in parallel.')

    dark_corrected_files = file_handler.get_files(['flux','star_center','science'], 'DARK_CORR', stage_output_folder)   
    flat_corrected_files = file_handler.get_files(['flux','star_center','science'], 'FLAT_CORR', stage_output_folder)   
    _ = correct_flats_multiprocessing(dark_corrected_files, flat_corrected_files, params)
    time.sleep(0.5)
    

    ### now split the files in right and left cube 
    right_files = file_handler.get_files(['flux','star_center','science'], 'RIGHT', stage_output_folder)  
    left_files = file_handler.get_files(['flux','star_center','science'], 'LEFT', stage_output_folder)  
    split_files_multiprocessing(flat_corrected_files, left_files, right_files, params)

    exp_time_science = routines.get_science_exp_time(file_handler.get_files('science', None, input_folder) )

    prepare_centering_files = file_handler.get_files('star_center', 'LEFT', stage_output_folder)\
                              + file_handler.get_files('star_center', 'RIGHT', stage_output_folder)

    ### and prepare for the centering
    routines.prepare_centering(prepare_centering_files, stage_output_folder)

    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return

      
def recentering_bad_pix_correction(input_folder, output_folder, nproc = None, overwrite=True): 

    if output_folder is None:
        output_folder = input_folder.replace('stage1', 'stage2')

    stage1_output_folder = os.path.join(output_folder, 'stage1')
    stage_output_folder = os.path.join(output_folder, 'stage2')
    stage3_output_folder = os.path.join(output_folder, 'stage3')

    file_handler = FILEHANDLER()
    file_handler.read_directory(input_folder)


    params = {'main_output_folder': output_folder,
              'output_folder': stage_output_folder,
              'input_folder': input_folder,
              'nproc': nproc,
              'overwrite': overwrite}

    if not os.path.exists(stage_output_folder):
        os.makedirs(stage_output_folder)

    if not os.path.exists(stage3_output_folder):
        os.makedirs(stage3_output_folder)


    # make the almost done Flux files 
    exp_time_science = routines.get_science_exp_time(file_handler.get_files('science', None, input_folder) )

    for L_or_R in ['LEFT','RIGHT']:
        for input_file, output_file in zip(file_handler.get_files('flux', L_or_R, stage1_output_folder),
                                           file_handler.get_files('flux', L_or_R + '_CUT_OUT', stage3_output_folder)):
            
            routines.create_flux_file(input_file, output_file, exp_time_science, L_or_R, overwrite=True, plot=False)

    for L_or_R in ['LEFT','RIGHT']:

        input_files = file_handler.get_files('science', L_or_R, stage1_output_folder)
        star_center_files = file_handler.get_files('star_center', L_or_R, stage1_output_folder)
        centered_cube = os.path.join(stage_output_folder, L_or_R.lower() + '_centered_cube.fits')
        output_angs = os.path.join(stage3_output_folder, L_or_R.lower() + '_parallactic_angles.txt')

        routines.make_the_cube(input_files, star_center_files[0], centered_cube, output_angs, plot = True)

    logging.info('Finished making the cubes, now comes the bad pixel removal part:')

    # lastly do the bad pix correction for flux and science files
    # for now only science files as flux seems good
    # for L_or_R in ['LEFT','RIGHT']:

    #     centered_cube = os.path.join(stage_output_folder, L_or_R.lower() + '_centered_cube.fits')
    #     bad_pix_corr_cube = os.path.join(stage_output_folder, L_or_R.lower() + '_centered_cube_bad_pix_corr.fits')
    #     routines.bad_pixel_correction(centered_cube, bad_pix_corr_cube, params)

    # and make the cubes quadratic
    for L_or_R in ['LEFT','RIGHT']:

        #bad_pix_corr_cubes = os.path.join(stage_output_folder, L_or_R.lower() + '_centered_cube_bad_pix_corr.fits')
        bad_pix_corr_cubes = os.path.join(stage_output_folder, L_or_R.lower() + '_centered_cube.fits')  
        output_cubes = os.path.join(stage3_output_folder, L_or_R.lower() + '_mastercube.fits')
        routines.make_cube_quadratic(bad_pix_corr_cubes, output_cubes, params)

    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return 

