import os
import time
import logging

from astropy.io import fits 
from os.path import basename, join 
from glob import glob 

from .multiprocessing_routines import split_files_multiprocessing, correct_darks_ispy_multiprocessing, correct_darks_multiprocessing, correct_flats_multiprocessing, correct_badpix_multiprocessing, make_cube_quadratic_multiprocessing
from . import routines
from .misc import FILEHANDLER,  make_dark_DF, make_flat_DF
import shutil


def make_bad_pix_masks(input_folder, output_folder, nproc = None, overwrite=True):

    bad_pix_mask_folder = os.path.join(output_folder, 'bad_pix_masks')

    logging.info('making DataFrames')
    DARK_DF = make_dark_DF(input_folder)

    params = {'main_output_folder': output_folder,
            'input_folder': input_folder,
            'nproc': nproc,
            'overwrite': overwrite,
            'DARK_DF':DARK_DF,
            }


    if not os.path.exists(bad_pix_mask_folder):
        os.makedirs(bad_pix_mask_folder)

    star_parent_dir = os.path.join(input_folder, 'DARK', 'IRDIS')
    all_obs_nights = os.listdir(star_parent_dir)
    for obs_night in all_obs_nights:
        obs_night_parent_dir = join(star_parent_dir,obs_night)
        obs_run = obs_night[:-3]
        filters = os.listdir(obs_night_parent_dir)

        for filter_name in filters:

            # create the bad_pix_mask
            bad_pix_mask_filepath = os.path.join(bad_pix_mask_folder, 'bad_pix_mask_{}_{}.fits'.format(obs_night, filter_name) )
            params['bad_pix_mask_filepath'] = bad_pix_mask_filepath
            routines.make_bad_pix_mask(bad_pix_mask_filepath, obs_night, filter_name, params)

    return


def make_all_masterflats(input_folder, output_folder, nproc = None, overwrite=True, skip_bad_pix = False):
    
    star_output_folder = os.path.join(output_folder, 'FLAT')
    master_flat_folder = os.path.join(output_folder, 'master_flat')

    if not os.path.exists(master_flat_folder):
        os.makedirs(master_flat_folder)

    if not os.path.exists(join(output_folder, 'DARK_CORR')):
        os.makedirs(join(output_folder, 'DARK_CORR'))

    if not os.path.exists(join(output_folder, 'BAD_PIX_CORR')):
        os.makedirs(join(output_folder, 'BAD_PIX_CORR'))

    stage_output_folder = os.path.join(star_output_folder, 'stage1')
    
    logging.info('making DataFrames')
    DARK_DF = make_dark_DF(input_folder)
    FLAT_DF = make_flat_DF(input_folder)

    params = {'main_output_folder': output_folder,
            'output_folder': stage_output_folder,
            'input_folder': input_folder,
            'nproc': nproc,
            'overwrite': overwrite,
            'DARK_DF':DARK_DF,
            'FLAT_DF':FLAT_DF,
            }

    logging.info('dark correcting all flat files:')
    star_parent_dir = os.path.join(input_folder, 'FLAT', 'IRDIS')
    all_obs_nights = os.listdir(star_parent_dir)
    for obs_night in all_obs_nights:
        obs_night_parent_dir = join(star_parent_dir,obs_night)
        obs_run = obs_night[:-3]
        filters = os.listdir(obs_night_parent_dir)

        for filter_name in filters:

            flats = FLAT_DF[(FLAT_DF['obs_night']== obs_night) & (FLAT_DF['filter']== filter_name)]['filepath']
            darks = DARK_DF[(DARK_DF['obs_run']== obs_run) & (DARK_DF['filter']== filter_name)]['filepath']


            logging.info('all input files for obs night {}:'.format(obs_night))
            logging.info('############################################################')
            logging.info('all {} files'.format('FLAT'))
            for file in flats:
                    logging.info('\t ->  {}'.format(file))

            logging.info('all {} files'.format('DARK'))
            for file in darks:
                    logging.info('\t ->  {}'.format(file))

            # define input and output filepths
            raw_files = flats            
            dark_corrected_files = [join(output_folder, 'DARK_CORR', basename(f).replace('FLAT', 'FLAT_DARK_CORR') ) for f in flats]
            
            logging.info('Correcting for dark current. Files are processed in parallel.')
            _ = correct_darks_ispy_multiprocessing(raw_files, dark_corrected_files, params)


    # delete the flats that were not processed from the file handler
    logging.info('Correcting for bad pixels. Files are processed in parallel.')
    all_dark_corr_flats = glob(output_folder + 'DARK_CORR/FLAT_DARK_CORR*.fits')
    all_dark_corr_flats.sort()
    bad_pix_corr_flats = [join(output_folder, 'BAD_PIX_CORR', basename(f).replace('FLAT_DARK_CORR', 'FLAT_BAD_PIX_CORR') ) for f in all_dark_corr_flats]
    

    if not skip_bad_pix:
        _ = correct_badpix_multiprocessing(all_dark_corr_flats, bad_pix_corr_flats, params)
    
    else:
        for input_file, output_file in zip(all_dark_corr_flats, bad_pix_corr_flats):
            logging.info('copy {} to {}'.format(input_file, output_file))
            shutil.copy2(input_file, output_file)

    logging.info('Masterflats are now created:')
    bad_pix_corr_flats = [basename(f).replace('FLAT_BAD_PIX_CORR', 'FLAT') for f in os.listdir(join(output_folder, 'BAD_PIX_CORR',))]
    for obs_night in all_obs_nights:
        obs_night_parent_dir = join(star_parent_dir,obs_night)
        filters = os.listdir(obs_night_parent_dir)

        for filter_name in filters:

            logging.info('Working on {} {}'.format(obs_night, filter_name))
            
            params['masterflat_path'] = os.path.join(master_flat_folder, 'masterflat_{}_{}.fits'.format(obs_night, filter_name) )

            if not os.path.exists(params['masterflat_path']):
                # Here comes the badpix correction
                logging.info('Correcting flats for bad pixels. Files are processed in parallel.')
    
                flats = FLAT_DF[(FLAT_DF['obs_night']== obs_night) & (FLAT_DF['filter']== filter_name)]['filename']
                flats_processed = [f for f in flats if f in bad_pix_corr_flats]
                
                if len(flats_processed) == 0:
                    logging.info('no flats processed, skipping {} {}'.format(obs_night, filter_name))
                    continue

                bad_pix_corrected_flat_files = [join(output_folder, 'BAD_PIX_CORR', basename(f).replace('FLAT', 'FLAT_BAD_PIX_CORR') ) for f in flats_processed]
 
                # Create the masterflat
                routines.make_master_flat(bad_pix_corrected_flat_files, params)
                logging.info('')
                logging.info('Master flat created: {}'.format(os.path.basename(params['masterflat_path'])))
                logging.info('')

    return 


def run_pre_processing_ispy(input_folder, star, obs_night, filter_name, output_folder, nproc = None, overwrite=True):

    logging.info('working on {}, {}, {}'.format(star, obs_night, filter_name))

    star_output_folder = os.path.join(output_folder, star, 'IRDIS', obs_night, filter_name)

    obs_run = obs_night[:-3]

    master_flat_folder = os.path.join(output_folder, 'master_flat')

    if not os.path.exists(master_flat_folder):
        os.makedirs(master_flat_folder)

    stage_output_folder = os.path.join(star_output_folder, 'stage1')

    params = {'main_output_folder': output_folder,
              'output_folder': stage_output_folder,
              'input_folder': input_folder,
              'star': star,
              'obs_run': obs_night[:-3],
              'obs_night': obs_night,
              'filter': filter_name,
              'nproc': nproc,
              'overwrite': overwrite}

    DARK_DF = make_dark_DF(input_folder)
    FLAT_DF = make_flat_DF(input_folder)

    flats = FLAT_DF[(FLAT_DF['obs_run']== obs_run) & (FLAT_DF['filter']== filter_name)]['filename']
    darks = DARK_DF[(DARK_DF['obs_run']== obs_run) & (DARK_DF['filter']== filter_name)]['filename']

    file_handler = FILEHANDLER()
    file_handler.read_directory_ispy(input_folder, flats, darks)

    logging.info('all input files:')
    logging.info('############################################################')
    for filetype in ['flat','science','flux','dark','star_center']:
        logging.info('all {} files'.format(filetype))
        for file in getattr(file_handler, filetype):
            logging.info('\t ->  {}'.format(file))
        logging.info('')

    ### first find the dark file that matches the science frame

    params = {'main_output_folder': output_folder,
            'output_folder': stage_output_folder,
            'input_folder': input_folder,
            'nproc': nproc,
            'overwrite': overwrite,
            'DARK_DF':DARK_DF,
            'FLAT_DF':FLAT_DF,
            }

    # check if the processed data directory exists, if not create it
    if not os.path.exists(stage_output_folder):
        os.makedirs(stage_output_folder)

    bad_pix_mask_folder = os.path.join(output_folder, 'bad_pix_masks')
    bad_pix_mask_filepath = os.path.join(bad_pix_mask_folder, 'bad_pix_mask_{}_{}.fits'.format(obs_night, filter_name) )
    params['bad_pix_mask_filepath'] = bad_pix_mask_filepath
    
    # Now correct for darks
    logging.info('Correcting for dark current. Files are processed in parallel.')

    # define input and output filepaths and run dark correction 
    raw_files = file_handler.get_files(['flux','star_center','science'], None, input_folder) 
    dark_corrected_files = file_handler.get_files(['flux','star_center','science'], 'DARK_CORR', stage_output_folder)   
    _ = correct_darks_ispy_multiprocessing(raw_files, dark_corrected_files, params)

    # delete the flats that were not processed from the file handler
    dark_corr_flats = glob(stage_output_folder + '/FLAT_DARK_CORR*.fits')
    dark_corr_flats.sort()
    dark_corr_flats = [os.path.basename(f).replace('_DARK_CORR', '') for f in dark_corr_flats]
    file_handler.flat = dark_corr_flats

    params['masterflat_path'] = os.path.join(master_flat_folder, 'masterflat_{}_{}.fits'.format(obs_night, filter_name) )
    logging.info('masterflat for this star: {}'.format(params['masterflat_path']))

    if not os.path.exists(params['masterflat_path']):
        # Here comes the badpix correction
        logging.info('Correcting flats for bad pixels. Files are processed in parallel.')
        dark_corrected_flat_files = file_handler.get_files('flat', 'DARK_CORR', stage_output_folder)
        bad_pix_corrected_flat_files = file_handler.get_files('flat', 'BAD_PIX_CORR', stage_output_folder)
        _ = correct_badpix_multiprocessing(dark_corrected_flat_files, bad_pix_corrected_flat_files, params)

        # Create the masterflat
        routines.make_master_flat(bad_pix_corrected_flat_files, params)
        logging.info('')
        logging.info('Master flat created: {}'.format(os.path.basename(params['masterflat_path'])))
        logging.info('')
    else:
        logging.info('Master flat already exists: {}'.format(os.path.basename(params['masterflat_path'])))


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

      
def recentering_bad_pix_correction_ispy(input_folder, star, obs_night, filter_name, output_folder, nproc = None, overwrite=True, skip_bad_pix=False): 

    logging.info('working on {}, {}, {}'.format(star, obs_night, filter_name))
    
    if output_folder is None:
        output_folder = input_folder.replace('stage1', 'stage2')

    star_output_folder = os.path.join(output_folder, star, 'IRDIS', obs_night, filter_name)

    stage1_output_folder = os.path.join(star_output_folder, 'stage1')
    stage_output_folder = os.path.join(star_output_folder, 'stage2')
    left_output_folder = os.path.join(star_output_folder, 'left')
    right_output_folder = os.path.join(star_output_folder, 'right')

    file_handler = FILEHANDLER()
    file_handler.read_directory_ispy(input_folder, None, None)

    if len(file_handler.star_center) == 0:
        logging.info('no starcenter files, is prob a ref star')

        # need to find the star_center files that are used for these targets


        return
    
    bad_pix_mask_folder = os.path.join(output_folder, 'bad_pix_masks')
    bad_pix_mask_filepath = os.path.join(bad_pix_mask_folder, 'bad_pix_mask_{}_{}.fits'.format(obs_night, filter_name) )

    params = {'main_output_folder': output_folder,
              'output_folder': stage_output_folder,
              'input_folder': input_folder,
              'nproc': nproc,
              'overwrite': overwrite,
              'bad_pix_mask_filepath':bad_pix_mask_filepath}

    if not os.path.exists(stage_output_folder):
        os.makedirs(stage_output_folder)

    if not os.path.exists(left_output_folder):
        os.makedirs(left_output_folder)

    if not os.path.exists(right_output_folder):
        os.makedirs(right_output_folder)

    # make the almost done Flux files 
    exp_time_science = routines.get_science_exp_time(file_handler.get_files('science', None, input_folder) )

    logging.info('creating flux files')
    for L_or_R in ['LEFT','RIGHT']:

        stage3_output_folder = os.path.join(star_output_folder, L_or_R.lower(), 'raw_files')

        input_files = file_handler.get_files('flux', L_or_R, stage1_output_folder)

        logging.info('working on {}'.format([basename(f) for f in input_files]))

        if len(input_files) == 1:
            output_files = [os.path.join(stage3_output_folder, L_or_R.lower() + '_flux.fits')]
        elif len(input_files) > 1:
            output_files = file_handler.get_files('flux', L_or_R + '_CUT_OUT', stage3_output_folder)
        else:
            raise ValueError('unkown number')

        for input_file, output_file in zip(input_files, output_files):
            routines.create_flux_file(input_file, output_file, exp_time_science, L_or_R, overwrite=overwrite, plot=False)

    logging.info('')
    logging.info('now creating the cubes')
    
    for L_or_R in ['LEFT','RIGHT']:
        
        stage3_output_folder = os.path.join(star_output_folder, L_or_R.lower(), 'raw_files')
        input_files = file_handler.get_files('science', L_or_R, stage1_output_folder)
        star_center_files = file_handler.get_files('star_center', L_or_R, stage1_output_folder)
        centered_cube = os.path.join(stage3_output_folder, L_or_R.lower() + '_master_cube_PCA.fits')
        output_angs = os.path.join(stage3_output_folder, L_or_R.lower() + '_parallactic_angles.txt')

        routines.make_the_cube(input_files, star_center_files[0], centered_cube, output_angs, params, plot = True)

    logging.info('Finished making the cubes')

    # lastly do the bad pix correction for flux and science files
    # for now only science files as flux seems good

    # if not skip_bad_pix:
    #     for L_or_R in ['LEFT','RIGHT']:
    #         centered_cube = os.path.join(stage_output_folder, L_or_R.lower() + '_centered_cube.fits')
    #         bad_pix_corr_cube = os.path.join(stage_output_folder, L_or_R.lower() + '_centered_cube_bad_pix_corr.fits')
    #         routines.bad_pixel_correction(centered_cube, bad_pix_corr_cube, params)

    # # and make the cubes quadratic
    # for L_or_R in ['LEFT','RIGHT']:

    #     stage3_output_folder = os.path.join(star_output_folder, L_or_R.lower())
    #     if skip_bad_pix:
    #         bad_pix_corr_cubes = os.path.join(stage_output_folder, L_or_R.lower() + '_centered_cube_corr.fits')
    #     else:
    #         bad_pix_corr_cubes = os.path.join(stage_output_folder, L_or_R.lower() + '_centered_cube_bad_pix_corr.fits')
    #     output_cubes = os.path.join(stage3_output_folder, L_or_R.lower() + '_mastercube.fits')
    #     routines.make_cube_quadratic(bad_pix_corr_cubes, output_cubes, params)

    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return 

