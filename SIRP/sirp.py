import os
import time
import logging
from os.path import basename, join 
from glob import glob 

from . import wrappers
from .misc import make_dark_DF, make_flat_DF, make_background_DF, make_star_DF, find_calib_files
from .filehandler import FILEHANDLER


def make_bad_pix_masks(params):

    logging.info('######################################################################################')
    logging.info('#### Creating all bad pixel masks! ###################################################')
    logging.info('######################################################################################')

    if not os.path.exists(params.bad_pix_mask_folder):
        os.makedirs(params.bad_pix_mask_folder)

    obs_runs = params.DARK_DF['obs_run'].unique()

    for obs_run in obs_runs:
        # create the bad_pix_mask
        wrappers.make_bad_pix_mask_wrapper(obs_run, params)

    # lastly create the clump mask for SPHERE/IRDIS
    wrappers.make_IRDIS_clump_mask_wrapper(params)

    return


def make_all_masterdarks(params):

    logging.info('######################################################################################')
    logging.info('#### Creating all master darks! ######################################################')
    logging.info('######################################################################################')

    if not os.path.exists(params.master_dark_folder):
        os.makedirs(params.master_dark_folder)

    # get all the indiviaul obs runs
    obs_runs = params.DARK_DF['obs_run'].unique()
    
    # loop through all obs runs
    for obs_run in obs_runs:

       # first define the bad pixel mask for each of the runs 
        params.bad_pix_mask_filepath = os.path.join(params.bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_run))
        #get all unique exp times to loop for them 
        exp_times = params.DARK_DF[(params.DARK_DF['obs_run']== obs_run)]['exp_time'].unique()

        # and create a master dark for each exp time
        for exp_time in exp_times:
            # set the params object
            #params.exp_time = exp_time
            params.masterdark_filepath = os.path.join(params.master_dark_folder, 'master_dark_{}_{}.fits'.format(obs_run, exp_time))

            #get all the input files 
            dark_filepaths = params.DARK_DF[(params.DARK_DF['obs_run']== obs_run) & (params.DARK_DF['exp_time'] == exp_time)]['filepath']

            logging.info('\n all input darks for obs run {}, exp_time {}:'.format(obs_run, exp_time))
            logging.info('\n'.join(dark_filepaths))
            logging.info('are combined to masterdark {} \n'.format(os.path.basename(params.masterdark_filepath)))

            # create the masterdark
            _ = wrappers.make_master_dark_wrapper(dark_filepaths, params)

    logging.info('######################################################################################')
    logging.info('#### Finished all master darks! ######################################################')
    logging.info('######################################################################################\n\n')

    return


def make_all_masterflats(params):
    
    logging.info('######################################################################################')
    logging.info('#### Creating all master flats! ######################################################')
    logging.info('######################################################################################')

    if not os.path.exists(params.master_flat_folder):
        os.makedirs(params.master_flat_folder)

    obs_nights = params.FLAT_DF['obs_night'].unique()
    for obs_night in obs_nights:
        obs_run = obs_night[:-3]

        # get all filters in obs night
        filters = params.FLAT_DF[(params.FLAT_DF['obs_night']== obs_night)]['filter'].unique()
        for filter_name in filters:
            
            logging.info('Working on {} {}'.format(obs_night, filter_name))

            # define the parameters object
            params.bad_pix_mask_filepath = os.path.join(params.bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_run))
            params.masterflat_filepath = os.path.join(params.master_flat_folder, 'master_flat_{}_{}.fits'.format(obs_night, filter_name))

            filter_and_obs_night_DF = params.FLAT_DF[(params.FLAT_DF['obs_run'] == obs_run) & (params.FLAT_DF['obs_night']== obs_night)]

            # for each file get the corresponding master dark
            exp_times = filter_and_obs_night_DF['exp_time']
            master_dark_list = [os.path.join(params.master_dark_folder, 'master_dark_{}_{}.fits'.format(obs_run, exp_time)) for exp_time in exp_times]
            flats_list = filter_and_obs_night_DF['filepath']

            wrappers.make_master_flat_wrapper(flats_list, master_dark_list, params)

    logging.info('######################################################################################')
    logging.info('#### Finished all master flats! ######################################################')
    logging.info('######################################################################################\n\n')
    return 


def make_all_masterbackgrounds(params):
    logging.info('######################################################################################')
    logging.info('#### Creating all master backgrounds! ################################################')
    logging.info('######################################################################################')

    if not os.path.exists(params.master_background_folder):
        os.makedirs(params.master_background_folder)

    obs_nights = params.FLAT_DF['obs_night'].unique()
    for obs_night in obs_nights:
        obs_run = obs_night[:-3]

        # get all filters in obs night
        filters = params.FLAT_DF[(params.FLAT_DF['obs_night']== obs_night)]['filter'].unique()
        for filter_name in filters:
            
            logging.info('Working on {} {}'.format(obs_night, filter_name))

            # define the parameters object
            params.bad_pix_mask_filepath = os.path.join(params.bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_run))
            params.masterflat_filepath = os.path.join(params.master_flat_folder, 'master_flat_{}_{}.fits'.format(obs_night, filter_name))

            filter_and_obs_night_DF = params.FLAT_DF[(params.FLAT_DF['obs_run'] == obs_run) & (params.FLAT_DF['obs_night']== obs_night)]

            # for each file get the corresponding master dark
            exp_times = filter_and_obs_night_DF['exp_time']
            master_dark_list = [os.path.join(params.master_dark_folder, 'master_dark_{}_{}.fits'.format(obs_run, exp_time)) for exp_time in exp_times]
            master_flat_list = [os.path.join(params.master_flat_folder, 'master_flat_{}_{}.fits'.format(obs_night, filter_name))] * len(exp_times)
            flats_list = filter_and_obs_night_DF['filepath']

            wrappers.make_master_background_wrapper(flats_list, master_dark_list, params)

    logging.info('######################################################################################')
    logging.info('#### Finished all master flats! ######################################################')
    logging.info('######################################################################################\n\n')



def pre_calibration_and_extraction(params):

    logging.info('working on {}, {}, {}'.format(params.star_name, params.obs_night, params.filter_name))
    obs_run = params.obs_night[:-3]

    # get all the calibration files and check if they exist, otherwise returns an error
    # todo for the future this shuld have an option to create the missing files
    params.STAR_DF = find_calib_files(params.STAR_DF, params)

    file_handler = FILEHANDLER()
    file_handler.read_directory_ispy(params.star_input_folder, None, None)

    logging.info('############################################################')
    logging.info('All input files:\n')    
    for filetype in ['science','flux','star_center','sky']:
        logging.info('all {} files'.format(filetype))
        for file in getattr(file_handler, filetype):
            logging.info('\t ->  {}'.format(file))
        logging.info('')

    # check if the processed data directory exists, if not create it
    if not os.path.exists(params.stage1_output_folder):
        os.makedirs(params.stage1_output_folder)
    if not os.path.exists(params.stage2_output_folder):
        os.makedirs(params.stage2_output_folder)

    # Now correct for darks
    logging.info('Correcting for masterdark, masterflat, and bad pixels as well as instrumental background for K-band observations.')

    # set up the list for all inputs 
    all_raw_files = params.STAR_DF['filepath'].to_list()
    masterdark_list = params.STAR_DF['master_dark'].to_list()
    masterflat_list = params.STAR_DF['master_flat'].to_list()
    params.bad_pix_mask_filepath = os.path.join(params.bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_run))
    all_calibrated_files = [os.path.join(params.stage1_output_folder, 'CALIB_' + basename(f)) for f in all_raw_files]

    _ = wrappers.calibrate_files_multiprocessing_wrapper(all_raw_files,
                                                        all_calibrated_files,
                                                        masterdark_list,
                                                        masterflat_list,
                                                        params)    
    
    logging.info('All raw files have been calibrated!')

    logging.info('Splitting the calibrated files. Files are processed in parallel.')
    # define the output paths for left and right files 
    right_files = [os.path.join(params.stage2_output_folder, 'RIGHT_' + basename(f)) for f in all_raw_files]
    left_files  = [os.path.join(params.stage2_output_folder, 'LEFT_'  + basename(f)) for f in all_raw_files]

    # split the files 
    _ = wrappers.split_files_multiprocessing_wrapper(all_calibrated_files, left_files, right_files, params)

    # prepare the centering for the star files 
    prepare_centering_files = [f for f in right_files + left_files if 'CENTER' in f]

    ### and prepare for the centering
    wrappers.prepare_centering_wrapper(prepare_centering_files, params.stage2_output_folder)

    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return

      
def centering_and_bad_pix_correction(params, ref_cubes_dict=None): 

    logging.info('working on {}, {}, {}'.format(params.star_name, params.obs_night, params.filter_name))

    file_handler = FILEHANDLER()
    file_handler.read_directory_ispy(params.star_input_folder, None, None)

    if isinstance(ref_cubes_dict, dict) and params.star_name in ref_cubes_dict.keys():
        if isinstance(ref_cubes_dict[params.star_name], dict):
            ref_stars = ref_cubes_dict[params.star_name][params.obs_night]
        elif isinstance(ref_cubes_dict[params.star_name], (list, str)):
            ref_stars = ref_cubes_dict[params.star_name]
        else:
            ref_stars = []
    else:
        ref_stars = []

    if isinstance(ref_stars, str):
        ref_stars = [ref_stars]
    elif ref_stars is None:
        ref_stars = []

    if len(ref_stars) == 0:

        logging.info('star has no ref stars!')
    else:
        logging.info('star has ref stars: {}'.format(ref_stars))

    if len(file_handler.star_center) == 0:
        logging.info('no starcenter files, is prob a ref star')
        logging.info('ref stars get processed with the star they are affiliated with')        
        return
    
    if not os.path.exists(params.left_output_folder):
        os.makedirs(params.left_output_folder)

    if not os.path.exists(params.right_output_folder):
        os.makedirs(params.right_output_folder)

    all_exp_times = params.STAR_DF[params.STAR_DF['type'] == 'OBJECT']['exp_time'].unique()
    if not len(all_exp_times) == 1:

        csv_filepath = join(params.star_output_folder, 'STAR_DF_{}_{}_{}.csv'.format(params.star_name, params.obs_night, params.filter_name))
        params.STAR_DF.to_csv(csv_filepath, index=False)
        raise ValueError('science files have different exposure times: {}.  Check the STAR_DF .csv file {}'.format(all_exp_times, csv_filepath))
    else:
        exp_time_science = all_exp_times[0]

    # make the almost done Flux files 
    # exp_time_science = wrappers.get_science_exp_time_wrapper(file_handler.get_files('science', None, params.star_input_folder) )

    for L_or_R in ['LEFT','RIGHT']:
        stage3_output_folder = os.path.join(params.star_output_folder, L_or_R.lower(), 'raw_files')
        params.L_or_R = L_or_R

        # check for sky files:
        all_sky_files = file_handler.get_files('sky', None, params.stage2_output_folder, prefix=L_or_R)

        if len(all_sky_files) == 0:
            logging.warning('no sky files found for {} {}'.format(params.star_name, L_or_R))
        
        elif isinstance(all_sky_files, list):
            logging.info('found {} sky files for {} {}'.format(len(all_sky_files), params.star_name, L_or_R))
            wrappers.make_mastersky_wrapper(all_sky_files, params)
            params.sky_files_exist = True
 
    logging.info('creating flux files')

    for L_or_R in ['LEFT','RIGHT']:       

        stage3_output_folder = os.path.join(params.star_output_folder, L_or_R.lower(), 'raw_files')
        params.L_or_R = L_or_R
        input_files = file_handler.get_files('flux', None, params.stage2_output_folder, prefix=L_or_R)

        if len(input_files) == 1:
            output_files = [os.path.join(stage3_output_folder, L_or_R.lower() + '_flux.fits')]
        elif len(input_files) > 1:

            output_files = [os.path.join(stage3_output_folder, L_or_R.lower() + f'_flux_{i+1}.fits') for i in range(len(input_files))]
        else:
            raise ValueError('unkown number')

        for input_file, output_file in zip(input_files, output_files):
            wrappers.create_flux_file_wrapper(input_file, output_file, exp_time_science, L_or_R, params)

    if params.only_flux:
        logging.info('only flux files are created, skipping the rest of the pre-processing')
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
        logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return
    
    logging.info('')
    logging.info('now creating the cubes')
    
    for L_or_R in ['LEFT','RIGHT']:
        
        params.L_or_R = L_or_R

        # define all input and output paths
        input_files = file_handler.get_files('science', None, params.stage2_output_folder, prefix=L_or_R)
        centered_cube = os.path.join(params.L_or_R_output_folder, L_or_R.lower() + '_master_cube.fits')
        output_angs = os.path.join(params.L_or_R_output_folder, L_or_R.lower() + '_parallactic_angles.txt')

        # get the center from the starcenter files, take the first one thats taken 
        star_center_files = file_handler.get_files('star_center', None, params.stage2_output_folder, prefix=L_or_R)
        shift_yx = wrappers.get_center_wrapper(star_center_files[0], params)

        # make the cubes
        wrappers.make_the_cube_wrapper(input_files, centered_cube, output_angs, shift_yx, star_center_files[0], params)

        logging.info('now onto the ref stars')
        
        for ref_star in ref_stars:
                
            logging.info('making the cube of {}'.format(ref_star))
            params.ref_star_name = ref_star

            if not os.path.exists(params.ref_L_or_R_output_folder):
                os.makedirs(params.ref_L_or_R_output_folder)

            #initialize the file handler
            ref_star_file_handler = FILEHANDLER()
            ref_star_file_handler.read_directory_ispy(params.ref_star_input_folder, None, None)

            # define all filepaths
            ref_star_input_files = ref_star_file_handler.get_files('science', None, params.ref_stage2_output_folder, prefix=L_or_R)
            ref_star_centered_cube = os.path.join(params.ref_L_or_R_output_folder, L_or_R.lower() + '_master_cube.fits')
            ref_star_output_angs = os.path.join(params.ref_L_or_R_output_folder, L_or_R.lower() + '_parallactic_angles.txt')

            logging.info('using files:')
            for file in ref_star_input_files:
                logging.info('\t -> {}'.format(basename(file)))

            # make the cubes
            wrappers.make_the_cube_wrapper(ref_star_input_files, ref_star_centered_cube, ref_star_output_angs, shift_yx, star_center_files[0], params)

    logging.info('Finished making the cubes')

    if params.clean_stage_1_and_2:
        logging.info('cleaning up stage 1 and 2 folders')
        if os.path.exists(params.stage1_output_folder):
            os.system('rm -r {}'.format(params.stage1_output_folder))
        if os.path.exists(params.stage2_output_folder):
            os.system('rm -r {}'.format(params.stage2_output_folder))

    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return 



def cleanup_all_folders(params):
    """" Cleanup all folders if wanted. """
    
    if params.clean_calibrations:

        logging.info('cleaning up all calibration folders\n')

        if os.path.exists(params.recentering_folder):
            logging.info('\t -> removing {}'.format(params.recentering_folder))
            os.system('rm -r {}'.format(params.recentering_folder))

        if os.path.exists(params.master_dark_folder):
            logging.info('\t -> removing {}'.format(params.master_dark_folder))
            os.system('rm -r {}'.format(params.master_dark_folder))

        if os.path.exists(params.master_flat_folder):
            logging.info('\t -> removing {}'.format(params.master_flat_folder))   
            os.system('rm -r {}'.format(params.master_flat_folder))

        if os.path.exists(params.bad_pix_mask_folder):
            logging.info('\t -> removing {}'.format(params.bad_pix_mask_folder))
            os.system('rm -r {}'.format(params.bad_pix_mask_folder))

        if os.path.exists(params.master_background_folder):
            logging.info('\t -> removing {}'.format(params.master_background_folder))
            os.system('rm -r {}'.format(params.master_background_folder))

        if os.path.exists(params.recentering_folder):
            logging.info('\t -> removing {}'.format(params.recentering_folder))
            os.system('rm -r {}'.format(params.recentering_folder))
    
    if params.clean_stage_1_and_2:    
        logging.info('cleaning up all stage 1 and 2 folders\n')
        all_stars = [f for f in os.listdir(params.raw_input_dir) if f not in ['FLAT','DARK', 'DB_H23', 'BACKGROUND', 'DISTORT', 'DARK,BACKGROUND']]

        for star in all_stars:

            star_parent_dir = os.path.join(params.raw_input_dir,star,'IRDIS')
            all_obs_nights = os.listdir(star_parent_dir)

            for obs_night in all_obs_nights:
                obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
                filters = os.listdir(obs_night_parent_dir)

                for filter_name in filters:
                    
                    params.star_name = star
                    params.obs_night = obs_night
                    params.filter_name = filter_name
                    params.get_star_dataframe()
                    logging.info('\t -> cleaning up stage 1 and 2 for {}, {}, {}'.format(star, obs_night, filter_name))
                    if os.path.exists(params.stage1_output_folder):
                        os.system('rm -r {}'.format(params.stage1_output_folder))
                    if os.path.exists(params.stage2_output_folder):
                        os.system('rm -r {}'.format(params.stage2_output_folder))