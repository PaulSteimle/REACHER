import os
import time
import logging
from os.path import basename, join 
from glob import glob 

from . import multiprocessing_routines as mp_routines 
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
            params.master_dark = os.path.join(params.master_dark_folder, 'master_dark_{}_{}.fits'.format(obs_run, exp_time))

            #get all the input files 
            dark_filepaths = params.DARK_DF[(params.DARK_DF['obs_run']== obs_run) & (params.DARK_DF['exp_time'] == exp_time)]['filepath']

            logging.info('\n all input darks for obs run {}, exp_time {}:'.format(obs_run, exp_time))
            logging.info('\n'.join(dark_filepaths))
            logging.info('are combined to masterdark {} \n'.format(os.path.basename(params.master_dark)))

            # create the masterdark
            _ = wrappers.make_master_dark(dark_filepaths, params)

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

            wrappers.make_master_flat(flats_list, master_dark_list, params)

    logging.info('######################################################################################')
    logging.info('#### Finished all master flats! ######################################################')
    logging.info('######################################################################################\n\n')
    return 


def make_all_masterbackgrounds(input_folder, output_folder, nproc = None, overwrite=True, skip_bad_pix=False):

    star_output_folder = os.path.join(output_folder, 'BACKGROUND')
    master_bkg_folder = os.path.join(output_folder, 'master_background')
    master_dark_folder = os.path.join(output_folder, 'master_dark')
    master_flat_folder = os.path.join(output_folder, 'master_flat')
    bad_pix_mask_folder = os.path.join(output_folder, 'bad_pix_masks')

    if not os.path.exists(master_bkg_folder):
        os.makedirs(master_bkg_folder)

    if not os.path.exists(join(output_folder, 'DARK_CORR')):
        os.makedirs(join(output_folder, 'DARK_CORR'))

    stage_output_folder = os.path.join(star_output_folder, 'stage1')
    
    logging.info('making DataFrames')
    DARK_DF = make_dark_DF(input_folder)
    FLAT_DF = make_flat_DF(input_folder)
    BACKGROUND_DF = make_background_DF(input_folder)

    params = {'main_output_folder': output_folder,
            'output_folder': stage_output_folder,
            'master_dark_folder': master_dark_folder,
            'input_folder': input_folder,
            'nproc': nproc,
            'overwrite': overwrite,
            'DARK_DF':DARK_DF,
            'FLAT_DF':FLAT_DF,
            'BACKGROUND_DF':BACKGROUND_DF,
            }
    
    # first correct all fiels for master-darks 
    logging.info('collection all instumental background files for dark correction\n')
    
    # set up the list for all inputs 
    all_raw_bkgs = []
    all_dark_corr_bkgs = []
    all_params = []

    # get all the individual obs runs
    obs_runs = BACKGROUND_DF['obs_run'].unique()
    for obs_run in obs_runs:

        # get all the unique exp times per obs run 
        exp_times = BACKGROUND_DF[(BACKGROUND_DF['obs_run'] == obs_run)]['exp_time'].unique()
        for exp_time in exp_times:

            # set all the parameter objects 
            params['obs_run'] = obs_run
            params['exp_time'] = exp_time
            params['master_dark'] = os.path.join(params['master_dark_folder'], 'master_dark_{}_{}.fits'.format(obs_run, exp_time))
            
            # define input and output filepths
            bkgs = BACKGROUND_DF[(BACKGROUND_DF['obs_run']== obs_run) & (BACKGROUND_DF['exp_time']== exp_time)]['filepath']
            dark_corrected_files = [join(output_folder, 'DARK_CORR', basename(f).replace('BACKGROUND', 'BACKGROUND_DARK_CORR') ) for f in bkgs]

            logging.info('\n all input bkgs for obs run {}, exp_time {}:'.format(obs_run, exp_time))
            logging.info('\n'.join([basename(f) for f in bkgs]))
            logging.info('are corrected with masterdark {} \n'.format(os.path.basename(params['master_dark'])))

            # extend and append the file and params lists
            all_raw_bkgs.extend(bkgs)
            all_dark_corr_bkgs.extend(dark_corrected_files)
            all_params.append(params.copy())

    #run dark correction in MP 
    logging.info('Correcting for dark current. Files are processed in parallel.')
    _ = mpr.correct_darks_ispy_multiprocessing_new(all_raw_bkgs, all_dark_corr_bkgs, all_params)

    logging.info('Stacking Background files to Masterbackground now:')

    # update bkgs only dark corr files are used in further analysis
    all_dark_corr_bkgs = glob(join(output_folder, 'DARK_CORR', 'BACKGROUND_DARK_CORR*.fits'))
    all_dark_corr_bkgs.sort()

    dark_corr_bkgs = [basename(f).replace('BACKGROUND_DARK_CORR', 'BACKGROUND') for f in all_dark_corr_bkgs]

    obs_nights = BACKGROUND_DF['obs_night'].unique()
    for obs_night in obs_nights:

        # get all filters in obs night
        filters = BACKGROUND_DF[(BACKGROUND_DF['obs_night']== obs_night)]['filter'].unique()
        for filter_name in filters:
                            
            # get all exp times 
            exp_times = BACKGROUND_DF[(BACKGROUND_DF['obs_night']== obs_night) &
                                      (BACKGROUND_DF['filter']== filter_name)]['exp_time'].unique()            
            for exp_time in exp_times:

                # get all nd filter configs 
                nd_filter = BACKGROUND_DF[(BACKGROUND_DF['obs_night']== obs_night) & 
                                          (BACKGROUND_DF['filter']== filter_name) & 
                                          (BACKGROUND_DF['exp_time']== exp_time)]['nd_filter'].unique()
                for nd in nd_filter:

                    logging.info('Working on master_background_{}_{}_{}_{}.fits\n'.format(obs_night, filter_name, exp_time, nd))

                    # get all background files for the observation in question 
                    bkgs = BACKGROUND_DF[(BACKGROUND_DF['obs_night']== obs_night) &
                                         (BACKGROUND_DF['filter']== filter_name) & 
                                         (BACKGROUND_DF['exp_time']== exp_time) & 
                                         (BACKGROUND_DF['nd_filter']== nd)]['filename']


                    # check which of the files were processed
                    bkg_processed = [f for f in bkgs if f in dark_corr_bkgs]

                    # define master bad pixel mask, masterflat and output filepath
                    params['bad_pix_mask_filepath'] = os.path.join(bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_night[:7]))
                    params['master_flat_path'] = os.path.join(master_flat_folder, 'master_flat_{}_{}.fits'.format(obs_night, filter_name))
                    params['master_bkg_path'] = os.path.join(master_bkg_folder, 'master_background_{}_{}_{}_{}.fits'.format(obs_night, filter_name, exp_time, nd))

                    # set other parameters that might be relevant 
                    params['exp_time'] = exp_time
                    params['nd_filter'] = nd
                    
                    if len(bkg_processed) == 0:
                        logging.info('no bkg dark corr files found, skipping {} {}'.format(obs_night, filter_name))
                        continue
                    if not os.path.exists(params['master_flat_path']):
                        logging.info('no master flat found, skipping {} {} {} {}'.format(obs_night, filter_name, exp_time, nd))
                        continue

                    dark_corrected_bkg_files = [join(output_folder, 'DARK_CORR', basename(f).replace('BACKGROUND', 'BACKGROUND_DARK_CORR') ) for f in bkg_processed]

                    # Create the masterbackground
                    wrappers.make_master_bkg(dark_corrected_bkg_files, params)
                    logging.info('')
                    logging.info('Master background created: {}'.format(os.path.basename(params['master_bkg_path'])))
                    logging.info('')

    return


def run_pre_processing_ispy(params):

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


    _ = mp_routines.calibrate_files_multiprocessing(all_raw_files,
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
    mp_routines.split_files_multiprocessing(all_calibrated_files, left_files, right_files, params)

    # prepare the centering for the star files 
    prepare_centering_files = [f for f in right_files + left_files if 'CENTER' in f]

    ### and prepare for the centering
    wrappers.prepare_centering(prepare_centering_files, params.stage2_output_folder)

    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return

      
def recentering_bad_pix_correction_ispy(params, ref_cubes_dict=None): 

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

    # make the almost done Flux files 
    exp_time_science = wrappers.get_science_exp_time(file_handler.get_files('science', None, params.star_input_folder) )

    for L_or_R in ['LEFT','RIGHT']:
        stage3_output_folder = os.path.join(params.star_output_folder, L_or_R.lower(), 'raw_files')
        params.L_or_R = L_or_R

        # check for sky files:
        all_sky_files = file_handler.get_files('sky', None, params.stage2_output_folder, prefix=L_or_R)

        if len(all_sky_files) == 0:
            logging.warning('no sky files found for {} {}'.format(params.star_name, L_or_R))
        
        elif isinstance(all_sky_files, list):
            logging.info('found {} sky files for {} {}'.format(len(all_sky_files), params.star_name, L_or_R))
            wrappers.make_mastersky(all_sky_files, params)
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
            wrappers.create_flux_file(input_file, output_file, exp_time_science, L_or_R, params)

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
        shift_yx = wrappers.get_center(star_center_files[0], plot= params.make_plots)

        # make the cubes
        wrappers.make_the_cube(input_files, centered_cube, output_angs, shift_yx, star_center_files[0], params)

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
            wrappers.make_the_cube(ref_star_input_files, ref_star_centered_cube, ref_star_output_angs, shift_yx, star_center_files[0], params)

    logging.info('Finished making the cubes')

    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return 

