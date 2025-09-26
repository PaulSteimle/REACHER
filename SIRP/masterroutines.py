import os
import time
import logging
from os.path import basename, join 
from glob import glob 

from . import multiprocessing_routines as mpr 
from . import wrappers
from .misc import make_dark_DF, make_flat_DF, make_background_DF, make_star_DF, find_calib_files
from .filehandler import FILEHANDLER


def make_bad_pix_masks(input_folder, output_folder, params):

    bad_pix_mask_folder = os.path.join(output_folder, 'bad_pix_masks')

    logging.info('making DataFrames')
    DARK_DF = make_dark_DF(input_folder)

    if not os.path.exists(bad_pix_mask_folder):
        os.makedirs(bad_pix_mask_folder)

    obs_runs = DARK_DF['obs_run'].unique()

    for obs_run in obs_runs:

        # create the bad_pix_mask
        bad_pix_mask_filepath = os.path.join(bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_run) )
        params['bad_pix_mask_filepath'] = bad_pix_mask_filepath
        wrappers.make_bad_pix_mask(bad_pix_mask_filepath, obs_run, params)

    return


def make_all_masterdarks(input_folder, output_folder, params):

    logging.info('creating master darks')
    master_dark_folder = os.path.join(output_folder, 'master_dark')
    bad_pix_mask_folder = os.path.join(output_folder, 'bad_pix_masks')

    if not os.path.exists(master_dark_folder):
        os.makedirs(master_dark_folder)

    logging.info('making DataFrames')
    DARK_DF = make_dark_DF(input_folder)

    # params = {'main_output_folder': output_folder,
    #         'input_folder': input_folder,
    #         'nproc': nproc,
    #         'overwrite': overwrite,
    #         'DARK_DF':DARK_DF,
    #         }

    # get all the indiviaul obs runs
    obs_runs = DARK_DF['obs_run'].unique()
    
    for obs_run in obs_runs:
        
        params['bad_pix_mask_filepath'] = os.path.join(bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_run))
        #get all unique exp times to loop for 
        exp_times = DARK_DF[(DARK_DF['obs_run']== obs_run)]['exp_time'].unique()

        for exp_time in exp_times:
            # set the params object
            params['exp_time'] = exp_time
            params['master_dark'] = os.path.join(master_dark_folder, 'master_dark_{}_{}.fits'.format(obs_run, exp_time))

            #get all the input files 
            dark_filepaths = DARK_DF[(DARK_DF['obs_run']== obs_run) & (DARK_DF['exp_time'] == exp_time)]['filepath']

            logging.info('\n all input darks for obs run {}, exp_time {}:'.format(obs_run, exp_time))
            logging.info('\n'.join(dark_filepaths))
            logging.info('are combined to masterdark {} \n'.format(os.path.basename(params['master_dark'])))

            # create the masterdark
            _ = wrappers.make_master_dark(dark_filepaths, params)
    return


def make_all_masterflats(input_folder, output_folder, nproc = None, overwrite=True, skip_bad_pix = False):
    
    logging.info('######################################################################################')
    logging.info('#### Creating all master flats! ######################################################')
    logging.info('######################################################################################')
    star_output_folder = os.path.join(output_folder, 'FLAT')
    master_flat_folder = os.path.join(output_folder, 'master_flat')
    master_dark_folder = os.path.join(output_folder, 'master_dark')
    bad_pix_mask_folder = os.path.join(output_folder, 'bad_pix_masks')

    if not os.path.exists(master_flat_folder):
        os.makedirs(master_flat_folder)

    if not os.path.exists(join(output_folder, 'DARK_CORR')):
        os.makedirs(join(output_folder, 'DARK_CORR'))

    stage_output_folder = os.path.join(star_output_folder, 'stage1')
    
    logging.info('making DataFrames')
    DARK_DF = make_dark_DF(input_folder)
    FLAT_DF = make_flat_DF(input_folder)

    params = {'main_output_folder': output_folder,
            'output_folder': stage_output_folder,
            'input_folder': input_folder,
            'master_dark_folder': master_dark_folder,
            'nproc': nproc,
            'overwrite': overwrite,
            'DARK_DF':DARK_DF,
            'FLAT_DF':FLAT_DF,
            }

    # first correct all fiels for master-darks 
    logging.info('collection all flat files for dark correction\n')
    
    # set up the list for all inputs 
    all_raw_flats = []
    all_dark_corr_flats = []
    all_params = []

    # get all the individual obs runs
    obs_runs = FLAT_DF['obs_run'].unique()
    for obs_run in obs_runs:

        # get all the unique exp times per obs run 
        exp_times = FLAT_DF[(FLAT_DF['obs_run']== obs_run)]['exp_time'].unique()
        for exp_time in exp_times:

            # set all the parameter objects 
            params['obs_run'] = obs_run
            params['exp_time'] = exp_time
            params['master_dark'] = os.path.join(params['master_dark_folder'], 'master_dark_{}_{}.fits'.format(obs_run, exp_time))
            
            # define input and output filepths
            flats = FLAT_DF[(FLAT_DF['obs_run']== obs_run) & (FLAT_DF['exp_time']== exp_time)]['filepath']
            dark_corrected_files = [join(output_folder, 'DARK_CORR', basename(f).replace('FLAT', 'FLAT_DARK_CORR') ) for f in flats]

            logging.info('\n all input flats for obs run {}, exp_time {}:'.format(obs_run, exp_time))
            logging.info('\n'.join([basename(f) for f in flats]))
            logging.info('are corrected with masterdark {} \n'.format(os.path.basename(params['master_dark'])))

            # extend and append the file and params lists
            all_raw_flats.extend(flats)
            all_dark_corr_flats.extend(dark_corrected_files)
            all_params.append(params.copy())

    #run dark correction in MP 
    logging.info('Correcting for dark current. Files are processed in parallel.')
    _ = mpr.correct_darks_ispy_multiprocessing_new(all_raw_flats, all_dark_corr_flats, all_params)

    logging.info('Stacking Flats to Masterflats now:')
    # update flats only dark corr files are used in further analysis
    all_dark_corr_flats = glob(output_folder + 'DARK_CORR/FLAT_DARK_CORR*.fits')
    all_dark_corr_flats.sort()
    dark_corr_flats = [basename(f).replace('FLAT_DARK_CORR', 'FLAT') for f in all_dark_corr_flats]

    # get all the individual obs runs
    obs_nights = FLAT_DF['obs_night'].unique()
    print(obs_nights)
    for obs_night in obs_nights:

        # get all filters in obs night
        filters = FLAT_DF[(FLAT_DF['obs_night']== obs_night)]['filter'].unique()
        for filter_name in filters:
            
            logging.info('Working on {} {}'.format(obs_night, filter_name))

            # define the parameters object
            params['bad_pix_mask_filepath'] = os.path.join(bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_run))
            params['master_flat_path'] = os.path.join(master_flat_folder, 'master_flat_{}_{}.fits'.format(obs_night, filter_name))

            # get all flats for observations in question
            flats = FLAT_DF[(FLAT_DF['obs_night']== obs_night) & (FLAT_DF['filter']== filter_name)]['filename']
            #check if they got dark corrected 
            flats_processed = [f for f in flats if f in dark_corr_flats]
            
            # check there are flats and if there is a bad pixel mask 
            if len(flats_processed) == 0:
                logging.info('no flats processed, skipping {} {}'.format(obs_night, filter_name))
            elif not os.path.exists(params['bad_pix_mask_filepath']):
                logging.info('no bad pixel mask found, skipping {} {}'.format(obs_night, filter_name))   
            else:
                # set the input paths for all corrected dark files  
                dark_corrected_flat_files = [join(output_folder, 'DARK_CORR', basename(f).replace('FLAT', 'FLAT_DARK_CORR') ) for f in flats_processed]

                # Create the masterflat
                wrappers.make_master_flat(dark_corrected_flat_files, params)
                logging.info('')
                logging.info('Master flat created: {}'.format(os.path.basename(params['master_flat_path'])))
                logging.info('')

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


def run_pre_processing_ispy(input_folder, star, obs_night, filter_name, output_folder, nproc = None, overwrite=True):

    logging.info('working on {}, {}, {}'.format(star, obs_night, filter_name))

    star_output_folder = os.path.join(output_folder, star, 'IRDIS', obs_night, filter_name)
    star_input_folder = os.path.join(input_folder, star, 'IRDIS', obs_night, filter_name)


    obs_run = obs_night[:-3]

    master_flat_folder = os.path.join(output_folder, 'master_flat')

    if not os.path.exists(master_flat_folder):
        os.makedirs(master_flat_folder)

    stage1_output_folder = os.path.join(star_output_folder, 'stage1')
    stage2_output_folder = os.path.join(star_output_folder, 'stage2')

    params = {'main_output_folder': output_folder,
              'output_folder': stage1_output_folder,
              'input_folder': input_folder,
              'master_dark_folder': join(output_folder, 'master_dark'),
              'master_background_folder': join(output_folder, 'master_background'),
              'master_flat_folder': join(output_folder, 'master_flat'),
              'bad_pix_mask_folder': join(output_folder, 'bad_pix_masks'),
              'star': star,
              'obs_run': obs_night[:-3],
              'obs_night': obs_night,
              'filter': filter_name,
              'nproc': nproc,
              'overwrite': overwrite}


    STAR_DF = make_star_DF(input_folder, star, obs_night, filter_name)

    # get all the calibration files and check if they exist, otherwise returns an error 
    #todo for the future this shuld have an option to create the missing files 
    STAR_DF = find_calib_files(STAR_DF, params)

    file_handler = FILEHANDLER()
    file_handler.read_directory_ispy(star_input_folder, None, None)

    logging.info('############################################################')
    logging.info('All input files:\n')    
    for filetype in ['science','flux','star_center','sky']:
        logging.info('all {} files'.format(filetype))
        for file in getattr(file_handler, filetype):
            logging.info('\t ->  {}'.format(file))
        logging.info('')


    # check if the processed data directory exists, if not create it
    if not os.path.exists(stage1_output_folder):
        os.makedirs(stage1_output_folder)
    if not os.path.exists(stage2_output_folder):
        os.makedirs(stage2_output_folder)

    # Now correct for darks
    logging.info('Correcting for masterdark, masterflat, and bad pixels as well as instrumental background for K-band observations.')

    # set up the list for all inputs 
    all_raw_files = []
    all_calibrated_files = []
    all_params = []

    for index, row in STAR_DF.iterrows():
        
        input_filepath = STAR_DF.loc[index, 'filepath']
        output_filepath = os.path.join(stage1_output_folder, 'CALIB_' + basename(input_filepath) )

        params['master_dark'] = STAR_DF.loc[index, 'master_dark']
        params['master_flat'] = STAR_DF.loc[index, 'master_flat'] 
        params['bad_pix_mask'] = STAR_DF.loc[index, 'bad_pix_mask']

        if 'DB_K12' in filter_name or 'BB_K' in filter_name:
            params['master_background'] = STAR_DF.loc[index, 'master_background'] 
        else:
            params['master_background'] = None

        all_raw_files.append(input_filepath)
        all_calibrated_files.append(output_filepath)
        all_params.append(params.copy())

    _ = mpr.calibrate_files_multiprocessing(all_raw_files, all_calibrated_files, all_params)    
    logging.info('All raw files have been calibrated!')

    # check for sky files:
    if len(getattr(file_handler, 'sky')) > 0:
        params['sky_file'] = [f for f in all_calibrated_files if 'SKY' in f]
    else:
        params['sky_file'] = None

    logging.info('Splitting the calibrated files. Files are processed in parallel.')
    # define the output paths for left and right files 
    right_files = [os.path.join(stage2_output_folder, 'RIGHT_' + basename(f)) for f in all_raw_files]
    left_files = [os.path.join(stage2_output_folder, 'LEFT_' + basename(f)) for f in all_raw_files]

    # split the files 
    mpr.split_files_multiprocessing(all_calibrated_files, left_files, right_files, params)

    # prepare the centering for the star files 
    prepare_centering_files = [f for f in [right_files + left_files] if 'CENTER' in f]

    ### and prepare for the centering
    wrappers.prepare_centering(prepare_centering_files, stage2_output_folder)

    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return

      
def recentering_bad_pix_correction_ispy(input_folder, star, obs_night, filter_name, output_folder, ref_cubes_dict= None, nproc = None, overwrite=True, skip_bad_pix=False, only_flux=False, starcenter_DF=None): 

    logging.info('working on {}, {}, {}'.format(star, obs_night, filter_name))
    star_input_folder = os.path.join(input_folder, star, 'IRDIS', obs_night, filter_name)

    if output_folder is None:
        output_folder = input_folder.replace('stage1', 'stage2')

    star_output_folder = os.path.join(output_folder, star, 'IRDIS', obs_night, filter_name)

    stage1_output_folder = os.path.join(star_output_folder, 'stage1')
    stage_output_folder = os.path.join(star_output_folder, 'stage2')
    left_output_folder = os.path.join(star_output_folder, 'left', 'raw_files')
    right_output_folder = os.path.join(star_output_folder, 'right', 'raw_files')

    file_handler = FILEHANDLER()
    file_handler.read_directory_ispy(star_input_folder, None, None)

    if isinstance(ref_cubes_dict, dict) and star in ref_cubes_dict.keys():
        if isinstance(ref_cubes_dict[star], dict):
            ref_stars = ref_cubes_dict[star][obs_night]
        elif isinstance(ref_cubes_dict[star], (list, str)):
            ref_stars = ref_cubes_dict[star]
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
    
    bad_pix_mask_folder = os.path.join(output_folder, 'bad_pix_masks')
    bad_pix_mask_filepath = os.path.join(bad_pix_mask_folder, 'bad_pix_mask_{}_{}.fits'.format(obs_night[:7], filter_name) )

    params = {'main_output_folder': output_folder,
              'output_folder': stage_output_folder,
              'input_folder': input_folder,
              'nproc': nproc,
              'overwrite': overwrite,
              'bad_pix_mask_filepath':bad_pix_mask_filepath,
              'star': star,
              'obs_night': obs_night,
              'filter': filter_name}

    if not os.path.exists(stage_output_folder):
        os.makedirs(stage_output_folder)

    if not os.path.exists(left_output_folder):
        os.makedirs(left_output_folder)

    if not os.path.exists(right_output_folder):
        os.makedirs(right_output_folder)

    # make the almost done Flux files 
    exp_time_science = wrappers.get_science_exp_time(file_handler.get_files('science', None, star_input_folder) )

    logging.info('creating flux files')
    for L_or_R in ['LEFT','RIGHT']:

        stage3_output_folder = os.path.join(star_output_folder, L_or_R.lower(), 'raw_files')

        input_files = file_handler.get_files('flux', L_or_R, stage1_output_folder)

        if len(input_files) == 1:
            output_files = [os.path.join(stage3_output_folder, L_or_R.lower() + '_flux_fft.fits')]
        elif len(input_files) > 1:
            output_files = file_handler.get_files('flux', L_or_R + '_CUT_OUT_fft', stage3_output_folder)
        else:
            raise ValueError('unkown number')

        for input_file, output_file in zip(input_files, output_files):
            wrappers.create_flux_file(input_file, output_file, exp_time_science, L_or_R, overwrite=overwrite, plot=False)

    if only_flux:
        logging.info('only flux files are created, skipping the rest of the pre-processing')
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
        logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return
    
    logging.info('')
    logging.info('now creating the cubes')
    
    for L_or_R in ['LEFT','RIGHT']:
        
        # define all input and output paths
        stage3_output_folder = os.path.join(star_output_folder, L_or_R.lower(), 'raw_files')
        input_files = file_handler.get_files('science', L_or_R, stage1_output_folder)
        centered_cube = os.path.join(stage3_output_folder, L_or_R.lower() + '_master_cube_PCA_fft.fits') #_master_cube_PCA_bad_pix
        output_angs = os.path.join(stage3_output_folder, L_or_R.lower() + '_parallactic_angles.txt')

        # get the center from the starcenter files, take the first one thats taken 
        star_center_files = file_handler.get_files('star_center', L_or_R, stage1_output_folder)
        shift_yx = wrappers.get_center(star_center_files[0])

        # make the cubes
        wrappers.make_the_cube(input_files, centered_cube, output_angs, shift_yx, star_center_files[0], params, plot = True)

        logging.info('now onto the ref stars')
        
        for ref_star in ref_stars:
                
            logging.info('making the cube of {}'.format(ref_star))
            # define all input and output paths
            ref_star_input_folder = os.path.join(input_folder, ref_star, 'IRDIS', obs_night, filter_name)
            ref_star_stage3_output_folder = os.path.join(output_folder, ref_star, 'IRDIS', obs_night, filter_name, L_or_R.lower(), 'raw_files')
            ref_star_stage1_output_folder = os.path.join(output_folder, ref_star, 'IRDIS', obs_night, filter_name, 'stage1')

            if not os.path.exists(ref_star_stage3_output_folder):
                os.makedirs(ref_star_stage3_output_folder)

            #initialize the file handler
            ref_star_file_handler = FILEHANDLER()
            ref_star_file_handler.read_directory_ispy(ref_star_input_folder, None, None)

            # define all filepaths
            ref_star_input_files = ref_star_file_handler.get_files('science', L_or_R, ref_star_stage1_output_folder)
            ref_star_centered_cube = os.path.join(ref_star_stage3_output_folder, L_or_R.lower() + '_master_cube_PCA.fits')
            ref_star_output_angs = os.path.join(ref_star_stage3_output_folder, L_or_R.lower() + '_parallactic_angles.txt')

            logging.info('using files:')
            for file in ref_star_input_files:
                logging.info('\t -> {}'.format(basename(file)))

            # make the cubes
            wrappers.make_the_cube(ref_star_input_files, ref_star_centered_cube, ref_star_output_angs, shift_yx, star_center_files[0], params, plot = True)

    logging.info('Finished making the cubes')

    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    logging.info('!!!!!!!!!!!!!!SUCCESFULLY FINISHED!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return 

