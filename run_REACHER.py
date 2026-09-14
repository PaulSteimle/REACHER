
########################################################################################################
#######                                                                                          #######
####### This file is an example script to run the REACHER pipeline on a multiple target survey.  #######
#######                                                                                          #######
########################################################################################################

# first import the REACHER module as well as other necessary modules
import REACHER 
import logging
import os

# define the parameters for the REACHER pipeline see also the REACHER.Parameters() class for more details
pars = REACHER.Parameters()

# define input and output directories
pars.raw_input_dir = '/path/to/raw/data/'
pars.reduced_output_dir = '/path/to/raw/reduced_cubes/'

# setup logging
REACHER.setup_logging(log_file=os.path.join(pars.reduced_output_dir, 'REACHER.log'), level=logging.INFO)

# set number of processors for parallel processing
pars.nproc = 100
# load all the calibration Dataframes where filenames, exposure times, observation types, etc. are stored
pars.get_dataframes()

# # setup the dictionary for reference cubes for each target star

ref_cubes_dict = {
            'star1': 'ref_star',
            'star2': None,
            }

ref_cubes_dict = {'HD188228':None}

# define all stars to be processed - in this case all stars found in ref_cubes_dict
all_stars = list(ref_cubes_dict.keys()) + list(ref_cubes_dict.values())

# but you can also define a custom list of stars to be processed
all_stars = ['star1', 'star2', 'ref_star']

# set additional parameters for the REACHER pipeline
pars.overwrite = True
pars.skip_dither_correction = True
pars.clean_calibrations = False
pars.clean_stage_1_and_2 = False
pars.save_fullsize_flux = True
pars.only_flux = True
pars.full_output = False
pars.calib_valdity_threshold = 7 
pars.star_loc = (400, 465)

if __name__ == "__main__":

    _ = REACHER.make_bad_pix_masks(pars)   

    _ = REACHER.make_all_masterdarks(pars)

    _ = REACHER.make_all_masterflats(pars)
    
    ### ! Caution ! ###
    ### The master background creation step is not working yet
    # _ = REACHER.make_all_masterbackgrounds(pars)

    # loop over all stars and observation nights to run the pre-calibration and extraction step
    # for star in all_stars:

        star_parent_dir = os.path.join(pars.raw_input_dir,star,'IRDIS')
        all_obs_nights = os.listdir(star_parent_dir)

        for obs_night in all_obs_nights:
            obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
            filters = [f for f in os.listdir(obs_night_parent_dir) if not 'ND' in f]

            for filter_name in filters:

                filter_parent_dir = os.path.join(obs_night_parent_dir,filter_name)
                
                pars.star_name = star
                pars.obs_night = obs_night
                pars.filter_name = filter_name
                pars.get_star_dataframe()

                _ = REACHER.pre_calibration_and_extraction(pars)

    # this is only neccessary if you want to grab all centering files and iscpect them before continuing
    # _ = REACHER.grab_all_files(pars, ref_cubes_dict.keys())
    # _ = REACHER.return_csv_files(pars, ref_cubes_dict.keys())
    
    # loop over all stars and observation nights to run the centering and bad pixel correction step
    for star in ref_cubes_dict.keys():

        star_parent_dir = os.path.join(pars.raw_input_dir,star,'IRDIS')
        all_obs_nights = REACHER.list_directory(star_parent_dir)

        for obs_night in all_obs_nights:
            obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
            filters = REACHER.list_directory(obs_night_parent_dir)

            for filter_name in filters:
                pars.star_name = star
                pars.obs_night = obs_night
                pars.filter_name = filter_name
                pars.get_star_dataframe()

                _ = REACHER.centering_and_bad_pix_correction(pars, ref_cubes_dict=ref_cubes_dict)

    # if desired, clean up all temporary folders created during the reduction
    _ = REACHER.cleanup_all_folders(pars)