
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
pars.raw_input_dir = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/additional_ISPY_raw_files/'
pars.reduced_output_dir = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/additional_ISPY_reduced_cubes/'

# pars.raw_input_dir = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/HD3670_data/raw_files/'
# pars.reduced_output_dir = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/HD3670_data/reduced_cubes/'

# pars.raw_input_dir = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/ISPY_raw_files/'
# pars.reduced_output_dir = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/ISPY_reduced_cubes/'

# setup logging
REACHER.setup_logging(log_file=os.path.join(pars.reduced_output_dir, 'REACHER.log'), level=logging.INFO)

# set number of processors for parallel processing
pars.nproc = 100
# load all the calibration Dataframes where filenames, exposure times, observation types, etc. are stored
pars.get_dataframes()

# # setup the dictionary for reference cubes for each target star
# ref_cubes_dict = {
#             'HD377': 'HD342',
#             # 'HD1461': 'HD2195',
#             'HD22049_epsEri': None,
#             # 'HD59659': 'HD58826',
#             'HD101412': 'TYC_8972-2028-1',
#             'HD105912': 'HD106976',
#             'HD121617': 'HD120275A',
#             'HD131511_VDEBoo': 'HD131302',
#             'HD169022_epsSgr': None,
#             'HD179218': None,
#             'HD181327': 'HD180134',
#             'HD191089': 'HD191052',
#             'HD207129': {'2022-10-14':'HD207400', 
#                          '2023-10-05':'HD206690',},
#             }


# ref_cubes_dict = {
#             'HD1461': 'HD2195',
#             'HD59659': None,
#             'TYC_8128_1946_1': None,
#             }

ref_cubes_dict = {'HD59659':None}

# define all stars to be processed - in this case all stars found in ref_cubes_dict
# all_stars = list(ref_cubes_dict.keys()) + list(ref_cubes_dict.values())
# all_stars = [star for star in all_stars if star is not None]

# but you can also define a custom list of stars to be processed
#all_stars = ['HD377', 'HD342', 'HD1461', 'HD2195', 'HD22049_epsEri', 'HD59659', 'HD58826', 'HD105912', 'HD101412', 'TYC_8972-2028-1',
            #  'HD106976', 'HD121617', 'HD120275A', 'HD131511_VDEBoo', 'HD131302', 'HD169022_epsSgr',
            #  'HD179218', 'HD181327', 'HD180134', 'HD191089', 'HD191052', 'HD207129', 'HD207400', 'HD206690',]


#all_stars = ['HD1461', 'HD2195', 'HD59659', 'TYC_8128_1946_1']

#all_stars = ['HD3670']
all_stars = ['HD59659']

# set additional parameters for the REACHER pipeline
pars.overwrite = True
pars.skip_dither_correction = True
pars.clean_calibrations = False
pars.clean_stage_1_and_2 = False
pars.save_fullsize_flux = False
pars.only_flux = False
pars.full_output = False
pars.star_loc = (464, 465)

if __name__ == "__main__":

    # _ = REACHER.make_bad_pix_masks(pars)   

    # _ = REACHER.make_all_masterdarks(pars)

    # _ = REACHER.make_all_masterflats(pars)
    
    ### ! Caution ! ###
    ### The master background creation step is not working yet
    # _ = REACHER.make_all_masterbackgrounds(pars)

    ### loop over all stars and observation nights to run the pre-calibration and extraction step
    # for star in all_stars:

    #     star_parent_dir = os.path.join(pars.raw_input_dir,star,'IRDIS')
    #     all_obs_nights = os.listdir(star_parent_dir)

    #     for obs_night in all_obs_nights:
    #         obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
    #         filters = os.listdir(obs_night_parent_dir)

    #         for filter_name in filters:

    #             if not 'ND' in filter_name:

    #                 filter_parent_dir = os.path.join(obs_night_parent_dir,filter_name)
                    
    #                 pars.star_name = star
    #                 pars.obs_night = obs_night
    #                 pars.filter_name = filter_name
    #                 pars.get_star_dataframe()

    #                 _ = REACHER.pre_calibration_and_extraction(pars)

    # this is only neccessary if you want to grab all centering files and iscpect them before continuing
    # _ = REACHER.grab_all_files(pars, ref_cubes_dict.keys())
    # _ = REACHER.return_csv_files(pars, ref_cubes_dict.keys())
    
    # loop over all stars and observation nights to run the centering and bad pixel correction step

    for star in ref_cubes_dict.keys():

        # in this case we have pre-defined master sky files for HD101412 as these are from a different obs-night 
        if star == 'HD101412':
            pars.fixed_user_mastersky['left'] = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/master_sky_files/master_sky_HD101412_left.fits'
            pars.fixed_user_mastersky['right'] = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/master_sky_files/master_sky_HD101412_right.fits'
            pars.sky_files_exist = True
        else: 
            pars.fixed_user_mastersky['left'] = None
            pars.fixed_user_mastersky['right'] = None
            pars.sky_files_exist = False

        star_parent_dir = os.path.join(pars.raw_input_dir,star,'IRDIS')
        all_obs_nights = REACHER.list_directory(star_parent_dir)

        for obs_night in all_obs_nights:
            obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
            filters = REACHER.list_directory(obs_night_parent_dir)

            if '2015' in obs_night:
                pars.star_center_index = 1
            else:
                pars.star_center_index = 0


            for filter_name in filters:
                pars.star_name = star
                pars.obs_night = obs_night
                pars.filter_name = filter_name
                pars.get_star_dataframe()

                _ = REACHER.centering_and_bad_pix_correction(pars, ref_cubes_dict=ref_cubes_dict)

    # if desired, clean up all temporary folders created during the reduction
    # _ = REACHER.cleanup_all_folders(pars)