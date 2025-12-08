
########################################################################################################
#######                                                                                          #######
####### This file is an example script to run the REACHER pipeline on a multiple target survey.  #######
#######                                                                                          #######
########################################################################################################

import REACHER 
import logging
import os

pars = REACHER.Parameters()

REACHER.setup_logging(log_file='/home/pasteimle/code/output_HD1461.log', level=logging.INFO)

pars.raw_input_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/HD1461_grouped/'
pars.reduced_output_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/HD1461_reduced_new/'

# pars.raw_input_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY2/'
# pars.reduced_output_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY_proc/'

pars.nproc = 100
# load all the calibration Dataframes 
pars.get_dataframes()

all_stars = [f for f in os.listdir(pars.raw_input_dir) if f not in ['FLAT','DARK', 'DB_H23', 'BACKGROUND', 'DISTORT', 'DARK,BACKGROUND', 'recentering']]

ref_cubes_dict = {
            'HD377': 'HD342',
            'HD1461': 'HD2195',
            'HD22049_epsEri': None,
            'HD59659': 'HD58826',
            'HD101412': 'TYC_8972-2028-1',
            'HD105912': 'HD106976',
            'HD121617': 'HD120275A',
            'HD131511_VDEBoo': 'HD131302',
            'HD169022_epsSgr': None,
            'HD179218': None,
            'HD181327': ['HD183028_mu.Tel','HD180134'],
            'HD191089': 'HD191052',
            'HD207129': {'2022-10-14':'HD207400', 
                         '2023-10-05':'HD206690',},
            }

pars.overwrite = True
pars.skip_dither_correction = True
pars.clean_calibrations = False
pars.clean_stage_1_and_2 = False
pars.save_fullsize_flux = False
pars.only_flux = False 

if __name__ == "__main__":

    # _ = REACHER.make_bad_pix_masks(pars)   

    # _ = REACHER.make_all_masterdarks(pars)

    # _ = REACHER.make_all_masterflats(pars)
    
    # _ = REACHER.make_all_masterbackgrounds(input_dir, output_dir, nproc=nproc, overwrite=True)
    
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

    # _ = REACHER.grab_all_files(pars, ref_cubes_dict.keys())
    
    # _ = REACHER.return_csv_files(pars, ref_cubes_dict.keys())
    
    for star in ['HD1461']: #ref_cubes_dict.keys():

        if star == 'HD101412':
            pars.fixed_user_mastersky['left'] = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY_proc/master_sky_files/master_sky_HD101412_left.fits'
            pars.fixed_user_mastersky['right'] = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY_proc/master_sky_files/master_sky_HD101412_right.fits'
            pars.sky_files_exist = True
        else: 
            pars.fixed_user_mastersky['left'] = None
            pars.fixed_user_mastersky['right'] = None
            pars.sky_files_exist = False

        star_parent_dir = os.path.join(pars.raw_input_dir,star,'IRDIS')
        all_obs_nights = os.listdir(star_parent_dir)
        all_obs_nights = ['2025-11-24', '2025-11-25']

        for obs_night in all_obs_nights:
            obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
            filters = os.listdir(obs_night_parent_dir)

            for filter_name in filters:
                
                pars.star_name = star
                pars.obs_night = obs_night
                pars.filter_name = filter_name
                pars.get_star_dataframe()

                _ = REACHER.centering_and_bad_pix_correction(pars, ref_cubes_dict=ref_cubes_dict)

    # _ = REACHER.cleanup_all_folders(pars)