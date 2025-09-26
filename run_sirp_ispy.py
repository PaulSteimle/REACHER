import SIRP 
import logging
import os

pars = SIRP.Parameters()

SIRP.setup_logging(log_file='/home/pasteimle/code/output_HD59659.log', level=logging.INFO)

pars.raw_input_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/ISPY_raw_files/'
pars.reduced_output_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/ISPY_reduced_cubes_new/'

# pars.raw_input_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY2/'
# pars.reduced_output_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY_proc/'

pars.nproc = 50
# load all the calibration Dataframes 
pars.get_dataframes()

all_stars = [f for f in os.listdir(pars.raw_input_dir) if f not in ['FLAT','DARK', 'DB_H23', 'BACKGROUND', 'DISTORT', 'DARK,BACKGROUND']]
#all_stars = ['HD377', 'HD342', 'HD101412', 'HD1461', 'HD2195', 'TYC_8972-2028-1']
all_stars = ['HD101412', 'TYC_8972-2028-1']

ref_cubes_dict = {
            #'HD377': 'HD342',
            # 'HD1461': 'HD2195',
            # 'HD22049_epsEri': None,
            # 'HD59659': 'HD58826',
            'HD101412': 'TYC_8972-2028-1',
            # 'HD105912': 'HD106976',
            # 'HD121617': 'HD120275A',
            # 'HD131511_VDEBoo': 'HD131302',
            # 'HD169022_epsSgr': None,
            # 'HD179218': None,
            # 'HD181327': ['HD183028_mu.Tel','HD180134'],
            # 'HD191089': 'HD191052',
            # 'HD207129': {'2022-10-14':'HD207400', 
            #              '2023-10-05':'HD206690',},
            }

pars.overwrite = True
pars.skip_dither_correction = True
pars.fixed_user_mastersky['left'] = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY_proc/master_sky_files/master_sky_HD101412_left.fits'
pars.fixed_user_mastersky['right'] = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY_proc/master_sky_files/master_sky_HD101412_right.fits'
pars.sky_files_exist = True

if __name__ == "__main__":

#     _ = SIRP.make_bad_pix_masks(pars)   

#     _ = SIRP.make_all_masterdarks(pars)

#     _ = SIRP.make_all_masterflats(pars)
    
#    # _ = SIRP.make_all_masterbackgrounds(input_dir, output_dir, nproc=nproc, overwrite=True)
    
#     for star in all_stars:

#         star_parent_dir = os.path.join(pars.raw_input_dir,star,'IRDIS')
#         all_obs_nights = os.listdir(star_parent_dir)
#         for obs_night in all_obs_nights:

#             obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
#             filters = os.listdir(obs_night_parent_dir)

#             for filter_name in filters:

#                 if not 'ND' in filter_name:

#                     filter_parent_dir = os.path.join(obs_night_parent_dir,filter_name)
                    
#                     pars.star_name = star
#                     pars.obs_night = obs_night
#                     pars.filter_name = filter_name
#                     pars.get_star_dataframe()

#                     _ = SIRP.run_pre_processing_ispy(pars)

#     _ = SIRP.grab_all_files(pars.raw_input_dir, all_stars)
    
#     _ = SIRP.return_csv_files(pars.raw_input_dir, all_stars)
    
    for star in all_stars:

        star_parent_dir = os.path.join(pars.raw_input_dir,star,'IRDIS')
        all_obs_nights = os.listdir(star_parent_dir)

        for obs_night in all_obs_nights:
            obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
            filters = os.listdir(obs_night_parent_dir)

            for filter_name in filters:
            
                if 'ND' in filter_name:
                    continue
                
                pars.star_name = star
                pars.obs_night = obs_night
                pars.filter_name = filter_name
                pars.get_star_dataframe()

                _ = SIRP.recentering_bad_pix_correction_ispy(pars, ref_cubes_dict=ref_cubes_dict)





# re_distribute all the recentering_files 
# first glob all files 

# input_folder = '/data/beegfs/astro-storage/groups/beuther/pasteimle/ispy_processed_data'
# all_recentering_pngs = glob(input_folder +'/*/*/*/*/*/*/*.png')
# all_center_csvs = glob(input_folder +'/*/*/*/*/*/*.csv')

# output_folder = os.path.join(input_folder, 'recentering')

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# for file in all_recentering_pngs:

#     filename = os.path.basename(file)
#     folder_name = file.split('/STAR_CENTER')[0].split('/ispy_processed_data/')[1].replace('/','_').replace('_recentering_pngs','')
#     print('copy', file, 'to', os.path.join(output_folder, folder_name, filename))


#     if not os.path.exists(os.path.join(output_folder, folder_name)):
#         os.makedirs(os.path.join(output_folder, folder_name))

#     #copy2(file, os.path.join(output_folder, folder_name, filename))#
#     copy2(os.path.join(output_folder, folder_name, filename),file)

# for file in all_center_csvs:

#     filename = os.path.basename(file)
#     folder_name = file.split('/centers.csv')[0].split('/ispy_processed_data/')[1].replace('/','_')
#     print('copy', file, 'to', os.path.join(output_folder, folder_name, filename))


#     if not os.path.exists(os.path.join(output_folder, folder_name)):
#         os.makedirs(os.path.join(output_folder, folder_name))

#     #copy2(file, os.path.join(output_folder, folder_name, filename))
#     copy2(os.path.join(output_folder, folder_name, filename),file)