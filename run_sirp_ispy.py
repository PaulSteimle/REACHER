from SIRP import run_pre_processing_ispy, recentering_bad_pix_correction_ispy, make_all_masterflats, setup_logging, make_bad_pix_masks
import logging
import os
from glob import glob
from shutil import copy2

setup_logging(log_file='/home/pasteimle/code/output.log', level=logging.INFO)

input_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/all_raw_data/'
output_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/ispy_processed_data/'


input_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/HD11397_raw_data/'
output_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/HD11397_processed_data/'

nproc = 100

all_stars = [f for f in os.listdir(input_dir) if f not in ['FLAT','DARK', 'DB_H23']]
#all_stars = ['HD121617']

if __name__ == "__main__":

    logging.info('running....')

    _ = make_bad_pix_masks(input_dir, output_dir, nproc=nproc, overwrite=False)   

    _ = make_all_masterflats(input_dir, output_dir, nproc=nproc, overwrite=False)

    obs_nights = []
    for star in all_stars:
        
        star_parent_dir = input_dir + star + '/IRDIS/'
        all_obs_nights = os.listdir(star_parent_dir)
        for obs_night in all_obs_nights:

            obs_night_parent_dir = star_parent_dir + obs_night + '/'
            filters = os.listdir(obs_night_parent_dir)

            for filter_name in filters:
                filter_parent_dir = obs_night_parent_dir + filter_name + '/'

                _ = run_pre_processing_ispy(filter_parent_dir, star, obs_night, filter_name, output_dir, nproc=nproc, overwrite=True)

                _ = recentering_bad_pix_correction_ispy(filter_parent_dir, star, obs_night, filter_name, output_dir, nproc=nproc, overwrite=True, skip_bad_pix=False)



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