from SIRP import run_pre_processing_ispy, recentering_bad_pix_correction_ispy, make_all_masterflats, setup_logging
import logging
import os

setup_logging(log_file='output.log', level=logging.INFO)

input_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/all_raw_data/'
output_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/ispy_processed_data/'

nproc = 40

all_stars = [f for f in os.listdir(input_dir) if f not in ['FLAT','DARK', 'DB_H23']]

if __name__ == "__main__":

    logging.info('running....')

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

                #_ = run_pre_processing_ispy(filter_parent_dir, star, obs_night, filter_name, output_dir, nproc=nproc, overwrite=False)

                #_ = recentering_bad_pix_correction_ispy(filter_parent_dir, star, obs_night, filter_name, output_dir, nproc=nproc, overwrite=False, skip_bad_pix=False)