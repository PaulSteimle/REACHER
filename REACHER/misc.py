import os
import re
import logging
import sys

from astropy.io import fits
from astropy.time import Time
import pandas as pd
from os.path import basename, join
import numpy as np
from shutil import copy2
from glob import glob 

def list_directory(path):
    return [f for f in os.listdir(path) if not '.DS' in f and not '._.DS' in f]

def sanitize_header(header):

    new_header = fits.Header()

    for card in header.cards:
        key, value, comment = card.keyword, card.value, card.comment

        # Skip blank or CONTINUE cards
        if key == '' or key == 'CONTINUE':
            continue

        # Try to fix strings that improperly include a '/' (i.e., value + comment mashed together)
        if isinstance(value, str) and '/' in value:
            match = re.match(r"^\s*([+-]?[\d\.eE]+)\s*/\s*(.+)$", value)
            if match:
                try:
                    fixed_value = float(match.group(1))
                    fixed_comment = match.group(2).strip()
                    new_header[key] = (fixed_value, fixed_comment)
                    continue
                except ValueError:
                    pass  # fallback to original

        # If it’s valid or doesn’t need fixing, keep it as is
        new_header[key] = (value, comment)

    return new_header

def setup_logging(log_file='log.log', level=logging.INFO):

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    class PrintLogger:
        def __init__(self, level=logging.INFO):
            self.level = level
            self.logger = logging.getLogger()
            self._original = sys.__stdout__ if level == logging.INFO else sys.__stderr__

        def write(self, message):
            message = message.strip()
            if message:
                self.logger.log(self.level, message)

        def flush(self):
            pass

        def fileno(self):
            return self._original.fileno()

    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger().error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler (overwrite each run)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Redirect print and errors
    sys.stdout = PrintLogger(logging.INFO)
    sys.stderr = PrintLogger(logging.ERROR)

    # Catch uncaught exceptions
    sys.excepthook = handle_uncaught_exception 

    return

def make_flat_DF(master_dir):
    
    star_parent_dir = master_dir + 'FLAT' + '/IRDIS/'
    all_obs_nights = os.listdir(star_parent_dir)

    FLAT_DF = pd.DataFrame()
    for obs_night in all_obs_nights:

        obs_night_parent_dir = star_parent_dir + obs_night + '/'
        filters = os.listdir(obs_night_parent_dir)

        for filter_name in filters:

            filter_parent_dir = obs_night_parent_dir + filter_name + '/'

            all_files = sorted(glob(filter_parent_dir + '*.fits'))

            for file in all_files:
                filename = basename(file) 
                exp_time = fits.getheader(file)['EXPTIME']
                ND_filter = fits.getheader(file)['HIERARCH ESO INS4 FILT2 NAME']
                
                DF_i = pd.DataFrame({'filename': [filename], 
                                    'obs_night': [obs_night],
                                    'obs_run': [obs_night[:-3]],
                                    'filter': [filter_name],
                                    'exp_time': [exp_time], 
                                    'filepath': [file],
                                    'nd_filter': [ND_filter]
                                    })

                FLAT_DF = pd.concat((FLAT_DF, DF_i), ignore_index=True)

    return FLAT_DF

def make_dark_DF(master_dir):
    DARK_DF = pd.DataFrame()

    star_parent_dir = master_dir + 'DARK' + '/IRDIS/'
    all_obs_nights = os.listdir(star_parent_dir)

    for obs_night in all_obs_nights:

        obs_night_parent_dir = star_parent_dir + obs_night + '/'
        
        filters = os.listdir(obs_night_parent_dir)

        for filter_name in filters:

            filter_parent_dir = obs_night_parent_dir + filter_name + '/'

            all_files = sorted(glob(filter_parent_dir + '*.fits'))

            for file in all_files:
                filename = basename(file) 
                exp_time = fits.getheader(file)['EXPTIME']
                filter = fits.getheader(file)['HIERARCH ESO INS COMB IFLT']
                ND_filter = fits.getheader(file)['HIERARCH ESO INS4 FILT2 NAME']
                type = fits.getheader(file)['HIERARCH ESO DPR TYPE']

                if 'BACKGROUND' in type and 'K' in filter:
                    pass
                else:
                    DF_i = pd.DataFrame({'filename': [filename], 
                                        'obs_night': [obs_night],
                                        'obs_run': [obs_night[:-3]],
                                        'filter': [filter_name],
                                        'exp_time': [exp_time], 
                                        'filepath': [file],
                                        'nd_filter': [ND_filter]
                                        })

                    DARK_DF = pd.concat((DARK_DF, DF_i), ignore_index=True)
                
    return DARK_DF

def make_background_DF(master_dir):
    BACKGROUND_DF = pd.DataFrame()

    star_parent_dir = join(master_dir, 'BACKGROUND', 'IRDIS')
    if not os.path.exists(star_parent_dir):
        return BACKGROUND_DF
    
    all_obs_nights = os.listdir(star_parent_dir)

    for obs_night in all_obs_nights:

        obs_night_parent_dir = join(star_parent_dir, obs_night)
        filters = os.listdir(obs_night_parent_dir)

        for filter_name in filters:

            filter_parent_dir = join(obs_night_parent_dir, filter_name)

            all_files = sorted(glob(join(filter_parent_dir, 'BACKGROUND*.fits')))

            for file in all_files:
                filename = basename(file) 
                exp_time = fits.getheader(file)['EXPTIME']
                ND_filter = fits.getheader(file)['HIERARCH ESO INS4 FILT2 NAME']
                
                DF_i = pd.DataFrame({'filename': [filename], 
                                    'obs_night': [obs_night],
                                    'obs_run': [obs_night[:-3]],
                                    'filter': [filter_name],
                                    'exp_time': [exp_time], 
                                    'filepath': [file],
                                    'nd_filter': [ND_filter]
                                    })

                BACKGROUND_DF = pd.concat((BACKGROUND_DF, DF_i), ignore_index=True)
                
    return BACKGROUND_DF

def make_star_DF(master_dir, star, obs_night, filter_name):
    STAR_DF = pd.DataFrame()

    filter_parent_dir = join(master_dir, star, 'IRDIS', obs_night, filter_name)
    all_files = sorted(glob(join(filter_parent_dir, '*SPHER*.fits')))

    for file in all_files:
        filename = basename(file) 

        header = fits.getheader(file)
        exp_time = header['EXPTIME']
        ND_filter = header['HIERARCH ESO INS4 FILT2 NAME']
        type = header['HIERARCH ESO DPR TYPE']
        
        DF_i = pd.DataFrame({'filename': [filename], 
                            'obs_night': [obs_night],
                            'obs_run': [obs_night[:-3]],
                            'filter': [filter_name],
                            'exp_time': [exp_time], 
                            'filepath': [file],
                            'nd_filter': [ND_filter],
                            'type': [type],
                            })

        STAR_DF = pd.concat((STAR_DF, DF_i), ignore_index=True)
                
    return STAR_DF

def find_calib_files(STAR_DF, params, calib_type='none'):

    stage_dict ={'bad_pix':1, 
                 'master_dark':2, 
                 'master_flat':3, 
                 'master_background':4,
                 'science':5,
                 'none':5,
                 }

    stage = stage_dict[calib_type]

    for index, row in STAR_DF.iterrows():

        obs_night = row['obs_night']
        filter_name = row['filter']
        exp_time = row['exp_time']
        nd_filter = row['nd_filter']
        obs_night_JD = Time(obs_night + 'T00:00:00.000', format='isot', scale='utc').jd

        # 1. Try to associate the calibration files that should be valid for the same obs night, 
        # 2. If same night calib files do not exist try to find the nearest one in time that is within the validity threshold. 
        # 3. If still no file is found within the validity threshold, raise an error, suggest to check the archive or to increase the validity threshold.

        # lets start with the bad pix file and master dark, which are both created from the same dark files and hence it should be the same for them.
        bad_pix_mask_filepath = os.path.join(params.bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_night))
        if not os.path.exists(bad_pix_mask_filepath):

            bad_pix_list = sorted(glob(os.path.join(params.bad_pix_mask_folder, 'bad_pix_mask_*.fits')))
            obs_night_list = [Time(f.split('_')[-1].split('.')[0] + 'T00:00:00.000', format='isot', scale='utc').jd for f in bad_pix_list]
            nearest_idx = np.argmin(np.abs(np.array(obs_night_list) - obs_night_JD))
            nearest_time =  np.min(np.abs(np.array(obs_night_list) - obs_night_JD))
            if nearest_time < params.calib_valdity_threshold:
                logging.warning('Bad pix mask file {} does not exist. Using nearest bad pix mask file {} which is {} days apart.'.format(bad_pix_mask_filepath, bad_pix_list[nearest_idx], nearest_time))
                bad_pix_mask_filepath = bad_pix_list[nearest_idx]
            else:
                raise FileExistsError('bad pix mask file does not exist: {}'.format(bad_pix_mask_filepath))
            
        STAR_DF.loc[index, 'bad_pix_mask'] = bad_pix_mask_filepath

        # now onto the masterdark 
        master_dark_filepath = os.path.join(params.master_dark_folder, 'master_dark_{}_{}.fits'.format(obs_night, exp_time))
        if not os.path.exists(master_dark_filepath):
        
            master_dark_filepath = 'None'
            master_dark_list = sorted(glob(os.path.join(params.master_dark_folder, 'master_dark_{}_{}.fits'.format('*', exp_time))))
            obs_night_list = [Time(f.split('_')[-2] + 'T00:00:00.000', format='isot', scale='utc').jd for f in master_dark_list]
            
            if len(obs_night_list) != 0:
                nearest_idx = np.argmin(np.abs(np.array(obs_night_list) - obs_night_JD))
                nearest_time =  np.min(np.abs(np.array(obs_night_list) - obs_night_JD))
                if nearest_time < params.calib_valdity_threshold:

                    logging.warning('Master dark file {} does not exist. Using nearest master dark file {} which is {} days apart.'.format('master_dark_{}_{}.fits'.format(obs_night, exp_time), master_dark_list[nearest_idx], nearest_time))
                    master_dark_filepath = master_dark_list[nearest_idx]

        # this is neccessary as in some cases flats do not have darks, however there are enough flats with darks to still produce a masterflat. Flats without darks will be skipped in the processing
        if (stage == 3 and master_dark_filepath == 'None') or master_dark_filepath != 'None':
              STAR_DF.loc[index, 'master_dark'] = master_dark_filepath       

        else:
            raise FileExistsError('master dark file does not exist: {}'.format(master_dark_filepath))
                
        # stage 4: up to flat 
        if stage >= 4:

            # now do the same for the master flat, which is also important to be correct, but less likely to be missing than the master dark.
            master_flat_filepath = os.path.join(params.master_flat_folder, 'master_flat_{}_{}.fits'.format(obs_night, filter_name))
            if not os.path.exists(master_flat_filepath):

                master_flat_list = sorted(glob(os.path.join(params.master_flat_folder, 'master_flat_{}_{}.fits'.format('*', filter_name))))
                obs_night_list = [Time(f.split('master_flat_')[1][:10] + 'T00:00:00.000', format='isot', scale='utc').jd for f in master_flat_list]
                nearest_idx = np.argmin(np.abs(np.array(obs_night_list) - obs_night_JD))
                nearest_time =  np.min(np.abs(np.array(obs_night_list) - obs_night_JD))
                if nearest_time < params.calib_valdity_threshold:
                    logging.warning('Master flat file {} does not exist. Using nearest master flat file {} which is {} days apart.'.format(master_flat_filepath, master_flat_list[nearest_idx], nearest_time))
                    master_flat_filepath = master_flat_list[nearest_idx]
                else:
                    raise FileExistsError('master flat file does not exist: {}. Nearest available file is {} days apart.'.format(master_flat_filepath, nearest_time))

            STAR_DF.loc[index, 'master_flat'] = master_flat_filepath

        # # stage 5: up to master_background this is only for full science obs
        # if stage >= 5:

        #     # now do the same for the master background, which is less important to be correct than the master dark and master flat, but still nice to have for the reduction if it is available.
        #     master_bkg_filepath =  os.path.join(params.master_background_folder, 'master_background_{}_{}_{}_{}.fits'.format(obs_night, filter_name, exp_time, nd_filter))
        #     if not os.path.exists(master_bkg_filepath):

        #         master_bkg_list = sorted(glob(os.path.join(params.master_background_folder, 'master_background_{}_{}_{}_{}.fits'.format('*', filter_name, exp_time, nd_filter))))
        #         obs_night_list = [Time(f.split('master_background_')[1][:10] + 'T00:00:00.000', format='isot', scale='utc').jd for f in master_bkg_list]
        #         nearest_idx = np.argmin(np.abs(np.array(obs_night_list) - obs_night_JD))
        #         nearest_time =  np.min(np.abs(np.array(obs_night_list) - obs_night_JD))
        #         if nearest_time < params.calib_valdity_threshold:
        #             master_bkg_filepath = master_bkg_list[nearest_idx]
        #             logging.warning('Master background file {} does not exist. Using nearest master background file {} which is {} days apart.'.format(master_bkg_filepath, master_bkg_list[nearest_idx], nearest_time))
        #         else:
        #             raise FileExistsError('master background file does not exist: {}'.format(master_bkg_filepath))

        #     STAR_DF.loc[index, 'master_background'] = master_bkg_filepath


    return STAR_DF

def grab_all_files(params, stars):

    logging.info('Grabbing all files for stars: {}'.format(stars))
    logging.info('Output folder for recentering files: {}'.format(params.recentering_folder))

    for star in stars:
        star_parent_dir = os.path.join(params.reduced_output_dir,star,'IRDIS')
        all_obs_nights = os.listdir(star_parent_dir)
        for obs_night in all_obs_nights:

            obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
            filters = os.listdir(obs_night_parent_dir)

            for filter_name in filters:

                all_recentering_pngs = glob(join(params.reduced_output_dir,'{}/*/*/*/stage2/recentering_pngs/*.png'.format(star))) 
                center_csv = glob(join(params.reduced_output_dir,'{}/*/*/*/stage2/centers.csv'.format(star)))[0]

                recentering_star_folder = join(params.recentering_folder, '_'.join([star, obs_night, filter_name]))
                if not os.path.exists(recentering_star_folder):
                    os.makedirs(recentering_star_folder)
            

                all_recentering_pngs_new = [join(recentering_star_folder, os.path.basename(f)) for f in all_recentering_pngs]
                new_center_csv = join(recentering_star_folder, 'centers.csv'.format(star))

                copy2(center_csv, new_center_csv)
                logging.info('\n')
                logging.info('\t -> copying {} center.csv'.format(star))
                for f_old, f_new in zip(all_recentering_pngs, all_recentering_pngs_new):
                    logging.info('\t -> copying {} to {}'.format(basename(f_old), basename(recentering_star_folder)))
                    copy2(f_old, f_new)
    return

def return_csv_files(params, stars):

    logging.info('Re-distributing all centers.csv files for stars: {} \n'.format(stars))
    for star in stars:
        star_parent_dir = os.path.join(params.reduced_output_dir,star,'IRDIS')
        all_obs_nights = os.listdir(star_parent_dir)
        for obs_night in all_obs_nights:
            obs_night_parent_dir = os.path.join(star_parent_dir,obs_night)
            filters = os.listdir(obs_night_parent_dir)
            for filter_name in filters:

                center_csv = glob(join(params.reduced_output_dir,'{}/*/*/*/stage2/centers.csv'.format(star)))[0]

                recentering_star_folder = join(params.recentering_folder, '_'.join([star, obs_night, filter_name]))
                if not os.path.exists(recentering_star_folder):
                    os.makedirs(recentering_star_folder)

                new_center_csv = join(recentering_star_folder, 'centers.csv'.format(star))

                copy2(new_center_csv, center_csv)
                logging.info('\t -> re-copying {}.center csv'.format(star))

    return



