import os
import re
import logging
import sys

from astropy.io import fits
import pandas as pd
from os.path import basename, join
import numpy as np
from shutil import copy2
from glob import glob 


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


def setup_logging(log_file='app.log', level=logging.INFO):
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


def find_calib_files(STAR_DF, params):

    for index, row in STAR_DF.iterrows():
        obs_night = row['obs_night']
        obs_run = row['obs_run']
        filter_name = row['filter']
        exp_time = row['exp_time']
        nd_filter = row['nd_filter']

        # run through all files and check if they exist 
        master_dark_filepath = os.path.join(params.master_dark_folder, 'master_dark_{}_{}.fits'.format(obs_run, exp_time))
        if not os.path.exists(master_dark_filepath):
            raise FileExistsError('master dark file does not exist: {}'.format(master_dark_filepath))
        STAR_DF.loc[index, 'master_dark'] = master_dark_filepath

        master_flat_filepath = os.path.join(params.master_flat_folder, 'master_flat_{}_{}.fits'.format(obs_night, filter_name))
        if not os.path.exists(master_flat_filepath):
            raise FileExistsError('master flat file does not exist: {}'.format(master_flat_filepath))
        STAR_DF.loc[index, 'master_flat'] = master_flat_filepath

        # master_bkg_filepath = os.path.join(params.master_background_folder, 'master_background_{}_{}_{}_{}.fits'.format(obs_night, filter_name, exp_time, nd_filter))
        # if not os.path.exists(master_bkg_filepath):
        #     raise FileExistsError('master background file does not exist: {}'.format(master_bkg_filepath))
        # STAR_DF.loc[index, 'master_background'] = master_bkg_filepath

        bad_pix_mask_filepath = os.path.join(params.bad_pix_mask_folder, 'bad_pix_mask_{}.fits'.format(obs_run))
        if not os.path.exists(bad_pix_mask_filepath):
            raise FileExistsError('bad pix mask file does not exist: {}'.format(bad_pix_mask_filepath))
        STAR_DF.loc[index, 'bad_pix_mask'] = bad_pix_mask_filepath

    return STAR_DF


def grab_all_files(params, stars):

    all_recentering_pngs = [f for f in glob(params.raw_input_dir +'/*/*/*/*/*/*/*.png') if f.split('/')[-7] in stars]
    all_center_csvs = [f for f in glob(params.raw_input_dir +'/*/*/*/*/*/*.csv') if f.split('/')[-6] in stars]

    output_folder = params.recentering_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    basefolder_name = os.path.basename(params.raw_input_dir)

    for file in all_recentering_pngs:

        filename = os.path.basename(file)
        folder_name = file.split('/STAR_CENTER')[0].split(f'/{basefolder_name}/')[1].replace('/','_').replace('_recentering_pngs','')
        print('copy', file, 'to', os.path.join(output_folder, folder_name, filename))

        output_folder_star = os.path.join(output_folder, folder_name)
        if not os.path.exists(output_folder_star):
            os.makedirs(output_folder_star)

        copy2(file, os.path.join(output_folder, folder_name, filename))


    for file in all_center_csvs:

        filename = os.path.basename(file)
        folder_name = file.split('/centers.csv')[0].split(f'/{basefolder_name}/')[1].replace('/','_')
        print('copy', file, 'to', os.path.join(output_folder, folder_name, filename))


        if not os.path.exists(os.path.join(output_folder, folder_name)):
            os.makedirs(os.path.join(output_folder, folder_name))

        copy2(file, os.path.join(output_folder, folder_name, filename))


    return


def return_csv_files(params, stars):
    all_center_csvs = [f for f in glob(params.raw_input_dir +'/*/*/*/*/*/*.csv') if f.split('/')[-6] in stars]

    output_folder = params.recentering_folder

    basefolder_name = os.path.basename(params.raw_input_dir)

    for file in all_center_csvs:

        filename = os.path.basename(file)
        folder_name = file.split('/centers.csv')[0].split(f'/{basefolder_name}/')[1].replace('/','_')
        print('copy', file, 'to', os.path.join(output_folder, folder_name, filename))

        copy2(os.path.join(output_folder, folder_name, filename),file)

    return



