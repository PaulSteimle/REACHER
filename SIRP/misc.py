import os
import re
import logging
import sys

from astropy.io import fits
import pandas as pd
from os.path import basename
import numpy as np
from glob import glob 

def get_file_name(filepath):
    """Get the file name from the full file path."""
    return os.path.basename(filepath).split('_SPHER')[0]


class FILEHANDLER():

    def __init__(self):
        pass

    def read_directory(self, input_dir):
        """Read the directory and return all files."""

        self.main_dir = input_dir

        self.all_files = glob(self.main_dir + '*.fits')
        self.all_files.sort()

        # self.input_flat = [f for f in self.all_files if 'FLAT' in os.path.basename(f) and '2023-07-24' in os.path.basename(f)]
        # self.input_science = [f for f in self.all_files if 'SCIENCE' in os.path.basename(f)]
        # self.input_flux = [f for f in self.all_files if 'FLUX' in os.path.basename(f)]
        # self.input_dark = [f for f in self.all_files if 'DARK' in os.path.basename(f)]
        # self.input_star_center = [f for f in self.all_files if 'STAR_CENTER' in os.path.basename(f)]

        self.flat = [os.path.basename(f) for f in self.all_files if 'FLAT' in os.path.basename(f) and '2023-07-24' in os.path.basename(f)]
        self.science = [os.path.basename(f) for f in self.all_files if 'SCIENCE' in os.path.basename(f)]
        self.flux = [os.path.basename(f) for f in self.all_files if 'FLUX' in os.path.basename(f)]
        self.dark = [os.path.basename(f) for f in self.all_files if 'DARK' in os.path.basename(f)]
        self.star_center = [os.path.basename(f) for f in self.all_files if 'STAR_CENTER' in os.path.basename(f)]

        self.flat.sort()
        self.science.sort()
        self.flux.sort()
        self.dark.sort()
        self.star_center.sort()

        return

    def read_directory_ispy(self, science_dir, flats, darks):
        """Read the directory and return all files."""

        self.all_files = glob(science_dir + '*.fits')
        self.all_files.sort()


        self.science = [os.path.basename(f) for f in self.all_files if 'SCIENCE' in os.path.basename(f)]
        self.flux = [os.path.basename(f) for f in self.all_files if 'FLUX' in os.path.basename(f)]
        self.star_center = [os.path.basename(f) for f in self.all_files if 'STAR_CENTER' in os.path.basename(f)]
        self.flux.sort()
        self.science.sort()
        self.star_center.sort()

        if flats is not None:
            self.flat = [os.path.basename(f) for f in flats]
            self.flat.sort()

        if darks is not None:   
            self.dark = [os.path.basename(f) for f in darks]
            self.dark.sort()    

        return


    # def update_flats(self, new_flats):
    #     """Update the flats list with new flats."""
    #     self.input_flat = new_flats
    #     self.flat = [os.path.basename(f) for f in new_flats]
    #     self.output_flat = [os.path.join(self.output_dir, f) for f in self.flat]

    #     self.flat.sort()
    #     self.output_flat.sort()
    #     self.input_flat.sort()

    #     return


    # def make_output_paths(self):

    #     for filetype in ['flat','science','flux','dark','star_center']:
    #         setattr(self, 'output_' + filetype, [os.path.join(self.output_dir , f) for f in getattr(self, filetype)])
    #     return


    # def get_raw_input_files(self):
    #     return (self.input_flat + self.input_flux + self.input_science + self.input_star_center)
    

    # def get_all_output_files(self):
    #     return (self.output_flat + self.output_flux + self.output_science + self.output_star_center)


    # def get_science_output_files(self):
    #     return (self.output_flux + self.output_science + self.output_star_center)


    def get_files(self, key, stage, output_folder = None):

        # first check what type key is: 
        if isinstance(key, str):
            key = [key]
    
        if isinstance(key, (np.ndarray, list)):
            all_output_files = []

            for k in key: 
                files = getattr(self, k)

                key_output_files = []

                if isinstance(files, str):
                    files = [files]

                for file in files:
                    file_type = get_file_name(file)

                    if stage is not None: 
                        output_file = file.replace(file_type, file_type + '_' + stage)
                    else: 
                        output_file = file 

                    if output_folder is not None:
                        output_file = os.path.join(output_folder, output_file)

                    key_output_files.append(output_file)

                all_output_files += key_output_files

        else: 
            raise KeyError('')
                
        return all_output_files
            


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
                
                DF_i = pd.DataFrame({'filename': [filename], 
                                    'obs_night': [obs_night],
                                    'obs_run': [obs_night[:-3]],
                                    'filter': [filter_name],
                                    'exp_time': [exp_time], 
                                    'filepath': [file],

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
                
                DF_i = pd.DataFrame({'filename': [filename], 
                                    'obs_night': [obs_night],
                                    'obs_run': [obs_night[:-3]],
                                    'filter': [filter_name],
                                    'exp_time': [exp_time], 
                                    'filepath': [file],

                                    })

                DARK_DF = pd.concat((DARK_DF, DF_i), ignore_index=True)
                
    return DARK_DF