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


def get_file_name(filepath):
    """Get the file name from the full file path."""
    return os.path.basename(filepath).split('_SPHER')[0]


class FILEHANDLER():

    def __init__(self):
        pass

    def read_directory(self, input_dir):
        """Read the directory and return all files."""

        self.main_dir = input_dir
        self.output_folder = None

        self.all_files = glob(join(self.main_dir,'*.fits'))
        self.all_files.sort()

        self.flat = [os.path.basename(f) for f in self.all_files if 'FLAT' in os.path.basename(f) and '2023-07-24' in os.path.basename(f)]
        self.science = [os.path.basename(f) for f in self.all_files if 'SCIENCE' in os.path.basename(f)]
        self.flux = [os.path.basename(f) for f in self.all_files if 'FLUX' in os.path.basename(f)]
        self.dark = [os.path.basename(f) for f in self.all_files if 'DARK' in os.path.basename(f)]
        self.star_center = [os.path.basename(f) for f in self.all_files if 'CENTER' in os.path.basename(f)]

        self.flat.sort()
        self.science.sort()
        self.flux.sort()
        self.dark.sort()
        self.star_center.sort()

        return


    def read_directory_ispy(self, science_dir, flats, darks):
        """Read the directory and return all files."""

        self.all_files = glob(join(science_dir,'*.fits'))
        self.all_files.sort()


        self.science = [os.path.basename(f) for f in self.all_files if 'SCIENCE' in os.path.basename(f)]
        self.flux = [os.path.basename(f) for f in self.all_files if 'FLUX' in os.path.basename(f)]
        self.star_center = [os.path.basename(f) for f in self.all_files if 'CENTER' in os.path.basename(f)]
        self.sky = [os.path.basename(f) for f in self.all_files if 'SKY' in os.path.basename(f)]
        self.flux.sort()
        self.science.sort()
        self.star_center.sort()
        self.sky.sort()
        
        if self.sky == []:
            logging.warning('no sky files found in {}'.format(science_dir))

        if flats is not None:
            self.flat = [os.path.basename(f) for f in flats]
            self.flat.sort()

        if darks is not None:   
            self.dark = [os.path.basename(f) for f in darks]
            self.dark.sort()    

        return


    def get_files(self, key, stage, output_folder = None, prefix=None):

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
                    if prefix is not None: 
                        output_file = file.replace(file_type, prefix + '_' + file_type)
                    else: 
                        output_file = file 

                    if output_folder is not None:
                        output_file = os.path.join(output_folder, output_file)

                    key_output_files.append(output_file)

                all_output_files += key_output_files

        else: 
            raise KeyError('')
                
        return all_output_files
    
