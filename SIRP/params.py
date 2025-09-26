"""
    Set here all the important parameters for the computation.

    Paul Steimle, 25.09.2025
"""
import logging
import os
import numpy as np
from .misc import make_dark_DF, make_flat_DF, make_background_DF, make_star_DF

directory_path = os.path.dirname(os.path.realpath(__file__))


class Parameters:
    """ Parameters object for all settings. """

    def __init__(self):

        # paths to define
        self.raw_input_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/ISPY_raw_files/'
        self.reduced_output_dir = '/data/beegfs/astro-storage/groups/beuther/pasteimle/ISPY_reduced_cubes/'

        self.bad_pix_mask_filepath = None
        self.masterflat_filepath = None
        self.masterdark_filepath = None
        self.master_background_filepath = None
        self.sky_files_exist = False
        self.fixed_user_mastersky = {
                                     'left': None, 
                                     'right': None
                                     }  # if not None, use this master sky file instead of making one

        # basic parameters
        self.nproc = None
        self.overwrite = True
        self.make_plots = True
        self.skip_bad_pix = False
        self.only_flux = False

        # all the dataframes
        self.DARK_DF = None
        self.FLAT_DF = None
        self.BACKGROUND_DF = None

        #star params 
        self.star_name = None
        self.ref_star_name = None
        self.obs_night = None
        self.filter_name = None
        self.L_or_R = None

        #comsic ray params
        self.box_radius=2
        self.n_sigma=9

        # VIP image transformation params 
        self.image_shift_library = 'vip-fft'  # 'opencv', 'ndimage-fourier', 'ndimage-interp', 'vip-fft'
        self.image_rotation_library = 'vip-fft'  # 'opencv', 'skimage', 'vip-fft

        self.image_shift_interpolation = 'bicubic' # 'opencv', 'ndimage-fourier', 'ndimage-interp', 'vip-fft'
        # Only used in case of imlib is set to 'opencv' or 'ndimage-interp' 
        # For Scipy.ndimage the options are: 'nearneig', bilinear', 'biquadratic', 'bicubic', 'biquartic' or 'biquintic'. 
        # The 'nearneig' interpolation is the fastest and the 'biquintic' the slowest. The 'nearneig' is the poorer option for interpolation of noisy astronomical images. 
        # For Opencv the options are: 'nearneig', 'bilinear', 'bicubic' or 'lanczos4'. The 'nearneig' interpolation is the fastest and the 'lanczos4' the slowest and accurate.
        # 'lanczos4' is the default for Opencv and 'biquartic' for Scipy.ndimage.

        self.image_rotation_interpolation = 'bicubic' # 
        # For Skimage the options are: ‘nearneig’, bilinear’, ‘biquadratic’, ‘bicubic’, ‘biquartic’ or ‘biquintic’. 
        # The ‘nearneig’ interpolation is the fastest and the ‘biquintic’ the slowest. 
        # The ‘nearneig’ is the poorer option for interpolation of noisy astronomical images. 
        # For Opencv the options are: ‘nearneig’, ‘bilinear’, ‘bicubic’ or ‘lanczos4’. 
        # The ‘nearneig’ interpolation is the fastest and the ‘lanczos4’ the slowest and more accurate. ‘lanczos4’ is the default for Opencv and ‘biquartic’ for Skimage.


        # parameters for making the final cube
        self.skip_dither_correction = False
        self.subtract_median_science_sky_bkg = False

        # 
        self.flux_cut_out_size = 41
        self.star_loc = (401, 466)  # expected star location (x, y) in the flux image, used for the cut out and gauss fit 

        self.full_output = True  # if True, save additioal cube to check for centering and bad pixel correction


        # telescope and instrument params
        self.sphere_offset = 135.99 # PA angles offset in degrees, from SPHERE manual

        #  
        self.clean_stage_1_and_2 = False
        self.clean_calibrations = False

    def get_dataframes(self):
        
        logging.info('making DataFrames')
        self.DARK_DF = make_dark_DF(self.raw_input_dir)
        self.FLAT_DF = make_flat_DF(self.raw_input_dir)
        self.BACKGROUND_DF = make_background_DF(self.raw_input_dir)

    def get_star_dataframe(self):
        self.STAR_DF = make_star_DF(self.raw_input_dir, self.star_name, self.obs_night, self.filter_name)

    @property
    def recentering_folder(self):
        return os.path.join(self.raw_input_dir, 'recentering')

    @property
    def master_flat_folder(self):
        return os.path.join(self.reduced_output_dir, 'master_flat')

    @property
    def master_dark_folder(self):
        return os.path.join(self.reduced_output_dir, 'master_dark')
    
    @property
    def master_background_folder(self):
        return os.path.join(self.reduced_output_dir, 'master_background')

    @property
    def bad_pix_mask_folder(self):
        return os.path.join(self.reduced_output_dir, 'bad_pix_masks')
    
    @property
    def clump_mask_filepath(self):
        return os.path.join(self.bad_pix_mask_folder, 'IRDIS_clump_mask.fits')
    
    @property
    def star_output_folder(self): 
        return os.path.join(self.reduced_output_dir, self.star_name, 'IRDIS', self.obs_night, self.filter_name)

    @property
    def star_input_folder(self):
        return os.path.join(self.raw_input_dir, self.star_name, 'IRDIS', self.obs_night, self.filter_name)

    @property
    def stage1_output_folder(self):
        return os.path.join(self.star_output_folder, 'stage1')
    
    @property
    def stage2_output_folder(self):
        return os.path.join(self.star_output_folder, 'stage2')

    @property
    def left_output_folder(self):
        return os.path.join(self.star_output_folder, 'left', 'raw_files')

    @property
    def right_output_folder(self):
        return os.path.join(self.star_output_folder, 'right', 'raw_files')
    
    @property
    def L_or_R_output_folder(self):
        if self.L_or_R == 'LEFT':
            return self.left_output_folder
        elif self.L_or_R == 'RIGHT':
            return self.right_output_folder
        else:
            raise ValueError('L_or_R not set correctly')

    @property 
    def ref_star_input_folder(self):
        return os.path.join(self.raw_input_dir, self.ref_star_name, 'IRDIS', self.obs_night, self.filter_name)
    
    @property
    def ref_star_output_folder(self): 
        return os.path.join(self.reduced_output_dir, self.ref_star_name, 'IRDIS', self.obs_night, self.filter_name)
    
    @property
    def ref_stage1_output_folder(self):
        return os.path.join(self.ref_star_output_folder, 'stage1')

    @property
    def ref_stage2_output_folder(self):
        return os.path.join(self.ref_star_output_folder, 'stage2')

    @property
    def ref_left_output_folder(self):
        return os.path.join(self.ref_star_output_folder, 'left', 'raw_files')

    @property
    def ref_right_output_folder(self):
        return os.path.join(self.ref_star_output_folder, 'right', 'raw_files')

    @property
    def ref_L_or_R_output_folder(self):
        if self.L_or_R == 'LEFT':
            return self.ref_left_output_folder
        elif self.L_or_R == 'RIGHT':
            return self.ref_right_output_folder
        else:
            raise ValueError('L_or_R not set correctly')
        
    @property
    def master_sky_filepath(self):
        if self.fixed_user_mastersky[self.L_or_R.lower()] is not None:
            return self.fixed_user_mastersky[self.L_or_R.lower()]
        else:
            return os.path.join(self.stage2_output_folder, 'master_sky_{}_{}.fits'.format(self.star_name, self.L_or_R))
