from astropy.io import fits
import os
import shutil
from glob import glob
from multiprocessing import Pool
import pandas as pd
from .misc import subtract_one_day


def get_info(files, rename_dict):

    if isinstance(files, str):
        files = [files]
    
    DF = pd.DataFrame()

    for file in files:
        header = fits.getheader(file)
        print('reading file: {}'.format(file) )
        try:

            # first check if these are all IRDIS files and not e.g. IFS
            ARM = header['HIERARCH ESO SEQ ARM']
            if ARM != 'IRDIS':
                continue

            _object = header['OBJECT']

            # for some reason some files are correupted and have no OBJECT keyword, so we skip those files
            if _object.strip() == '':
                continue
        
            exp_time = header['EXPTIME']
            obs_date = header['DATE-OBS'].split('T')[0]
            obs_time = header['DATE-OBS'].split('T')[1]

            if int(obs_time[:2]) < 14:
                obs_night = subtract_one_day(obs_date)
            else: 
                obs_night = obs_date

            filter_name = header['HIERARCH ESO INS COMB IFLT']
            nd_filter = header['HIERARCH ESO INS4 FILT2 NAME']
            _type = header['HIERARCH ESO DPR TYPE']

        except KeyError as e:
            print(f"KeyError: {e} in file {file}")
            print(header)
            return

        if _type == 'OBJECT':
            _type = 'SCIENCE'
        elif 'OBJECT' in _type:
            _type = _type.replace('OBJECT,','')
        if 'DARK,BACKGROUND' in _type:
            _type = _type.replace('DARK,BACKGROUND','DARK')

        if 'DARK,BACKGROUND' in _object:
            _object = _object.replace('DARK,BACKGROUND','DARK')  


        for string in ['DARK,',',LAMP','LAMP,']: #,
            if string in _type:
                _type = _type.replace(string, '')
            if string in _object:
                _object = _object.replace(string, '')

        _object = _object.replace(' ','')    

        if _object in rename_dict:
            _object = rename_dict[_object]

        folderpath = os.path.join(_object, 'IRDIS')
        folderpath = os.path.join(folderpath, obs_night)

        folderpath = os.path.join(folderpath, filter_name)

        filepath = os.path.join(folderpath, _type + '_' + os.path.basename(file))


        DF_i = pd.DataFrame({
            "orig_file": [file],
            "object": [_object],
            "exp_time": [exp_time],
            "obs_night": [obs_night],
            "filter_name": [filter_name],
            "nd_filter": [nd_filter],
            "type": [_type],
            "filepath": [filepath],
            "folderpath": [folderpath],})

        DF = pd.concat([DF, DF_i], ignore_index=True)

    DF = DF.sort_values(by=["object", "obs_night", "filter_name", "exp_time", "orig_file"], ascending=[True, True, True, True, True])
    
    return DF


def move_files_from_download_dir(download_directory, new_directory, rename_dict):
    """
    Move files from download directory to new directory with organized structure. And rename objects based on rename_dict.
    
    :param download_directory: Description
    :param new_directory: Description
    :param rename_dict: Description
    """

    all_files = glob(os.path.join(download_directory, 'SPHER*.fits'))
    DF = get_info(all_files, rename_dict)

    DF = DF[DF['object'] != 'OBJECT']

    for directory in DF['folderpath'].unique():
        full_path = os.path.join(new_directory, directory)
        os.makedirs(full_path, exist_ok=True)

    i = 0 
    print('Copying files to new directory...')
    for orig_file, filepath in zip(DF['orig_file'],DF['filepath'].to_list()):
        if i % 100 == 0:
            print(i)
        orig_filepath = os.path.join(orig_file)
        new_filepath = os.path.join(new_directory, filepath)
        shutil.copy2(orig_filepath, new_filepath)
        i += 1

if __name__ == "__main__":

    rename_dict = {
        'epsSgr' : 'HD169022_epsSgr',
        'epsEri' : 'HD22049_epsEri',
        'VDEBoo' : 'HD131511_VDEBoo',
        'TYC8972-2028-1' : 'TYC_8972-2028-1',
        'mu.Tel' : 'HD183028_mu.Tel',
        'HIP_36312' : 'HD59659',
        'TYC_8128_1946_1' : 'HD59659',
    }

    download_directory = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY/'
    new_directory = '/data/beegfs/astro-storage/groups/beuther/pasteimle/Background_data_ISPY2/'

    move_files_from_download_dir(download_directory, new_directory, rename_dict)
