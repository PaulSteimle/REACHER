import REACHER

rename_dict = {
    'epsSgr' : 'HD169022_epsSgr',
}

download_directory = '/path/to/downloaded/data/'
new_directory = '/path/to/structured/data/'

REACHER.move_files_from_download_dir(download_directory, new_directory, rename_dict)