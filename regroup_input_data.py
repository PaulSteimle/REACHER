import REACHER

rename_dict = {
    'star_name_1' : 'starname1',
    'star_name_alt' : 'starname1'
}

download_directory = '/path/to/raw/download/folder'
new_directory = '/path/to/raw/data/'

REACHER.move_files_from_download_dir(download_directory, new_directory, rename_dict)



 