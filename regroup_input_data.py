import REACHER

rename_dict = {
    #'epsSgr' : 'HD169022_epsSgr',
}

download_directory = '/Users/pasteimle/SPHERE_test_data'
new_directory = '/Users/pasteimle/SPHERE_test_data'

REACHER.move_files_from_download_dir(download_directory, new_directory, rename_dict)



