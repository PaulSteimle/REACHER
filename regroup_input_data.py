import REACHER

rename_dict = {
    #'epsSgr' : 'HD169022_epsSgr',
}

download_directory = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/HD3670/download_dir'
new_directory = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/HD3670/raw_files'

download_directory = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/download'
new_directory = '/nexus/posix0/MIA-astro-env/lkreidberg/pasteimle/additional_ISPY_raw_files'

REACHER.move_files_from_download_dir(download_directory, new_directory, rename_dict)



