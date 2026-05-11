import os, logging, numpy as np
from glob import glob
from astropy.time import Time
from ..exceptions import CalibrationFileNotFoundError

def find_nearest_calib_file(target_jd, calib_folder, pattern, validity_threshold_days):
    """Find nearest calibration file in time within validity threshold.
    
    Args:
        target_jd: Target Julian Date
        calib_folder: Folder containing calibration files
        pattern: Glob pattern for files (e.g., 'bad_pix_mask_*.fits')
        validity_threshold_days: Max allowed time difference in days
    
    Returns:
        str: Path to nearest calibration file
        
    Raises:
        CalibrationFileNotFoundError: If no file found within threshold
    """
    file_list = sorted(glob(os.path.join(calib_folder, pattern)))
    if not file_list:
        raise CalibrationFileNotFoundError(f"No calibration files matching {pattern} in {calib_folder}")
    
    jd_list = []
    for f in file_list:
        try:
            date_str = f.split('_')[-1].split('.')[0]
            jd = Time(date_str + 'T00:00:00.000', format='isot', scale='utc').jd
            jd_list.append(jd)
        except:
            continue
    
    if not jd_list:
        raise CalibrationFileNotFoundError(f"Could not extract dates from files matching {pattern}")
    
    nearest_idx = np.argmin(np.abs(np.array(jd_list) - target_jd))
    nearest_delta = np.min(np.abs(np.array(jd_list) - target_jd))
    
    if nearest_delta > validity_threshold_days:
        raise CalibrationFileNotFoundError(
            f"No calibration file within {validity_threshold_days} days. Nearest file is {nearest_delta:.1f} days away."
        )
    
    if nearest_delta > 0:
        logging.warning(f"Using calibration file {nearest_delta:.1f} days away from target")
    
    return file_list[nearest_idx]
