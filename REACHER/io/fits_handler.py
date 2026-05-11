import os, logging
from astropy.io import fits
from ..exceptions import FITSReadError, FITSWriteError, HeaderKeyError
import re

def sanitize_header(header):
    """Remove problematic characters from FITS header."""
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

def read_fits(filepath):
    """Read FITS file and return data + header."""
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = sanitize_header(hdul[0].header)
        return data, header
    except Exception as e:
        raise FITSReadError(f"Could not read FITS file {filepath}: {str(e)}")

def write_fits(filepath, data, header=None, overwrite=False):
    """Write data to FITS file."""
    try:
        hdu = fits.PrimaryHDU(data)
        if header:
            hdu.header.update(header)
        hdu.writeto(filepath, overwrite=overwrite)
    except Exception as e:
        raise FITSWriteError(f"Could not write FITS file {filepath}: {str(e)}")

def get_fits_keyword(filepath, keyword):
    """Get specific keyword from FITS header."""
    try:
        with fits.open(filepath) as hdul:
            return sanitize_header(hdul[0].header)[keyword]
    except KeyError:
        raise HeaderKeyError(f"Keyword {keyword} not found in {filepath}")
    except Exception as e:
        raise FITSReadError(f"Could not read FITS file {filepath}: {str(e)}")
