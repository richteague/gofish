import os
import numpy as np
from astropy.io import fits
import scipy.constants as sc

class imagecube:

    # Disk specific units.
    msun = 1.988e30
    fwhm = 2. * np.sqrt(2 * np.log(2))
    disk_coords_niter = 20

    def __init__(self, path, kelvin=False, clip=None, resample=1, verbose=None,
                 suppress_warnings=True, dx0=0.0, dy0=0.0):
        """
        Load up a FITS image cube.

        Args:
            path (str): Relative path to the FITS cube.
            kelvin (Optional[bool/str]): Convert the brightness units to [K].
                If True, use the full Planck law, or if 'RJ' use the
                Rayleigh-Jeans approximation. This is not as accurate but does
                not suffer as much in the low intensity regime.
            clip (Optional[float]): Clip the image cube down to a FOV spanning
                (2 * clip) in [arcseconds].
            resample (Optional[int]): Resample the data spectrally, averaging
                over `resample` number of channels.
            verbose (Optional[bool]): Print out warning messages messages.
            suppress_warnings (Optional[bool]): Suppress warnings from other
                Python pacakges (for example numpy). If this is selected then
                verbose will be set to False unless specified.
            dx0 (Optional[float]): Recenter the image to this right ascencion
                offset [arcsec].
            dy0 (Optional[float]): Recenter the image to this declination
                offset [arcsec].
        Returns:
            None)
        """
