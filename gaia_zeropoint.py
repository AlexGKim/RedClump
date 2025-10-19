# import pandas as pd
import numpy as np
# from typing import Dict, Optional, Union
# import warnings
import astropy.units as u
from astropy.io       import fits


# Import the g2 UniformDisk class
import sys
# sys.path.append('/Users/akim/Projects/g2')
# from g2.models.sources.simple import UniformDisk

# https://www.aanda.org/articles/aa/full_html/2021/05/aa39587-20/T3.html

# Physical constants
SPEED_OF_LIGHT = 2.99792458e8  # m/s

# Gaia photometric system constants
# G band (broad band, ~400-1000 nm)
GAIA_G_EFFECTIVE_WAVELENGTH = 639.07e-9  # meters (623 nm)
GAIA_G_EFFECTIVE_FREQUENCY = SPEED_OF_LIGHT / GAIA_G_EFFECTIVE_WAVELENGTH  # Hz

# BP band (blue photometer, ~330-680 nm)
GAIA_BP_EFFECTIVE_WAVELENGTH = 518.26e-9  # meters (532 nm)
GAIA_BP_EFFECTIVE_FREQUENCY = SPEED_OF_LIGHT / GAIA_BP_EFFECTIVE_WAVELENGTH  # Hz

# RP band (red photometer, ~630-1050 nm)
GAIA_RP_EFFECTIVE_WAVELENGTH = 782.51e-9  # meters (797 nm)
GAIA_RP_EFFECTIVE_FREQUENCY = SPEED_OF_LIGHT / GAIA_RP_EFFECTIVE_WAVELENGTH  # Hz

# # Unit conversion constants
# MAS_TO_RAD = np.pi / (180 * 3600 * 1000)  # milliarcseconds to radians


with fits.open('alpha_lyr_mod_004.fits') as hdul:
    data = hdul[1].data
    lam = data['WAVELENGTH']  * u.AA
    flux = data['FLUX']  * u.erg/(u.s * u.cm**2 * u.AA)

fnu_array = flux.to(u.W/(u.m**2 * u.Hz),
                    equivalencies=u.spectral_density(lam))

# Vega flux densities at Gaia effective wavelengths
VEGA_FLUX_G = np.interp(GAIA_G_EFFECTIVE_WAVELENGTH*1e10, lam.value, fnu_array.value)
VEGA_FLUX_BP = np.interp(GAIA_BP_EFFECTIVE_WAVELENGTH*1e10, lam.value, fnu_array.value)
VEGA_FLUX_RP = np.interp(GAIA_RP_EFFECTIVE_WAVELENGTH*1e10, lam.value, fnu_array.value)
