"""
Gaia to SATLAS RadialGrid Converter
===================================

This module provides functions to create g2.models.sources.radial_grid.RadialGrid2 objects
from SATLAS limb-darkening data, specifically using:
- Radial profiles in astronomical B, V, R, I, H, and K bands
- Radius data in milliarcseconds
- Wavelengths: 4450, 5510, 6580, 8060, 16300, and 21900 Angstroms

The module handles proper unit conversions and creates RadialGrid2 objects
for accurate visibility calculations using polar DFT algorithms.

Author: Generated for RedClump project
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
import warnings
import astropy.units as u
from pathlib import Path

# Import the g2 RadialGrid class
import sys
sys.path.append('/Users/akim/Projects/g2')
from g2.models.sources.radial_grid import RadialGrid2

# Physical constants
SPEED_OF_LIGHT = 2.99792458e8  # m/s
MAS_TO_RAD = np.pi / (180 * 3600 * 1000)  # milliarcseconds to radians
PARSEC_TO_METERS = 3.0857e16  # meters per parsec

# Default distance for SATLAS data (95.594 parsecs)
DEFAULT_DISTANCE_PC = 95.594
DEFAULT_DISTANCE_M = DEFAULT_DISTANCE_PC * PARSEC_TO_METERS

# SATLAS band information
SATLAS_BANDS = {
    'B': {'wavelength': 4450.0, 'column': 'I/I0_B'},
    'V': {'wavelength': 5510.0, 'column': 'I/I0_V'},
    'R': {'wavelength': 6580.0, 'column': 'I/I0_R'},
    'I': {'wavelength': 8060.0, 'column': 'I/I0_I'},
    'H': {'wavelength': 16300.0, 'column': 'I/I0_H'},
    'K': {'wavelength': 21900.0, 'column': 'I/I0_K'}
}


def create_radial_grid_from_satlas(file_path: str,
                                   star_row: pd.Series,
                                   g_mag_col: str = 'phot_g_mean_mag',
                                   bp_mag_col: str = 'phot_bp_mean_mag',
                                   rp_mag_col: str = 'phot_rp_mean_mag',
                                   s: float = 1.0) -> RadialGrid2:
    """
    Create RadialGrid2 object directly from SATLAS data file in one method.
    
    This function loads SATLAS limb-darkening data, validates it, and creates
    a RadialGrid2 object with proper unit conversions in a single operation.
    The specific flux is calculated from Gaia magnitudes using methods from
    gaia_uniform_disk and interpolated to SATLAS wavelengths.
    
    Parameters
    ----------
    file_path : str
        Path to the SATLAS data file
    star_row : pd.Series
        A row from a Gaia DataFrame containing magnitude data
    g_mag_col : str, default 'phot_g_mean_mag'
        Name of the column containing Gaia G magnitudes
    bp_mag_col : str, default 'phot_bp_mean_mag'
        Name of the column containing Gaia BP magnitudes
    rp_mag_col : str, default 'phot_rp_mean_mag'
        Name of the column containing Gaia RP magnitudes
    s : float, default 1.0
        Size parameter
        
    Returns
    -------
    RadialGrid2
        Configured RadialGrid2 object
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('extended_data_table_2.csv')
    >>> hd_360_row = df[df['Star'] == 'HD 360'].iloc[0]
    >>> radial_grid = create_radial_grid_from_satlas(
    ...     'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt',
    ...     star_row=hd_360_row
    ... )
    >>> print(f"Wavelength range: {radial_grid.get_spectrum_info()['wavelength_range_angstrom']}")
    >>> print(f"Radial range: {radial_grid.get_spectrum_info()['radial_range_rad']}")
    
    Notes
    -----
    The function performs the following operations:
    1. Loads SATLAS data from text file (skipping header)
    2. Validates data structure and content
    3. Converts radius from milliarcseconds to radians
    4. Extracts wavelength and intensity data
    5. Calculates specific flux from Gaia magnitudes and interpolates to SATLAS wavelengths
    6. Creates RadialGrid2 object with proper data types
    """
    try:
        # Load the data file, skipping the header line
        data = pd.read_csv(file_path, sep=r'\s+', comment='#')
        
        # Set proper column names if not already set
        expected_columns = ['r(mas)', 'I/I0_B', 'I/I0_V', 'I/I0_R', 'I/I0_I', 'I/I0_H', 'I/I0_K']
        if len(data.columns) == len(expected_columns):
            data.columns = expected_columns
        
    except Exception as e:
        raise ValueError(f"Error loading SATLAS data from {file_path}: {str(e)}")
    
    # Validate the loaded SATLAS data
    required_columns = ['r(mas)'] + [SATLAS_BANDS[band]['column'] for band in SATLAS_BANDS.keys()]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if data.empty:
        raise ValueError("SATLAS data is empty")
    
    # Check for negative radii
    if (data['r(mas)'] < 0).any():
        raise ValueError("Negative radii found in SATLAS data")
    
    # Check for invalid intensity values
    for band in SATLAS_BANDS.keys():
        col = SATLAS_BANDS[band]['column']
        if (data[col] < 0).any():
            warnings.warn(f"Negative intensity values found in {band} band")
        if (data[col] > 1.5).any():
            warnings.warn(f"Intensity values > 1.5 found in {band} band (expected normalized values)")
    
    # Extract radius data and convert from mas to radians (angular coordinates)
    radius_mas = data['r(mas)'].values.astype(np.float64)  # Ensure float64 precision
    p_rays = radius_mas * MAS_TO_RAD  # Convert to radians
    
    # Create wavelength grid
    wavelengths = np.array([SATLAS_BANDS[band]['wavelength'] for band in SATLAS_BANDS.keys()])
    
    # Create intensity array: shape (n_wavelengths, n_radial_points)
    n_wavelengths = len(wavelengths)
    n_radial_points = len(radius_mas)
    I_nu_p = np.zeros((n_wavelengths, n_radial_points))
    
    for i, band in enumerate(SATLAS_BANDS.keys()):
        col = SATLAS_BANDS[band]['column']
        I_nu_p[i, :] = data[col].values.astype(np.float64)  # Ensure float64 precision
    
    # Calculate specific_flux from Gaia magnitudes using methods from gaia_uniform_disk
    specific_flux = calculate_star_intensity_at_satlas_wavelengths(
        star_row, g_mag_col, bp_mag_col, rp_mag_col
    )
    
    # Debug: Check radius range
    max_radius_mas = radius_mas.max()
    max_radius_rad = max_radius_mas * MAS_TO_RAD
    
    print(f"Debug SATLAS radius info:")
    print(f"  Max radius: {max_radius_mas:.3f} mas")
    print(f"  Max radius: {max_radius_rad:.2e} rad")
    print(f"  Size parameter: {s}")
    
    # Create RadialGrid2 object with specified size parameter
    radial_grid = RadialGrid2(
        specific_flux=specific_flux,
        lambdas=wavelengths,
        I_nu_p=I_nu_p,
        p_rays=p_rays,
        s=s
    )
    
    return radial_grid


def get_satlas_band_info() -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Get information about SATLAS bands.
    
    Returns
    -------
    dict
        Dictionary with band information including wavelengths and column names
    """
    return SATLAS_BANDS.copy()


def calculate_visibility_at_bands(radial_grid: RadialGrid2,
                                  baseline: np.ndarray,
                                  bands: Optional[list] = None) -> Dict[str, complex]:
    """
    Calculate visibility at specific SATLAS bands.
    
    Parameters
    ----------
    radial_grid : RadialGrid2
        RadialGrid2 object created from SATLAS data
    baseline : array_like, shape (3,)
        Baseline vector in meters [Bx, By, Bz]
    bands : list, optional
        List of bands to calculate. If None, calculates all bands.
        
    Returns
    -------
    dict
        Dictionary mapping band names to complex visibility values
    """
    if bands is None:
        bands = list(SATLAS_BANDS.keys())
    
    visibilities = {}
    
    for band in bands:
        if band not in SATLAS_BANDS:
            warnings.warn(f"Unknown band: {band}")
            continue
            
        # Convert wavelength to frequency
        wavelength_angstrom = SATLAS_BANDS[band]['wavelength']
        wavelength_m = wavelength_angstrom * 1e-10
        frequency = SPEED_OF_LIGHT / wavelength_m
        
        # Calculate visibility
        visibility = radial_grid.V(frequency, baseline)
        visibilities[band] = visibility
    
    return visibilities


def get_radial_grid_properties(radial_grid: RadialGrid2) -> Dict[str, any]:
    """
    Get comprehensive properties of the RadialGrid2 object.
    
    Parameters
    ----------
    radial_grid : RadialGrid2
        RadialGrid2 object to analyze
        
    Returns
    -------
    dict
        Dictionary with detailed properties
    """
    spectrum_info = radial_grid.get_spectrum_info()
    
    properties = {
        'spectrum_info': spectrum_info,
        'bands': list(SATLAS_BANDS.keys()),
        'wavelengths_angstrom': [SATLAS_BANDS[band]['wavelength'] for band in SATLAS_BANDS.keys()],
        's': radial_grid.s,
        'max_radius_mas': np.max(radial_grid.p_rays) / MAS_TO_RAD,
        'angular_resolution_mas': (np.max(radial_grid.p_rays) - np.min(radial_grid.p_rays)) / MAS_TO_RAD / len(radial_grid.p_rays)
    }
    
    return properties


def create_radial_grid_from_gaia(df: pd.DataFrame,
                                 satlas_file: str,
                                 star_name_col: str = 'Star',
                                 g_mag_col: str = 'phot_g_mean_mag',
                                 bp_mag_col: str = 'phot_bp_mean_mag',
                                 rp_mag_col: str = 'phot_rp_mean_mag',
                                 s: float = 1.0) -> Dict[str, RadialGrid2]:
    """
    Create RadialGrid2 objects from Gaia data using SATLAS limb-darkening profiles.
    
    This function creates RadialGrid2 objects for multiple stars from a Gaia DataFrame,
    using SATLAS limb-darkening data. Each star gets absolute intensities calculated
    from Gaia magnitudes, normalized such that specific_flux matches the star's
    actual flux density.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stellar data with Gaia measurements.
    satlas_file : str
        Path to the SATLAS data file containing limb-darkening profiles.
    star_name_col : str, default 'Star'
        Name of the column containing star identifiers.
    g_mag_col : str, default 'phot_g_mean_mag'
        Name of the column containing Gaia G magnitudes.
    bp_mag_col : str, default 'phot_bp_mean_mag'
        Name of the column containing Gaia BP magnitudes.
    rp_mag_col : str, default 'phot_rp_mean_mag'
        Name of the column containing Gaia RP magnitudes.
    s : float, default 1.0
        Size parameter for RadialGrid2 objects.
        
    Returns
    -------
    star_grids : dict
        Dictionary mapping star names to RadialGrid2 objects.
        Keys are star names (str), values are RadialGrid2 instances with
        absolute intensities.
        
    Raises
    ------
    ValueError
        If DataFrame is invalid or SATLAS file cannot be loaded.
    TypeError
        If input is not a pandas DataFrame.
        
    Notes
    -----
    The function performs the following operations:
    1. Validates the input DataFrame
    2. Loads SATLAS limb-darkening data once (normalized profiles)
    3. For each star:
       - Calculates absolute flux densities at SATLAS wavelengths from Gaia magnitudes
       - Scales the normalized SATLAS profiles to match the star's flux densities
       - Creates a RadialGrid2 object with absolute intensities
    
    Rows with NaN values in required columns are skipped with a warning.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('extended_data_table_2.csv')
    >>> satlas_file = 'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt'
    >>> star_grids = create_radial_grid_from_gaia(df, satlas_file)
    >>>
    >>> # Access specific star
    >>> hd_360 = star_grids['HD 360']
    >>> print(f"HD 360 wavelength range: {hd_360.get_spectrum_info()['wavelength_range_angstrom']}")
    >>>
    >>> # Calculate visibility
    >>> import numpy as np
    >>> baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W baseline
    >>> frequency = SPEED_OF_LIGHT / (5510e-10)  # V band frequency
    >>> visibility = hd_360.V(frequency, baseline)
    >>> print(f"Visibility: {abs(visibility):.3f}")
    """
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if star_name_col not in df.columns:
        raise ValueError(f"Star name column '{star_name_col}' not found in DataFrame")
    
    # Check for any valid data in star name column
    if df[star_name_col].isna().all():
        raise ValueError(f"Column '{star_name_col}' contains no valid data")
    
    # Load SATLAS data once (will be reused for all stars)
    try:
        # Load the data file, skipping the header line
        satlas_data = pd.read_csv(satlas_file, sep=r'\s+', comment='#')
        
        # Set proper column names if not already set
        expected_columns = ['r(mas)', 'I/I0_B', 'I/I0_V', 'I/I0_R', 'I/I0_I', 'I/I0_H', 'I/I0_K']
        if len(satlas_data.columns) == len(expected_columns):
            satlas_data.columns = expected_columns
        
    except Exception as e:
        raise ValueError(f"Error loading SATLAS data from {satlas_file}: {str(e)}")
    
    # Validate the loaded SATLAS data
    required_columns = ['r(mas)'] + [SATLAS_BANDS[band]['column'] for band in SATLAS_BANDS.keys()]
    missing_columns = [col for col in required_columns if col not in satlas_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in SATLAS file: {missing_columns}")
    
    if satlas_data.empty:
        raise ValueError("SATLAS data is empty")
    
    # Check for negative radii
    if (satlas_data['r(mas)'] < 0).any():
        raise ValueError("Negative radii found in SATLAS data")
    
    # Check for invalid intensity values
    for band in SATLAS_BANDS.keys():
        col = SATLAS_BANDS[band]['column']
        if (satlas_data[col] < 0).any():
            warnings.warn(f"Negative intensity values found in {band} band")
        if (satlas_data[col] > 1.5).any():
            warnings.warn(f"Intensity values > 1.5 found in {band} band (expected normalized values)")
    
    # Extract radius data and convert from mas to radians (angular coordinates)
    radius_mas = satlas_data['r(mas)'].values.astype(np.float64)
    p_rays = radius_mas * MAS_TO_RAD
    p_rays = p_rays.astype(np.float64)
    
    # Create wavelength grid
    wavelengths = np.array([SATLAS_BANDS[band]['wavelength'] for band in SATLAS_BANDS.keys()])
    
    # Create intensity array: shape (n_wavelengths, n_radial_points)
    n_wavelengths = len(wavelengths)
    n_radial_points = len(radius_mas)
    I_nu_p = np.zeros((n_wavelengths, n_radial_points))
    
    for i, band in enumerate(SATLAS_BANDS.keys()):
        col = SATLAS_BANDS[band]['column']
        I_nu_p[i, :] = satlas_data[col].values.astype(np.float64)
    
    # Create RadialGrid2 objects for each star
    star_grids = {}
    skipped_stars = []
    
    for idx, row in df.iterrows():
        star_name = row[star_name_col]
        
        # Check for missing star name
        if pd.isna(star_name):
            skipped_stars.append(f"Row {idx}")
            continue
        
        # Check for required magnitude columns
        required_values = [row[g_mag_col], row[bp_mag_col], row[rp_mag_col]]
        if any(pd.isna(val) for val in required_values):
            skipped_stars.append(star_name)
            continue
        
        try:
            # Calculate absolute flux densities for this star at SATLAS wavelengths
            star_flux_densities = calculate_star_intensity_at_satlas_wavelengths(
                row, g_mag_col, bp_mag_col, rp_mag_col
            )
            
            print(star_flux_densities)
            # Create RadialGrid2 object with absolute intensities
            radial_grid = RadialGrid2(
                specific_flux=star_flux_densities,
                lambdas=wavelengths,
                I_nu_p=I_nu_p,
                p_rays=p_rays,
                s=s
            )
            
            # Store in dictionary
            star_grids[star_name] = radial_grid
            
        except Exception as e:
            warnings.warn(f"Error processing {star_name}: {str(e)}")
            skipped_stars.append(star_name)
            continue
        
        break
    
    # Report summary
    total_stars = len(df)
    successful_stars = len(star_grids)
    
    print(f"Successfully created RadialGrid2 objects for {successful_stars}/{total_stars} stars")
    if skipped_stars:
        print(f"Skipped {len(skipped_stars)} stars due to missing/invalid data: {skipped_stars[:5]}{'...' if len(skipped_stars) > 5 else ''}")
    
    return star_grids


def calculate_star_intensity_at_satlas_wavelengths(star_row: pd.Series,
                                                   g_mag_col: str = 'phot_g_mean_mag',
                                                   bp_mag_col: str = 'phot_bp_mean_mag',
                                                   rp_mag_col: str = 'phot_rp_mean_mag') -> np.ndarray:
    """
    Calculate intensity values for a Gaia star at SATLAS wavelengths.
    
    This function takes Gaia G, BP, and RP magnitudes, converts them to flux densities,
    and interpolates to the SATLAS wavelengths (B, V, R, I, H, K bands).
    
    Parameters
    ----------
    star_row : pd.Series
        A row from a Gaia DataFrame containing magnitude data.
    g_mag_col : str, default 'phot_g_mean_mag'
        Name of the column containing Gaia G magnitudes.
    bp_mag_col : str, default 'phot_bp_mean_mag'
        Name of the column containing Gaia BP magnitudes.
    rp_mag_col : str, default 'phot_rp_mean_mag'
        Name of the column containing Gaia RP magnitudes.
        
    Returns
    -------
    intensities : np.ndarray
        Array of intensity values at SATLAS wavelengths (6 values for B, V, R, I, H, K).
        Units: W m^-2 Hz^-1
        
    Raises
    ------
    ValueError
        If required magnitude columns are missing or contain NaN values.
        
    Notes
    -----
    The function performs the following steps:
    1. Extracts Gaia G, BP, RP magnitudes from the star row
    2. Converts magnitudes to flux densities using functions from gaia_uniform_disk
    3. Creates wavelength-flux density pairs for Gaia bands
    4. Interpolates (log-linear) to SATLAS wavelengths
    
    The interpolation is performed in log space to better handle the exponential
    nature of stellar spectra.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('extended_data_table_2.csv')
    >>> hd_360_row = df[df['Star'] == 'HD 360'].iloc[0]
    >>> intensities = calculate_star_intensity_at_satlas_wavelengths(hd_360_row)
    >>> print(f"Intensities at SATLAS wavelengths: {intensities}")
    """
    from gaia_uniform_disk import (
        g_magnitude_to_flux_density,
        bp_magnitude_to_flux_density,
        rp_magnitude_to_flux_density
    )
    from gaia_zeropoint import (
        GAIA_G_EFFECTIVE_WAVELENGTH,
        GAIA_BP_EFFECTIVE_WAVELENGTH,
        GAIA_RP_EFFECTIVE_WAVELENGTH
    )
    
    # Check for required columns
    required_cols = [g_mag_col, bp_mag_col, rp_mag_col]
    for col in required_cols:
        if col not in star_row.index:
            raise ValueError(f"Required column '{col}' not found in star data")
        if pd.isna(star_row[col]):
            raise ValueError(f"Column '{col}' contains NaN value")
    
    # Extract magnitudes
    g_mag = star_row[g_mag_col]
    bp_mag = star_row[bp_mag_col]
    rp_mag = star_row[rp_mag_col]
    
    # Convert magnitudes to flux densities
    flux_g = g_magnitude_to_flux_density(g_mag)
    flux_bp = bp_magnitude_to_flux_density(bp_mag)
    flux_rp = rp_magnitude_to_flux_density(rp_mag)
    
    # Gaia wavelengths in Angstroms
    gaia_wavelengths = np.array([
        GAIA_BP_EFFECTIVE_WAVELENGTH * 1e10,  # Convert m to Angstroms
        GAIA_G_EFFECTIVE_WAVELENGTH * 1e10,
        GAIA_RP_EFFECTIVE_WAVELENGTH * 1e10
    ])
    
    # Gaia flux densities
    gaia_flux_densities = np.array([flux_bp, flux_g, flux_rp])
    
    # SATLAS wavelengths in Angstroms
    satlas_wavelengths = np.array([SATLAS_BANDS[band]['wavelength'] for band in SATLAS_BANDS.keys()])
    
    # Perform log-linear interpolation
    # This is more appropriate for stellar spectra which are approximately power laws
    log_gaia_wavelengths = np.log10(gaia_wavelengths)
    log_gaia_flux = np.log10(gaia_flux_densities)
    log_satlas_wavelengths = np.log10(satlas_wavelengths)
    
    # Interpolate in log space
    log_satlas_flux = np.interp(log_satlas_wavelengths, log_gaia_wavelengths, log_gaia_flux)
    
    # Convert back from log space
    satlas_intensities = 10**log_satlas_flux
    
    return satlas_intensities

def create_radial_from_gaia(df: pd.DataFrame,
                            teff_col: str = 'teff_gspphot',
                            logg_col: str = 'logg_gspphot',
                            feh_col: str = 'mh_gspphot',
                            g_mag_col: str = 'phot_g_mean_mag',
                            bp_mag_col: str = 'phot_bp_mean_mag',
                            rp_mag_col: str = 'phot_rp_mean_mag',
                            star_name_col: str = 'Star',
                            vel: float = 1.0,
                            s: float = 1.0,
                            n_radial_points: int = 100,
                            satlas_log_dir: Optional[str] = None) -> Dict[str, RadialGrid2]:
    """
    Create RadialGrid2 objects from Gaia data using PHOENIX limb-darkening models.
    
    This function creates RadialGrid2 objects for stars from a Gaia DataFrame by:
    1. Matching each star's parameters (Teff, log g, [Fe/H]) to the closest PHOENIX model
    2. Getting limb-darkening functions for each SATLAS band from LD_phoenix
    3. Creating gridded intensity profiles for each band
    4. Building RadialGrid2 objects with absolute flux densities from Gaia magnitudes
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stellar data with Gaia measurements and parameters.
    teff_col : str, default 'teff_gspphot'
        Name of the column containing effective temperature (K).
    logg_col : str, default 'logg_gspphot'
        Name of the column containing surface gravity (log g).
    feh_col : str, default 'mh_gspphot'
        Name of the column containing metallicity [M/H] or [Fe/H].
    g_mag_col : str, default 'phot_g_mean_mag'
        Name of the column containing Gaia G magnitudes.
    bp_mag_col : str, default 'phot_bp_mean_mag'
        Name of the column containing Gaia BP magnitudes.
    rp_mag_col : str, default 'phot_rp_mean_mag'
        Name of the column containing Gaia RP magnitudes.
    star_name_col : str, default 'Star'
        Name of the column containing star identifiers.
    vel : float, default 1.0
        Microturbulent velocity [km/s] for PHOENIX models.
    s : float, default 1.0
        Size parameter for RadialGrid2 objects.
    n_radial_points : int, default 100
        Number of radial grid points from center to limb.
    satlas_log_dir : str, optional
        Path to directory containing SATLAS log file with radius and distance info.
        If provided, will read radius (r) and distance (D) from log file to properly
        calculate the angular coordinate relationship: gamma = r/D * sin(t), where
        t is the angular coordinate in the intensity profile.
        
    Returns
    -------
    star_grids : dict
        Dictionary mapping star names to RadialGrid2 objects.
        Keys are star names (str), values are RadialGrid2 instances with
        PHOENIX limb-darkening profiles and absolute intensities.
        
    Raises
    ------
    ValueError
        If DataFrame is invalid or required columns are missing.
    TypeError
        If input is not a pandas DataFrame.
        
    Notes
    -----
    The function performs the following operations for each star:
    1. Extracts stellar parameters (Teff, log g, [Fe/H])
    2. Matches to closest PHOENIX model using match_phoenix_parameters
    3. Gets limb-darkening functions for each SATLAS band (B, V, R, I, H, K)
    4. If satlas_log_dir provided, reads radius r and distance D from log file
    5. Creates radial intensity grid using the limb-darkening functions:
       - For each angular coordinate t: gamma = r/D * sin(t)
       - Converts gamma to mu = cos(gamma)
       - Evaluates limb-darkening function at mu
    6. Calculates absolute flux densities from Gaia magnitudes
    7. Creates RadialGrid2 object with PHOENIX profiles and absolute intensities
    
    The SATLAS bands are mapped to PHOENIX filters as follows:
    - B band (4450 Å) → PHOENIX 'B' filter
    - V band (5510 Å) → PHOENIX 'V' filter
    - R band (6580 Å) → PHOENIX 'R' filter
    - I band (8060 Å) → PHOENIX 'I' filter
    - H band (16300 Å) → PHOENIX 'H' filter
    - K band (21900 Å) → PHOENIX 'K' filter
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('extended_data_table_2.csv')
    >>> star_grids = create_radial_from_gaia(df)
    >>>
    >>> # Access specific star
    >>> hd_360 = star_grids['HD 360']
    >>> print(f"HD 360 wavelength range: {hd_360.get_spectrum_info()['wavelength_range_angstrom']}")
    >>>
    >>> # Calculate visibility
    >>> import numpy as np
    >>> baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W baseline
    >>> frequency = SPEED_OF_LIGHT / (5510e-10)  # V band frequency
    >>> visibility = hd_360.V(frequency, baseline)
    >>> print(f"Visibility: {abs(visibility):.3f}")
    """
    from LD_phoenix import LimbDarkening
    from gaia_phoenix import match_phoenix_parameters
    
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for required columns
    required_cols = [star_name_col, teff_col, logg_col, g_mag_col, bp_mag_col, rp_mag_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize LimbDarkening class
    ld = LimbDarkening()
    
    # Create wavelength grid for SATLAS bands
    wavelengths = np.array([SATLAS_BANDS[band]['wavelength'] for band in SATLAS_BANDS.keys()])
    n_wavelengths = len(wavelengths)
    
    # Map SATLAS bands to PHOENIX filters
    satlas_to_phoenix = {
        'B': 'B',
        'V': 'V',
        'R': 'R',
        'I': 'I',
        'H': 'H',
        'K': 'K'
    }
    
    # Read radius and distance from SATLAS log if provided
    radius_rsun = None
    distance_pc = None
    if satlas_log_dir is not None:
        log_path = Path(satlas_log_dir) / 'log'
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    for line in f:
                        if 'Radius' in line and 'distance' in line:
                            # Parse line like: "Radius = 9.312 Rsun ; distance = 95.594 pc"
                            parts = line.split(';')
                            radius_part = parts[0].split('=')[1].strip().split()[0]
                            distance_part = parts[1].split('=')[1].strip().split()[0]
                            radius_rsun = float(radius_part)
                            distance_pc = float(distance_part)
                            print(f"Read from SATLAS log: r = {radius_rsun} R_sun, D = {distance_pc} pc")
                            break
            except Exception as e:
                warnings.warn(f"Could not read radius/distance from {log_path}: {str(e)}")
    
    # Create angular coordinate grid t (from 0 to pi/2, center to limb)
    t_grid = np.linspace(0, np.pi/2, n_radial_points, dtype=np.float64)
    
    # Create RadialGrid2 objects for each star
    star_grids = {}
    skipped_stars = []
    
    for idx, row in df.iterrows():
        star_name = row[star_name_col]
        
        # Check for missing star name
        if pd.isna(star_name):
            skipped_stars.append(f"Row {idx}")
            continue
        
        # Check for required parameter columns
        required_values = [row[teff_col], row[logg_col], row[g_mag_col], row[bp_mag_col], row[rp_mag_col]]
        if any(pd.isna(val) for val in required_values):
            skipped_stars.append(star_name)
            continue
        
        try:
            # Extract stellar parameters
            teff = row[teff_col]
            logg = row[logg_col]
            feh = row[feh_col] if feh_col in df.columns and not pd.isna(row[feh_col]) else 0.0
            
            # Match to closest PHOENIX model
            teff_phoenix, logg_phoenix, feh_phoenix = match_phoenix_parameters(teff, logg, feh)
            
            print(f"\n{star_name}:")
            print(f"  Gaia params: Teff={teff:.0f}K, logg={logg:.2f}, [Fe/H]={feh:.2f}")
            print(f"  PHOENIX match: Teff={teff_phoenix:.0f}K, logg={logg_phoenix:.2f}, [Fe/H]={feh_phoenix:.2f}")
            
            # Create intensity array: shape (n_wavelengths, n_radial_points)
            I_nu_p = np.zeros((n_wavelengths, n_radial_points))
            
            # Get limb-darkening function for each band
            for i, band in enumerate(SATLAS_BANDS.keys()):
                phoenix_filter = satlas_to_phoenix[band]
                
                # Get limb-darkening function from PHOENIX (mu-based)
                ld_func = ld.get_limb_darkening_function(
                    logg_phoenix, teff_phoenix, feh_phoenix, vel, phoenix_filter
                )
                
                # Calculate intensity profile based on angular coordinate relationship
                if radius_rsun is not None and distance_pc is not None:
                    # Convert radius to meters and distance to meters
                    r_meters = radius_rsun * 6.957e8  # R_sun to meters
                    D_meters = distance_pc * PARSEC_TO_METERS
                    
                    # For each angular coordinate t, calculate gamma = (r/D) * sin(t)
                    # gamma is the angle from the line of sight to the surface normal
                    gamma_angles = (r_meters / D_meters) * np.sin(t_grid)
                    
                    # Convert gamma to mu = cos(gamma)
                    mu_values = np.cos(gamma_angles)
                    
                    # Evaluate limb-darkening function at mu values
                    I_nu_p[i, :] = ld_func(mu_values)
                else:
                    # Fallback: use t_grid directly as angles from surface normal
                    # This assumes normalized coordinates where t goes from 0 (center) to pi/2 (limb)
                    mu_values = np.cos(t_grid)
                    I_nu_p[i, :] = ld_func(mu_values)
            
            # Calculate absolute flux densities for this star at SATLAS wavelengths
            star_flux_densities = calculate_star_intensity_at_satlas_wavelengths(
                row, g_mag_col, bp_mag_col, rp_mag_col
            )
            
            # Convert angular coordinates to radians for p_rays
            # p_rays represents the angular coordinate in the sky plane
            if radius_rsun is not None and distance_pc is not None:
                # Calculate angular radius: theta = r/D (in radians)
                r_meters = radius_rsun * 6.957e8  # R_sun to meters
                D_meters = distance_pc * PARSEC_TO_METERS
                max_angular_radius_rad = r_meters / D_meters
                
                # p_rays = angular radius * sin(t), where t is the angular coordinate
                p_rays = max_angular_radius_rad * np.sin(t_grid)
                p_rays = p_rays.astype(np.float64)
                
                print(f"  Angular radius: {max_angular_radius_rad / MAS_TO_RAD:.3f} mas")
            elif 'LD' in df.columns and not pd.isna(row['LD']):
                # LD is limb-darkened angular diameter in mas
                max_radius_mas = float(row['LD']) / 2.0  # Convert diameter to radius
                max_angular_radius_rad = max_radius_mas * MAS_TO_RAD
                
                # p_rays = angular radius * sin(t)
                p_rays = max_angular_radius_rad * np.sin(t_grid)
                p_rays = p_rays.astype(np.float64)
            else:
                # Use a typical stellar radius estimate as fallback
                max_radius_mas = 0.5  # mas, typical for red clump stars
                max_angular_radius_rad = max_radius_mas * MAS_TO_RAD
                
                # p_rays = angular radius * sin(t)
                p_rays = max_angular_radius_rad * np.sin(t_grid)
                p_rays = p_rays.astype(np.float64)
            
            # Create RadialGrid2 object with PHOENIX limb-darkening and absolute intensities
            radial_grid = RadialGrid2(
                specific_flux=star_flux_densities,
                lambdas=wavelengths,
                I_nu_p=I_nu_p,
                p_rays=p_rays,
                s=s
            )
            
            # Store in dictionary
            star_grids[star_name] = radial_grid
            
        except Exception as e:
            import traceback
            print(f"\nDetailed error for {star_name}:")
            print(traceback.format_exc())
            warnings.warn(f"Error processing {star_name}: {str(e)}")
            skipped_stars.append(star_name)
            continue
    
    # Report summary
    total_stars = len(df)
    successful_stars = len(star_grids)
    
    print(f"\n{'='*60}")
    print(f"Successfully created RadialGrid2 objects for {successful_stars}/{total_stars} stars")
    if skipped_stars:
        print(f"Skipped {len(skipped_stars)} stars due to missing/invalid data: {skipped_stars[:5]}{'...' if len(skipped_stars) > 5 else ''}")
    
    return star_grids



if __name__ == "__main__":
    # Example usage and testing
    print("SATLAS to RadialGrid2 Converter")
    print("=" * 40)
    
    # Define the SATLAS data file path
    satlas_file = 'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt'
    
    try:
        # Load Gaia data for example star
        df = pd.read_csv('extended_data_table_2.csv')
        hd_360_row = df[df['Star'] == 'HD 360'].iloc[0]
        
        # Create RadialGrid2 from SATLAS data with Gaia magnitudes
        print(f"Loading SATLAS data from: {satlas_file}")
        print(f"Using HD 360 magnitudes: G={hd_360_row['phot_g_mean_mag']:.3f}, BP={hd_360_row['phot_bp_mean_mag']:.3f}, RP={hd_360_row['phot_rp_mean_mag']:.3f}")
        radial_grid = create_radial_grid_from_satlas(satlas_file, hd_360_row)
        
        # Display properties
        properties = get_radial_grid_properties(radial_grid)
        print(f"\nRadialGrid2 Properties:")
        print(f"  Wavelength range: {properties['spectrum_info']['wavelength_range_angstrom']} Å")
        print(f"  Frequency range: {properties['spectrum_info']['frequency_range_hz']} Hz")
        print(f"  Radial range: {properties['spectrum_info']['radial_range_rad']} rad")
        print(f"  Max radius: {properties['max_radius_mas']:.3f} mas")
        print(f"  Angular resolution: {properties['angular_resolution_mas']:.6f} mas")
        print(f"  Bands: {properties['bands']}")
        print(f"  Wavelengths: {properties['wavelengths_angstrom']} Å")
        print(f"  Size parameter: {properties['s']}")
        
        # Example visibility calculation
        baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W baseline
        print(f"\nVisibility calculation (100m E-W baseline):")
        
        visibilities = calculate_visibility_at_bands(radial_grid, baseline)
        for band, visibility in visibilities.items():
            wavelength = SATLAS_BANDS[band]['wavelength']
            print(f"  {band} band ({wavelength} Å): |V| = {abs(visibility):.6f}")
        
        # Test specific flux calculation
        print(f"\nFlux density calculations:")
        for band in ['V', 'I', 'K']:  # Test a few bands
            wavelength_angstrom = SATLAS_BANDS[band]['wavelength']
            wavelength_m = wavelength_angstrom * 1e-10
            frequency = SPEED_OF_LIGHT / wavelength_m
            flux = radial_grid.specific_flux(frequency)
            print(f"  {band} band: F_ν = {flux:.2e}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {str(e)}")
        print("Please ensure the data files exist in the specified location")
    except Exception as e:
        print(f"Error: {str(e)}")
        
    # Display band information
    print(f"\nSATLAS Band Information:")
    band_info = get_satlas_band_info()
    for band, info in band_info.items():
        print(f"  {band}: {info['wavelength']} Å ({info['column']})")