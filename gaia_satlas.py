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
                                   s: float = 1.0) -> RadialGrid2:
    """
    Create RadialGrid2 object directly from SATLAS data file in one method.
    
    This function loads SATLAS limb-darkening data, validates it, and creates
    a RadialGrid2 object with proper unit conversions in a single operation.
    
    Parameters
    ----------
    file_path : str
        Path to the SATLAS data file
    s : float, default 1.0
        Size parameter
        
    Returns
    -------
    RadialGrid2
        Configured RadialGrid2 object
        
    Examples
    --------
    >>> radial_grid = create_radial_grid_from_satlas(
    ...     'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt'
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
    5. Creates RadialGrid2 object with proper data types
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
    p_rays = p_rays.astype(np.float64)  # Ensure float64 precision
    
    # Create wavelength grid
    wavelengths = np.array([SATLAS_BANDS[band]['wavelength'] for band in SATLAS_BANDS.keys()])
    
    # Create intensity array: shape (n_wavelengths, n_radial_points)
    n_wavelengths = len(wavelengths)
    n_radial_points = len(radius_mas)
    I_nu_p = np.zeros((n_wavelengths, n_radial_points))
    
    for i, band in enumerate(SATLAS_BANDS.keys()):
        col = SATLAS_BANDS[band]['column']
        I_nu_p[i, :] = data[col].values.astype(np.float64)  # Ensure float64 precision
    
    # Debug: Check radius range
    max_radius_mas = radius_mas.max()
    max_radius_rad = max_radius_mas * MAS_TO_RAD
    
    print(f"Debug SATLAS radius info:")
    print(f"  Max radius: {max_radius_mas:.3f} mas")
    print(f"  Max radius: {max_radius_rad:.2e} rad")
    print(f"  Size parameter: {s}")
    
    # Create RadialGrid2 object with specified size parameter
    radial_grid = RadialGrid2(
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
            
            # Create absolute intensity array by scaling normalized profiles
            # The normalized profile I/I0 needs to be scaled such that when integrated
            # over the disk, it gives the correct specific flux
            I_nu_p_absolute = np.zeros((n_wavelengths, n_radial_points))
            
            for i in range(n_wavelengths):
                # First, create a temporary RadialGrid2 with the normalized profile
                # to calculate what the integrated flux would be
                I_temp = np.zeros((1, n_radial_points))
                I_temp[0, :] = I_nu_p[i, :]
                
                temp_grid = RadialGrid2(
                    lambdas=np.array([wavelengths[i]]),
                    I_nu_p=I_temp,
                    p_rays=p_rays,
                    s=s
                )
                
                # Calculate the specific flux for the normalized profile
                wavelength_m = wavelengths[i] * 1e-10
                frequency = SPEED_OF_LIGHT / wavelength_m
                normalized_flux = temp_grid.specific_flux(frequency)
                
                # Scale factor to match the star's actual flux density
                if normalized_flux > 0:
                    scale_factor = star_flux_densities[i] / normalized_flux
                else:
                    scale_factor = star_flux_densities[i]
                
                # Apply the scale factor to get absolute intensities
                I_nu_p_absolute[i, :] = scale_factor * I_nu_p[i, :]
            
            # Create RadialGrid2 object with absolute intensities
            radial_grid = RadialGrid2(
                lambdas=wavelengths,
                I_nu_p=I_nu_p_absolute,
                p_rays=p_rays,
                s=s
            )
            
            # Store in dictionary
            star_grids[star_name] = radial_grid
            
        except Exception as e:
            warnings.warn(f"Error processing {star_name}: {str(e)}")
            skipped_stars.append(star_name)
            continue
    
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


if __name__ == "__main__":
    # Example usage and testing
    print("SATLAS to RadialGrid2 Converter")
    print("=" * 40)
    
    # Define the SATLAS data file path
    satlas_file = 'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt'
    
    try:
        # Create RadialGrid2 from SATLAS data
        print(f"Loading SATLAS data from: {satlas_file}")
        radial_grid = create_radial_grid_from_satlas(satlas_file)
        
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
        
    except FileNotFoundError:
        print(f"Error: SATLAS data file not found: {satlas_file}")
        print("Please ensure the data file exists in the specified location")
    except Exception as e:
        print(f"Error: {str(e)}")
        
    # Display band information
    print(f"\nSATLAS Band Information:")
    band_info = get_satlas_band_info()
    for band, info in band_info.items():
        print(f"  {band}: {info['wavelength']} Å ({info['column']})")