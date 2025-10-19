"""
Gaia to UniformDisk Converter
============================

This module provides functions to create g2.models.sources.simple.UniformDisk objects
from Gaia photometric and astrometric data, specifically using:
- Gaia RP magnitude (phot_rp_mean_mag)
- Parallax (parallax) 
- Angular diameter (LD)

The module handles proper unit conversions and uses standard Gaia photometric
zero-points for accurate flux density calculations.

Author: Generated for RedClump project
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import warnings
import astropy.units as u
from astropy.io       import fits
from gaia_zeropoint import VEGA_FLUX_G ,VEGA_FLUX_BP ,VEGA_FLUX_RP,    GAIA_G_EFFECTIVE_FREQUENCY, GAIA_BP_EFFECTIVE_FREQUENCY, GAIA_RP_EFFECTIVE_FREQUENCY

# Import the g2 UniformDisk class
import sys
sys.path.append('/Users/akim/Projects/g2')
from g2.models.sources.simple import UniformDisk


# # Physical constants
# SPEED_OF_LIGHT = 2.99792458e8  # m/s

# # Gaia photometric system constants
# # G band (broad band, ~400-1000 nm)
# GAIA_G_EFFECTIVE_WAVELENGTH = 623e-9  # meters (623 nm)
# GAIA_G_EFFECTIVE_FREQUENCY = SPEED_OF_LIGHT / GAIA_G_EFFECTIVE_WAVELENGTH  # Hz

# # BP band (blue photometer, ~330-680 nm)
# GAIA_BP_EFFECTIVE_WAVELENGTH = 532e-9  # meters (532 nm)
# GAIA_BP_EFFECTIVE_FREQUENCY = SPEED_OF_LIGHT / GAIA_BP_EFFECTIVE_WAVELENGTH  # Hz

# # RP band (red photometer, ~630-1050 nm)
# GAIA_RP_EFFECTIVE_WAVELENGTH = 797e-9  # meters (797 nm)
# GAIA_RP_EFFECTIVE_FREQUENCY = SPEED_OF_LIGHT / GAIA_RP_EFFECTIVE_WAVELENGTH  # Hz

# # Unit conversion constants
MAS_TO_RAD = np.pi / (180 * 3600 * 1000)  # milliarcseconds to radians
# with fits.open('alpha_lyr_mod_004.fits') as hdul:
#     data = hdul[1].data
#     lam = data['WAVELENGTH']  * u.AA
#     flux = data['FLUX']  * u.erg/(u.s * u.cm**2 * u.AA)

# fnu_array = flux.to(u.W/(u.m**2 * u.Hz),
#                     equivalencies=u.spectral_density(lam))

# # Vega flux densities at Gaia effective wavelengths


# # For backward compatibility
# VEGA_FLUX = VEGA_FLUX_RP


# <Quantity [1.69e-09] erg / (Angstrom cm2)>

def g_magnitude_to_flux_density(g_mag: Union[float, np.ndarray],
                               reference_frequency: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Convert Gaia G magnitude to flux density in W m^-2 Hz^-1.
    
    Uses the standard Gaia G photometric zero-point to convert apparent magnitude
    to flux density at the Gaia G effective frequency.
    
    Parameters
    ----------
    g_mag : float or array_like
        Gaia G magnitude(s) from phot_g_mean_mag column.
    reference_frequency : float, optional
        Reference frequency in Hz. If None, uses Gaia G effective frequency.
        
    Returns
    -------
    flux_density : float or array_like
        Flux density in W m^-2 Hz^-1 at the reference frequency.
        
    Notes
    -----
    The conversion uses the standard magnitude-flux relation:
    F_ν = F_ν,0 * 10^(-0.4 * (m - m_0))
    
    where F_ν,0 is the Vega flux density at the G band effective wavelength.
    
    Examples
    --------
    >>> flux = g_magnitude_to_flux_density(5.716)  # HD 360
    >>> print(f"Flux density: {flux:.2e} W/m²/Hz")
    """
    if reference_frequency is None:
        reference_frequency = GAIA_G_EFFECTIVE_FREQUENCY
    
    flux_density = VEGA_FLUX_G * 10**(-0.4 * g_mag)
    return flux_density


def bp_magnitude_to_flux_density(bp_mag: Union[float, np.ndarray],
                                reference_frequency: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Convert Gaia BP magnitude to flux density in W m^-2 Hz^-1.
    
    Uses the standard Gaia BP photometric zero-point to convert apparent magnitude
    to flux density at the Gaia BP effective frequency.
    
    Parameters
    ----------
    bp_mag : float or array_like
        Gaia BP magnitude(s) from phot_bp_mean_mag column.
    reference_frequency : float, optional
        Reference frequency in Hz. If None, uses Gaia BP effective frequency.
        
    Returns
    -------
    flux_density : float or array_like
        Flux density in W m^-2 Hz^-1 at the reference frequency.
        
    Notes
    -----
    The conversion uses the standard magnitude-flux relation:
    F_ν = F_ν,0 * 10^(-0.4 * (m - m_0))
    
    where F_ν,0 is the Vega flux density at the BP band effective wavelength.
    
    Examples
    --------
    >>> flux = bp_magnitude_to_flux_density(6.228)  # HD 360
    >>> print(f"Flux density: {flux:.2e} W/m²/Hz")
    """
    if reference_frequency is None:
        reference_frequency = GAIA_BP_EFFECTIVE_FREQUENCY
    
    flux_density = VEGA_FLUX_BP * 10**(-0.4 * bp_mag)
    return flux_density


def rp_magnitude_to_flux_density(rp_mag: Union[float, np.ndarray],
                                reference_frequency: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Convert Gaia RP magnitude to flux density in W m^-2 Hz^-1.
    
    Uses the standard Gaia RP photometric zero-point to convert apparent magnitude
    to flux density at the Gaia RP effective frequency.
    
    Parameters
    ----------
    rp_mag : float or array_like
        Gaia RP magnitude(s) from phot_rp_mean_mag column.
    reference_frequency : float, optional
        Reference frequency in Hz. If None, uses Gaia RP effective frequency.
        
    Returns
    -------
    flux_density : float or array_like
        Flux density in W m^-2 Hz^-1 at the reference frequency.
        
    Notes
    -----
    The conversion uses the standard magnitude-flux relation:
    F_ν = F_ν,0 * 10^(-0.4 * (m - m_0))
    
    where:
    - F_ν,0 is the zero-point flux density
    - m is the observed magnitude
    - m_0 is the zero-point magnitude
    
    Examples
    --------
    >>> flux = rp_magnitude_to_flux_density(5.044)  # HD 360
    >>> print(f"Flux density: {flux:.2e} W/m²/Hz")
    """
    if reference_frequency is None:
        reference_frequency = GAIA_RP_EFFECTIVE_FREQUENCY
    
    # Convert magnitude to flux density using standard relation
    # F_ν = F_ν,0 * 10^(-0.4 * (m - m_0))
    # flux_density = 3631e-26 * 10**(-0.4 * (rp_mag - GAIA_RP_ZERO_POINT_MAG)) # W⋅m−2⋅Hz−1

    flux_density = VEGA_FLUX_RP * 10**(-0.4 * rp_mag)
   
    return flux_density


def angular_diameter_to_radius(angular_diameter_mas: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert angular diameter from milliarcseconds to angular radius in radians.
    
    Parameters
    ----------
    angular_diameter_mas : float or array_like
        Angular diameter in milliarcseconds (from LD column).
        
    Returns
    -------
    radius_rad : float or array_like
        Angular radius in radians.
        
    Notes
    -----
    Conversion: radius = diameter / 2 * (π / (180 * 3600 * 1000))
    
    Examples
    --------
    >>> radius = angular_diameter_to_radius(0.906)  # HD 360
    >>> print(f"Angular radius: {radius:.2e} radians")
    """
    # Convert diameter to radius and mas to radians
    radius_rad = (angular_diameter_mas / 2.0) * MAS_TO_RAD
    return radius_rad


def parallax_to_distance(parallax_mas: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert parallax from milliarcseconds to distance in parsecs.
    
    Parameters
    ----------
    parallax_mas : float or array_like
        Parallax in milliarcseconds.
        
    Returns
    -------
    distance_pc : float or array_like
        Distance in parsecs.
        
    Raises
    ------
    ValueError
        If parallax is negative or zero.
        
    Notes
    -----
    Uses the simple relation: distance = 1000 / parallax_mas
    
    This assumes parallax is in milliarcseconds and returns distance in parsecs.
    For more sophisticated distance estimates, consider using methods that
    account for parallax uncertainties and the Lutz-Kelker bias.
    
    Examples
    --------
    >>> distance = parallax_to_distance(9.011)  # HD 360
    >>> print(f"Distance: {distance:.1f} pc")
    """
    # Check for invalid parallax values
    if np.any(parallax_mas <= 0):
        raise ValueError("Parallax must be positive for distance calculation")
    
    # Convert parallax to distance: d = 1000 / π (where π is in mas)
    distance_pc = 1000.0 / parallax_mas
    return distance_pc


def validate_dataframe(df: pd.DataFrame,
                      star_name_col: str,
                      g_mag_col: str,
                      bp_mag_col: str,
                      rp_mag_col: str,
                      parallax_col: str,
                      angular_diameter_col: str) -> None:
    """
    Validate that the input DataFrame has the required columns and data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    star_name_col : str
        Name of the star name column.
    g_mag_col : str
        Name of the G magnitude column.
    bp_mag_col : str
        Name of the BP magnitude column.
    rp_mag_col : str
        Name of the RP magnitude column.
    parallax_col : str
        Name of the parallax column.
    angular_diameter_col : str
        Name of the angular diameter column.
        
    Raises
    ------
    ValueError
        If DataFrame is empty or missing required columns.
    TypeError
        If input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    required_columns = [star_name_col, g_mag_col, bp_mag_col, rp_mag_col, parallax_col, angular_diameter_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for any completely missing data in required columns
    for col in required_columns:
        if df[col].isna().all():
            raise ValueError(f"Column '{col}' contains no valid data")


def create_uniform_disk_from_gaia(df: pd.DataFrame,
                                 star_name_col: str = 'Star',
                                 g_mag_col: str = 'phot_g_mean_mag',
                                 bp_mag_col: str = 'phot_bp_mean_mag',
                                 rp_mag_col: str = 'phot_rp_mean_mag',
                                 parallax_col: str = 'parallax',
                                 angular_diameter_col: str = 'LD') -> Dict[str, UniformDisk]:
    """
    Create UniformDisk objects from Gaia photometric and astrometric data.
    
    This function converts Gaia G, BP, and RP magnitudes, parallaxes, and angular diameters
    into the parameters needed for g2.models.sources.simple.UniformDisk objects with
    frequency-dependent flux densities.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stellar data with Gaia measurements.
    star_name_col : str, default 'Star'
        Name of the column containing star identifiers.
    g_mag_col : str, default 'phot_g_mean_mag'
        Name of the column containing Gaia G magnitudes.
    bp_mag_col : str, default 'phot_bp_mean_mag'
        Name of the column containing Gaia BP magnitudes.
    rp_mag_col : str, default 'phot_rp_mean_mag'
        Name of the column containing Gaia RP magnitudes.
    parallax_col : str, default 'parallax'
        Name of the column containing parallax values in milliarcseconds.
    angular_diameter_col : str, default 'LD'
        Name of the column containing angular diameters in milliarcseconds.
        
    Returns
    -------
    star_disks : dict
        Dictionary mapping star names to UniformDisk objects.
        Keys are star names (str), values are UniformDisk instances with
        frequency-dependent flux densities.
        
    Raises
    ------
    ValueError
        If DataFrame is invalid or contains invalid data.
    TypeError
        If input is not a pandas DataFrame.
        
    Notes
    -----
    The function performs the following conversions:
    1. G, BP, RP magnitudes → flux densities (W m^-2 Hz^-1) using Gaia zero-points
    2. Angular diameter (mas) → angular radius (radians)
    3. Parallax (mas) → distance (pc) for validation
    
    The UniformDisk objects are created with frequency-dependent flux densities
    using the three Gaia photometric bands (G, BP, RP).
    
    Rows with NaN values in required columns are skipped with a warning.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('extended_data_table_2.csv')
    >>> star_disks = create_uniform_disk_from_gaia(df)
    >>>
    >>> # Access specific star
    >>> hd_360 = star_disks['HD 360']
    >>> print(f"HD 360 flux at G band: {hd_360._interpolate_flux_density(GAIA_G_EFFECTIVE_FREQUENCY):.2e} W/m²/Hz")
    >>> print(f"HD 360 radius: {hd_360.radius:.2e} radians")
    >>>
    >>> # Calculate visibility
    >>> import numpy as np
    >>> baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W baseline
    >>> nu_0 = GAIA_G_EFFECTIVE_FREQUENCY  # G band frequency
    >>> visibility = hd_360.V(nu_0, baseline)
    >>> print(f"Visibility: {abs(visibility):.3f}")
    """
    # Validate input DataFrame
    validate_dataframe(df, star_name_col, g_mag_col, bp_mag_col, rp_mag_col, parallax_col, angular_diameter_col)
    
    star_disks = {}
    skipped_stars = []
    
    for idx, row in df.iterrows():
        star_name = row[star_name_col]
        
        # Check for missing data in this row
        required_values = [row[g_mag_col], row[bp_mag_col], row[rp_mag_col],
                          row[parallax_col], row[angular_diameter_col]]
        if any(pd.isna(val) for val in required_values):
            skipped_stars.append(star_name)
            continue
        
        try:
            # Extract values
            g_mag = row[g_mag_col]
            bp_mag = row[bp_mag_col]
            rp_mag = row[rp_mag_col]
            parallax_mas = row[parallax_col]
            angular_diameter_mas = row[angular_diameter_col]
            
            # Validate parallax (must be positive for distance calculation)
            if parallax_mas <= 0:
                warnings.warn(f"Skipping {star_name}: invalid parallax ({parallax_mas} mas)")
                skipped_stars.append(star_name)
                continue
            
            # Validate angular diameter (must be positive)
            if angular_diameter_mas <= 0:
                warnings.warn(f"Skipping {star_name}: invalid angular diameter ({angular_diameter_mas} mas)")
                skipped_stars.append(star_name)
                continue
            
            # Convert magnitudes to flux densities at their respective effective frequencies
            flux_g = g_magnitude_to_flux_density(g_mag, GAIA_G_EFFECTIVE_FREQUENCY)
            flux_bp = bp_magnitude_to_flux_density(bp_mag, GAIA_BP_EFFECTIVE_FREQUENCY)
            flux_rp = rp_magnitude_to_flux_density(rp_mag, GAIA_RP_EFFECTIVE_FREQUENCY)
            
            # Create frequency and flux density arrays
            frequencies = np.array([GAIA_G_EFFECTIVE_FREQUENCY, GAIA_BP_EFFECTIVE_FREQUENCY, GAIA_RP_EFFECTIVE_FREQUENCY])
            flux_densities = np.array([flux_g, flux_bp, flux_rp])
            
            # Convert angular diameter to radius
            angular_radius = angular_diameter_to_radius(angular_diameter_mas)
            
            # Calculate distance for reference (not used in UniformDisk but useful for validation)
            distance_pc = parallax_to_distance(parallax_mas)
            
            # Create UniformDisk object with frequency-dependent flux densities using array constructors
            uniform_disk = UniformDisk(flux_density=flux_densities, frequencies=frequencies, radius=angular_radius)
            
            # Store in dictionary
            star_disks[star_name] = uniform_disk
            
        except Exception as e:
            warnings.warn(f"Error processing {star_name}: {str(e)}")
            skipped_stars.append(star_name)
            continue
    
    # Report summary
    total_stars = len(df)
    successful_stars = len(star_disks)
    
    print(f"Successfully created UniformDisk objects for {successful_stars}/{total_stars} stars")
    if skipped_stars:
        print(f"Skipped {len(skipped_stars)} stars due to missing/invalid data: {skipped_stars[:5]}{'...' if len(skipped_stars) > 5 else ''}")
    
    return star_disks


def get_star_properties(star_disks: Dict[str, UniformDisk], star_name: str,
                       frequency: Optional[float] = None) -> Dict[str, float]:
    """
    Get properties of a specific star's UniformDisk object.
    
    Parameters
    ----------
    star_disks : dict
        Dictionary of star names to UniformDisk objects.
    star_name : str
        Name of the star to query.
    frequency : float, optional
        Frequency in Hz for flux density calculation. If None, uses Gaia G effective frequency.
        
    Returns
    -------
    properties : dict
        Dictionary containing star properties.
        
    Examples
    --------
    >>> props = get_star_properties(star_disks, 'HD 360')
    >>> print(f"Flux density at G band: {props['flux_density']:.2e} W/m²/Hz")
    >>> print(f"Angular radius: {props['angular_radius_rad']:.2e} rad")
    >>> print(f"Angular diameter: {props['angular_diameter_mas']:.3f} mas")
    """
    if star_name not in star_disks:
        raise KeyError(f"Star '{star_name}' not found in star_disks")
    
    disk = star_disks[star_name]
    
    if frequency is None:
        frequency = GAIA_G_EFFECTIVE_FREQUENCY
    
    # Convert radius back to diameter in mas for reference
    angular_diameter_mas = (disk.radius * 2.0) / MAS_TO_RAD
    
    # Get flux density at specified frequency
    flux_density = disk._interpolate_flux_density(frequency)
    
    # Get surface brightness at specified frequency
    surface_brightness = disk._get_surface_brightness(frequency)
    
    return {
        'flux_density': flux_density,
        'angular_radius_rad': disk.radius,
        'angular_diameter_mas': angular_diameter_mas,
        'surface_brightness': surface_brightness,
        'frequency': frequency
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Gaia to UniformDisk Converter (Multi-band)")
    print("=" * 50)
    
    # Load the data
    try:
        df = pd.read_csv('extended_data_table_2.csv')
        print(f"Loaded data for {len(df)} stars")
        
        # Create UniformDisk objects with frequency-dependent flux densities
        star_disks = create_uniform_disk_from_gaia(df)
        
        # Example: Get properties for HD 360
        if 'HD 360' in star_disks:
            print(f"\nHD 360 Properties:")
            
            # Show properties at different frequencies
            for band_name, frequency in [('G', GAIA_G_EFFECTIVE_FREQUENCY),
                                        ('BP', GAIA_BP_EFFECTIVE_FREQUENCY),
                                        ('RP', GAIA_RP_EFFECTIVE_FREQUENCY)]:
                props = get_star_properties(star_disks, 'HD 360', frequency)
                print(f"  {band_name} band ({frequency:.2e} Hz):")
                print(f"    Flux density: {props['flux_density']:.2e} W/m²/Hz")
                print(f"    Surface brightness: {props['surface_brightness']:.2e} W/m²/Hz/sr")
            
            props = get_star_properties(star_disks, 'HD 360')  # Default to G band
            print(f"  Angular radius: {props['angular_radius_rad']:.2e} radians")
            print(f"  Angular diameter: {props['angular_diameter_mas']:.3f} mas")
            
            # Calculate visibilities at different frequencies
            hd_360_disk = star_disks['HD 360']
            baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W baseline
            
            print(f"\n  Visibility (100m E-W baseline):")
            for band_name, frequency in [('G', GAIA_G_EFFECTIVE_FREQUENCY),
                                        ('BP', GAIA_BP_EFFECTIVE_FREQUENCY),
                                        ('RP', GAIA_RP_EFFECTIVE_FREQUENCY)]:
                visibility = hd_360_disk.V(frequency, baseline)
                print(f"    {band_name} band: |V| = {abs(visibility):.4f}")
        
    except FileNotFoundError:
        print("Error: extended_data_table_2.csv not found")
        print("Please ensure the data file is in the current directory")
    except Exception as e:
        print(f"Error: {str(e)}")