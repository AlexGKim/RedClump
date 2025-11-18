"""
Match Gaia stars to closest PHOENIX model parameters.

This module provides functionality to find the nearest PHOENIX model grid point
for a given set of stellar parameters (Teff, log g, [Fe/H]).
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union


def match_phoenix_parameters(
    teff: float,
    logg: float,
    feh: float = 0.0,
    phoenix_grid: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[float, float, float]:
    """
    Match stellar parameters to the closest PHOENIX model grid point.
    
    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin
    logg : float
        Surface gravity (log g) in cgs units
    feh : float, optional
        Metallicity [Fe/H] in dex (default: 0.0 for solar)
    phoenix_grid : dict, optional
        Dictionary containing PHOENIX grid parameters with keys:
        'teff', 'logg', 'feh'. If None, uses default PHOENIX grid.
    
    Returns
    -------
    teff_model : float
        Matched effective temperature from PHOENIX grid
    logg_model : float
        Matched log g from PHOENIX grid
    feh_model : float
        Matched [Fe/H] from PHOENIX grid
    
    Examples
    --------
    >>> teff_m, logg_m, feh_m = match_phoenix_parameters(5777, 4.44, 0.0)
    >>> print(f"Matched: Teff={teff_m}K, logg={logg_m}, [Fe/H]={feh_m}")
    """
    if phoenix_grid is None:
        phoenix_grid = get_default_phoenix_grid()
    
    # Extract grid arrays
    teff_grid = phoenix_grid['teff']
    logg_grid = phoenix_grid['logg']
    feh_grid = phoenix_grid['feh']
    
    # Find closest match for each parameter
    teff_idx = np.argmin(np.abs(teff_grid - teff))
    logg_idx = np.argmin(np.abs(logg_grid - logg))
    feh_idx = np.argmin(np.abs(feh_grid - feh))
    
    teff_model = teff_grid[teff_idx]
    logg_model = logg_grid[logg_idx]
    feh_model = feh_grid[feh_idx]
    
    return teff_model, logg_model, feh_model


def match_phoenix_parameters_3d(
    teff: float,
    logg: float,
    feh: float = 0.0,
    phoenix_grid: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[float, float, float, float]:
    """
    Match stellar parameters to closest PHOENIX model using 3D distance metric.
    
    This function finds the closest model in the full 3D parameter space,
    accounting for the different scales of Teff, log g, and [Fe/H].
    
    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin
    logg : float
        Surface gravity (log g) in cgs units
    feh : float, optional
        Metallicity [Fe/H] in dex (default: 0.0)
    phoenix_grid : dict, optional
        Dictionary containing PHOENIX grid parameters. If None, uses default.
    
    Returns
    -------
    teff_model : float
        Matched effective temperature from PHOENIX grid
    logg_model : float
        Matched log g from PHOENIX grid
    feh_model : float
        Matched [Fe/H] from PHOENIX grid
    distance : float
        Normalized distance to the matched model
    
    Examples
    --------
    >>> teff_m, logg_m, feh_m, dist = match_phoenix_parameters_3d(5777, 4.44, 0.0)
    """
    if phoenix_grid is None:
        phoenix_grid = get_default_phoenix_grid()
    
    # Extract grid arrays
    teff_grid = phoenix_grid['teff']
    logg_grid = phoenix_grid['logg']
    feh_grid = phoenix_grid['feh']
    
    # Create meshgrid for all combinations
    teff_mesh, logg_mesh, feh_mesh = np.meshgrid(
        teff_grid, logg_grid, feh_grid, indexing='ij'
    )
    
    # Normalize parameters for distance calculation
    # Typical scales: Teff ~1000K, logg ~0.5 dex, [Fe/H] ~0.5 dex
    teff_norm = (teff_mesh - teff) / 1000.0
    logg_norm = (logg_mesh - logg) / 0.5
    feh_norm = (feh_mesh - feh) / 0.5
    
    # Calculate 3D distance
    distance = np.sqrt(teff_norm**2 + logg_norm**2 + feh_norm**2)
    
    # Find minimum distance
    min_idx = np.unravel_index(np.argmin(distance), distance.shape)
    
    teff_model = teff_grid[min_idx[0]]
    logg_model = logg_grid[min_idx[1]]
    feh_model = feh_grid[min_idx[2]]
    min_distance = distance[min_idx]
    
    return teff_model, logg_model, feh_model, min_distance


def get_default_phoenix_grid() -> Dict[str, np.ndarray]:
    """
    Get default PHOENIX model grid parameters.
    
    Returns a dictionary with typical PHOENIX grid coverage.
    Based on the PHOENIX BT-Settl grid.
    
    Returns
    -------
    grid : dict
        Dictionary with keys 'teff', 'logg', 'feh' containing numpy arrays
        of available grid points.
    """
    # PHOENIX BT-Settl typical grid
    # Teff: 2300K to 12000K (extended range available)
    teff_grid = np.concatenate([
        np.arange(2300, 7000, 100),   # 100K steps for cool stars
        np.arange(7000, 12100, 200)   # 200K steps for hot stars
    ])
    
    # log g: typically 0.0 to 6.0 in 0.5 dex steps
    logg_grid = np.arange(0.0, 6.5, 0.5)
    
    # [Fe/H]: typically -4.0 to +1.0 in 0.5 dex steps
    # Some grids have finer sampling around solar
    feh_grid = np.array([-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 
                         -0.3, 0.0, 0.3, 0.5, 1.0])
    
    return {
        'teff': teff_grid,
        'logg': logg_grid,
        'feh': feh_grid
    }


def match_gaia_star(
    gaia_params: Dict[str, float],
    phoenix_grid: Optional[Dict[str, np.ndarray]] = None,
    method: str = '3d'
) -> Dict[str, float]:
    """
    Match a Gaia star to PHOENIX model parameters.
    
    Parameters
    ----------
    gaia_params : dict
        Dictionary containing Gaia stellar parameters:
        - 'teff': effective temperature (K)
        - 'logg': surface gravity (log g)
        - 'feh': metallicity [Fe/H] (optional, default 0.0)
    phoenix_grid : dict, optional
        PHOENIX grid parameters. If None, uses default grid.
    method : str, optional
        Matching method: '3d' for 3D distance, 'independent' for
        independent parameter matching (default: '3d')
    
    Returns
    -------
    matched_params : dict
        Dictionary with matched PHOENIX parameters:
        - 'teff': matched temperature
        - 'logg': matched log g
        - 'feh': matched [Fe/H]
        - 'distance': distance metric (only for '3d' method)
    
    Examples
    --------
    >>> gaia = {'teff': 5777, 'logg': 4.44, 'feh': 0.0}
    >>> phoenix = match_gaia_star(gaia)
    >>> print(f"Matched Teff: {phoenix['teff']}K")
    """
    teff = gaia_params['teff']
    logg = gaia_params['logg']
    feh = gaia_params.get('feh', 0.0)
    
    if method == '3d':
        teff_m, logg_m, feh_m, dist = match_phoenix_parameters_3d(
            teff, logg, feh, phoenix_grid
        )
        return {
            'teff': teff_m,
            'logg': logg_m,
            'feh': feh_m,
            'distance': dist
        }
    elif method == 'independent':
        teff_m, logg_m, feh_m = match_phoenix_parameters(
            teff, logg, feh, phoenix_grid
        )
        return {
            'teff': teff_m,
            'logg': logg_m,
            'feh': feh_m
        }
    else:
        raise ValueError(f"Unknown method: {method}. Use '3d' or 'independent'")


def batch_match_gaia_stars(
    gaia_catalog: Union[Dict[str, np.ndarray], np.ndarray],
    phoenix_grid: Optional[Dict[str, np.ndarray]] = None,
    method: str = '3d'
) -> Dict[str, np.ndarray]:
    """
    Match multiple Gaia stars to PHOENIX models in batch.
    
    Parameters
    ----------
    gaia_catalog : dict or structured array
        Catalog of Gaia stars with fields 'teff', 'logg', and optionally 'feh'
    phoenix_grid : dict, optional
        PHOENIX grid parameters
    method : str, optional
        Matching method (default: '3d')
    
    Returns
    -------
    matched_catalog : dict
        Dictionary with matched parameters for all stars
    
    Examples
    --------
    >>> catalog = {
    ...     'teff': np.array([5777, 6000, 4500]),
    ...     'logg': np.array([4.44, 4.0, 2.5]),
    ...     'feh': np.array([0.0, -0.5, 0.2])
    ... }
    >>> matched = batch_match_gaia_stars(catalog)
    """
    # Handle structured array input
    if isinstance(gaia_catalog, np.ndarray):
        teff_arr = gaia_catalog['teff']
        logg_arr = gaia_catalog['logg']
        feh_arr = gaia_catalog['feh'] if 'feh' in gaia_catalog.dtype.names else np.zeros_like(teff_arr)
    else:
        teff_arr = np.asarray(gaia_catalog['teff'])
        logg_arr = np.asarray(gaia_catalog['logg'])
        feh_arr = np.asarray(gaia_catalog.get('feh', np.zeros_like(teff_arr)))
    
    n_stars = len(teff_arr)
    
    # Initialize output arrays
    teff_matched = np.zeros(n_stars)
    logg_matched = np.zeros(n_stars)
    feh_matched = np.zeros(n_stars)
    
    if method == '3d':
        distance = np.zeros(n_stars)
    
    # Match each star
    for i in range(n_stars):
        gaia_params = {
            'teff': teff_arr[i],
            'logg': logg_arr[i],
            'feh': feh_arr[i]
        }
        
        matched = match_gaia_star(gaia_params, phoenix_grid, method)
        
        teff_matched[i] = matched['teff']
        logg_matched[i] = matched['logg']
        feh_matched[i] = matched['feh']
        
        if method == '3d':
            distance[i] = matched['distance']
    
    result = {
        'teff': teff_matched,
        'logg': logg_matched,
        'feh': feh_matched
    }
    
    if method == '3d':
        result['distance'] = distance
    
    return result


if __name__ == '__main__':
    # Example usage with real Gaia data from Table1.dat
    print("PHOENIX Model Matching Examples")
    print("=" * 85)
    
    # Read data from Table1.dat
    import os
    data_file = 'data/Table1.dat'
    
    if os.path.exists(data_file):
        # Read the data file
        hd_names = []
        teff_list = []
        logg_list = []
        feh_list = []
        
        with open(data_file, 'r') as f:
            lines = f.readlines()
            # Skip header line
            for line in lines[1:]:
                # Split the line by tabs first
                fields = line.split('\t')
                if len(fields) >= 5:
                    # Extract HD name
                    hd_names.append(f"HD {fields[0].strip()}")
                    
                    # Extract values (before ±) from each field
                    # Field 2: log Teff
                    log_teff_str = fields[2].split('±')[0].strip()
                    log_teff = float(log_teff_str)
                    teff_list.append(10**log_teff)
                    
                    # Field 3: log g
                    logg_str = fields[3].split('±')[0].strip()
                    logg_list.append(float(logg_str))
                    
                    # Field 4: [Fe/H]
                    feh_str = fields[4].split('±')[0].strip()
                    # Handle minus sign (−) vs hyphen (-)
                    feh_str = feh_str.replace('−', '-')
                    feh_list.append(float(feh_str))
        
        # Use first 10 stars for example
        n_stars = min(10, len(hd_names))
        catalog = {
            'teff': np.array(teff_list[:n_stars]),
            'logg': np.array(logg_list[:n_stars]),
            'feh': np.array(feh_list[:n_stars])
        }
        star_names = hd_names[:n_stars]
        
        # Batch matching with detailed output
        print("\nMatching Gaia stars to closest PHOENIX model parameters:")
        print("-" * 85)
        
        matched_catalog = batch_match_gaia_stars(catalog, method='3d')
        
        # Print header
        print(f"\n{'Star':<12} {'Gaia Parameters':<35} {'PHOENIX Parameters':<35} {'Dist':<6}")
        print(f"{'Name':<12} {'Teff(K)  logg  [Fe/H]':<35} {'Teff(K)  logg  [Fe/H]':<35} {'':<6}")
        print("-" * 85)
        
        # Print each star's parameters
        for i in range(len(catalog['teff'])):
            gaia_str = f"{catalog['teff'][i]:6.0f}   {catalog['logg'][i]:4.2f}  {catalog['feh'][i]:+5.2f}"
            phoenix_str = f"{matched_catalog['teff'][i]:6.0f}   {matched_catalog['logg'][i]:4.2f}  {matched_catalog['feh'][i]:+5.2f}"
            dist_str = f"{matched_catalog['distance'][i]:5.3f}"
            print(f"{star_names[i]:<12} {gaia_str:<35} {phoenix_str:<35} {dist_str:<6}")
        
        print("-" * 85)
        print(f"\nSummary: Matched {len(catalog['teff'])} stars from Table1.dat")
        print(f"Mean distance: {np.mean(matched_catalog['distance']):.3f}")
        print(f"Max distance:  {np.max(matched_catalog['distance']):.3f}")
    else:
        print(f"\nWarning: {data_file} not found. Using example data instead.")
        # Fallback to example data
        catalog = {
            'teff': np.array([5777, 6000, 4500]),
            'logg': np.array([4.44, 4.0, 2.5]),
            'feh': np.array([0.0, -0.5, 0.2])
        }
        matched_catalog = batch_match_gaia_stars(catalog, method='3d')
        print(f"Matched {len(catalog['teff'])} example stars")