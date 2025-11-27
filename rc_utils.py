import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional

import pandas as pd
import astropy.units as u
from astropy.io       import fits

from g2.core import Observation

def get_filter_properties_df(vega_spectrum_path='data/alpha_lyr_mod_004.fits'):
    """
    Create a DataFrame containing zeropoints and effective wavelengths for 
    GAIA (G, BP, RP), Johnson (B, V), and 2MASS (H, K) filters.
    
    Parameters
    ----------
    vega_spectrum_path : str
        Path to the Vega spectrum FITS file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Filter, λ_eff (nm), λ_eff (Å), ν_eff (Hz), 
        F_ν (W/m²/Hz), F_λ (W/m²/nm)
    """
    # Load Vega spectrum
    with fits.open(vega_spectrum_path) as hdul:
        data = hdul[1].data
        lam = data['WAVELENGTH'] * u.AA
        flux = data['FLUX'] * u.erg/(u.s * u.cm**2 * u.AA)
    
    # Convert to F_nu
    fnu_array = flux.to(u.W/(u.m**2 * u.Hz), 
                        equivalencies=u.spectral_density(lam))
    
    # Physical constants
    c = 2.99792458e8  # m/s
    
    # Define filter effective wavelengths (in nm)
    filters = {
        'G': 639.07,    # from the code
        'BP': 518.26,   # from the code
        'RP': 782.51,   # from the code
        'B': 445.0,  # Standard Johnson B
        'V': 551.0,  # Standard Johnson V
        'H': 1662.0,   # 2MASS H band
        'K': 2159.0,    # 2MASS K band (Ks)
        'I': 806.0,    # Standard Johnson I
        'R': 658.0     # Standard Johnson R
    }
    
    filter_data = []
    
    for filter_name, lambda_eff_nm in filters.items():
        lambda_eff_m = lambda_eff_nm * 1e-9  # Convert to meters
        lambda_eff_angstrom = lambda_eff_nm * 10  # Convert to Angstroms
        nu_eff = c / lambda_eff_m  # Effective frequency in Hz
        
        # Interpolate Vega flux at this wavelength
        f_nu_vega = np.interp(lambda_eff_angstrom, lam.value, fnu_array.value)
        
        # Convert to F_lambda (W/m²/nm)
        # F_lambda = F_nu * (c / lambda^2)
        f_lambda_vega = f_nu_vega * (c / (lambda_eff_m**2)) * 1e-9  # W/m²/nm
        
        filter_data.append({
            'Filter': filter_name,
            'λ_eff_nm': lambda_eff_nm,
            'λ_eff_angstrom': lambda_eff_angstrom,
            'ν_eff_Hz': nu_eff,
            'F_ν_vega_W_m2_Hz': f_nu_vega,
            'F_λ_vega_W_m2_nm': f_lambda_vega,
            'Vega_mag': 0.0  # Vega is defined as magnitude 0 in all bands
        })
    
    df = pd.DataFrame(filter_data)
    
    # Add AB magnitude zeropoints for reference
    # AB magnitude zeropoint: F_ν = 3631 Jy = 3.631e-23 W/m²/Hz
    df['F_ν_AB_zeropoint_W_m2_Hz'] = 3.631e-23
    
    return df

def read_table_b1(filepath):
    """
    Reads the TableB1.txt file and returns it as a pandas DataFrame.
    
    Parameters:
    -----------
    filepath : str
        Path to the TableB1.txt file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the astronomical data
    """
    # Read the file and skip the header to parse manually
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the header line and data lines
    header_idx = None
    for i, line in enumerate(lines):
        if 'Star' in line and 'K(mag)' in line:
            header_idx = i
            break
    
    if header_idx is None:
        header_idx = 0  # Assume first line is header
    
    # Parse data rows
    data = []
    for line in lines[header_idx + 1:]:
        if line.strip():  # Skip empty lines
            parts = line.split()
            if len(parts) >= 11:  # Valid data row
                # Star name is "HD" + number (first two parts)
                star = parts[0] + ' ' + parts[1]
                k_mag = parts[2]
                h_mag = parts[3]
                parallax = ' '.join(parts[4:7])  # e.g., "8.97 ± 0.13"
                ebv = parts[7]
                date = parts[8]
                baselines = parts[9]
                sp_channels = parts[10]
                bin_flag = parts[-1]
                calibrator = ' '.join(parts[11:-1])
                
                data.append([star, k_mag, h_mag, parallax, ebv, date, 
                           baselines, sp_channels, calibrator, bin_flag])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'Star', 'K(mag)', 'H(mag)', 'π(mas)', 'E(B–V)', 'Date',
        'Baselines', 'Sp. channels', 'Calibrator HD', 'Bin.'
    ])
    
    # Convert numeric columns
    df['K(mag)'] = pd.to_numeric(df['K(mag)'], errors='coerce')
    df['H(mag)'] = pd.to_numeric(df['H(mag)'], errors='coerce')
    df['E(B–V)'] = pd.to_numeric(df['E(B–V)'], errors='coerce')
    df['Sp. channels'] = pd.to_numeric(df['Sp. channels'], errors='coerce')
    
    # Replace em-dash with NaN in Bin. column
    df['Bin.'] = df['Bin.'].replace('–', None)
    
    return df



def get_table1_df() -> pd.DataFrame:
    """
    Load the Table1.txt file into a pandas DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing all columns from Table1.txt with values and uncertainties.
    """
    print("Loading stellar parameters from Table1.txt...")
    table1_data = []
    
    with open('data/Table1.dat', 'r') as f:
        lines = f.readlines()
        
        # Find header line
        header_idx = 0
        for i, line in enumerate(lines):
            if 'HD' in line and 'θLD' in line:
                header_idx = i
                break
        
        # Parse data rows
        for line in lines[header_idx + 1:]:
            if line.strip():  # Skip empty lines
                fields = line.split('\t')
                if len(fields) >= 8:
                    hd_num = fields[0].strip()
                    
                    # Helper function to parse value ± uncertainty
                    def parse_value_uncertainty(s):
                        s = s.strip().replace('−', '-')
                        if '±' in s:
                            parts = s.split('±')
                            val = float(parts[0].strip())
                            unc = float(parts[1].strip()) if len(parts) > 1 else None
                            return val, unc
                        else:
                            return float(s) if s else None, None
                    
                    theta_ld, theta_ld_unc = parse_value_uncertainty(fields[1])
                    log_teff, log_teff_unc = parse_value_uncertainty(fields[2])
                    logg, logg_unc = parse_value_uncertainty(fields[3])
                    feh, feh_unc = parse_value_uncertainty(fields[4])
                    ebv, ebv_unc = parse_value_uncertainty(fields[5])
                    log_r, log_r_unc = parse_value_uncertainty(fields[6])
                    log_l, log_l_unc = parse_value_uncertainty(fields[7])
                    
                    # Calculate Teff from log Teff
                    teff = 10**log_teff if log_teff is not None else None
                    
                    table1_data.append({
                        'Star': f"HD {hd_num}",
                        'theta_ld': theta_ld,
                        'theta_ld_unc': theta_ld_unc,
                        'log_teff': log_teff,
                        'log_teff_unc': log_teff_unc,
                        'teff': teff,
                        'logg': logg,
                        'logg_unc': logg_unc,
                        'feh': feh,
                        'feh_unc': feh_unc,
                        'ebv': ebv,
                        'ebv_unc': ebv_unc,
                        'log_r': log_r,
                        'log_r_unc': log_r_unc,
                        'log_l': log_l,
                        'log_l_unc': log_l_unc
                    })
    
    return pd.DataFrame(table1_data)

def master_df():
    # Load your data

    gaia_df = pd.read_csv('data/extended_data_table_2.csv')
    table1_df = get_table1_df()
    chara_df = read_table_b1('data/TableB1.txt')

    # # Get the star names from each dataframe
    # gaia_stars = set(gaia_df['Star'])
    # table1_stars = set(table1_df['Star'])
    # chara_stars = set(chara_df['Star'])
    # # Create a summary dataframe
    # all_stars = gaia_stars | table1_stars | chara_stars

    # summary = pd.DataFrame({
    #     'Star': sorted(all_stars),
    #     'in_gaia': [s in gaia_stars for s in sorted(all_stars)],
    #     'in_table1': [s in table1_stars for s in sorted(all_stars)],
    #     'in_chara': [s in chara_stars for s in sorted(all_stars)]
    # })

    # # Add a column showing how many dataframes contain each star
    # summary['count'] = summary[['in_gaia', 'in_table1', 'in_chara']].sum(axis=1)

    # # Filter to show stars not in all three
    # stars_not_in_all = summary[summary['count'] < 3]
    # print(stars_not_in_all)

    # # Summary statistics
    # print(f"\nStars in all 3 dataframes: {(summary['count'] == 3).sum()}")
    # print(f"Stars in exactly 2 dataframes: {(summary['count'] == 2).sum()}")
    # print(f"Stars in exactly 1 dataframe: {(summary['count'] == 1).sum()}")

    combined_df = gaia_df.merge(table1_df, on='Star', how='inner') \
                     .merge(chara_df, on='Star', how='inner')
    combined_df.rename(columns={'phot_g_mean_mag': 'G', 'phot_bp_mean_mag': 'BP', 'phot_rp_mean_mag': 'RP', 'H(mag)': 'H'}, inplace=True)

    return combined_df

  

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_stellar_seds(df, filter_df, atlas_filters=['V', 'R', 'I', 'H', 'K'], 
                      all_filters=['V', 'G', 'BP', 'RP', 'H', 'K'], 
                      num_stars=None, save_plots=False):
    """
    Plot specific flux vs wavelength for stars.
    
    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with stellar data
    filter_df : pd.DataFrame
        Filter properties dataframe
    atlas_filters : list
        Filters from original catalog (plotted as triangles)
    all_filters : list
        All filters to plot (plotted as circles/squares)
    num_stars : int, optional
        Number of stars to plot (None = all stars)
    save_plots : bool
        Whether to save plots to files
        
    Returns
    -------
    dict
        Dictionary with star names as keys and atlas filter flux arrays as values
    """
    # Create filter lookup dictionary
    filter_info = {}
    for _, row in filter_df.iterrows():
        filter_info[row['Filter']] = {
            'wavelength': row['λ_eff_nm'],
            'F_nu_vega': row['F_ν_vega_W_m2_Hz']
        }
    
    # Dictionary to store atlas filter fluxes for each star
    atlas_flux_results = {}
    
    # Limit number of stars if specified
    stars_to_plot = df.head(num_stars) if num_stars else df
    
    # Create subplots
    n_stars = len(stars_to_plot)
    n_cols = min(3, n_stars)
    n_rows = int(np.ceil(n_stars / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_stars == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (star_idx, star_row) in enumerate(stars_to_plot.iterrows()):
        star_name = star_row['Star']
        ax = axes[idx]
        
        # Step 1: Collect ALL available magnitudes for interpolation basis
        available_data = []
        for filt in df.columns:
            if filt in filter_info and pd.notna(star_row[filt]):
                mag = star_row[filt]
                wavelength = filter_info[filt]['wavelength']
                f_nu_vega = filter_info[filt]['F_nu_vega']
                f_nu = f_nu_vega * 10**(-mag / 2.5)
                available_data.append({
                    'filter': filt,
                    'wavelength': wavelength,
                    'f_nu': f_nu,
                    'mag': mag
                })
        
        if len(available_data) == 0:
            ax.text(0.5, 0.5, f'{star_name}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            atlas_flux_results[star_name] = np.array([np.nan] * len(atlas_filters))
            continue
        
        # Sort by wavelength for interpolation
        available_data = sorted(available_data, key=lambda x: x['wavelength'])
        wave_available = np.array([d['wavelength'] for d in available_data])
        flux_available = np.array([d['f_nu'] for d in available_data])
        
        # Step 2: Create interpolation function if we have multiple points
        interp_func = None
        if len(wave_available) > 1:
            interp_func = interp1d(np.log10(wave_available), np.log10(flux_available), 
                                  kind='linear', fill_value='extrapolate')
        
        # Step 3: Process ATLAS_FILTERS and store results
        atlas_measured = {'wave': [], 'flux': [], 'filters': []}
        atlas_interpolated = {'wave': [], 'flux': [], 'filters': []}
        atlas_flux_vector = []
        
        for filt in atlas_filters:
            if filt not in filter_info:
                print(f"Warning: Atlas filter {filt} not in filter_info, skipping")
                atlas_flux_vector.append(np.nan)
                continue
                
            wavelength = filter_info[filt]['wavelength']
            
            # Check if we have a direct measurement
            if filt in star_row.index and pd.notna(star_row[filt]):
                mag = star_row[filt]
                f_nu_vega = filter_info[filt]['F_nu_vega']
                f_nu = f_nu_vega * 10**(-mag / 2.5)
                atlas_measured['wave'].append(wavelength)
                atlas_measured['flux'].append(f_nu)
                atlas_measured['filters'].append(filt)
                atlas_flux_vector.append(f_nu)
            elif interp_func is not None:
                f_nu = 10**interp_func(np.log10(wavelength))
                atlas_interpolated['wave'].append(wavelength)
                atlas_interpolated['flux'].append(f_nu)
                atlas_interpolated['filters'].append(filt)
                atlas_flux_vector.append(f_nu)
            else:
                atlas_flux_vector.append(np.nan)
        
        # Store atlas flux vector
        atlas_flux_results[star_name] = np.array(atlas_flux_vector)
        
        # Step 4: Process ALL_FILTERS
        all_measured = {'wave': [], 'flux': [], 'filters': []}
        all_interpolated = {'wave': [], 'flux': [], 'filters': []}
        
        for filt in all_filters:
            if filt not in filter_info:
                print(f"Warning: Filter {filt} not in filter_info, skipping")
                continue
            
            # Skip if already in atlas_filters (to avoid double plotting)
            if filt in atlas_filters:
                continue
                
            wavelength = filter_info[filt]['wavelength']
            
            if filt in star_row.index and pd.notna(star_row[filt]):
                mag = star_row[filt]
                f_nu_vega = filter_info[filt]['F_nu_vega']
                f_nu = f_nu_vega * 10**(-mag / 2.5)
                all_measured['wave'].append(wavelength)
                all_measured['flux'].append(f_nu)
                all_measured['filters'].append(filt)
            elif interp_func is not None:
                f_nu = 10**interp_func(np.log10(wavelength))
                all_interpolated['wave'].append(wavelength)
                all_interpolated['flux'].append(f_nu)
                all_interpolated['filters'].append(filt)
        
        # Plot ATLAS filters (measured)
        if len(atlas_measured['wave']) > 0:
            ax.scatter(atlas_measured['wave'], atlas_measured['flux'], 
                      s=150, c='green', zorder=4, label='Atlas (measured)', 
                      edgecolors='black', linewidth=2, marker='^')
            for i, filt in enumerate(atlas_measured['filters']):
                ax.annotate(filt, (atlas_measured['wave'][i], atlas_measured['flux'][i]), 
                           xytext=(5, 8), textcoords='offset points', fontsize=10, 
                           fontweight='bold', color='green')
        
        # Plot ATLAS filters (interpolated)
        if len(atlas_interpolated['wave']) > 0:
            ax.scatter(atlas_interpolated['wave'], atlas_interpolated['flux'], 
                      s=130, c='lightgreen', zorder=3, label='Atlas (interp.)', 
                      edgecolors='darkgreen', linewidth=1.5, marker='^', alpha=0.7)
            for i, filt in enumerate(atlas_interpolated['filters']):
                ax.annotate(filt, (atlas_interpolated['wave'][i], 
                                  atlas_interpolated['flux'][i]), 
                           xytext=(5, -12), textcoords='offset points', 
                           fontsize=9, style='italic', color='darkgreen')
        
        # Plot ALL_FILTERS (measured)
        if len(all_measured['wave']) > 0:
            ax.scatter(all_measured['wave'], all_measured['flux'], 
                      s=120, c='red', zorder=3, label='Other (measured)', 
                      edgecolors='black', linewidth=1.5, marker='o')
            for i, filt in enumerate(all_measured['filters']):
                ax.annotate(filt, (all_measured['wave'][i], all_measured['flux'][i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9, 
                           fontweight='bold', color='red')
        
        # Plot ALL_FILTERS (interpolated)
        if len(all_interpolated['wave']) > 0:
            ax.scatter(all_interpolated['wave'], all_interpolated['flux'], 
                      s=100, c='blue', zorder=2, label='Other (interp.)', 
                      edgecolors='black', linewidth=1.5, marker='s', alpha=0.7)
            for i, filt in enumerate(all_interpolated['filters']):
                ax.annotate(filt, (all_interpolated['wave'][i], 
                                  all_interpolated['flux'][i]), 
                           xytext=(5, -10), textcoords='offset points', 
                           fontsize=9, style='italic', color='blue')
        
        # Plot smooth interpolation curve
        if interp_func is not None:
            wave_smooth = np.logspace(np.log10(wave_available.min()), 
                                     np.log10(wave_available.max()), 200)
            flux_smooth = 10**interp_func(np.log10(wave_smooth))
            ax.plot(wave_smooth, flux_smooth, 'gray', alpha=0.3, 
                   linewidth=2, linestyle='--', label='SED model')
        
        # Formatting
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_ylabel(r'$F_\nu$ (W m$^{-2}$ Hz$^{-1}$)', fontsize=11)
        ax.set_title(f'{star_name}', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=8)
        
    # Hide unused subplots
    for idx in range(n_stars, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('stellar_seds.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'stellar_seds.png'")
    
    plt.show()
    
    # Print atlas flux results
    print("\nAtlas Filter Fluxes:")
    print(f"Filters: {atlas_filters}")
    for star_name, fluxes in atlas_flux_results.items():
        print(f"{star_name}: {fluxes}")
    
    return atlas_flux_results

def get_stellar_fluxes(df, filter_df, filters=['V', 'R', 'I', 'H', 'K'], verbose=True):
    """
    Calculate specific fluxes for all stars in specified filters.
    Uses measured magnitudes when available, interpolates otherwise.
    
    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with stellar data
    filter_df : pd.DataFrame
        Filter properties dataframe
    filters : list
        List of filters to calculate fluxes for
    verbose : bool
        Print progress information
        
    Returns
    -------
    flux_df : pd.DataFrame
        DataFrame with columns: Star, filter1_flux, filter2_flux, ...
        and additional columns for flux_type (measured/interpolated)
    flux_dict : dict
        Dictionary with star names as keys and flux arrays as values
    """
    if verbose:
        print(f"Processing {len(df)} stars for {len(filters)} filters...")
    
    # Create filter lookup dictionary
    filter_info = {}
    for _, row in filter_df.iterrows():
        filter_info[row['Filter']] = {
            'wavelength': row['λ_eff_nm'],
            'F_nu_vega': row['F_ν_vega_W_m2_Hz']
        }
    
    results = []
    flux_dict = {}
    
    # Iterate through all stars
    for idx in range(len(df)):
        star_row = df.iloc[idx]
        star_name = star_row['Star']
        
        if verbose and (idx + 1) % 10 == 0:
            print(f"  Processing star {idx + 1}/{len(df)}: {star_name}")
        
        # Collect ALL available magnitudes for interpolation basis
        available_data = []
        for col in df.columns:
            if col in filter_info and pd.notna(star_row[col]):
                mag = star_row[col]
                wavelength = filter_info[col]['wavelength']
                f_nu_vega = filter_info[col]['F_nu_vega']
                f_nu = f_nu_vega * 10**(-mag / 2.5)
                available_data.append({
                    'filter': col,
                    'wavelength': wavelength,
                    'f_nu': f_nu,
                    'mag': mag
                })
        
        # Initialize result row
        row_data = {'Star': star_name}
        flux_vector = []
        flux_types = []
        
        if len(available_data) == 0:
            # No data available
            for filt in filters:
                row_data[f'{filt}_flux'] = np.nan
                row_data[f'{filt}_type'] = 'no_data'
                flux_vector.append(np.nan)
                flux_types.append('no_data')
        else:
            # Sort by wavelength for interpolation
            available_data = sorted(available_data, key=lambda x: x['wavelength'])
            wave_available = np.array([d['wavelength'] for d in available_data])
            flux_available = np.array([d['f_nu'] for d in available_data])
            
            # Create interpolation function if we have multiple points
            interp_func = None
            if len(wave_available) > 1:
                interp_func = interp1d(np.log10(wave_available), np.log10(flux_available), 
                                      kind='linear', fill_value='extrapolate')
            
            # Calculate flux for each requested filter
            for filt in filters:
                if filt not in filter_info:
                    if verbose and idx == 0:  # Only warn once
                        print(f"Warning: Filter {filt} not in filter_info")
                    row_data[f'{filt}_flux'] = np.nan
                    row_data[f'{filt}_type'] = 'unknown_filter'
                    flux_vector.append(np.nan)
                    flux_types.append('unknown_filter')
                    continue
                    
                wavelength = filter_info[filt]['wavelength']
                
                # Check if we have a direct measurement
                if filt in star_row.index and pd.notna(star_row[filt]):
                    mag = star_row[filt]
                    f_nu_vega = filter_info[filt]['F_nu_vega']
                    f_nu = f_nu_vega * 10**(-mag / 2.5)
                    row_data[f'{filt}_flux'] = f_nu
                    row_data[f'{filt}_type'] = 'measured'
                    flux_vector.append(f_nu)
                    flux_types.append('measured')
                elif interp_func is not None:
                    # Interpolate
                    f_nu = 10**interp_func(np.log10(wavelength))
                    row_data[f'{filt}_flux'] = f_nu
                    row_data[f'{filt}_type'] = 'interpolated'
                    flux_vector.append(f_nu)
                    flux_types.append('interpolated')
                else:
                    # Only one data point, can't interpolate
                    row_data[f'{filt}_flux'] = np.nan
                    row_data[f'{filt}_type'] = 'insufficient_data'
                    flux_vector.append(np.nan)
                    flux_types.append('insufficient_data')
        
        results.append(row_data)
        flux_dict[star_name] = {
            'fluxes': np.array(flux_vector),
            'types': flux_types,
            'filters': filters
        }
    
    flux_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\nCompleted! Processed {len(flux_df)} stars.")
        print(f"DataFrame shape: {flux_df.shape}")
    
    return flux_df, flux_dict

def master_df_with_Fnu():
    filter_df = get_filter_properties_df()
    print(filter_df.columns.tolist())

    df = master_df()
    # print(df.columns.tolist())
    atlas_filters = ['V', 'R', 'I', 'H', 'K']
    all_filters = ['V', 'G', 'BP', 'RP', 'H', 'K']  # Added BP for completeness
    flux_df, flux_dict = get_stellar_fluxes(df, filter_df, filters=atlas_filters)
    print(flux_df.columns.tolist())
    ans = df.merge(flux_df, on='Star', how='inner')
    return ans

def calculate_inverse_noise(
    df: pd.DataFrame,
    filter_df: pd.DataFrame,
    observation: Observation,
    filters: list[str] = ['V', 'R', 'I', 'H', 'K'],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate inverse noise for specific fluxes in each filter.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing stellar fluxes (with columns like 'V_flux', 'R_flux', etc.)
    filter_df : pd.DataFrame
        Filter properties dataframe with effective frequencies
    observation : Observation
        Observation parameters (integration time, telescope area, etc.)
    filters : list[str]
        List of filters to calculate inverse noise for
    verbose : bool
        Print progress information
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with additional inverse_noise columns for each filter
        
    Notes
    -----
    The inverse noise is calculated as:
        inverse_noise = (photon_rate_per_nu) * sqrt(t_int / jitter) * (128π)^(-1/4)
    
    where photon_rate_per_nu = (throughput * area * flux) / (h * nu_0)
    """
    if verbose:
        print(f"Calculating inverse noise for {len(df)} stars in {len(filters)} filters...")
    
    # Physical constants
    h = 6.62607015e-34  # Planck constant (J⋅s)
    
    # Create filter lookup for frequencies
    filter_freq = {}
    for _, row in filter_df.iterrows():
        filter_freq[row['Filter']] = row['ν_eff_Hz']
    
    # Common factor that doesn't depend on flux or frequency
    common_factor = np.sqrt(observation.integration_time / observation.detector_jitter) * (128 * np.pi)**(-0.25)
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate inverse noise for each filter
    for filt in filters:
        flux_col = f'{filt}_flux'
        inv_noise_col = f'{filt}_inv_noise'
        
        if flux_col not in df.columns:
            if verbose:
                print(f"Warning: {flux_col} not found in dataframe, skipping {filt}")
            continue
        
        if filt not in filter_freq:
            if verbose:
                print(f"Warning: Filter {filt} not in filter_df, skipping")
            continue
        
        # Get central frequency for this filter
        nu_0 = filter_freq[filt]
        
        # Calculate photon energy
        photon_energy = h * nu_0
        
        # Get flux values (W/m²/Hz)
        flux_values = df[flux_col].values
        
        # Calculate photon rate per frequency
        # flux is in W/m²/Hz = J/(s⋅m²⋅Hz)
        # photon_rate_per_nu has units of photons/(s⋅Hz)
        photon_rate_per_nu = (
            observation.throughput * 
            observation.telescope_area * 
            flux_values / 
            photon_energy
        )
        
        # Calculate inverse noise
        inverse_noise = photon_rate_per_nu * common_factor
        
        # Add to dataframe
        result_df[inv_noise_col] = inverse_noise
        
        if verbose:
            # Calculate statistics (excluding NaN)
            valid_inv_noise = inverse_noise[~np.isnan(inverse_noise)]
            if len(valid_inv_noise) > 0:
                print(f"  {filt}: mean={np.mean(valid_inv_noise):.2e}, "
                      f"median={np.median(valid_inv_noise):.2e}, "
                      f"min={np.min(valid_inv_noise):.2e}, "
                      f"max={np.max(valid_inv_noise):.2e}")
    
    if verbose:
        print(f"\nCompleted! Added {len(filters)} inverse noise columns.")
    
    return result_df
def master_df_with_inverse_noise():
    filter_df = get_filter_properties_df()

    df = master_df_with_Fnu()
    
    # Observation parameters (same as fisher_matrix_table.py)
    observation = Observation(
        integration_time=3600,  # 3600 seconds
        telescope_area=np.pi * (1.0)**2,  # π*(5m)^2
        throughput=0.3,  # 0.3
        detector_jitter=42.4e-12/2.555 #130e-12/2.555  # 130 ps FWHM to stddev
    )
    
    new_df = calculate_inverse_noise(df, filter_df, observation)
    return new_df