#!/usr/bin/env python3
"""
Plot HD 360 H-band intensity profiles and visibility squared for SATLAS and PHOENIX models.

Creates a single PDF with two subplots:
- Left: Intensity profiles for both models
- Right: Visibility squared for both models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/akim/Projects/g2')

from gaia_satlas import create_radial_grid_from_satlas, create_radial_from_gaia, SATLAS_BANDS
from gaia_zeropoint import SPEED_OF_LIGHT

# Constants
MAS_TO_RAD = np.pi / (180 * 3600 * 1000)

def main():
    """Main function to create the comparison plots."""
    print("HD 360 H-band Comparison: SATLAS vs PHOENIX")
    print("=" * 60)
    
    # Load Gaia data
    print("Loading Gaia data...")
    df = pd.read_csv('extended_data_table_2.csv')
    
    # Load stellar parameters from Table1.dat
    print("Loading stellar parameters from Table1.dat...")
    table1_data = []
    with open('data/Table1.dat', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            fields = line.split('\t')
            if len(fields) >= 5:
                hd_name = f"HD {fields[0].strip()}"
                log_teff_str = fields[2].split('±')[0].strip()
                log_teff = float(log_teff_str)
                teff = 10**log_teff
                logg_str = fields[3].split('±')[0].strip()
                logg = float(logg_str)
                feh_str = fields[4].split('±')[0].strip().replace('−', '-')
                feh = float(feh_str)
                table1_data.append({
                    'Star': hd_name,
                    'teff_gspphot': teff,
                    'logg_gspphot': logg,
                    'mh_gspphot': feh
                })
    
    df_table1 = pd.DataFrame(table1_data)
    df = df.merge(df_table1, on='Star', how='left')
    
    # Select HD 360
    star_name = 'HD 360'
    if star_name not in df['Star'].values:
        print(f"Error: {star_name} not found in the data")
        return
    
    star_row = df[df['Star'] == star_name].iloc[0]
    print(f"✓ Selected star: {star_name}")
    
    # Path to SATLAS data file
    satlas_file = 'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt'
    satlas_log_dir = 'data/output_ld-satlas_1762763642809'
    
    # Create RadialGrid from SATLAS data
    print("Creating RadialGrid from SATLAS data...")
    radial_grid_satlas = create_radial_grid_from_satlas(satlas_file, star_row)
    
    # Create RadialGrid from PHOENIX models
    print("Creating RadialGrid from PHOENIX models...")
    radial_grids_phoenix = create_radial_from_gaia(
        df[df['Star'] == star_name]
    )
    radial_grid_phoenix = radial_grids_phoenix[star_name]
    
    # Get H band information
    h_band_wavelength = SATLAS_BANDS['H']['wavelength']  # Angstroms
    h_band_frequency = SPEED_OF_LIGHT / (h_band_wavelength * 1e-10)  # Hz
    
    # Find H band index (should be index 4 in the SATLAS bands order: B, V, R, I, H, K)
    h_band_idx = 4
    
    print(f"\nH band wavelength: {h_band_wavelength} Å")
    print(f"H band frequency: {h_band_frequency:.3e} Hz")
    
    # Extract intensity profiles for H band
    # SATLAS
    r_mas_satlas = radial_grid_satlas.p_rays / MAS_TO_RAD
    intensity_satlas = radial_grid_satlas.I_nu_p[h_band_idx, :]
    
    # PHOENIX
    r_mas_phoenix = radial_grid_phoenix.p_rays / MAS_TO_RAD
    intensity_phoenix = radial_grid_phoenix.I_nu_p[h_band_idx, :]
    
    # Debug: Check for NaN values
    print(f"\nDebug - SATLAS intensity:")
    print(f"  Min: {np.nanmin(intensity_satlas):.3f}, Max: {np.nanmax(intensity_satlas):.3f}")
    print(f"  NaN count: {np.isnan(intensity_satlas).sum()}")
    
    print(f"\nDebug - PHOENIX intensity:")
    print(f"  Min: {np.nanmin(intensity_phoenix):.3f}, Max: {np.nanmax(intensity_phoenix):.3f}")
    print(f"  NaN count: {np.isnan(intensity_phoenix).sum()}")
    print(f"  Sample values: {intensity_phoenix[:5]}")
    
    # Calculate visibility squared for a range of baselines
    baseline_lengths = np.linspace(10, 250, 100)  # meters
    vis_squared_satlas = []
    vis_squared_phoenix = []
    
    print("\nCalculating visibility squared...")
    for baseline_length in baseline_lengths:
        baseline = np.array([baseline_length, 0.0, 0.0])  # E-W baseline
        
        # SATLAS
        vis_satlas = radial_grid_satlas.V(h_band_frequency, baseline)
        vis_squared_satlas.append(np.abs(vis_satlas)**2)
        
        # PHOENIX
        vis_phoenix = radial_grid_phoenix.V(h_band_frequency, baseline)
        vis_squared_phoenix.append(np.abs(vis_phoenix)**2)
    
    vis_squared_satlas = np.array(vis_squared_satlas)
    vis_squared_phoenix = np.array(vis_squared_phoenix)
    
    # Create the plot
    print("\nCreating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Intensity profiles
    ax1.plot(r_mas_satlas, intensity_satlas, 'o-', color='blue', 
             markersize=4, linewidth=2, label='SATLAS', alpha=0.7)
    ax1.plot(r_mas_phoenix, intensity_phoenix, 's-', color='red', 
             markersize=4, linewidth=2, label='PHOENIX', alpha=0.7)
    
    ax1.set_xlabel('Radial Distance (mas)', fontsize=12)
    ax1.set_ylabel('Normalized Intensity I(r)/I(0)', fontsize=12)
    ax1.set_title(f'{star_name} - H Band Intensity Profile\n({h_band_wavelength:.0f} Å)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, max(r_mas_satlas.max(), r_mas_phoenix.max()) * 1.05)
    ax1.set_ylim(0, 1.1)
    
    # Right plot: Visibility squared
    ax2.plot(baseline_lengths, vis_squared_satlas, '-', color='blue', 
             linewidth=2.5, label='SATLAS', alpha=0.7)
    ax2.plot(baseline_lengths, vis_squared_phoenix, '-', color='red', 
             linewidth=2.5, label='PHOENIX', alpha=0.7)
    
    ax2.set_xlabel('Baseline Length (m)', fontsize=12)
    ax2.set_ylabel('Visibility Squared |V|²', fontsize=12)
    ax2.set_title(f'{star_name} - H Band Visibility Squared\n({h_band_wavelength:.0f} Å)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xlim(baseline_lengths[0], baseline_lengths[-1])
    ax2.set_yscale('log')
    
    # Overall title
    fig.suptitle(f'HD 360 H-band Comparison: SATLAS vs PHOENIX Models', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save to PDF
    output_filename = 'hd360_h_band_comparison.pdf'
    fig.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"✓ Plot saved to {output_filename}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("-" * 60)
    print(f"SATLAS:")
    print(f"  Radial points: {len(r_mas_satlas)}")
    print(f"  Max radius: {r_mas_satlas.max():.3f} mas")
    print(f"  Intensity range: {intensity_satlas.min():.3f} - {intensity_satlas.max():.3f}")
    print(f"  |V|² at 100m: {vis_squared_satlas[np.argmin(np.abs(baseline_lengths - 100))]:.6f}")
    
    print(f"\nPHOENIX:")
    print(f"  Radial points: {len(r_mas_phoenix)}")
    print(f"  Max radius: {r_mas_phoenix.max():.3f} mas")
    print(f"  Intensity range: {intensity_phoenix.min():.3f} - {intensity_phoenix.max():.3f}")
    print(f"  |V|² at 100m: {vis_squared_phoenix[np.argmin(np.abs(baseline_lengths - 100))]:.6f}")
    
    print("=" * 60)
    print(f"Plot generation completed successfully!")
    print(f"Output: {output_filename}")
    print("=" * 60)
    
    plt.close(fig)

if __name__ == "__main__":
    main()