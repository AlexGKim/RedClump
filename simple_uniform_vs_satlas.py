exit#!/usr/bin/env python3
"""
Simple Uniform Disk vs SATLAS Comparison Script

This script compares visibility calculations between UniformDisk and SATLAS RadialGrid2
for HD 360, focusing on |V|^2 vs baseline without Fisher matrix calculations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('/Users/akim/Projects/g2')

from gaia_uniform_disk import create_uniform_disk_from_gaia
from gaia_satlas import create_radial_grid_from_satlas, SATLAS_BANDS
from gaia_zeropoint import (
    GAIA_G_EFFECTIVE_WAVELENGTH,
    GAIA_BP_EFFECTIVE_WAVELENGTH,
    GAIA_RP_EFFECTIVE_WAVELENGTH,
    GAIA_G_EFFECTIVE_FREQUENCY,
    GAIA_BP_EFFECTIVE_FREQUENCY,
    GAIA_RP_EFFECTIVE_FREQUENCY
)

def calculate_visibility_for_source(source_name, source_obj, baseline_lengths, wavelengths, frequencies, band_names):
    """Calculate |V|^2 for a single source across all baselines and wavelengths."""
    results = []
    
    for wavelength, frequency, band_name in zip(wavelengths, frequencies, band_names):
        vis_squared_values = []
        baseline_values = []
        
        for baseline_length in baseline_lengths:
            try:
                # Create E-W baseline
                baseline = np.array([baseline_length, 0.0, 0.0])
                
                # Calculate visibility squared
                visibility_squared = source_obj.V_squared(frequency, baseline)
                
                # Store valid results
                if not np.isnan(visibility_squared) and visibility_squared >= 0:
                    vis_squared_values.append(visibility_squared)
                    baseline_values.append(baseline_length)
                
            except Exception as e:
                print(f"Warning: Error calculating for baseline {baseline_length}m at {wavelength*1e9:.1f}nm: {str(e)}")
                continue
        
        results.append({
            'source_name': source_name,
            'wavelength_nm': wavelength * 1e9,
            'band_name': band_name,
            'baselines': np.array(baseline_values),
            'visibility_squared': np.array(vis_squared_values)
        })
    
    return results

def plot_visibility_comparison(uniform_results, satlas_results):
    """Create comparison plots for |V|^2 vs baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Uniform Disk vs SATLAS: Visibility Squared Comparison (HD 360)', fontsize=16, fontweight='bold')
    
    # Colors for each band
    colors = ['blue', 'green', 'red']
    band_names = ['G', 'BP', 'RP']
    
    for i, (uniform_result, satlas_result, color, band_name) in enumerate(zip(uniform_results, satlas_results, colors, band_names)):
        ax = axes[i]
        
        # Plot uniform disk results
        ax.semilogy(uniform_result['baselines'], uniform_result['visibility_squared'],
                   'o-', color=color, markersize=4, linewidth=2, alpha=0.7,
                   label=f'Uniform Disk')
        
        # Plot SATLAS results
        ax.semilogy(satlas_result['baselines'], satlas_result['visibility_squared'],
                   's--', color=color, markersize=4, linewidth=2, alpha=0.7,
                   label=f'SATLAS')
        
        ax.set_xlabel('Baseline Length (m)')
        ax.set_ylabel('Visibility Squared |V|²')
        ax.set_title(f'{band_name} Band ({uniform_result["wavelength_nm"]:.1f} nm)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 250)
        ax.legend()
        
        # Add statistics text box
        uniform_min, uniform_max = uniform_result['visibility_squared'].min(), uniform_result['visibility_squared'].max()
        satlas_min, satlas_max = satlas_result['visibility_squared'].min(), satlas_result['visibility_squared'].max()
        
        stats_text = (
            f"Uniform: {uniform_min:.2e} - {uniform_max:.2e}\n"
            f"SATLAS: {satlas_min:.2e} - {satlas_max:.2e}"
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_intensity_profiles(uniform_disk, satlas_grid, star_name):
    """Create comparison plots for intensity profiles from both sources."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Intensity Profile Comparison: Uniform Disk vs SATLAS ({star_name})', fontsize=16, fontweight='bold')
    
    # Define radial coordinates for sampling (in radians)
    max_radius_rad = np.max(satlas_grid.p_rays*1.2)
    radial_coords = np.linspace(0, max_radius_rad, 20)
    
    # Wavelengths and frequencies for the three bands
    wavelengths = [GAIA_G_EFFECTIVE_WAVELENGTH, GAIA_BP_EFFECTIVE_WAVELENGTH, GAIA_RP_EFFECTIVE_WAVELENGTH]
    frequencies = [GAIA_G_EFFECTIVE_FREQUENCY, GAIA_BP_EFFECTIVE_FREQUENCY, GAIA_RP_EFFECTIVE_FREQUENCY]
    band_names = ['G', 'BP', 'RP']
    colors = ['blue', 'green', 'red']
    
    for i, (wavelength, frequency, band_name, color) in enumerate(zip(wavelengths, frequencies, band_names, colors)):
        ax = axes[i]
        
        # Calculate intensity profiles for both sources
        uniform_intensities = []
        satlas_intensities = []
        
        for r in radial_coords:
            # Direction vector for this radial coordinate
            n_hat = np.array([r, 0.0])  # Along x-axis
            
            try:
                # Uniform disk intensity
                uniform_intensity = uniform_disk.intensity(frequency, n_hat)
                uniform_intensities.append(uniform_intensity)
                
                # SATLAS intensity
                satlas_intensity = satlas_grid.intensity(frequency, n_hat)
                satlas_intensities.append(satlas_intensity)
                
            except Exception as e:
                # Handle edge cases
                uniform_intensities.append(0.0)
                satlas_intensities.append(0.0)
        
        # Convert to numpy arrays
        uniform_intensities = np.array(uniform_intensities)
        satlas_intensities = np.array(satlas_intensities)
        
        # Convert radial coordinates to milliarcseconds for plotting
        radial_coords_mas = radial_coords / (np.pi / (180 * 3600 * 1000))
        
        # Normalize intensities to their maximum for comparison
        if np.max(uniform_intensities) > 0:
            uniform_intensities_norm = uniform_intensities / np.max(uniform_intensities)
        else:
            uniform_intensities_norm = uniform_intensities
            
        if np.max(satlas_intensities) > 0:
            satlas_intensities_norm = satlas_intensities / np.max(satlas_intensities)
        else:
            satlas_intensities_norm = satlas_intensities
        
        # Plot intensity profiles
        ax.plot(radial_coords_mas, uniform_intensities_norm,
               'o-', color=color, markersize=3, linewidth=2, alpha=0.7,
               label='Uniform Disk')
        
        ax.plot(radial_coords_mas, satlas_intensities_norm,
               's--', color=color, markersize=3, linewidth=2, alpha=0.7,
               label='SATLAS')
        
        ax.set_xlabel('Radial Distance (mas)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'{band_name} Band ({wavelength*1e9:.1f} nm)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, np.max(radial_coords_mas))
        ax.set_ylim(0, 1.1)
        
        # Add statistics text box
        uniform_nonzero = uniform_intensities_norm[uniform_intensities_norm > 0]
        satlas_nonzero = satlas_intensities_norm[satlas_intensities_norm > 0]
        
        stats_text = f"Uniform: {len(uniform_nonzero)} non-zero points\nSATLAS: {len(satlas_nonzero)} non-zero points"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to create visibility comparison plots."""
    print("Simple Uniform Disk vs SATLAS Visibility Comparison")
    print("=" * 60)
    
    # Load Gaia data
    print("Loading Gaia data...")
    df = pd.read_csv('extended_data_table_2.csv')
    print(f"✓ Loaded {len(df)} stars")
    
    # Select HD 360 for plotting
    star_name = 'HD 360'
    if star_name not in df['Star'].values:
        print(f"Error: {star_name} not found in the data")
        available_stars = df['Star'].tolist()
        print(f"Available stars: {available_stars}")
        return
    print(f"Selected star: {star_name}")
    
    # Create UniformDisk objects
    print("Creating UniformDisk objects...")
    star_disks = create_uniform_disk_from_gaia(df)
    uniform_disk = star_disks[star_name]
    print(f"✓ Created UniformDisk object for {star_name}")
    
    # Create SATLAS RadialGrid2 object
    print("Creating SATLAS RadialGrid2 object...")
    satlas_file = 'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt'
    satlas_grid = create_radial_grid_from_satlas(satlas_file)
    print(f"✓ Created SATLAS RadialGrid2 object")
    
    # Baseline lengths from 25 to 245 in 10m intervals
    baseline_lengths = np.arange(5, 251, 5)
    
    # Wavelengths and frequencies
    wavelengths = [GAIA_G_EFFECTIVE_WAVELENGTH, GAIA_BP_EFFECTIVE_WAVELENGTH, GAIA_RP_EFFECTIVE_WAVELENGTH]
    frequencies = [GAIA_G_EFFECTIVE_FREQUENCY, GAIA_BP_EFFECTIVE_FREQUENCY, GAIA_RP_EFFECTIVE_FREQUENCY]
    band_names = ['G', 'BP', 'RP']
    
    print(f"\nParameters:")
    print(f"  Wavelengths: {[w*1e9 for w in wavelengths]} nm")
    print(f"  Baseline range: {baseline_lengths[0]}-{baseline_lengths[-1]} m ({len(baseline_lengths)} points)")
    
    # Calculate visibility for uniform disk
    print(f"\nCalculating visibility for Uniform Disk ({star_name})...")
    uniform_results = calculate_visibility_for_source(
        f"Uniform Disk ({star_name})", uniform_disk, baseline_lengths, 
        wavelengths, frequencies, band_names
    )
    
    # Calculate visibility for SATLAS
    print(f"Calculating visibility for SATLAS...")
    satlas_results = calculate_visibility_for_source(
        "SATLAS", satlas_grid, baseline_lengths, 
        wavelengths, frequencies, band_names
    )

    # Print summary of results
    print(f"\nUniform Disk Results:")
    for result in uniform_results:
        print(f"  {result['band_name']} Band ({result['wavelength_nm']:.1f} nm): "
            f"{len(result['baselines'])} valid points")
        if len(result['visibility_squared']) > 0:
            print(f"    |V|² range: {result['visibility_squared'].min():.2e} - {result['visibility_squared'].max():.2e}")
    
    print(f"\nSATLAS Results:")
    for result in satlas_results:
        print(f"  {result['band_name']} Band ({result['wavelength_nm']:.1f} nm): "
            f"{len(result['baselines'])} valid points")
        if len(result['visibility_squared']) > 0:
            print(f"    |V|² range: {result['visibility_squared'].min():.2e} - {result['visibility_squared'].max():.2e}")
    
    # Create comparison plots and save to single PDF
    print(f"\nCreating comparison plots...")
    output_filename = f'simple_uniform_vs_satlas_{star_name.replace(" ", "_")}.pdf'
    
    with PdfPages(output_filename) as pdf:
        # Page 1: Visibility comparison
        print(f"  Creating visibility comparison plots...")
        fig_visibility = plot_visibility_comparison(uniform_results, satlas_results)
        pdf.savefig(fig_visibility, bbox_inches='tight', dpi=300)
        plt.close(fig_visibility)
        
        # Page 2: Intensity profile comparison
        print(f"  Creating intensity profile comparison plots...")
        fig_intensity = plot_intensity_profiles(uniform_disk, satlas_grid, star_name)
        pdf.savefig(fig_intensity, bbox_inches='tight', dpi=300)
        plt.close(fig_intensity)
    
    print(f"✓ Combined comparison plots saved to {output_filename}")
    
    print(f"\n" + "=" * 60)
    print("Simple visibility comparison completed successfully!")
    print(f"Generated combined plot file: {output_filename}")
    print(f"  - Page 1: Visibility comparison (|V|² vs baseline)")
    print(f"  - Page 2: Intensity profile comparison")
    print("=" * 60)

if __name__ == "__main__":
    main()