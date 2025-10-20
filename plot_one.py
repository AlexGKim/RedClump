#!/usr/bin/env python3
"""
Single Star Fisher Matrix Plotting Script

This script plots |V|^2 and sqrt_inv_Fisher_lnrlnr as a function of baseline
for one star from extended_data_table_2.csv, using baseline intervals from 50 to 650m in 25m steps.

Based on algorithms from fisher_matrix_table.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/akim/Projects/g2')

from gaia_uniform_disk import create_uniform_disk_from_gaia
from gaia_zeropoint import (
    GAIA_G_EFFECTIVE_WAVELENGTH,
    GAIA_BP_EFFECTIVE_WAVELENGTH,
    GAIA_RP_EFFECTIVE_WAVELENGTH,
    GAIA_G_EFFECTIVE_FREQUENCY,
    GAIA_BP_EFFECTIVE_FREQUENCY,
    GAIA_RP_EFFECTIVE_FREQUENCY
)
from g2.core import Observation, fisher_matrix, inverse_noise

def calculate_parameters_for_star(star_name, star_disk, observation, baseline_lengths, wavelengths, frequencies, band_names):
    """Calculate |V|^2 and sqrt_inv_Fisher_lnrlnr for a single star across all baselines and wavelengths."""
    results = []
    
    for wavelength, frequency, band_name in zip(wavelengths, frequencies, band_names):
        vis_squared_values = []
        sqrt_inv_fisher_values = []
        baseline_values = []
        
        # Calculate inverse noise once per wavelength (independent of baseline)
        try:
            inv_noise = inverse_noise(star_disk, frequency, observation)
        except Exception as e:
            print(f"Warning: Error calculating inverse noise for {wavelength*1e9:.1f}nm: {str(e)}")
            inv_noise = np.nan
        
        for baseline_length in baseline_lengths:
            try:
                # Create E-W baseline
                baseline = np.array([baseline_length, 0.0, 0.0])
                
                # Calculate visibility squared
                visibility_squared = star_disk.V_squared(frequency, baseline)
                
                # Calculate Fisher matrix for this baseline
                F = fisher_matrix(star_disk, frequency, baseline, observation)
                
                # Extract Fisher matrix element for radius parameter
                if F.size > 0:
                    fisher_lnrlnr = F[0, 0]  # Fisher matrix element for radius
                    
                    # Calculate sqrt of inverse Fisher matrix element
                    if fisher_lnrlnr > 0:
                        sqrt_inv_fisher_lnrlnr = 1.0 / np.sqrt(fisher_lnrlnr)
                    else:
                        sqrt_inv_fisher_lnrlnr = np.inf
                else:
                    sqrt_inv_fisher_lnrlnr = np.inf
                
                # Store valid results
                if not np.isnan(visibility_squared) and not np.isinf(sqrt_inv_fisher_lnrlnr):
                    vis_squared_values.append(visibility_squared)
                    sqrt_inv_fisher_values.append(sqrt_inv_fisher_lnrlnr)
                    baseline_values.append(baseline_length)
                
            except Exception as e:
                print(f"Warning: Error calculating for baseline {baseline_length}m at {wavelength*1e9:.1f}nm: {str(e)}")
                continue
        
        results.append({
            'wavelength_nm': wavelength * 1e9,
            'band_name': band_name,
            'baselines': np.array(baseline_values),
            'visibility_squared': np.array(vis_squared_values),
            'sqrt_inv_fisher': np.array(sqrt_inv_fisher_values),
            'inverse_noise': inv_noise
        })
    
    return results

def plot_star_parameters(star_name, results, observation):
    """Create plots for |V|^2 and sqrt_inv_Fisher_lnrlnr vs baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Star: {star_name}', fontsize=16, fontweight='bold')
    
    # Colors and labels for each wavelength
    colors = ['blue', 'green', 'red']
    
    # Plot |V|^2 vs baseline (left subplot)
    ax1 = axes[0]
    for i, result in enumerate(results):
        ax1.semilogy(result['baselines'], result['visibility_squared'],
                    'o-', color=colors[i], markersize=4, linewidth=2,
                    label=f'{result["band_name"]} ({result["wavelength_nm"]:.1f} nm)')
    
    ax1.set_xlabel('Baseline Length (m)')
    ax1.set_ylabel('Visibility Squared |V|²')
    ax1.set_title('Visibility Squared vs Baseline')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 250)
    ax1.legend()
    
    # Add inverse noise values in a text box
    inverse_noise_text = "Inverse Noise:\n"
    for i, result in enumerate(results):
        if not np.isnan(result['inverse_noise']):
            inverse_noise_text += f"{result['band_name']}: {result['inverse_noise']:.2e}\n"
        else:
            inverse_noise_text += f"{result['band_name']}: N/A\n"
    
    # Add text box to the left subplot
    ax1.text(0.02, 0.98, inverse_noise_text.strip(), transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=9)
    
    # Plot sqrt_inv_Fisher_lnrlnr vs baseline (right subplot)
    ax2 = axes[1]
    for i, result in enumerate(results):
        ax2.semilogy(result['baselines'], result['sqrt_inv_fisher'],
                    'o-', color=colors[i], markersize=4, linewidth=2,
                    label=f'{result["band_name"]} ({result["wavelength_nm"]:.1f} nm)')
    
    ax2.set_xlabel('Baseline Length (m)')
    ax2.set_ylabel(r'$\sigma_{\ln{r}}$')
    ax2.set_title('Parameter Uncertainty vs Baseline')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 250)
    ax2.legend()
    
    # Add observation parameters as text box
    obs_text = (
        f"Observation Parameters:\n"
        f"Integration time: {observation.integration_time} s\n"
        f"Telescope area: {observation.telescope_area:.1f} m²\n"
        f"Throughput: {observation.throughput}\n"
        f"Detector jitter: {observation.detector_jitter*1e9:.1f} ns"
    )
    
    # Add text box to the right subplot
    ax2.text(0.02, 0.98, obs_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to create plots for one star."""
    print("Single Star Fisher Matrix Plotting Script")
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
    star_disk = star_disks[star_name]
    print(f"✓ Created UniformDisk object for {star_name}")
    
    # Observation parameters (same as fisher_matrix_table.py)
    observation = Observation(
        integration_time=3600,  # 3600 seconds
        telescope_area=np.pi * (5.0)**2,  # π*(5m)^2
        throughput=0.3,  # 0.3
        detector_jitter=130e-12/2.555  # 130 ps FWHM to stddev
    )
    
    # Baseline lengths from 50 to 650 in 25m intervals
    baseline_lengths = np.arange(25, 251, 10)  # 50, 75, 100, ..., 650
    
    # Wavelengths and frequencies
    wavelengths = [GAIA_G_EFFECTIVE_WAVELENGTH, GAIA_BP_EFFECTIVE_WAVELENGTH, GAIA_RP_EFFECTIVE_WAVELENGTH]
    frequencies = [GAIA_G_EFFECTIVE_FREQUENCY, GAIA_BP_EFFECTIVE_FREQUENCY, GAIA_RP_EFFECTIVE_FREQUENCY]
    band_names = ['G', 'BP', 'RP']
    
    print(f"\nObservation parameters:")
    print(f"  Integration time: {observation.integration_time} s")
    print(f"  Telescope area: {observation.telescope_area:.2f} m²")
    print(f"  Throughput: {observation.throughput}")
    print(f"  Detector jitter: {observation.detector_jitter*1e9:.1f} ns")
    print(f"  Wavelengths: {[w*1e9 for w in wavelengths]} nm")
    print(f"  Baseline range: {baseline_lengths[0]}-{baseline_lengths[-1]} m ({len(baseline_lengths)} points)")
    
    # Calculate parameters for the selected star
    print(f"\nCalculating parameters for {star_name}...")
    results = calculate_parameters_for_star(
        star_name, star_disk, observation, baseline_lengths, 
        wavelengths, frequencies, band_names
    )
    
    # Print summary of results
    print(f"\nResults summary:")
    for result in results:
        print(f"  {result['band_name']} Band ({result['wavelength_nm']:.1f} nm): "
              f"{len(result['baselines'])} valid points")
        print(f"    |V|² range: {result['visibility_squared'].min():.2e} - {result['visibility_squared'].max():.2e}")
        print(f"    sigma_lnr: {result['sqrt_inv_fisher'].min():.2e} - {result['sqrt_inv_fisher'].max():.2e}")
    
    # Create plots
    print(f"\nCreating plots...")
    fig = plot_star_parameters(star_name, results, observation)
    
    # Save plot
    output_filename = f'plot_one_{star_name.replace(" ", "_")}.pdf'
    fig.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"✓ Plot saved to {output_filename}")
    
    print(f"\n" + "=" * 60)
    print("Single star plotting completed successfully!")
    print(f"Generated plot for {star_name} saved to {output_filename}")
    print("=" * 60)

if __name__ == "__main__":
    main()