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
from gaia_satlas import create_radial_grid_from_satlas, create_radial_from_gaia, get_radial_grid_properties

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
    """Calculate |V|^2, sqrt_inv_Fisher_lnrlnr, inverse_noise, and specific_flux for a single star across all baselines and wavelengths."""
    results = []
    
    for wavelength, frequency, band_name in zip(wavelengths, frequencies, band_names):
        vis_squared_values = []
        sqrt_inv_fisher_values = []
        inverse_noise_values = []
        specific_flux_values = []
        baseline_values = []
        
        for baseline_length in baseline_lengths:
            try:
                # Create E-W baseline
                baseline = np.array([baseline_length, 0.0, 0.0])
                
                # Calculate visibility squared
                visibility_squared = star_disk.V_squared(frequency, baseline)
                
                # Calculate inverse_noise for this baseline
                inv_noise_baseline = inverse_noise(star_disk, frequency, observation)
                
                # Calculate specific_flux
                spec_flux = star_disk.specific_flux(frequency)
                
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
                    inverse_noise_values.append(inv_noise_baseline)
                    specific_flux_values.append(spec_flux)
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
            'inverse_noise_per_baseline': np.array(inverse_noise_values),
            'specific_flux': np.array(specific_flux_values)
        })
    
    return results

def plot_star_parameters(star_name, results, observation, star_source=None, wavelengths=None, band_names=None):
    """Create plots for |V|^2, sqrt_inv_Fisher_lnrlnr, inverse_noise, specific_flux, and intensity profile vs baseline."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[:, 2])  # Intensity profile spans both rows
    
    fig.suptitle(f'Star: {star_name}', fontsize=16, fontweight='bold')
    
    # Colors and labels for each wavelength
    colors = ['blue', 'green', 'red']
    
    # Plot |V|^2 vs baseline (top-left subplot)
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
    
    # Plot sqrt_inv_Fisher_lnrlnr vs baseline (top-center subplot)
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
    
    # Plot inverse_noise vs baseline (bottom-left subplot)
    for i, result in enumerate(results):
        ax3.semilogy(result['baselines'], result['inverse_noise_per_baseline'],
                    'o-', color=colors[i], markersize=4, linewidth=2,
                    label=f'{result["band_name"]} ({result["wavelength_nm"]:.1f} nm)')
    
    ax3.set_xlabel('Baseline Length (m)')
    ax3.set_ylabel(r'Inverse Noise $\sigma^{-1}$')
    ax3.set_title('Inverse Noise vs Baseline')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 250)
    ax3.legend()
    
    # Plot specific_flux vs baseline (bottom-center subplot)
    for i, result in enumerate(results):
        # specific_flux is constant for all baselines, so just plot it as a horizontal line
        ax4.semilogy(result['baselines'], result['specific_flux'],
                    'o-', color=colors[i], markersize=4, linewidth=2,
                    label=f'{result["band_name"]} ({result["wavelength_nm"]:.1f} nm)')
    
    ax4.set_xlabel('Baseline Length (m)')
    ax4.set_ylabel(r'Specific Flux (W/m²/Hz)')
    ax4.set_title('Specific Flux vs Baseline')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 250)
    ax4.legend()
    
    # Plot intensity profile (right side, spanning both rows)
    if star_source is not None and wavelengths is not None and band_names is not None:
        # Check if source has radial profile (RadialGrid2) or is uniform disk
        has_radial_profile = hasattr(star_source, 'I_nu_p') and hasattr(star_source, 'p_rays')
        
        if has_radial_profile:
            # For RadialGrid2 sources, plot the actual intensity profiles
            # Convert p_rays from radians to mas for plotting
            MAS_TO_RAD = np.pi / (180 * 3600 * 1000)
            r_mas = star_source.p_rays / MAS_TO_RAD
            
            for i, (wavelength, band_name) in enumerate(zip(wavelengths, band_names)):
                # Get intensity profile for this wavelength
                # I_nu_p has shape (n_wavelengths, n_radial_points)
                if i < star_source.I_nu_p.shape[0]:
                    intensity = star_source.I_nu_p[i, :]
                    ax5.plot(r_mas, intensity, 'o-', color=colors[i],
                           markersize=3, linewidth=2,
                           label=f'{band_name} ({wavelength*1e9:.1f} nm)')
            
            ax5.set_xlabel('Radial Distance (mas)')
            ax5.set_ylabel('Normalized Intensity I(r)/I(0)')
            ax5.set_title('Limb-Darkened Intensity Profile')
            ax5.set_xlim(0, r_mas.max() * 1.05)
            ax5.set_ylim(0, 1.1)
        else:
            # For uniform disk, plot a flat profile
            # Get the radius from the source
            if hasattr(star_source, 'r'):
                radius_rad = star_source.r
                MAS_TO_RAD = np.pi / (180 * 3600 * 1000)
                radius_mas = radius_rad / MAS_TO_RAD
                
                r_mas = np.linspace(0, radius_mas, 100)
                intensity = np.ones_like(r_mas)
                intensity[r_mas > radius_mas] = 0
                
                for i, (wavelength, band_name) in enumerate(zip(wavelengths, band_names)):
                    ax5.plot(r_mas, intensity, '-', color=colors[i],
                           linewidth=2, label=f'{band_name} ({wavelength*1e9:.1f} nm)')
                
                ax5.set_xlabel('Radial Distance (mas)')
                ax5.set_ylabel('Normalized Intensity I(r)/I(0)')
                ax5.set_title('Uniform Disk Intensity Profile')
                ax5.set_xlim(0, radius_mas * 1.2)
                ax5.set_ylim(-0.1, 1.2)
            else:
                ax5.text(0.5, 0.5, 'No radial profile available',
                       ha='center', va='center', transform=ax5.transAxes)
        
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'Intensity profile not provided',
               ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Intensity Profile')
    
    return fig

def plot_intensity_profile(star_name, star_source, wavelengths, band_names):
    """Create intensity profile plot for a source."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f'Intensity Profile: {star_name}', fontsize=16, fontweight='bold')
    
    # Colors for each wavelength
    colors = ['blue', 'green', 'red']
    
    # Check if source has radial profile (RadialGrid2) or is uniform disk
    has_radial_profile = hasattr(star_source, 'I_nu_p') and hasattr(star_source, 'p_rays')
    
    if has_radial_profile:
        # For RadialGrid2 sources, plot the actual intensity profiles
        # Convert p_rays from radians to mas for plotting
        MAS_TO_RAD = np.pi / (180 * 3600 * 1000)
        r_mas = star_source.p_rays / MAS_TO_RAD
        
        for i, (wavelength, band_name) in enumerate(zip(wavelengths, band_names)):
            # Get intensity profile for this wavelength
            # I_nu_p has shape (n_wavelengths, n_radial_points)
            if i < star_source.I_nu_p.shape[0]:
                intensity = star_source.I_nu_p[i, :]
                ax.plot(r_mas, intensity, 'o-', color=colors[i],
                       markersize=3, linewidth=2,
                       label=f'{band_name} ({wavelength*1e9:.1f} nm)')
        
        ax.set_xlabel('Radial Distance (mas)')
        ax.set_ylabel('Normalized Intensity I(r)/I(0)')
        ax.set_title('Limb-Darkened Intensity Profile')
        ax.set_xlim(0, r_mas.max() * 1.05)
        ax.set_ylim(0, 1.1)
    else:
        # For uniform disk, plot a flat profile
        # Get the radius from the source
        if hasattr(star_source, 'r'):
            radius_rad = star_source.r
            MAS_TO_RAD = np.pi / (180 * 3600 * 1000)
            radius_mas = radius_rad / MAS_TO_RAD
            
            r_mas = np.linspace(0, radius_mas, 100)
            intensity = np.ones_like(r_mas)
            intensity[r_mas > radius_mas] = 0
            
            for i, (wavelength, band_name) in enumerate(zip(wavelengths, band_names)):
                ax.plot(r_mas, intensity, '-', color=colors[i],
                       linewidth=2, label=f'{band_name} ({wavelength*1e9:.1f} nm)')
            
            ax.set_xlabel('Radial Distance (mas)')
            ax.set_ylabel('Normalized Intensity I(r)/I(0)')
            ax.set_title('Uniform Disk Intensity Profile')
            ax.set_xlim(0, radius_mas * 1.2)
            ax.set_ylim(-0.1, 1.2)
        else:
            ax.text(0.5, 0.5, 'No radial profile available',
                   ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
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
    
    # Merge with Gaia data
    df = df.merge(df_table1, on='Star', how='left')
    print(f"✓ Merged stellar parameters for {df['teff_gspphot'].notna().sum()} stars")
    
    # # Select HD 360 for plotting
    # star_name = 'HD 360'
    # if star_name not in df['Star'].values:
    #     print(f"Error: {star_name} not found in the data")
    #     available_stars = df['Star'].tolist()
    #     print(f"Available stars: {available_stars}")
    #     return
    # print(f"Selected star: {star_name}")
    
    # Create UniformDisk objects
    print("Creating UniformDisk objects...")
    star_disks = create_uniform_disk_from_gaia(df)

    # Get first star row for SATLAS
    first_star_name = list(star_disks.keys())[0]
    first_star_row = df[df['Star'] == first_star_name].iloc[0]
    
    # Path to SATLAS data file
    satlas_file = 'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt'
    print("Creating RadialGrid from SATLAS data...")
    radial_grid_satlas = create_radial_grid_from_satlas(satlas_file, first_star_row)
    
    # Create RadialGrid from PHOENIX models
    print("Creating RadialGrid from PHOENIX models...")
    radial_grids_phoenix = create_radial_from_gaia(df)
    first_radial_phoenix = radial_grids_phoenix[first_star_name]


    # star_disk = star_disks[star_name]
    # print(f"✓ Created UniformDisk object for {star_name}")
    
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
    
    for star_name in star_disks.keys():  # Loop over selected star(s)
        # Calculate parameters for the selected star
        print(f"\nCalculating parameters for {star_name}...")
        results = calculate_parameters_for_star(
            star_name, star_disks[star_name], observation, baseline_lengths, 
            wavelengths, frequencies, band_names
        )

        # Calculate for SATLAS radial grid
        results_radial_satlas = calculate_parameters_for_star(
            star_name, radial_grid_satlas, observation, baseline_lengths,
            wavelengths, frequencies, band_names
        )
        
        # Calculate for PHOENIX radial grid
        results_radial_phoenix = calculate_parameters_for_star(
            star_name, first_radial_phoenix, observation, baseline_lengths,
            wavelengths, frequencies, band_names
        )

        # Print summary of results
        print(f"\nResults summary:")
        for result in results:
            print(f"  {result['band_name']} Band ({result['wavelength_nm']:.1f} nm): "
                f"{len(result['baselines'])} valid points")
            if len(result['baselines']) > 0:
                print(f"    |V|² range: {result['visibility_squared'].min():.2e} - {result['visibility_squared'].max():.2e}")
                print(f"    sigma_lnr: {result['sqrt_inv_fisher'].min():.2e} - {result['sqrt_inv_fisher'].max():.2e}")
                print(f"    Inverse noise range: {result['inverse_noise_per_baseline'].min():.2e} - {result['inverse_noise_per_baseline'].max():.2e}")
            else:
                print(f"    No valid data points")
        
        # Create plots
        print(f"\nCreating plots...")
        
        # Plot for uniform disk (with intensity profile)
        fig_uniform = plot_star_parameters(star_name + " (Uniform Disk)", results, observation,
                                          star_source=star_disks[star_name],
                                          wavelengths=wavelengths, band_names=band_names)
        output_filename_uniform = f'plot_one_{star_name.replace(" ", "_")}_uniform.pdf'
        fig_uniform.savefig(output_filename_uniform, bbox_inches='tight', dpi=300)
        print(f"✓ Uniform disk plot saved to {output_filename_uniform}")

        # Plot for SATLAS radial grid (with intensity profile)
        fig_radial_satlas = plot_star_parameters(star_name + " (SATLAS)", results_radial_satlas, observation,
                                                 star_source=radial_grid_satlas,
                                                 wavelengths=wavelengths, band_names=band_names)
        output_filename_satlas = f'plot_one_{star_name.replace(" ", "_")}_satlas.pdf'
        fig_radial_satlas.savefig(output_filename_satlas, bbox_inches='tight', dpi=300)
        print(f"✓ SATLAS radial plot saved to {output_filename_satlas}")
        
        # Plot for PHOENIX radial grid (with intensity profile)
        fig_radial_phoenix = plot_star_parameters(star_name + " (PHOENIX)", results_radial_phoenix, observation,
                                                  star_source=first_radial_phoenix,
                                                  wavelengths=wavelengths, band_names=band_names)
        output_filename_phoenix = f'plot_one_{star_name.replace(" ", "_")}_phoenix.pdf'
        fig_radial_phoenix.savefig(output_filename_phoenix, bbox_inches='tight', dpi=300)
        print(f"✓ PHOENIX radial plot saved to {output_filename_phoenix}")

        plt.close(fig_uniform)
        plt.close(fig_radial_satlas)
        plt.close(fig_radial_phoenix)


        
        print(f"\n" + "=" * 60)
        print("Single star plotting completed successfully!")
        print(f"Generated 3 plots for {star_name} (each with intensity profile):")
        print(f"  - Uniform disk: {output_filename_uniform}")
        print(f"  - SATLAS radial: {output_filename_satlas}")
        print(f"  - PHOENIX radial: {output_filename_phoenix}")
        print("=" * 60)
        break

if __name__ == "__main__":
    main()