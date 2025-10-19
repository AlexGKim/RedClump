#!/usr/bin/env python3
"""
Fisher Matrix Analysis for Stars in extended_data_table_2.csv

This script calculates visibility squared and sqrt inverse Fisher matrix
for the 'radius' parameter of stars using intensity interferometry observations.

Based on demo_workflow.py and uses the g2 library for Fisher matrix calculations.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/akim/Projects/g2')

from gaia_uniform_disk import (
    create_uniform_disk_from_gaia
)
from gaia_zeropoint import (
    GAIA_G_EFFECTIVE_WAVELENGTH,
    GAIA_BP_EFFECTIVE_WAVELENGTH,
    GAIA_RP_EFFECTIVE_WAVELENGTH,
    GAIA_G_EFFECTIVE_FREQUENCY,
    GAIA_BP_EFFECTIVE_FREQUENCY,
    GAIA_RP_EFFECTIVE_FREQUENCY
)
from g2.core import Observation, fisher_matrix, inverse_noise

def main():
    """Calculate Fisher matrix table for all stars."""
    print("Fisher Matrix Analysis for Extended Data Table 2")
    print("=" * 60)
    
    # Load Gaia data
    print("Loading Gaia data...")
    df = pd.read_csv('extended_data_table_2.csv')
    print(f"✓ Loaded {len(df)} stars")
    
    # Create UniformDisk objects
    print("Creating UniformDisk objects...")
    star_disks = create_uniform_disk_from_gaia(df)
    print(f"✓ Created {len(star_disks)} UniformDisk objects")
    
    # Observation parameters as specified
    observation = Observation(
        integration_time=3600,  # 3600 seconds
        telescope_area=np.pi * (5.0)**2,  # π*(5m)^2
        throughput=0.3,  # 0.3
        detector_jitter=130e-12/2.555  # 130 ps FWHM to stddev
    )
    
    # Baseline lengths to test
    baseline_lengths = [50, 100, 150, 200, 250, 300, 350]  # meters
    
    # Wavelengths and frequencies to cycle through
    wavelengths = [GAIA_G_EFFECTIVE_WAVELENGTH, GAIA_BP_EFFECTIVE_WAVELENGTH, GAIA_RP_EFFECTIVE_WAVELENGTH]
    frequencies = [GAIA_G_EFFECTIVE_FREQUENCY, GAIA_BP_EFFECTIVE_FREQUENCY, GAIA_RP_EFFECTIVE_FREQUENCY]
    band_names = ['G', 'BP', 'RP']
    
    print(f"\nObservation parameters:")
    print(f"  Integration time: {observation.integration_time} s")
    print(f"  Telescope area: {observation.telescope_area:.2f} m²")
    print(f"  Throughput: {observation.throughput}")
    print(f"  Detector jitter: {observation.detector_jitter*1e9:.1f} ns")
    print(f"  Wavelengths: {[w*1e9 for w in wavelengths]} nm")
    print(f"  Frequencies: {[f for f in frequencies]} Hz")
    print(f"  Baselines: {baseline_lengths} m")
    
    # Prepare results table
    results = []
    
    print(f"\nCalculating Fisher matrices and visibilities...")
    print(f"{'Star':<12} {'Wavelength (nm)':<15} {'Baseline (m)':<12} {'|V|²':<15} {'1/inv_σ':<15} {'SNR':<15} {'sqrt(F^-1_rr)':<15} {'σ_r/r':<15}")
    print("-" * 120)
    
    for star_name, disk in star_disks.items():
        for wavelength, frequency, band_name in zip(wavelengths, frequencies, band_names):
            for baseline_length in baseline_lengths:
                try:
                    # Create E-W baseline
                    baseline = np.array([baseline_length, 0.0, 0.0])
                    
                    # Calculate visibility squared
                    visibility_squared = disk.V_squared(frequency, baseline)
                    
                    # Calculate Fisher matrix for this baseline
                    F = fisher_matrix(disk, frequency, baseline, observation)
                    
                    # Extract Fisher matrix element for radius parameter
                    # The radius parameter should be the first (and only) parameter
                    if F.size > 0:
                        fisher_rr = F[0, 0]  # Fisher matrix element for radius
                        
                        # Calculate sqrt of inverse Fisher matrix element
                        if fisher_rr > 0:
                            sqrt_inv_fisher_rr = 1.0 / np.sqrt(fisher_rr)
                        else:
                            sqrt_inv_fisher_rr = np.inf
                        
                        # Calculate 1/inverse_sigma using core.inverse_noise
                        inverse_sigma = inverse_noise(disk, frequency, observation)
                        one_over_inverse_sigma = 1.0 / inverse_sigma if inverse_sigma > 0 else 0.0
                        
                        # Calculate SNR = |V|² * inv_sigma
                        snr = visibility_squared * inverse_sigma
                    else:
                        fisher_rr = 0.0
                        sqrt_inv_fisher_rr = np.inf
                        # Calculate 1/inverse_sigma using core.inverse_noise even when Fisher matrix fails
                        inverse_sigma = inverse_noise(disk, frequency, observation)
                        one_over_inverse_sigma = 1.0 / inverse_sigma if inverse_sigma > 0 else 0.0
                        
                        # Calculate SNR = |V|² * inv_sigma
                        snr = visibility_squared * inverse_sigma if not np.isnan(visibility_squared) else np.nan
                    
                    # Get the radius from the UniformDisk object
                    radius = disk.radius  # angular radius in radians
                    
                    # Calculate sigma_r/r (relative uncertainty)
                    if radius > 0 and not np.isinf(sqrt_inv_fisher_rr):
                        sigma_r_over_r = sqrt_inv_fisher_rr / radius
                    else:
                        sigma_r_over_r = np.inf
                    
                    # Store results
                    results.append({
                        'Star': star_name,
                        'Wavelength_nm': wavelength * 1e9,  # Convert to nm
                        'Baseline_m': baseline_length,
                        'Visibility_squared': visibility_squared,
                        '1/inverse_sigma': one_over_inverse_sigma,
                        'SNR': snr,
                        'sqrt_inv_Fisher_rr': sqrt_inv_fisher_rr,
                        'sigma_r/r': sigma_r_over_r
                    })
                    
                    print(f"{star_name:<12} {wavelength*1e9:<15.1f} {baseline_length:<12} {visibility_squared:<15.6f} {one_over_inverse_sigma:<15.2e} {snr:<15.2e} {sqrt_inv_fisher_rr:<15.2e} {sigma_r_over_r:<15.2e}")
                    
                except Exception as e:
                    print(f"{star_name:<12} {wavelength*1e9:<15.1f} {baseline_length:<12} ERROR: {str(e)}")
                    results.append({
                        'Star': star_name,
                        'Wavelength_nm': wavelength * 1e9,  # Convert to nm
                        'Baseline_m': baseline_length,
                        'Visibility_squared': np.nan,
                        '1/inverse_sigma': np.nan,
                        'SNR': np.nan,
                        'sqrt_inv_Fisher_rr': np.nan,
                        'sigma_r/r': np.nan
                    })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Create pivot tables for better visualization
    print(f"\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    # Show results for each wavelength separately
    for wavelength, band_name in zip(wavelengths, band_names):
        wavelength_nm = wavelength * 1e9
        print(f"\n\n{band_name} Band ({wavelength_nm:.1f} nm) Results:")
        print("=" * 60)
        
        # Filter data for this wavelength
        wave_data = results_df[results_df['Wavelength_nm'] == wavelength_nm]
        
        # Visibility squared table
        print(f"\nVisibility Squared |V|² for each star and baseline:")
        vis_pivot = wave_data.pivot(index='Star', columns='Baseline_m', values='Visibility_squared')
        print(vis_pivot.to_string(float_format='%.6f'))
        
        # sqrt inverse Fisher matrix table
        print(f"\nsqrt(F^-1_rr) - Parameter uncertainty bounds:")
        sqrt_inv_pivot = wave_data.pivot(index='Star', columns='Baseline_m', values='sqrt_inv_Fisher_rr')
        print(sqrt_inv_pivot.to_string(float_format='%.2e'))
        
        # 1/inverse_sigma table
        print(f"\n1/inverse_sigma - sqrt(Fisher matrix element):")
        one_over_inv_sigma_pivot = wave_data.pivot(index='Star', columns='Baseline_m', values='1/inverse_sigma')
        print(one_over_inv_sigma_pivot.to_string(float_format='%.2e'))
        
        # SNR table
        print(f"\nSNR - Signal-to-Noise Ratio (|V|² × inv_σ):")
        snr_pivot = wave_data.pivot(index='Star', columns='Baseline_m', values='SNR')
        print(snr_pivot.to_string(float_format='%.2e'))
        
        # sigma_r/r table
        print(f"\nσ_r/r - Relative parameter uncertainty:")
        sigma_r_over_r_pivot = wave_data.pivot(index='Star', columns='Baseline_m', values='sigma_r/r')
        print(sigma_r_over_r_pivot.to_string(float_format='%.2e'))
    
    # Save to CSV
    output_file = 'fisher_matrix_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Additional analysis
    print(f"\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Find best baselines for each star and wavelength (lowest uncertainty)
    print(f"\nBest baseline for each star and wavelength (lowest parameter uncertainty):")
    for wavelength, band_name in zip(wavelengths, band_names):
        wavelength_nm = wavelength * 1e9
        print(f"\n{band_name} Band ({wavelength_nm:.1f} nm):")
        wave_data = results_df[results_df['Wavelength_nm'] == wavelength_nm]
        best_baselines = wave_data.loc[wave_data.groupby('Star')['sqrt_inv_Fisher_rr'].idxmin()]
        for _, row in best_baselines.iterrows():
            if not np.isnan(row['sqrt_inv_Fisher_rr']):
                print(f"  {row['Star']:<12}: {row['Baseline_m']:>3}m (1/inv_σ = {row['1/inverse_sigma']:.2e}, SNR = {row['SNR']:.2e}, σ_r = {row['sqrt_inv_Fisher_rr']:.2e}, σ_r/r = {row['sigma_r/r']:.2e}, |V|² = {row['Visibility_squared']:.6f})")
    
    # Statistics by baseline and wavelength
    print(f"\nParameter uncertainty statistics by baseline and wavelength:")
    for wavelength, band_name in zip(wavelengths, band_names):
        wavelength_nm = wavelength * 1e9
        print(f"\n{band_name} Band ({wavelength_nm:.1f} nm):")
        wave_data = results_df[results_df['Wavelength_nm'] == wavelength_nm]
        baseline_stats = wave_data.groupby('Baseline_m')['sqrt_inv_Fisher_rr'].agg(['mean', 'std', 'min', 'max'])
        print(baseline_stats.to_string(float_format='%.2e'))
    
    print(f"\nVisibility statistics by baseline and wavelength:")
    for wavelength, band_name in zip(wavelengths, band_names):
        wavelength_nm = wavelength * 1e9
        print(f"\n{band_name} Band ({wavelength_nm:.1f} nm):")
        wave_data = results_df[results_df['Wavelength_nm'] == wavelength_nm]
        vis_stats = wave_data.groupby('Baseline_m')['Visibility_squared'].agg(['mean', 'std', 'min', 'max'])
        print(vis_stats.to_string(formatters={'mean': '{:.6f}'.format, 'std': '{:.6f}'.format, 'min': '{:.6f}'.format, 'max': '{:.6f}'.format}))
    
    print(f"\n" + "=" * 80)
    print("Analysis completed!")
    print(f"Total measurements: {len(results_df)}")
    print(f"Successful calculations: {len(results_df[~results_df['sqrt_inv_Fisher_rr'].isna()])}")
    print(f"Measurements per wavelength: {len(results_df) // 3}")

if __name__ == "__main__":
    main()