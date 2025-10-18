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
    create_uniform_disk_from_gaia,
    GAIA_RP_EFFECTIVE_FREQUENCY
)
from g2.core import Observation, fisher_matrix

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
    
    # Frequency
    nu_0 = GAIA_RP_EFFECTIVE_FREQUENCY
    
    print(f"\nObservation parameters:")
    print(f"  Integration time: {observation.integration_time} s")
    print(f"  Telescope area: {observation.telescope_area:.2f} m²")
    print(f"  Throughput: {observation.throughput}")
    print(f"  Detector jitter: {observation.detector_jitter*1e9:.1f} ns")
    print(f"  Frequency: {nu_0:.2e} Hz")
    print(f"  Baselines: {baseline_lengths} m")
    
    # Prepare results table
    results = []
    
    print(f"\nCalculating Fisher matrices and visibilities...")
    print(f"{'Star':<12} {'Baseline (m)':<12} {'F_nu (W/m²/Hz)':<15} {'|V|²':<15} {'sqrt(F^-1_rr)':<15} {'σ_r/r':<15}")
    print("-" * 90)
    
    for star_name, disk in star_disks.items():
        # Calculate specific flux (same for all baselines)
        specific_flux = disk.specific_flux(nu_0)
        
        for baseline_length in baseline_lengths:
            try:
                # Create E-W baseline
                baseline = np.array([baseline_length, 0.0, 0.0])
                
                # Calculate visibility squared
                visibility_squared = disk.V_squared(nu_0, baseline)
                
                # Calculate Fisher matrix for this baseline
                F = fisher_matrix(disk, nu_0, baseline, observation)
                
                # Extract Fisher matrix element for radius parameter
                # The radius parameter should be the first (and only) parameter
                if F.size > 0:
                    fisher_rr = F[0, 0]  # Fisher matrix element for radius
                    
                    # Calculate sqrt of inverse Fisher matrix element
                    if fisher_rr > 0:
                        sqrt_inv_fisher_rr = 1.0 / np.sqrt(fisher_rr)
                    else:
                        sqrt_inv_fisher_rr = np.inf
                else:
                    fisher_rr = 0.0
                    sqrt_inv_fisher_rr = np.inf
                
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
                    'Baseline_m': baseline_length,
                    'F_nu': specific_flux,
                    'Visibility_squared': visibility_squared,
                    'sqrt_inv_Fisher_rr': sqrt_inv_fisher_rr,
                    'sigma_r/r': sigma_r_over_r
                })
                
                print(f"{star_name:<12} {baseline_length:<12} {specific_flux:<15.2e} {visibility_squared:<15.6f} {sqrt_inv_fisher_rr:<15.2e} {sigma_r_over_r:<15.2e}")
                
            except Exception as e:
                print(f"{star_name:<12} {baseline_length:<12} ERROR: {str(e)}")
                results.append({
                    'Star': star_name,
                    'Baseline_m': baseline_length,
                    'F_nu': specific_flux,
                    'Visibility_squared': np.nan,
                    'sqrt_inv_Fisher_rr': np.nan,
                    'sigma_r/r': np.nan
                })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Create pivot tables for better visualization
    print(f"\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    # Specific flux table (same for all baselines, so just show one column)
    print("\nSpecific Flux F_nu (W/m²/Hz) for each star:")
    flux_data = results_df.drop_duplicates('Star')[['Star', 'F_nu']].set_index('Star')
    print(flux_data.to_string(float_format='%.2e'))
    
    # Visibility squared table
    print(f"\n\nVisibility Squared |V|² for each star and baseline:")
    vis_pivot = results_df.pivot(index='Star', columns='Baseline_m', values='Visibility_squared')
    print(vis_pivot.to_string(float_format='%.6f'))
    
    # sqrt inverse Fisher matrix table
    print(f"\n\nsqrt(F^-1_rr) - Parameter uncertainty bounds:")
    sqrt_inv_pivot = results_df.pivot(index='Star', columns='Baseline_m', values='sqrt_inv_Fisher_rr')
    print(sqrt_inv_pivot.to_string(float_format='%.2e'))
    
    # sigma_r/r table
    print(f"\n\nσ_r/r - Relative parameter uncertainty:")
    sigma_r_over_r_pivot = results_df.pivot(index='Star', columns='Baseline_m', values='sigma_r/r')
    print(sigma_r_over_r_pivot.to_string(float_format='%.2e'))
    
    # Save to CSV
    output_file = 'fisher_matrix_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Additional analysis
    print(f"\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Find best baselines for each star (lowest uncertainty)
    print(f"\nBest baseline for each star (lowest parameter uncertainty):")
    best_baselines = results_df.loc[results_df.groupby('Star')['sqrt_inv_Fisher_rr'].idxmin()]
    for _, row in best_baselines.iterrows():
        if not np.isnan(row['sqrt_inv_Fisher_rr']):
            print(f"  {row['Star']:<12}: {row['Baseline_m']:>3}m (F_nu = {row['F_nu']:.2e}, σ_r = {row['sqrt_inv_Fisher_rr']:.2e}, σ_r/r = {row['sigma_r/r']:.2e}, |V|² = {row['Visibility_squared']:.6f})")
    
    # Statistics by baseline
    print(f"\nParameter uncertainty statistics by baseline:")
    baseline_stats = results_df.groupby('Baseline_m')['sqrt_inv_Fisher_rr'].agg(['mean', 'std', 'min', 'max'])
    print(baseline_stats.to_string(float_format='%.2e'))
    
    print(f"\nVisibility statistics by baseline:")
    vis_stats = results_df.groupby('Baseline_m')['Visibility_squared'].agg(['mean', 'std', 'min', 'max'])
    print(vis_stats.to_string(formatters={'mean': '{:.6f}'.format, 'std': '{:.6f}'.format, 'min': '{:.6f}'.format, 'max': '{:.6f}'.format}))
    
    print(f"\n" + "=" * 80)
    print("Analysis completed!")
    print(f"Total measurements: {len(results_df)}")
    print(f"Successful calculations: {len(results_df[~results_df['sqrt_inv_Fisher_rr'].isna()])}")

if __name__ == "__main__":
    main()