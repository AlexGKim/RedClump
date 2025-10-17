#!/usr/bin/env python3
"""
Complete workflow demonstration for the Gaia to UniformDisk converter.

This script demonstrates the complete workflow from loading Gaia data
to performing visibility calculations with UniformDisk objects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/akim/Projects/g2')

from gaia_uniform_disk import (
    create_uniform_disk_from_gaia,
    get_star_properties,
    GAIA_RP_EFFECTIVE_FREQUENCY
)

def main():
    """Demonstrate the complete workflow."""
    print("Gaia to UniformDisk Converter - Complete Workflow Demo")
    print("=" * 60)
    
    # Step 1: Load Gaia data
    print("Step 1: Loading Gaia data from CSV...")
    df = pd.read_csv('extended_data_table_2.csv')
    print(f"✓ Loaded {len(df)} stars")
    print(f"  Columns: {list(df.columns)}")
    
    # Step 2: Create UniformDisk objects
    print(f"\nStep 2: Converting Gaia data to UniformDisk objects...")
    star_disks = create_uniform_disk_from_gaia(df)
    print(f"✓ Created {len(star_disks)} UniformDisk objects")
    
    # Step 3: Explore individual star properties
    print(f"\nStep 3: Exploring star properties...")
    example_star = 'HD 360'
    if example_star in star_disks:
        props = get_star_properties(star_disks, example_star)
        print(f"\n{example_star} properties:")
        print(f"  Flux density: {props['flux_density']:.2e} W/m²/Hz")
        print(f"  Angular radius: {props['angular_radius_rad']:.2e} radians")
        print(f"  Angular diameter: {props['angular_diameter_mas']:.3f} mas")
        print(f"  Surface brightness: {props['surface_brightness']:.2e} W/m²/Hz/sr")
    
    # Step 4: Calculate visibilities for different baselines
    print(f"\nStep 4: Calculating visibilities for different baselines...")
    if example_star in star_disks:
        disk = star_disks[example_star]
        nu_0 = GAIA_RP_EFFECTIVE_FREQUENCY
        
        # Test different baseline lengths
        baseline_lengths = [10, 50, 100, 200, 500, 1000]  # meters
        print(f"\nVisibility vs baseline length for {example_star}:")
        print(f"{'Baseline (m)':<12} {'|V|':<10} {'Phase (rad)':<12}")
        print("-" * 35)
        
        for B in baseline_lengths:
            baseline = np.array([B, 0.0, 0.0])  # E-W baseline
            visibility = disk.V(nu_0, baseline)
            print(f"{B:<12} {abs(visibility):<10.4f} {np.angle(visibility):<12.3f}")
    
    # Step 5: Compare different stars
    print(f"\nStep 5: Comparing different stars...")
    comparison_stars = ['HD 360', 'HD 9362', 'HD 16815', 'HD 219784']
    baseline = np.array([100.0, 0.0, 0.0])  # 100m baseline
    
    print(f"\nComparison at 100m baseline:")
    print(f"{'Star':<12} {'Diameter (mas)':<15} {'|V|':<10} {'First null (m)':<15}")
    print("-" * 55)
    
    for star_name in comparison_stars:
        if star_name in star_disks:
            disk = star_disks[star_name]
            props = get_star_properties(star_disks, star_name)
            visibility = disk.V(nu_0, baseline)
            
            # Calculate first null baseline (approximate)
            # For uniform disk: first null at B ≈ 1.22λ/(2θ)
            wavelength = 3e8 / nu_0  # c/ν
            theta_rad = props['angular_radius_rad']
            first_null_baseline = 1.22 * wavelength / (2 * theta_rad)
            
            print(f"{star_name:<12} {props['angular_diameter_mas']:<15.3f} "
                  f"{abs(visibility):<10.4f} {first_null_baseline:<15.0f}")
    
    # Step 6: Demonstrate visibility function behavior
    print(f"\nStep 6: Demonstrating visibility function behavior...")
    if example_star in star_disks:
        disk = star_disks[example_star]
        
        # Create baseline array for plotting
        baselines = np.logspace(1, 3.5, 50)  # 10m to ~3000m
        visibilities = []
        
        for B in baselines:
            baseline = np.array([B, 0.0, 0.0])
            vis = disk.V(nu_0, baseline)
            visibilities.append(abs(vis))
        
        # Find first few nulls
        visibilities = np.array(visibilities)
        null_indices = []
        for i in range(1, len(visibilities)-1):
            if (visibilities[i] < visibilities[i-1] and 
                visibilities[i] < visibilities[i+1] and 
                visibilities[i] < 0.1):
                null_indices.append(i)
        
        print(f"\nVisibility nulls for {example_star}:")
        for i, idx in enumerate(null_indices[:3]):  # First 3 nulls
            print(f"  Null {i+1}: ~{baselines[idx]:.0f}m baseline, |V| = {visibilities[idx]:.4f}")
    
    # Step 7: Summary statistics
    print(f"\nStep 7: Dataset summary statistics...")
    all_diameters = []
    all_fluxes = []
    
    for star_name, disk in star_disks.items():
        props = get_star_properties(star_disks, star_name)
        all_diameters.append(props['angular_diameter_mas'])
        all_fluxes.append(props['flux_density'])
    
    all_diameters = np.array(all_diameters)
    all_fluxes = np.array(all_fluxes)
    
    print(f"\nDataset statistics:")
    print(f"  Number of stars: {len(star_disks)}")
    print(f"  Angular diameter range: {all_diameters.min():.3f} - {all_diameters.max():.3f} mas")
    print(f"  Mean angular diameter: {all_diameters.mean():.3f} ± {all_diameters.std():.3f} mas")
    print(f"  Flux density range: {all_fluxes.min():.2e} - {all_fluxes.max():.2e} W/m²/Hz")
    print(f"  Mean flux density: {all_fluxes.mean():.2e} W/m²/Hz")
    
    # Step 8: Usage recommendations
    print(f"\nStep 8: Usage recommendations...")
    print(f"\nFor intensity interferometry observations:")
    print(f"  • Small stars (< 1 mas): Use baselines > 200m for good visibility contrast")
    print(f"  • Large stars (> 2 mas): Use baselines 50-200m to avoid first null")
    print(f"  • Bright stars: Better SNR, suitable for longer baselines")
    print(f"  • All stars show circular symmetry (uniform disk model)")
    
    print(f"\nNext steps:")
    print(f"  1. Use star_disks dictionary to access any star by name")
    print(f"  2. Call disk.V(frequency, baseline) for visibility calculations")
    print(f"  3. Experiment with different baseline configurations")
    print(f"  4. Consider observational constraints (SNR, atmospheric effects)")
    
    print(f"\n" + "=" * 60)
    print("Workflow demonstration completed!")
    print(f"All {len(star_disks)} stars are ready for intensity interferometry modeling.")

if __name__ == "__main__":
    main()