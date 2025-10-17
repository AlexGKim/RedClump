#!/usr/bin/env python3
"""
Complete test and example usage of the Gaia to UniformDisk converter.

This script demonstrates the full workflow of creating UniformDisk objects
from Gaia data and using them for visibility calculations.
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
    print("Complete Gaia to UniformDisk Converter Test")
    print("=" * 60)
    
    # Load the full dataset
    df = pd.read_csv('extended_data_table_2.csv')
    print(f"Loaded {len(df)} stars from extended_data_table_2.csv")
    
    # Create UniformDisk objects for all stars
    print("\nCreating UniformDisk objects for all stars...")
    star_disks = create_uniform_disk_from_gaia(df)
    
    print(f"\nSuccessfully created UniformDisk objects for {len(star_disks)} stars")
    
    # Show properties of a few example stars
    example_stars = ['HD 360', 'HD 9362', 'HD 16815', 'HD 219784']
    
    print(f"\nExample star properties:")
    print("-" * 80)
    print(f"{'Star':<12} {'Flux (W/m²/Hz)':<15} {'Radius (rad)':<15} {'Diameter (mas)':<15} {'Visibility':<10}")
    print("-" * 80)
    
    baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W baseline
    nu_0 = GAIA_RP_EFFECTIVE_FREQUENCY
    
    for star_name in example_stars:
        if star_name in star_disks:
            props = get_star_properties(star_disks, star_name)
            disk = star_disks[star_name]
            visibility = disk.V(nu_0, baseline)
            
            print(f"{star_name:<12} {props['flux_density']:<15.2e} {props['angular_radius_rad']:<15.2e} "
                  f"{props['angular_diameter_mas']:<15.3f} {abs(visibility):<10.4f}")
    
    # Demonstrate visibility vs baseline length
    print(f"\nVisibility vs baseline length for HD 360:")
    print("-" * 50)
    
    if 'HD 360' in star_disks:
        hd_360 = star_disks['HD 360']
        baseline_lengths = np.logspace(1, 3, 10)  # 10m to 1000m
        visibilities = []
        
        print(f"{'Baseline (m)':<12} {'|V|':<10}")
        print("-" * 25)
        
        for B in baseline_lengths:
            baseline = np.array([B, 0.0, 0.0])
            vis = hd_360.V(nu_0, baseline)
            visibilities.append(abs(vis))
            print(f"{B:<12.1f} {abs(vis):<10.4f}")
    
    # Summary statistics
    print(f"\nDataset Summary:")
    print("-" * 30)
    
    flux_densities = [disk.flux_density for disk in star_disks.values()]
    angular_radii = [disk.radius for disk in star_disks.values()]
    angular_diameters = [radius * 2 / (np.pi / (180 * 3600 * 1000)) for radius in angular_radii]
    
    print(f"Number of stars: {len(star_disks)}")
    print(f"Flux density range: {min(flux_densities):.2e} - {max(flux_densities):.2e} W/m²/Hz")
    print(f"Angular diameter range: {min(angular_diameters):.3f} - {max(angular_diameters):.3f} mas")
    print(f"Mean angular diameter: {np.mean(angular_diameters):.3f} mas")
    
    # Test different baseline orientations for one star
    print(f"\nVisibility vs baseline orientation for HD 360:")
    print("-" * 50)
    
    if 'HD 360' in star_disks:
        hd_360 = star_disks['HD 360']
        B_length = 100.0  # 100m baseline
        angles = np.linspace(0, 180, 7)  # 0 to 180 degrees
        
        print(f"{'Angle (deg)':<12} {'Baseline':<20} {'|V|':<10}")
        print("-" * 45)
        
        for angle in angles:
            angle_rad = np.radians(angle)
            baseline = np.array([B_length * np.cos(angle_rad), 
                               B_length * np.sin(angle_rad), 
                               0.0])
            vis = hd_360.V(nu_0, baseline)
            baseline_str = f"[{baseline[0]:.1f}, {baseline[1]:.1f}, 0]"
            print(f"{angle:<12.0f} {baseline_str:<20} {abs(vis):<10.4f}")
    
    print(f"\n" + "=" * 60)
    print("Test completed successfully!")
    print(f"All {len(star_disks)} stars are ready for visibility calculations.")

if __name__ == "__main__":
    main()