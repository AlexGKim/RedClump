#!/usr/bin/env python3
"""
Direct test of the Gaia to UniformDisk converter function.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/akim/Projects/g2')

from gaia_uniform_disk import create_uniform_disk_from_gaia, get_star_properties

def main():
    print("Direct test of create_uniform_disk_from_gaia function")
    print("=" * 60)
    
    # Load the data
    df = pd.read_csv('extended_data_table_2.csv')
    print(f"Loaded {len(df)} stars from CSV")
    
    # Show the columns we'll use
    print(f"Required columns present: {['Star', 'phot_rp_mean_mag', 'parallax', 'LD']}")
    
    # Test with just the first few rows to start
    test_df = df.head(3)
    print(f"\nTesting with first 3 stars:")
    for i, row in test_df.iterrows():
        print(f"  {row['Star']}: RP={row['phot_rp_mean_mag']:.3f}, parallax={row['parallax']:.3f}, LD={row['LD']:.3f}")
    
    # Create UniformDisk objects
    print(f"\nCalling create_uniform_disk_from_gaia...")
    try:
        star_disks = create_uniform_disk_from_gaia(test_df)
        print(f"✓ Function completed successfully!")
        print(f"Created {len(star_disks)} UniformDisk objects")
        
        # Test each star
        for star_name in star_disks.keys():
            print(f"\n{star_name}:")
            props = get_star_properties(star_disks, star_name)
            print(f"  Flux density: {props['flux_density']:.2e} W/m²/Hz")
            print(f"  Angular radius: {props['angular_radius_rad']:.2e} rad")
            print(f"  Angular diameter: {props['angular_diameter_mas']:.3f} mas")
            
            # Test visibility calculation
            disk = star_disks[star_name]
            baseline = np.array([100.0, 0.0, 0.0])  # 100m baseline
            nu_0 = 3.76e14  # Gaia RP frequency
            visibility = disk.V(nu_0, baseline)
            print(f"  Visibility (100m): |V| = {abs(visibility):.4f}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()