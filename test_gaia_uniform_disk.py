#!/usr/bin/env python3
"""
Test script for the Gaia to UniformDisk converter function.

This script tests the create_uniform_disk_from_gaia function using the
extended_data_table_2.csv file containing stellar data.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/akim/Projects/g2')

from gaia_uniform_disk import (
    create_uniform_disk_from_gaia,
    get_star_properties,
    rp_magnitude_to_flux_density,
    angular_diameter_to_radius,
    parallax_to_distance,
    GAIA_RP_EFFECTIVE_FREQUENCY
)

def test_individual_conversions():
    """Test individual conversion functions with known values."""
    print("Testing individual conversion functions:")
    print("-" * 50)
    
    # Test data from HD 360 (first row of the CSV)
    rp_mag = 5.044772
    angular_diameter_mas = 0.906
    parallax_mas = 9.011463712285309
    
    # Test magnitude to flux conversion
    flux = rp_magnitude_to_flux_density(rp_mag)
    print(f"RP magnitude {rp_mag} → Flux density: {flux:.2e} W/m²/Hz")
    
    # Test angular diameter to radius conversion
    radius = angular_diameter_to_radius(angular_diameter_mas)
    print(f"Angular diameter {angular_diameter_mas} mas → Radius: {radius:.2e} radians")
    print(f"  (equivalent to {radius * 180 * 3600 * 1000 / np.pi:.3f} mas radius)")
    
    # Test parallax to distance conversion
    distance = parallax_to_distance(parallax_mas)
    print(f"Parallax {parallax_mas:.3f} mas → Distance: {distance:.1f} pc")
    
    print()

def test_dataframe_loading():
    """Test loading and processing the CSV file."""
    print("Testing DataFrame loading and processing:")
    print("-" * 50)
    
    try:
        # Load the CSV file
        df = pd.read_csv('extended_data_table_2.csv')
        print(f"✓ Successfully loaded CSV with {len(df)} rows")
        
        # Check required columns
        required_cols = ['Star', 'phot_rp_mean_mag', 'parallax', 'LD']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"✗ Missing columns: {missing_cols}")
            return None
        else:
            print(f"✓ All required columns present: {required_cols}")
        
        # Show first few rows of relevant columns
        print("\nFirst 5 rows of relevant data:")
        print(df[required_cols].head())
        
        return df
        
    except FileNotFoundError:
        print("✗ Error: extended_data_table_2.csv not found")
        return None
    except Exception as e:
        print(f"✗ Error loading CSV: {str(e)}")
        return None

def test_uniform_disk_creation(df):
    """Test creating UniformDisk objects from the DataFrame."""
    print("\nTesting UniformDisk creation:")
    print("-" * 50)
    
    try:
        # Create UniformDisk objects
        star_disks = create_uniform_disk_from_gaia(df)
        
        print(f"✓ Successfully created {len(star_disks)} UniformDisk objects")
        
        # Test a few specific stars
        test_stars = ['HD 360', 'HD 3750', 'HD 4211']
        
        for star_name in test_stars:
            if star_name in star_disks:
                props = get_star_properties(star_disks, star_name)
                print(f"\n{star_name}:")
                print(f"  Flux density: {props['flux_density']:.2e} W/m²/Hz")
                print(f"  Angular radius: {props['angular_radius_rad']:.2e} rad")
                print(f"  Angular diameter: {props['angular_diameter_mas']:.3f} mas")
                print(f"  Surface brightness: {props['surface_brightness']:.2e} W/m²/Hz/sr")
            else:
                print(f"✗ {star_name} not found in results")
        
        return star_disks
        
    except Exception as e:
        print(f"✗ Error creating UniformDisk objects: {str(e)}")
        return None

def test_visibility_calculation(star_disks):
    """Test visibility calculations with the UniformDisk objects."""
    print("\nTesting visibility calculations:")
    print("-" * 50)
    
    if not star_disks or 'HD 360' not in star_disks:
        print("✗ Cannot test visibility - HD 360 not available")
        return
    
    try:
        hd_360_disk = star_disks['HD 360']
        
        # Test different baselines
        baselines = [
            ([10.0, 0.0, 0.0], "10m E-W"),
            ([100.0, 0.0, 0.0], "100m E-W"),
            ([0.0, 100.0, 0.0], "100m N-S"),
            ([100.0, 100.0, 0.0], "100m diagonal")
        ]
        
        nu_0 = GAIA_RP_EFFECTIVE_FREQUENCY
        
        print(f"Visibility calculations for HD 360 at {nu_0:.2e} Hz:")
        
        for baseline_vec, description in baselines:
            baseline = np.array(baseline_vec)
            visibility = hd_360_disk.V(nu_0, baseline)
            print(f"  {description:15}: |V| = {abs(visibility):.4f}, phase = {np.angle(visibility):.3f} rad")
        
        print("✓ Visibility calculations completed successfully")
        
    except Exception as e:
        print(f"✗ Error in visibility calculation: {str(e)}")

def main():
    """Run all tests."""
    print("Gaia to UniformDisk Converter - Test Suite")
    print("=" * 60)
    
    # Test individual conversion functions
    test_individual_conversions()
    
    # Test DataFrame loading
    df = test_dataframe_loading()
    if df is None:
        print("Cannot proceed with further tests due to data loading failure")
        return
    
    # Test UniformDisk creation
    star_disks = test_uniform_disk_creation(df)
    if star_disks is None:
        print("Cannot proceed with visibility tests due to UniformDisk creation failure")
        return
    
    # Test visibility calculations
    test_visibility_calculation(star_disks)
    
    print("\n" + "=" * 60)
    print("Test suite completed!")

if __name__ == "__main__":
    main()