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
    g_magnitude_to_flux_density,
    bp_magnitude_to_flux_density,
    rp_magnitude_to_flux_density,
    angular_diameter_to_radius,
    parallax_to_distance,
    GAIA_G_EFFECTIVE_FREQUENCY,
    GAIA_BP_EFFECTIVE_FREQUENCY,
    GAIA_RP_EFFECTIVE_FREQUENCY
)

def test_individual_conversions():
    """Test individual conversion functions with known values."""
    print("Testing individual conversion functions:")
    print("-" * 50)
    
    # Test data from HD 360 (first row of the CSV)
    g_mag = 5.7160897
    bp_mag = 6.227563
    rp_mag = 5.044772
    angular_diameter_mas = 0.906
    parallax_mas = 9.011463712285309
    
    # Test magnitude to flux conversions for all bands
    flux_g = g_magnitude_to_flux_density(g_mag)
    flux_bp = bp_magnitude_to_flux_density(bp_mag)
    flux_rp = rp_magnitude_to_flux_density(rp_mag)
    
    print(f"G magnitude {g_mag} → Flux density: {flux_g:.2e} W/m²/Hz")
    print(f"BP magnitude {bp_mag} → Flux density: {flux_bp:.2e} W/m²/Hz")
    print(f"RP magnitude {rp_mag} → Flux density: {flux_rp:.2e} W/m²/Hz")
    
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
        required_cols = ['Star', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax', 'LD']
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
                print(f"\n{star_name}:")
                
                # Show properties at different frequencies
                for band_name, frequency in [('G', GAIA_G_EFFECTIVE_FREQUENCY),
                                            ('BP', GAIA_BP_EFFECTIVE_FREQUENCY),
                                            ('RP', GAIA_RP_EFFECTIVE_FREQUENCY)]:
                    props = get_star_properties(star_disks, star_name, frequency)
                    print(f"  {band_name} band:")
                    print(f"    Flux density: {props['flux_density']:.2e} W/m²/Hz")
                    print(f"    Surface brightness: {props['surface_brightness']:.2e} W/m²/Hz/sr")
                
                # Show geometric properties (frequency-independent)
                props = get_star_properties(star_disks, star_name)
                print(f"  Angular radius: {props['angular_radius_rad']:.2e} rad")
                print(f"  Angular diameter: {props['angular_diameter_mas']:.3f} mas")
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
        
        print(f"Visibility calculations for HD 360:")
        
        # Test at different frequencies
        for band_name, nu_0 in [('G', GAIA_G_EFFECTIVE_FREQUENCY),
                               ('BP', GAIA_BP_EFFECTIVE_FREQUENCY),
                               ('RP', GAIA_RP_EFFECTIVE_FREQUENCY)]:
            print(f"\n  {band_name} band ({nu_0:.2e} Hz):")
            
            for baseline_vec, description in baselines:
                baseline = np.array(baseline_vec)
                visibility = hd_360_disk.V(nu_0, baseline)
                print(f"    {description:15}: |V| = {abs(visibility):.4f}, phase = {np.angle(visibility):.3f} rad")
        
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