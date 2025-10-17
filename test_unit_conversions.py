#!/usr/bin/env python3
"""
Unit tests for the Gaia to UniformDisk converter functions.

This module contains comprehensive unit tests for all conversion functions
to ensure accuracy and reliability.
"""

import unittest
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/akim/Projects/g2')

from gaia_uniform_disk import (
    rp_magnitude_to_flux_density,
    angular_diameter_to_radius,
    parallax_to_distance,
    validate_dataframe,
    create_uniform_disk_from_gaia,
    GAIA_RP_ZERO_POINT_MAG,
    GAIA_RP_ZERO_POINT_FLUX,
    MAS_TO_RAD
)

class TestConversionFunctions(unittest.TestCase):
    """Test individual conversion functions."""
    
    def test_rp_magnitude_to_flux_density(self):
        """Test RP magnitude to flux density conversion."""
        # Test with zero-point magnitude (should give zero-point flux)
        flux = rp_magnitude_to_flux_density(GAIA_RP_ZERO_POINT_MAG)
        self.assertAlmostEqual(flux, GAIA_RP_ZERO_POINT_FLUX, places=10)
        
        # Test with brighter magnitude (should give higher flux)
        brighter_mag = GAIA_RP_ZERO_POINT_MAG - 1.0
        brighter_flux = rp_magnitude_to_flux_density(brighter_mag)
        expected_flux = GAIA_RP_ZERO_POINT_FLUX * 10**(0.4 * 1.0)
        self.assertAlmostEqual(brighter_flux, expected_flux, places=10)
        
        # Test with array input
        mags = np.array([5.0, 6.0, 7.0])
        fluxes = rp_magnitude_to_flux_density(mags)
        self.assertEqual(len(fluxes), 3)
        self.assertTrue(fluxes[0] > fluxes[1] > fluxes[2])  # Brighter = higher flux
    
    def test_angular_diameter_to_radius(self):
        """Test angular diameter to radius conversion."""
        # Test known conversion
        diameter_mas = 1000.0  # 1 arcsecond
        radius_rad = angular_diameter_to_radius(diameter_mas)
        expected_radius = (1000.0 / 2.0) * MAS_TO_RAD
        self.assertAlmostEqual(radius_rad, expected_radius, places=12)
        
        # Test that radius is half of diameter
        diameter_mas = 2.0
        radius_rad = angular_diameter_to_radius(diameter_mas)
        # Convert back to check
        radius_mas = radius_rad / MAS_TO_RAD
        self.assertAlmostEqual(radius_mas, diameter_mas / 2.0, places=10)
        
        # Test array input
        diameters = np.array([1.0, 2.0, 3.0])
        radii = angular_diameter_to_radius(diameters)
        self.assertEqual(len(radii), 3)
        self.assertAlmostEqual(radii[1], 2 * radii[0], places=10)
    
    def test_parallax_to_distance(self):
        """Test parallax to distance conversion."""
        # Test known conversion: 1 mas parallax = 1000 pc
        parallax_mas = 1.0
        distance_pc = parallax_to_distance(parallax_mas)
        self.assertAlmostEqual(distance_pc, 1000.0, places=10)
        
        # Test 10 mas parallax = 100 pc
        parallax_mas = 10.0
        distance_pc = parallax_to_distance(parallax_mas)
        self.assertAlmostEqual(distance_pc, 100.0, places=10)
        
        # Test that distance is inversely proportional to parallax
        parallax1, parallax2 = 5.0, 10.0
        dist1 = parallax_to_distance(parallax1)
        dist2 = parallax_to_distance(parallax2)
        self.assertAlmostEqual(dist1, 2 * dist2, places=10)
        
        # Test error handling for negative parallax
        with self.assertRaises(ValueError):
            parallax_to_distance(-1.0)
        
        with self.assertRaises(ValueError):
            parallax_to_distance(0.0)
    
    def test_validate_dataframe(self):
        """Test DataFrame validation function."""
        # Test valid DataFrame
        valid_df = pd.DataFrame({
            'Star': ['HD 1', 'HD 2'],
            'phot_rp_mean_mag': [5.0, 6.0],
            'parallax': [10.0, 20.0],
            'LD': [1.0, 2.0]
        })
        
        # Should not raise any exception
        validate_dataframe(valid_df, 'Star', 'phot_rp_mean_mag', 'parallax', 'LD')
        
        # Test missing column
        invalid_df = valid_df.drop('parallax', axis=1)
        with self.assertRaises(ValueError):
            validate_dataframe(invalid_df, 'Star', 'phot_rp_mean_mag', 'parallax', 'LD')
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            validate_dataframe(empty_df, 'Star', 'phot_rp_mean_mag', 'parallax', 'LD')
        
        # Test non-DataFrame input
        with self.assertRaises(TypeError):
            validate_dataframe("not a dataframe", 'Star', 'phot_rp_mean_mag', 'parallax', 'LD')

class TestIntegration(unittest.TestCase):
    """Test the complete integration with real data."""
    
    def setUp(self):
        """Set up test data."""
        # Create a small test DataFrame with known values
        self.test_df = pd.DataFrame({
            'Star': ['Test Star 1', 'Test Star 2'],
            'phot_rp_mean_mag': [5.0, 6.0],
            'parallax': [10.0, 20.0],  # 100 pc and 50 pc
            'LD': [1.0, 2.0]  # 1 and 2 mas diameter
        })
    
    def test_create_uniform_disk_from_gaia(self):
        """Test the main function with test data."""
        star_disks = create_uniform_disk_from_gaia(self.test_df)
        
        # Should create 2 UniformDisk objects
        self.assertEqual(len(star_disks), 2)
        self.assertIn('Test Star 1', star_disks)
        self.assertIn('Test Star 2', star_disks)
        
        # Check that objects have correct types and properties
        disk1 = star_disks['Test Star 1']
        disk2 = star_disks['Test Star 2']
        
        # Test Star 2 should have lower flux (fainter magnitude)
        self.assertGreater(disk1.flux_density, disk2.flux_density)
        
        # Test Star 2 should have larger radius (larger diameter)
        self.assertGreater(disk2.radius, disk1.radius)
        
        # Test that radii are reasonable (should be very small)
        self.assertLess(disk1.radius, 1e-6)  # Less than 1 microrad
        self.assertGreater(disk1.radius, 1e-12)  # Greater than 1 picorad
    
    def test_with_invalid_data(self):
        """Test handling of invalid data."""
        # Test with negative parallax
        invalid_df = pd.DataFrame({
            'Star': ['Bad Star'],
            'phot_rp_mean_mag': [5.0],
            'parallax': [-1.0],  # Invalid
            'LD': [1.0]
        })
        
        star_disks = create_uniform_disk_from_gaia(invalid_df)
        self.assertEqual(len(star_disks), 0)  # Should skip invalid star
        
        # Test with zero angular diameter
        zero_diameter_df = pd.DataFrame({
            'Star': ['Zero Diameter Star'],
            'phot_rp_mean_mag': [5.0],
            'parallax': [10.0],
            'LD': [0.0]  # Invalid - zero diameter
        })
        
        star_disks = create_uniform_disk_from_gaia(zero_diameter_df)
        self.assertEqual(len(star_disks), 0)  # Should skip invalid star

class TestPhysicalRealism(unittest.TestCase):
    """Test that results are physically realistic."""
    
    def test_flux_density_range(self):
        """Test that flux densities are in reasonable range."""
        # Typical stellar flux densities should be much less than 1 W/m²/Hz
        # but much greater than 1e-30 W/m²/Hz
        mag = 5.0  # Bright star
        flux = rp_magnitude_to_flux_density(mag)
        self.assertLess(flux, 1.0)
        self.assertGreater(flux, 1e-30)
    
    def test_angular_size_range(self):
        """Test that angular sizes are in reasonable range for stars."""
        # Stellar angular diameters are typically 0.1 to 10 milliarcseconds
        diameter_mas = 1.0  # 1 mas
        radius_rad = angular_diameter_to_radius(diameter_mas)
        
        # Convert to arcseconds for checking
        radius_arcsec = radius_rad * 180 * 3600 / np.pi
        self.assertLess(radius_arcsec, 0.01)  # Less than 0.01 arcsec
        self.assertGreater(radius_arcsec, 1e-6)  # Greater than 1 microarcsec
    
    def test_distance_range(self):
        """Test that distances are in reasonable range."""
        # For nearby stars, parallax should give distances of 10-1000 pc
        parallax_mas = 10.0  # 10 mas parallax
        distance_pc = parallax_to_distance(parallax_mas)
        self.assertEqual(distance_pc, 100.0)  # Should be exactly 100 pc
        
        # Very nearby star
        parallax_mas = 100.0  # 100 mas parallax (very close)
        distance_pc = parallax_to_distance(parallax_mas)
        self.assertEqual(distance_pc, 10.0)  # Should be exactly 10 pc

def run_tests():
    """Run all unit tests."""
    print("Running unit tests for Gaia to UniformDisk converter...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConversionFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicalRealism))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✓ All tests passed successfully!")
    else:
        print(f"✗ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)