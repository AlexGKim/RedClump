"""
Simple Uniform Disk vs SATLAS Comparison
========================================

This script compares visibility calculations between UniformDisk and SATLAS RadialGrid2
for HD 360, focusing on comparing the intensity profile and |V|^2 vs baseline without 
Fisher matrix calculations.

The comparison includes:
1. Loading HD 360 data from Gaia catalog
2. Creating UniformDisk model from Gaia photometry
3. Creating RadialGrid2 model from SATLAS limb-darkening data
4. Comparing intensity profiles
5. Comparing |V|² vs baseline length
6. Plotting results for visual comparison

Author: Generated for RedClump project
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from pathlib import Path
import sys

# Add g2 to path
sys.path.append('/Users/akim/Projects/g2')

# Import required modules
from gaia_uniform_disk import create_uniform_disk_from_gaia, get_star_properties
from gaia_satlas import create_radial_grid_from_satlas, get_radial_grid_properties
from gaia_zeropoint import GAIA_RP_EFFECTIVE_FREQUENCY

# Physical constants
SPEED_OF_LIGHT = 2.99792458e8  # m/s
MAS_TO_RAD = np.pi / (180 * 3600 * 1000)  # milliarcseconds to radians


def create_uniform_disk_model():
    """
    Create UniformDisk model for HD 360 from Gaia data.
    
    Returns
    -------
    UniformDisk
        Configured UniformDisk object for HD 360
    dict
        HD 360 properties dictionary
    """
    print("\nCreating UniformDisk model from Gaia data...")
    
    # Load HD 360 data directly from CSV
    try:
        df = pd.read_csv('extended_data_table_2.csv')
        hd360_row = df[df['Star'] == 'HD 360']
        
        if hd360_row.empty:
            raise ValueError("HD 360 not found in data table")
        
        # Create UniformDisk objects using the full dataframe but filter for HD 360
        star_disks = create_uniform_disk_from_gaia(hd360_row)
        
        if 'HD 360' not in star_disks:
            raise ValueError("Failed to create UniformDisk for HD 360")
        
        uniform_disk = star_disks['HD 360']
        
        # Extract HD 360 data for return
        hd360_data = {
            'star_name': 'HD 360',
            'phot_g_mean_mag': hd360_row['phot_g_mean_mag'].iloc[0],
            'phot_bp_mean_mag': hd360_row['phot_bp_mean_mag'].iloc[0],
            'phot_rp_mean_mag': hd360_row['phot_rp_mean_mag'].iloc[0],
            'parallax': hd360_row['parallax'].iloc[0],
            'LD': hd360_row['LD'].iloc[0]  # Angular diameter in mas
        }
        
        # Get properties at RP band frequency (commonly used for comparison)
        props = get_star_properties(star_disks, 'HD 360', GAIA_RP_EFFECTIVE_FREQUENCY)
        
        print(f"UniformDisk created successfully:")
        print(f"  Angular radius: {props['angular_radius_rad']:.2e} rad")
        print(f"  Angular diameter: {props['angular_diameter_mas']:.3f} mas")
        print(f"  Flux density (RP): {props['flux_density']:.2e} W/m²/Hz")
        print(f"  Surface brightness (RP): {props['surface_brightness']:.2e} W/m²/Hz/sr")
        
        return uniform_disk, hd360_data
        
    except Exception as e:
        print(f"Error creating UniformDisk model: {str(e)}")
        raise

def create_satlas_model(hd360_row):
    """
    Create RadialGrid2 model from SATLAS limb-darkening data.
    
    Parameters
    ----------
    hd360_row : pd.Series
        HD 360 data row containing Gaia magnitudes
    
    Returns
    -------
    RadialGrid2
        Configured RadialGrid2 object from SATLAS data
    dict
        SATLAS model properties
    """
    print("\nCreating SATLAS RadialGrid2 model...")
    
    # Path to SATLAS data file
    satlas_file = 'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt'
    
    if not Path(satlas_file).exists():
        raise FileNotFoundError(f"SATLAS data file not found: {satlas_file}")
    
    # Create RadialGrid2 from SATLAS data with Gaia magnitudes
    radial_grid = create_radial_grid_from_satlas(satlas_file, hd360_row)
    
    # Get properties
    props = get_radial_grid_properties(radial_grid)
    
    print(f"SATLAS RadialGrid2 created successfully:")
    print(f"  Max radius: {props['max_radius_mas']:.3f} mas")
    print(f"  Wavelength range: {props['wavelengths_angstrom']} Å")
    print(f"  Bands: {props['bands']}")
    print(f"  Size parameter: {props['s']}")
    
    return radial_grid, props

def compare_intensity_profiles(uniform_disk, radial_grid, hd360_data, pdf_pages, frequency=None):
    """
    Compare intensity profiles between UniformDisk and SATLAS models.
    
    Parameters
    ----------
    uniform_disk : UniformDisk
        UniformDisk model
    radial_grid : RadialGrid2
        SATLAS RadialGrid2 model
    hd360_data : dict
        HD 360 properties
    frequency : float, optional
        Frequency for comparison. Default uses RP band frequency.
    """
    if frequency is None:
        frequency = GAIA_RP_EFFECTIVE_FREQUENCY
    
    print(f"\nComparing intensity profiles at {frequency:.2e} Hz...")
    
    # Create radial grid for comparison
    max_radius_mas = hd360_data['LD']  # Use angular diameter from Gaia
    max_radius_rad = max_radius_mas * MAS_TO_RAD / 2.0  # Convert diameter to radius
    
    # Radial points from 0 to 1.5 times the stellar radius
    r_points = np.linspace(0, 1.5 * max_radius_rad, 100)
    
    # Calculate intensity profiles
    uniform_intensities = []
    
    for r in r_points:
        # For intensity profiles, we evaluate along a radial line (theta_x = r, theta_y = 0)
        n_hat = np.array([r, 0.0])
        
        # UniformDisk intensity
        uniform_intensity = uniform_disk.intensity(frequency, n_hat)
        uniform_intensities.append(uniform_intensity)
    
    # Convert to arrays
    uniform_intensities = np.array(uniform_intensities)
    
    # For SATLAS, we'll create a normalized profile based on the limb-darkening data
    # This is a simplified approach - extract the radial profile from SATLAS data
    try:
        # Get the radial profile data from SATLAS
        # Access the internal data structure
        satlas_radii_rad = radial_grid.p_rays  # Already in radians
        
        # Convert frequency to wavelength for SATLAS band selection
        wavelength_m = SPEED_OF_LIGHT / frequency
        wavelength_angstrom = wavelength_m * 1e10
        
        # Find closest wavelength in SATLAS data (R band is closest to RP)
        satlas_wavelengths = np.array([4450, 5510, 6580, 8060, 16300, 21900])  # Angstroms
        closest_idx = np.argmin(np.abs(satlas_wavelengths - wavelength_angstrom))
        
        # Get the intensity profile for the closest band
        satlas_intensity_profile = radial_grid.I_nu_p[closest_idx, :]
        
        # Interpolate SATLAS profile to our radial grid
        satlas_intensities = np.interp(r_points, satlas_radii_rad, satlas_intensity_profile)
        
        # Normalize SATLAS intensities to match the scale (they are I/I0 ratios)
        # Scale by the uniform disk central intensity
        central_uniform_intensity = uniform_intensities[0]
        satlas_intensities = satlas_intensities * central_uniform_intensity
        
    except Exception as e:
        print(f"Warning: Could not extract SATLAS intensity profile: {e}")
        # Fallback to zeros
        satlas_intensities = np.zeros_like(uniform_intensities)
    
    # Convert radial points to milliarcseconds for plotting
    r_mas = r_points / MAS_TO_RAD
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(r_mas, uniform_intensities, 'b-', linewidth=2, label='UniformDisk')
    plt.plot(r_mas, satlas_intensities, 'r--', linewidth=2, label='SATLAS RadialGrid2')
    plt.axvline(x=max_radius_mas/2, color='gray', linestyle=':', alpha=0.7, label='Stellar radius')
    
    plt.xlabel('Radial distance (mas)')
    plt.ylabel('Specific intensity (W m⁻² Hz⁻¹ sr⁻¹)')
    plt.title(f'Intensity Profile Comparison - HD 360\nFrequency: {frequency:.2e} Hz')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.5 * max_radius_mas/2)
    pdf_pages.savefig(bbox_inches='tight')
    plt.close()
    
    return r_mas, uniform_intensities, satlas_intensities

def compare_visibility_vs_baseline(uniform_disk, radial_grid, pdf_pages, frequency=None, max_baseline=1000):
    """
    Compare |V|² vs baseline length between UniformDisk and SATLAS models.
    
    Parameters
    ----------
    uniform_disk : UniformDisk
        UniformDisk model
    radial_grid : RadialGrid2
        SATLAS RadialGrid2 model
    frequency : float, optional
        Frequency for comparison. Default uses RP band frequency.
    max_baseline : float, optional
        Maximum baseline length in meters. Default is 1000m.
    """
    if frequency is None:
        frequency = GAIA_RP_EFFECTIVE_FREQUENCY
    
    print(f"\nComparing |V|² vs baseline at {frequency:.2e} Hz...")
    
    # Create baseline length array
    baseline_lengths = np.logspace(0, np.log10(max_baseline), 50)  # 1m to max_baseline
    
    # Calculate visibilities
    uniform_vis_squared = []
    satlas_vis_squared = []
    
    for B in baseline_lengths:
        # East-West baseline
        baseline = np.array([B, 0.0, 0.0])
        
        # UniformDisk visibility
        uniform_vis = uniform_disk.V(frequency, baseline)
        uniform_vis_squared.append(abs(uniform_vis)**2)
        
        # SATLAS visibility
        satlas_vis = radial_grid.V(frequency, baseline)
        satlas_vis_squared.append(abs(satlas_vis)**2)
    
    # Convert to arrays
    uniform_vis_squared = np.array(uniform_vis_squared)
    satlas_vis_squared = np.array(satlas_vis_squared)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Linear scale plot
    plt.subplot(2, 1, 1)
    plt.plot(baseline_lengths, uniform_vis_squared, 'b-', linewidth=2, label='UniformDisk')
    plt.plot(baseline_lengths, satlas_vis_squared, 'r--', linewidth=2, label='SATLAS RadialGrid2')
    plt.xlabel('Baseline length (m)')
    plt.ylabel('|V|²')
    plt.title(f'Visibility Amplitude Squared vs Baseline - HD 360\nFrequency: {frequency:.2e} Hz')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_baseline)
    plt.ylim(0, 1.1)
    
    # Log scale plot
    plt.subplot(2, 1, 2)
    plt.semilogx(baseline_lengths, uniform_vis_squared, 'b-', linewidth=2, label='UniformDisk')
    plt.semilogx(baseline_lengths, satlas_vis_squared, 'r--', linewidth=2, label='SATLAS RadialGrid2')
    plt.xlabel('Baseline length (m)')
    plt.ylabel('|V|²')
    plt.title('Visibility Amplitude Squared vs Baseline (Log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, max_baseline)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    pdf_pages.savefig(bbox_inches='tight')
    plt.close()
    
    return baseline_lengths, uniform_vis_squared, satlas_vis_squared

def calculate_visibility_differences(baseline_lengths, uniform_vis_squared, satlas_vis_squared, pdf_pages):
    """
    Calculate and analyze differences between visibility models.
    
    Parameters
    ----------
    baseline_lengths : array_like
        Baseline lengths in meters
    uniform_vis_squared : array_like
        |V|² values for UniformDisk model
    satlas_vis_squared : array_like
        |V|² values for SATLAS model
    """
    print("\nAnalyzing visibility differences...")
    
    # Calculate absolute and relative differences
    abs_diff = np.abs(uniform_vis_squared - satlas_vis_squared)
    rel_diff = abs_diff / np.maximum(uniform_vis_squared, 1e-10)  # Avoid division by zero
    
    # Find maximum differences
    max_abs_diff_idx = np.argmax(abs_diff)
    max_rel_diff_idx = np.argmax(rel_diff)
    
    print(f"Maximum absolute difference: {abs_diff[max_abs_diff_idx]:.6f}")
    print(f"  at baseline: {baseline_lengths[max_abs_diff_idx]:.1f} m")
    print(f"Maximum relative difference: {rel_diff[max_rel_diff_idx]:.2%}")
    print(f"  at baseline: {baseline_lengths[max_rel_diff_idx]:.1f} m")
    
    # Calculate RMS differences
    rms_abs_diff = np.sqrt(np.mean(abs_diff**2))
    rms_rel_diff = np.sqrt(np.mean(rel_diff**2))
    
    print(f"RMS absolute difference: {rms_abs_diff:.6f}")
    print(f"RMS relative difference: {rms_rel_diff:.2%}")
    
    # Plot difference analysis
    plt.figure(figsize=(12, 8))
    
    # Absolute differences
    plt.subplot(2, 1, 1)
    plt.semilogx(baseline_lengths, abs_diff, 'g-', linewidth=2)
    plt.xlabel('Baseline length (m)')
    plt.ylabel('|V|² absolute difference')
    plt.title('Absolute Difference: |V²_uniform - V²_satlas|')
    plt.grid(True, alpha=0.3)
    
    # Relative differences
    plt.subplot(2, 1, 2)
    plt.semilogx(baseline_lengths, rel_diff * 100, 'orange', linewidth=2)
    plt.xlabel('Baseline length (m)')
    plt.ylabel('Relative difference (%)')
    plt.title('Relative Difference: |V²_uniform - V²_satlas| / V²_uniform')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf_pages.savefig(bbox_inches='tight')
    plt.close()
    
    return abs_diff, rel_diff

def main():
    """
    Main function to run the complete comparison.
    """
    print("Simple Uniform Disk vs SATLAS Comparison")
    print("=" * 50)
    
    try:
        # Load HD 360 data
        df = pd.read_csv('extended_data_table_2.csv')
        hd360_row = df[df['Star'] == 'HD 360'].iloc[0]
        
        # Create models
        uniform_disk, hd360_data = create_uniform_disk_model()
        radial_grid, satlas_props = create_satlas_model(hd360_row)
        
        # Set comparison frequency (RP band)
        frequency = GAIA_RP_EFFECTIVE_FREQUENCY
        wavelength_angstrom = SPEED_OF_LIGHT / frequency * 1e10
        
        print(f"\nComparison frequency: {frequency:.2e} Hz")
        print(f"Corresponding wavelength: {wavelength_angstrom:.1f} Å")
        
        # Create PDF file for all plots
        pdf_filename = 'simple_uniform_vs_satlas_HD_360.pdf'
        with PdfPages(pdf_filename) as pdf_pages:
            # Compare intensity profiles
            print("\n" + "="*30)
            print("INTENSITY PROFILE COMPARISON")
            print("="*30)
            r_mas, uniform_intensities, satlas_intensities = compare_intensity_profiles(
                uniform_disk, radial_grid, hd360_data, pdf_pages, frequency
            )
            
            # Compare visibility vs baseline
            print("\n" + "="*30)
            print("VISIBILITY COMPARISON")
            print("="*30)
            baseline_lengths, uniform_vis_squared, satlas_vis_squared = compare_visibility_vs_baseline(
                uniform_disk, radial_grid, pdf_pages, frequency, max_baseline=1000
            )
            
            # Analyze differences
            print("\n" + "="*30)
            print("DIFFERENCE ANALYSIS")
            print("="*30)
            abs_diff, rel_diff = calculate_visibility_differences(
                baseline_lengths, uniform_vis_squared, satlas_vis_squared, pdf_pages
            )
        
        print(f"\nComparison completed successfully!")
        print(f"All plots saved to: {pdf_filename}")
        print(f"Models compared:")
        print(f"  - UniformDisk: {hd360_data['LD']:.3f} mas diameter")
        print(f"  - SATLAS RadialGrid2: {satlas_props['max_radius_mas']:.3f} mas max radius")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        raise

if __name__ == "__main__":
    main()