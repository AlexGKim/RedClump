#!/usr/bin/env python3
"""
Compare UniformDisk vs RadialGrid2 (SATLAS) for HD 360
=======================================================

This script creates both UniformDisk and RadialGrid2 objects for HD 360
and compares:
1. Intensity profiles (radial structure) using intensity() method
2. Visibility squared as a function of baseline
3. Flux densities at different wavelengths
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/akim/Projects/g2')

from gaia_uniform_disk import create_uniform_disk_from_gaia
from gaia_satlas import create_radial_grid_from_gaia, SATLAS_BANDS, SPEED_OF_LIGHT, MAS_TO_RAD
from gaia_zeropoint import (
    GAIA_G_EFFECTIVE_WAVELENGTH,
    GAIA_BP_EFFECTIVE_WAVELENGTH,
    GAIA_RP_EFFECTIVE_WAVELENGTH,
    GAIA_G_EFFECTIVE_FREQUENCY,
    GAIA_BP_EFFECTIVE_FREQUENCY,
    GAIA_RP_EFFECTIVE_FREQUENCY
)

def compare_flux_densities(star_name, uniform_disk, radial_grid):
    """Compare flux densities at Gaia wavelengths."""
    print(f"\n{'='*60}")
    print(f"Flux Density Comparison for {star_name}")
    print(f"{'='*60}")
    
    bands = [
        ('G', GAIA_G_EFFECTIVE_WAVELENGTH, GAIA_G_EFFECTIVE_FREQUENCY),
        ('BP', GAIA_BP_EFFECTIVE_WAVELENGTH, GAIA_BP_EFFECTIVE_FREQUENCY),
        ('RP', GAIA_RP_EFFECTIVE_WAVELENGTH, GAIA_RP_EFFECTIVE_FREQUENCY)
    ]
    
    for band_name, wavelength, frequency in bands:
        uniform_flux = uniform_disk._interpolate_flux_density(frequency)
        radial_flux = radial_grid.specific_flux(frequency)
        
        print(f"\n{band_name} band ({wavelength*1e9:.1f} nm, {frequency:.2e} Hz):")
        print(f"  UniformDisk flux:  {uniform_flux:.3e} W/m²/Hz")
        print(f"  RadialGrid2 flux:  {radial_flux:.3e} W/m²/Hz")
        
        if uniform_flux > 0:
            ratio = radial_flux / uniform_flux
            print(f"  Ratio (RG/UD):     {ratio:.4f}")

def calculate_visibility_squared_comparison(uniform_disk, radial_grid, baseline_lengths, wavelengths, frequencies, band_names):
    """Calculate |V|^2 for both models across all baselines and wavelengths."""
    results = {'uniform': [], 'radial': []}
    
    for wavelength, frequency, band_name in zip(wavelengths, frequencies, band_names):
        uniform_vis_squared = []
        radial_vis_squared = []
        baseline_values = []
        
        for baseline_length in baseline_lengths:
            try:
                # Create E-W baseline
                baseline = np.array([baseline_length, 0.0, 0.0])
                
                # Calculate visibility squared for both models
                uniform_v2 = uniform_disk.V_squared(frequency, baseline)
                radial_v2 = radial_grid.V_squared(frequency, baseline)
                
                # Store valid results
                if not np.isnan(uniform_v2) and not np.isnan(radial_v2):
                    uniform_vis_squared.append(uniform_v2)
                    radial_vis_squared.append(radial_v2)
                    baseline_values.append(baseline_length)
                
            except Exception as e:
                print(f"Warning: Error at baseline {baseline_length}m, {wavelength*1e9:.1f}nm: {str(e)}")
                continue
        
        results['uniform'].append({
            'wavelength_nm': wavelength * 1e9,
            'band_name': band_name,
            'baselines': np.array(baseline_values),
            'visibility_squared': np.array(uniform_vis_squared)
        })
        
        results['radial'].append({
            'wavelength_nm': wavelength * 1e9,
            'band_name': band_name,
            'baselines': np.array(baseline_values),
            'visibility_squared': np.array(radial_vis_squared)
        })
    
    return results

def plot_comparison(star_name, results, uniform_disk, radial_grid, baseline_lengths):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'UniformDisk vs RadialGrid2 (SATLAS) Comparison - {star_name}',
                 fontsize=16, fontweight='bold')
    
    # Colors for each wavelength
    colors = ['blue', 'green', 'red']
    
    # Determine baseline range for axis labels
    baseline_min = baseline_lengths.min()
    baseline_max = baseline_lengths.max()
    baseline_label = f'Baseline Length ({baseline_min:.0f}-{baseline_max:.0f} m)'
    
    # Plot 1: UniformDisk |V|^2 vs baseline
    ax1 = fig.add_subplot(gs[0, 0])
    for i, result in enumerate(results['uniform']):
        if len(result['baselines']) > 0:
            ax1.semilogy(result['baselines'], result['visibility_squared'],
                        'o-', color=colors[i], markersize=4, linewidth=2,
                        label=f'{result["band_name"]} ({result["wavelength_nm"]:.1f} nm)')
    
    ax1.set_xlabel(baseline_label, fontsize=11)
    ax1.set_ylabel('Visibility Squared |V|²', fontsize=11)
    ax1.set_title('UniformDisk Model', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Plot 2: RadialGrid2 |V|^2 vs baseline
    ax2 = fig.add_subplot(gs[0, 1])
    for i, result in enumerate(results['radial']):
        if len(result['baselines']) > 0:
            ax2.semilogy(result['baselines'], result['visibility_squared'],
                        'o-', color=colors[i], markersize=4, linewidth=2,
                        label=f'{result["band_name"]} ({result["wavelength_nm"]:.1f} nm)')
    
    ax2.set_xlabel(baseline_label, fontsize=11)
    ax2.set_ylabel('Visibility Squared |V|²', fontsize=11)
    ax2.set_title('RadialGrid2 (SATLAS) Model', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Ratio of |V|^2 (RadialGrid2 / UniformDisk)
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(len(results['uniform'])):
        if len(results['uniform'][i]['baselines']) > 0:
            ratio = results['radial'][i]['visibility_squared'] / results['uniform'][i]['visibility_squared']
            ax3.plot(results['uniform'][i]['baselines'], ratio,
                    'o-', color=colors[i], markersize=4, linewidth=2,
                    label=f'{results["uniform"][i]["band_name"]} ({results["uniform"][i]["wavelength_nm"]:.1f} nm)')
    
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel(baseline_label, fontsize=11)
    ax3.set_ylabel('Ratio: RadialGrid2 / UniformDisk', fontsize=11)
    ax3.set_title('|V|² Ratio', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Plot 4: UniformDisk intensity profile using intensity() method
    ax4 = fig.add_subplot(gs[1, 0])
    radius_mas = (uniform_disk.radius / MAS_TO_RAD)
    
    # Create radial points in mas
    radii_mas = np.linspace(0, radius_mas * 1.2, 200)
    radii_rad = radii_mas * MAS_TO_RAD
    
    # Calculate intensity at G band using intensity() method
    # intensity(nu, n_hat) expects a direction vector [nx, ny, nz]
    # For radial profile, use direction vectors along x-axis at different angles
    intensities_g = []
    for r in radii_rad:
        # Create direction vector: n_hat = [sin(theta), 0, cos(theta)]
        # where theta is the angle from z-axis corresponding to radial distance r
        # For small angles: sin(theta) ≈ r, cos(theta) ≈ 1
        n_hat = np.array([r, 0.0, 1.0])
        # Normalize
        n_hat = n_hat / np.linalg.norm(n_hat)
        intensity = uniform_disk.intensity(GAIA_G_EFFECTIVE_FREQUENCY, n_hat)
        intensities_g.append(intensity)
    
    intensities_g = np.array(intensities_g)
    
    ax4.plot(radii_mas, intensities_g, 'b-', linewidth=2,
             label=f'G band ({GAIA_G_EFFECTIVE_WAVELENGTH*1e9:.1f} nm)')
    ax4.axvline(x=radius_mas, color='red', linestyle='--', linewidth=1,
                label=f'Radius = {radius_mas:.3f} mas')
    ax4.set_xlabel('Radial Distance (mas)', fontsize=11)
    ax4.set_ylabel('Intensity (W/m²/Hz/sr)', fontsize=11)
    ax4.set_title('UniformDisk Intensity Profile', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, radius_mas * 1.2)
    ax4.legend(fontsize=9)
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 5: RadialGrid2 intensity profiles using intensity() method
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Get radial grid points from RadialGrid2
    p_rays = radial_grid.p_rays
    p_rays_mas = p_rays / MAS_TO_RAD
    
    # Calculate intensity at different wavelengths using intensity() method
    # Plot for Gaia bands
    gaia_bands = [
        ('G', GAIA_G_EFFECTIVE_FREQUENCY, 'blue'),
        ('BP', GAIA_BP_EFFECTIVE_FREQUENCY, 'green'),
        ('RP', GAIA_RP_EFFECTIVE_FREQUENCY, 'red')
    ]
    
    for band_name, frequency, color in gaia_bands:
        intensities = []
        for p in p_rays:
            # Create direction vector for this radial distance
            n_hat = np.array([p, 0.0, 1.0])
            n_hat = n_hat / np.linalg.norm(n_hat)
            intensity = radial_grid.intensity(frequency, n_hat)
            intensities.append(intensity)
        intensities = np.array(intensities)
        ax5.plot(p_rays_mas, intensities, '-', color=color, linewidth=2,
                label=f'{band_name} band')
    
    ax5.set_xlabel('Radial Distance (mas)', fontsize=11)
    ax5.set_ylabel('Intensity (W/m²/Hz/sr)', fontsize=11)
    ax5.set_title('RadialGrid2 (SATLAS) Intensity Profile', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, p_rays_mas.max() * 1.05)
    ax5.legend(fontsize=9)
    ax5.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 6: Comparison statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate statistics
    stats_text = f"Comparison Statistics\n{'='*30}\n\n"
    stats_text += f"Star: {star_name}\n\n"
    stats_text += f"UniformDisk:\n"
    stats_text += f"  Radius: {radius_mas:.3f} mas\n"
    stats_text += f"  Radius: {uniform_disk.radius:.2e} rad\n\n"
    
    stats_text += f"RadialGrid2:\n"
    stats_text += f"  Max radius: {p_rays_mas.max():.3f} mas\n"
    stats_text += f"  Max radius: {p_rays.max():.2e} rad\n"
    stats_text += f"  Size param: {radial_grid.s}\n\n"
    
    # Add flux comparison
    stats_text += f"Flux at G band:\n"
    uniform_flux_g = uniform_disk._interpolate_flux_density(GAIA_G_EFFECTIVE_FREQUENCY)
    radial_flux_g = radial_grid.specific_flux(GAIA_G_EFFECTIVE_FREQUENCY)
    stats_text += f"  UD: {uniform_flux_g:.2e} W/m²/Hz\n"
    stats_text += f"  RG: {radial_flux_g:.2e} W/m²/Hz\n"
    if uniform_flux_g > 0:
        stats_text += f"  Ratio: {radial_flux_g/uniform_flux_g:.4f}\n\n"
    
    # Add intensity comparison at center
    stats_text += f"Intensity at center (G band):\n"
    n_hat_center = np.array([0.0, 0.0, 1.0])  # Looking straight down z-axis
    uniform_int_center = uniform_disk.intensity(GAIA_G_EFFECTIVE_FREQUENCY, n_hat_center)
    radial_int_center = radial_grid.intensity(GAIA_G_EFFECTIVE_FREQUENCY, n_hat_center)
    stats_text += f"  UD: {uniform_int_center:.2e} W/m²/Hz/sr\n"
    stats_text += f"  RG: {radial_int_center:.2e} W/m²/Hz/sr\n"
    if uniform_int_center > 0:
        stats_text += f"  Ratio: {radial_int_center/uniform_int_center:.4f}\n"
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
             verticalalignment='top', fontfamily='monospace',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig

def main():
    """Main comparison function."""
    print("UniformDisk vs RadialGrid2 (SATLAS) Comparison")
    print("=" * 60)
    
    # Load Gaia data
    print("Loading Gaia data...")
    df = pd.read_csv('extended_data_table_2.csv')
    print(f"✓ Loaded {len(df)} stars")
    
    # Select HD 360
    star_name = 'HD 360'
    if star_name not in df['Star'].values:
        print(f"Error: {star_name} not found in the data")
        return
    print(f"Selected star: {star_name}")
    
    # Create UniformDisk
    print("\nCreating UniformDisk objects...")
    star_disks = create_uniform_disk_from_gaia(df)
    uniform_disk = star_disks[star_name]
    print(f"✓ Created UniformDisk for {star_name}")
    
    # Create RadialGrid2
    print("\nCreating RadialGrid2 objects...")
    satlas_file = 'data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt'
    star_grids = create_radial_grid_from_gaia(df, satlas_file)
    radial_grid = star_grids[star_name]
    print(f"✓ Created RadialGrid2 for {star_name}")
    
    # Compare flux densities
    compare_flux_densities(star_name, uniform_disk, radial_grid)
    
    # Calculate visibility squared for both models
    print(f"\n{'='*60}")
    print("Calculating visibility squared profiles...")
    print(f"{'='*60}")
    
    baseline_lengths = np.arange(25, 251, 10)
    wavelengths = [GAIA_G_EFFECTIVE_WAVELENGTH, GAIA_BP_EFFECTIVE_WAVELENGTH, GAIA_RP_EFFECTIVE_WAVELENGTH]
    frequencies = [GAIA_G_EFFECTIVE_FREQUENCY, GAIA_BP_EFFECTIVE_FREQUENCY, GAIA_RP_EFFECTIVE_FREQUENCY]
    band_names = ['G', 'BP', 'RP']
    
    results = calculate_visibility_squared_comparison(
        uniform_disk, radial_grid, baseline_lengths,
        wavelengths, frequencies, band_names
    )
    
    # Print summary
    print(f"\nResults summary:")
    for i, band_name in enumerate(band_names):
        uniform_result = results['uniform'][i]
        radial_result = results['radial'][i]
        
        print(f"\n{band_name} Band ({uniform_result['wavelength_nm']:.1f} nm):")
        print(f"  UniformDisk: {len(uniform_result['baselines'])} valid points")
        if len(uniform_result['baselines']) > 0:
            print(f"    |V|² range: {uniform_result['visibility_squared'].min():.2e} - {uniform_result['visibility_squared'].max():.2e}")
        
        print(f"  RadialGrid2: {len(radial_result['baselines'])} valid points")
        if len(radial_result['baselines']) > 0:
            print(f"    |V|² range: {radial_result['visibility_squared'].min():.2e} - {radial_result['visibility_squared'].max():.2e}")
    
    # Create comparison plots
    print(f"\nCreating comparison plots...")
    fig = plot_comparison(star_name, results, uniform_disk, radial_grid, baseline_lengths)
    
    # Save plot
    output_filename = f'compare_uniform_vs_satlas_{star_name.replace(" ", "_")}.pdf'
    fig.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"✓ Plot saved to {output_filename}")
    
    print(f"\n{'='*60}")
    print("Comparison completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()