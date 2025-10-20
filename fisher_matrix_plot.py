#!/usr/bin/env python3
"""
Fisher Matrix Plotting Script

This script reads the fisher_matrix_results.csv file created by fisher_matrix_table.py
and creates plots for SNR, sqrt(F^-1_lnrlnr), and Visibility Squared |V|².

For each parameter, it creates:
1. Parameter as a function of baseline with each star plotted in a single plot
2. Parameter as a function of star with each baseline plotted in a single plot

All plots are saved to a PDF file with observation parameters included.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import matplotlib.cm as cm

def load_data():
    """Load the Fisher matrix results data."""
    try:
        df = pd.read_csv('fisher_matrix_results.csv')
        print(f"✓ Loaded {len(df)} measurements from fisher_matrix_results.csv")
        return df
    except FileNotFoundError:
        print("Error: fisher_matrix_results.csv not found. Please run fisher_matrix_table.py first.")
        return None

def get_observation_parameters():
    """Get observation parameters from the original script."""
    return {
        'integration_time': 3600,  # seconds
        'telescope_area': np.pi * (5.0)**2,  # π*(5m)^2
        'throughput': 0.3,
        'detector_jitter': 130e-12/2.555,  # 130 ps FWHM to stddev
        'wavelengths': [639.07, 518.26, 782.51],  # nm
        'baselines': [60*np.sqrt(2), 120, 120*np.sqrt(2), 630]  # meters
    }

def add_observation_info(ax, obs_params):
    """Add observation parameters as text box to the plot."""
    info_text = (
        f"Observation Parameters:\n"
        f"Integration time: {obs_params['integration_time']} s\n"
        f"Telescope area: {obs_params['telescope_area']:.1f} m²\n"
        f"Throughput: {obs_params['throughput']}\n"
        f"Detector jitter: {obs_params['detector_jitter']*1e9:.1f} ns\n"
        f"Wavelengths: {obs_params['wavelengths']} nm\n"
        f"Baselines: {[f'{b:.1f}' for b in obs_params['baselines']]} m"
    )
    
    # Add text box in upper right corner
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=8)

def plot_parameter_vs_baseline(df, parameter, ylabel, title_prefix, obs_params, log_scale=False):
    """Plot parameter as a function of baseline for each star - creates two plots with half the stars each."""
    wavelengths = df['Wavelength_nm'].unique()
    wavelengths.sort()
    
    # Get unique stars and split them into two groups
    stars = sorted(df['Star'].unique())
    mid_point = len(stars) // 2
    stars_group1 = stars[:mid_point]
    stars_group2 = stars[mid_point:]
    
    # Create colors for all stars to maintain consistency
    all_colors = plt.cm.tab20(np.linspace(0, 1, len(stars)))
    star_colors = dict(zip(stars, all_colors))
    
    figures = []
    
    # Create first plot with first half of stars
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle(f'{title_prefix} vs Baseline Length (Stars 1-{mid_point})', fontsize=16, fontweight='bold')
    
    for i, wavelength in enumerate(wavelengths):
        ax = axes1[i]
        wave_data = df[df['Wavelength_nm'] == wavelength]
        
        # Plot first group of stars
        for star in stars_group1:
            star_data = wave_data[wave_data['Star'] == star]
            if not star_data.empty:
                # Remove NaN and infinite values
                valid_data = star_data.dropna(subset=[parameter])
                valid_data = valid_data[np.isfinite(valid_data[parameter])]
                
                if not valid_data.empty:
                    ax.plot(valid_data['Baseline_m'], valid_data[parameter],
                           'o-', color=star_colors[star], label=star, alpha=0.7, markersize=4)
        
        ax.set_xlabel('Baseline Length (m)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{wavelength:.1f} nm')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        # Add legend only to the first subplot to avoid clutter
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add observation parameters to the last subplot
    add_observation_info(axes1[-1], obs_params)
    plt.tight_layout()
    figures.append(fig1)
    
    # Create second plot with second half of stars
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f'{title_prefix} vs Baseline Length (Stars {mid_point+1}-{len(stars)})', fontsize=16, fontweight='bold')
    
    for i, wavelength in enumerate(wavelengths):
        ax = axes2[i]
        wave_data = df[df['Wavelength_nm'] == wavelength]
        
        # Plot second group of stars
        for star in stars_group2:
            star_data = wave_data[wave_data['Star'] == star]
            if not star_data.empty:
                # Remove NaN and infinite values
                valid_data = star_data.dropna(subset=[parameter])
                valid_data = valid_data[np.isfinite(valid_data[parameter])]
                
                if not valid_data.empty:
                    ax.plot(valid_data['Baseline_m'], valid_data[parameter],
                           'o-', color=star_colors[star], label=star, alpha=0.7, markersize=4)
        
        ax.set_xlabel('Baseline Length (m)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{wavelength:.1f} nm')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        # Add legend only to the first subplot to avoid clutter
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add observation parameters to the last subplot
    add_observation_info(axes2[-1], obs_params)
    plt.tight_layout()
    figures.append(fig2)
    
    return figures

def plot_parameter_vs_star(df, parameter, ylabel, title_prefix, obs_params, log_scale=False):
    """Plot parameter as a function of star for each baseline - creates two plots with half the stars each."""
    wavelengths = df['Wavelength_nm'].unique()
    wavelengths.sort()
    
    # Define target baselines to plot
    target_baselines = [50, 100, 150, 200, 300, 400, 630]
    
    # Get actual baselines from data and find closest matches
    actual_baselines = df['Baseline_m'].unique()
    actual_baselines.sort()
    
    # Find closest actual baseline for each target baseline
    selected_baselines = []
    for target in target_baselines:
        closest_baseline = min(actual_baselines, key=lambda x: abs(x - target))
        # Only include if reasonably close (within 50m) or if it's 630m (exact match expected)
        if abs(closest_baseline - target) < 50 or target == 630:
            selected_baselines.append(closest_baseline)
    
    # Remove duplicates while preserving order
    selected_baselines = list(dict.fromkeys(selected_baselines))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_baselines)))
    baseline_colors = dict(zip(selected_baselines, colors))
    
    # Get all unique stars and split them into two groups
    all_stars = sorted(df['Star'].unique())
    mid_point = len(all_stars) // 2
    stars_group1 = all_stars[:mid_point]
    stars_group2 = all_stars[mid_point:]
    
    figures = []
    
    # Create first plot with first half of stars
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle(f'{title_prefix} vs Star (Stars 1-{mid_point})', fontsize=16, fontweight='bold')
    
    for i, wavelength in enumerate(wavelengths):
        ax = axes1[i]
        wave_data = df[df['Wavelength_nm'] == wavelength]
        
        # Filter data to only include first group of stars
        wave_data_group1 = wave_data[wave_data['Star'].isin(stars_group1)]
        
        # Plot each selected baseline
        for baseline in selected_baselines:
            baseline_data = wave_data_group1[wave_data_group1['Baseline_m'] == baseline]
            if not baseline_data.empty:
                # Remove NaN and infinite values
                valid_data = baseline_data.dropna(subset=[parameter])
                valid_data = valid_data[np.isfinite(valid_data[parameter])]
                
                if not valid_data.empty:
                    # Sort by star name for consistent plotting
                    valid_data = valid_data.sort_values('Star')
                    # Create indices based on position in stars_group1
                    star_indices = [stars_group1.index(star) for star in valid_data['Star']]
                    
                    ax.plot(star_indices, valid_data[parameter],
                           'o', color=baseline_colors[baseline],
                           label=f'{baseline:.1f} m', alpha=0.7, markersize=4)
        
        # Set star names as x-tick labels for first group
        ax.set_xticks(range(len(stars_group1)))
        ax.set_xticklabels(stars_group1, rotation=45, ha='right', fontsize=6)
        
        ax.set_xlabel('Star')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{wavelength:.1f} nm')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add observation parameters to the last subplot
    add_observation_info(axes1[-1], obs_params)
    plt.tight_layout()
    figures.append(fig1)
    
    # Create second plot with second half of stars
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f'{title_prefix} vs Star (Stars {mid_point+1}-{len(all_stars)})', fontsize=16, fontweight='bold')
    
    for i, wavelength in enumerate(wavelengths):
        ax = axes2[i]
        wave_data = df[df['Wavelength_nm'] == wavelength]
        
        # Filter data to only include second group of stars
        wave_data_group2 = wave_data[wave_data['Star'].isin(stars_group2)]
        
        # Plot each selected baseline
        for baseline in selected_baselines:
            baseline_data = wave_data_group2[wave_data_group2['Baseline_m'] == baseline]
            if not baseline_data.empty:
                # Remove NaN and infinite values
                valid_data = baseline_data.dropna(subset=[parameter])
                valid_data = valid_data[np.isfinite(valid_data[parameter])]
                
                if not valid_data.empty:
                    # Sort by star name for consistent plotting
                    valid_data = valid_data.sort_values('Star')
                    # Create indices based on position in stars_group2
                    star_indices = [stars_group2.index(star) for star in valid_data['Star']]
                    
                    ax.plot(star_indices, valid_data[parameter],
                           'o', color=baseline_colors[baseline],
                           label=f'{baseline:.1f} m', alpha=0.7, markersize=4)
        
        # Set star names as x-tick labels for second group
        ax.set_xticks(range(len(stars_group2)))
        ax.set_xticklabels(stars_group2, rotation=45, ha='right', fontsize=6)
        
        ax.set_xlabel('Star')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{wavelength:.1f} nm')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add observation parameters to the last subplot
    add_observation_info(axes2[-1], obs_params)
    plt.tight_layout()
    figures.append(fig2)
    
    return figures

def create_all_plots(df, obs_params):
    """Create all plots and return list of figures."""
    figures = []
    
    # Define parameters to plot
    parameters = [
        ('SNR', 'Signal-to-Noise Ratio', 'SNR', True),
        ('sqrt_inv_Fisher_lnrlnr', 'sqrt(F⁻¹ₗₙᵣₗₙᵣ)', 'Parameter Uncertainty', True),
        ('Visibility_squared', 'Visibility Squared |V|²', 'Visibility Squared', True)
    ]
    
    for param_col, ylabel, title_prefix, log_scale in parameters:
        print(f"Creating plots for {title_prefix}...")
        
        # Parameter vs Baseline plots (returns list of 2 figures)
        baseline_figs = plot_parameter_vs_baseline(df, param_col, ylabel, title_prefix, obs_params, log_scale)
        figures.extend(baseline_figs)
        
        # Parameter vs Star plots (returns list of 2 figures)
        star_figs = plot_parameter_vs_star(df, param_col, ylabel, title_prefix, obs_params, log_scale)
        figures.extend(star_figs)
    
    return figures

def save_plots_to_pdf(figures, filename='fisher_matrix_plots.pdf'):
    """Save all figures to a PDF file."""
    print(f"Saving plots to {filename}...")
    
    with PdfPages(filename) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)  # Close figure to free memory
    
    print(f"✓ All plots saved to {filename}")

def print_data_summary(df):
    """Print summary of the data."""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"Total measurements: {len(df)}")
    print(f"Number of stars: {df['Star'].nunique()}")
    print(f"Number of wavelengths: {df['Wavelength_nm'].nunique()}")
    print(f"Number of baselines: {df['Baseline_m'].nunique()}")
    
    print(f"\nWavelengths (nm): {sorted(df['Wavelength_nm'].unique())}")
    print(f"Baselines (m): {sorted(df['Baseline_m'].unique())}")
    print(f"Stars: {sorted(df['Star'].unique())}")
    
    # Check for missing data
    print(f"\nMissing data:")
    for col in ['SNR', 'sqrt_inv_Fisher_lnrlnr', 'Visibility_squared']:
        missing = df[col].isna().sum()
        infinite = np.isinf(df[col]).sum()
        print(f"  {col}: {missing} NaN, {infinite} infinite values")

def main():
    """Main function to create all plots."""
    print("Fisher Matrix Plotting Script")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Print data summary
    print_data_summary(df)
    
    # Get observation parameters
    obs_params = get_observation_parameters()
    
    # Create all plots
    print(f"\nCreating plots...")
    figures = create_all_plots(df, obs_params)
    
    # Save to PDF
    save_plots_to_pdf(figures, 'fisher_matrix_plots.pdf')
    
    print(f"\n" + "="*60)
    print("Plotting completed successfully!")
    print(f"Generated {len(figures)} plots saved to fisher_matrix_plots.pdf")
    print("="*60)

if __name__ == "__main__":
    main()