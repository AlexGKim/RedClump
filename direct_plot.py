#!/usr/bin/env python3
"""
direct_plot.py

Plot the Phoenix limb‑darkening intensity profile together with the
Satlas reference profile for a selected photometric band.

Phoenix law:
    I(μ) / I(1) = 1 - Σ_{k=1}^{4} a_k * (1 - μ^{k/2})

The a_k coefficients are read from *table12.dat* (fixed‑width format).
The Satlas profile is read from
data/output_ld‑satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt
(the first two columns are μ and I/I0_H).
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_table12 import read_table12
import scipy.special


RSUN_M   = 6.957_000_000e8   # metres per solar radius
PARSEC_M = 3.085_677_581_491_367e16  # metres per parsec

# Radius = 9.312 Rsun ; distance = 95.594 pc
radius_rsun = 9.312
distance_pc = 95.594
radius_m   = radius_rsun * RSUN_M
distance_m = distance_pc * PARSEC_M


# HD      θLD (mas)       log Teff(K)     log g   [Fe/H] (dex)    E(B-V ) log R/R⊙        log L/L⊙
# 360     0.906 ± 0.015   3.678 ± 0.002   2.62 ± 0.11     −0.12 ± 0.05    0.009   1.036 ± 0.009   1.736 ± 0.019


# ----------------------------------------------------------------------
# Helper: compute the Phoenix intensity for a given filter
# ----------------------------------------------------------------------
def _phoenix_intensity(mu2: np.ndarray, coeffs: pd.Series) -> np.ndarray:
    """
    Evaluate the Phoenix limb‑darkening law.

    Parameters
    ----------
    mu : ndarray
        Cosine of the angle (μ) – values in [0, 1].
    coeffs : pandas.Series
        Must contain the four coefficients ``a1``, ``a2``, ``a3``, ``a4``.

    Returns
    -------
    ndarray
        Normalised intensity I(μ)/I(1).
    """
    ans = np.zeros(len(mu2))
    w = mu2 >= 0.0
    mu = np.sqrt(mu2[w])
    a1, a2, a3, a4 = coeffs[["a1", "a2", "a3", "a4"]].values
    # μ^{k/2} for k = 1..4
    mu_pow = np.stack([mu ** (0.5 * k) for k in range(1, 5)], axis=0)
    term = np.dot(np.array([a1, a2, a3, a4]), (1.0 - mu_pow))
    ans[w] = 1.0 - term
    return ans


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def plot_intensity(
    filter_name: str,
    satlas_path: str | pathlib.Path,
    table12_path: str | pathlib.Path,
    *,
    mu_grid: int = 200,
    u_max: float = 3,          # highest spatial frequency (in 1/mas)
    n_u: int = 200,              # number of u points
    ax: plt.Axes | None = None,
    save_as: str | pathlib.Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot the Phoenix intensity profile together with the Satlas reference
    **and** the visibility curve obtained from the Hankel transform.

    Parameters
    ----------
    filter_name, satlas_path, table12_path, mu_grid, ax, save_as, show
        as before.
    u_max : float, optional
        Maximum spatial frequency (units: 1/mas). Default 0.5 mas⁻¹.
    n_u : int, optional
        Number of points in the u‑grid.
    """
    # ------------------------------------------------------------------
    # 1) Load coefficients
    # ------------------------------------------------------------------
    coeffs_df = read_table12(pathlib.Path(table12_path))

    needed = [f"a1{filter_name}", f"a2{filter_name}",
              f"a3{filter_name}", f"a4{filter_name}"]
    if not set(needed).issubset(coeffs_df.columns):
        raise ValueError(
            f'Filter "{filter_name}" not recognised. '
            f'Available filters: {sorted({c[-1] for c in coeffs_df.columns if c.startswith("a1")})}'
        )

    from gaia_phoenix import match_phoenix_parameters
    teff, logg, feh = 4800, 2.5, 1.0
    teff_phoenix, logg_phoenix, feh_phoenix = match_phoenix_parameters(teff, logg, feh)

    # rows that match the Phoenix atmospheric parameters
    mask = (
        (coeffs_df["Teff"] == teff_phoenix) &
        (coeffs_df["logg"] == logg_phoenix)  # column name may be "FeH" or "feh"
    )
    if not mask.any():
        raise ValueError(
            f'No entry in table12.dat for Teff={teff_phoenix}, '
            f'logg={logg_phoenix}, FeH={feh_phoenix}.'
        )
    # keep only the four needed columns and grab the first matching row
    coeff_row: pd.Series = coeffs_df.loc[mask, needed].iloc[0]   # Series
    coeff_row.index = ["a1", "a2", "a3", "a4"]                  # rename to generic keys

    # ------------------------------------------------------------------
    # 2) Load Satlas data
    # ------------------------------------------------------------------
    satlas_df = pd.read_csv(
        pathlib.Path(satlas_path),
        sep=r'\s+',           # replaces deprecated delim_whitespace
        header=0,
        comment="#",
    )

    # Add a row of zeros to extenend the profile beyond the radius
    tr_max = 1.0000001 * radius_m/distance_m
    rmas_max = tr_max * (180 / np.pi) * 3600 * 1000
    zero_row = pd.Series(0.0, index=satlas_df.columns)
    zero_row["r(mas)"] = rmas_max
    satlas_df = pd.concat([satlas_df, zero_row.to_frame().T], ignore_index=True)


    # ------------------------------------------------------------------
    # 3) Build μ grid and compute Phoenix profile
    # ------------------------------------------------------------------
    # Convert the angular radius (mas) → radians
    theta_rad = satlas_df["r(mas)"].values * (np.pi / 180 / 3600 / 1000)
    # μ = cos(θ) for a uniform‐disk geometry
    # mu = np.sqrt(1.0 - (distance_m * theta_rad / radius_m) ** 2)
    mu2 = 1.0 - (distance_m * theta_rad / radius_m) ** 2

    phoenix_I = _phoenix_intensity(mu2, coeff_row)                     # shape (N,)

    # ------------------------------------------------------------------
    # 4) Prepare the second‑subplot: visibility vs. spatial frequency u
    # ------------------------------------------------------------------
    # u‑grid (1/mas)
    u_grid = np.linspace(0.0, u_max, n_u)

    # scaled rays = the same radial coordinate used for the Hankel transform
    scaled_rays = satlas_df["r(mas)"].values.astype(float)   # (N,)

    # Containers for the two visibility curves
    vis_phoenix = np.empty_like(u_grid)
    vis_satlas  = np.empty_like(u_grid)

    # Pre‑compute the denominator (the "zero‑baseline" integral) – common to both
    denom_phoenix = np.trapezoid(phoenix_I * scaled_rays, scaled_rays)
    denom_satlas  = np.trapezoid(satlas_df["I/I0_H"].values * scaled_rays, scaled_rays)

    for i, u in enumerate(u_grid):
        J0 = scipy.special.j0(2.0 * np.pi * u * scaled_rays)   # J0(2πuρ)

        # numerator for Phoenix
        num_pho = np.trapezoid(phoenix_I * J0 * scaled_rays, scaled_rays)
        vis_phoenix[i] = num_pho / denom_phoenix

        # numerator for Satlas
        num_sat = np.trapezoid(satlas_df["I/I0_H"].values * J0 * scaled_rays,
                           scaled_rays)
        vis_satlas[i] = num_sat / denom_satlas

    # ------------------------------------------------------------------
    # 5) Plot (three sub‑plots: intensity on left, visibility and difference stacked on right)
    # ------------------------------------------------------------------
    if ax is None:
        fig = plt.figure(figsize=(12, 5))
        # Create gridspec with 1 column on left, 1 column on right
        gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, 
                              bottom=0.11, top=0.9, wspace=0.3)
        ax_int = fig.add_subplot(gs[0, 0])
        
        # Split the right column into two rows
        gs_right = gs[0, 1].subgridspec(2, 1, hspace=0.15)
        ax_vis = fig.add_subplot(gs_right[0])
        ax_diff = fig.add_subplot(gs_right[1], sharex=ax_vis)
    else:
        fig = ax.figure
        ax_int = ax
        # You would need to handle the layout manually if ax is provided
        gs_right = fig.add_gridspec(2, 1, left=0.55, right=0.98, 
                                    bottom=0.11, top=0.9, hspace=0.05)
        ax_vis = fig.add_subplot(gs_right[0])
        ax_diff = fig.add_subplot(gs_right[1], sharex=ax_vis)

    # ---- intensity ----------------------------------------------------
    ax_int.plot(satlas_df["r(mas)"], phoenix_I,
                label="PHOENIX", color="tab:blue")
    ax_int.plot(satlas_df["r(mas)"], satlas_df["I/I0_H"],
                label="SATLAS", color="tab:red", linestyle="--")
    ax_int.set_xlabel(r"$r\;(\mathrm{mas})$")
    ax_int.set_ylabel(r"$I(r)/I(0)$")
    ax_int.grid(True, which="both", ls=":", alpha=0.6)
    ax_int.legend()

    # ---- visibility ----------------------------------------------------


    ax_vis.plot(u_grid, vis_phoenix**2,
                label="PHOENIX", color="tab:blue")
    ax_vis.plot(u_grid, vis_satlas**2,
                label="SATLAS", color="tab:red", linestyle="--")
    ax_vis.set_ylabel(r"$|V|^2$")
    # ax_vis.set_yscale("log")
    ax_vis.grid(True, which="both", ls=":", alpha=0.6)
    ax_vis.set_yscale("log")
    ax_vis.set_ylim(1e-4,1.1)
    ax_vis.legend()
    ax_vis.tick_params(labelbottom=False)  # Hide x-axis labels on top plot

    # ---- difference in |V|² -----------------------------------------
    diff_v2 = vis_phoenix**2 - vis_satlas**2
    ax_diff.plot(u_grid, diff_v2, color="tab:green", linewidth=1.5)
    ax_diff.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.7)
    ax_diff.set_xlabel(r"$u\;(\mathrm{mas}^{-1})$")
    ax_diff.set_ylabel(r"$\Delta|V|^2$")
    ax_diff.grid(True, which="both", ls=":", alpha=0.6)
    ax_diff.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax_diff.yaxis.get_offset_text().set_position((-0.1, -1))  # Move offset to left

    # ------------------------------------------------------------------
    # 6) Save / show
    # ------------------------------------------------------------------
    if save_as:
        plt.savefig(save_as, bbox_inches="tight")
    if show:
        plt.show()

    # Returning the intensity axis is kept for backward compatibility.
    return ax_int
# ----------------------------------------------------------------------
# Helper – remove the “I/I0_” prefix that Satlas columns contain
# ----------------------------------------------------------------------
def _strip_prefix(col_name: str, prefix: str = "I/I0_") -> str:
    """
    Return *col_name* with *prefix* stripped if it starts with that prefix.
    """
    if col_name.startswith(prefix):
        return col_name[len(prefix) :]
    return col_name

# ----------------------------------------------------------------------
# New public API --------------------------------------------------------
# ----------------------------------------------------------------------
def plot_satlas_vis2_all_bands(
    satlas_path: str | pathlib.Path,
    *,
    u_max: float = 3.0,               # 1/mas
    n_u: int = 200,
    radius_m: float = radius_m,       # already defined globally
    distance_m: float = distance_m,   # already defined globally
    save_as: str | pathlib.Path = "satlas_vis2_all_bands.pdf",
    show: bool = True,
) -> plt.Axes:
    """
    Compute and plot |V|² for **every** photometric band present in the
    Satlas surface‑intensity file.

    Parameters
    ----------
    satlas_path
        Path to ``ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt``.
    u_max, n_u
        Spatial‑frequency grid (1/mas).
    radius_m, distance_m
        Stellar radius and distance (same geometry used for the Phoenix
        profile).  Defaults are the module‑level values.
    save_as
        Filename for the PDF that will contain the plot.
    show
        Call ``plt.show()`` after saving.
    """
    # --------------------------------------------------------------
    # 1) Load the Satlas table
    # --------------------------------------------------------------
    satlas_df = pd.read_csv(
        pathlib.Path(satlas_path),
        sep=r"\s+",
        header=0,
        comment="#",
    )

    # The first column is the angular radius (mas); everything else is a band.
    radius_col = "r(mas)"
    if radius_col not in satlas_df.columns:
        raise KeyError(f"Expected column '{radius_col}' not found in {satlas_path}")

    band_cols = [c for c in satlas_df.columns if c != radius_col]

    # Extend the profile with a zero‑intensity point just beyond the stellar edge,
    # mirroring what ``plot_intensity`` does.
    tr_max = 1.0000001 * radius_m / distance_m
    rmas_max = tr_max * (180.0 / np.pi) * 3600.0 * 1000.0
    zero_row = pd.Series(0.0, index=satlas_df.columns)
    zero_row[radius_col] = rmas_max
    satlas_df = pd.concat([satlas_df, zero_row.to_frame().T], ignore_index=True)

    # --------------------------------------------------------------
    # 2) Geometry: scaled radial coordinate (mas) for the Hankel transform
    # --------------------------------------------------------------
    scaled_rays = satlas_df[radius_col].values.astype(float)      # (N,)

    # --------------------------------------------------------------
    # 3) u‑grid
    # --------------------------------------------------------------
    u_grid = np.linspace(0.0, u_max, n_u)

    # --------------------------------------------------------------
    # 4) Allocate results container
    # --------------------------------------------------------------
    vis2_dict: dict[str, np.ndarray] = {band: np.empty_like(u_grid) for band in band_cols}

    # --------------------------------------------------------------
    # 5) Pre‑compute denominators (zero‑baseline integrals) for each band
    # --------------------------------------------------------------
    denom = {
        band: np.trapezoid(satlas_df[band].values * scaled_rays, scaled_rays)
        for band in band_cols
    }

    # --------------------------------------------------------------
    # 6) Loop over u and evaluate the Hankel transform
    # --------------------------------------------------------------
    for i, u in enumerate(u_grid):
        J0 = scipy.special.j0(2.0 * np.pi * u * scaled_rays)   # J0(2πuρ)

        for band in band_cols:
            num = np.trapezoid(satlas_df[band].values * J0 * scaled_rays, scaled_rays)
            vis = num / denom[band]
            vis2_dict[band][i] = vis ** 2

    # --------------------------------------------------------------
    # 7) Plot
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use a cycle of colours/linestyles that works for an arbitrary number of bands
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]
    for idx, band in enumerate(band_cols):
        col = colors[idx % len(colors)]
        ls = linestyles[(idx // len(colors)) % len(linestyles)]
        label = _strip_prefix(band)          # <-- strip “I/I0_” here
        ax.plot(u_grid, vis2_dict[band], label=label, color=col, linestyle=ls)
        # ax.plot(u_grid, vis2_dict[band], label=band, color=col, linestyle=ls)

    ax.set_xlabel(r"$u\;(\mathrm{mas}^{-1})$")
    ax.set_ylabel(r"$|V|^{2}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-3,1.1)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(title="Band", loc="upper right", fontsize="small", ncol=2)

    # --------------------------------------------------------------
    # 8) Save / show
    # --------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(save_as, bbox_inches="tight")
    if show:
        plt.show()
    return ax

def compute_visibility_derivative(
    intensity: np.ndarray,
    scaled_rays: np.ndarray,
    u_grid: np.ndarray,
    s: float = 1.0,
) -> np.ndarray:
    """
    Compute ∂V/∂s for the visibility amplitude.
    
    Parameters
    ----------
    intensity : np.ndarray
        Intensity profile I(θ), shape (N,)
    scaled_rays : np.ndarray
        Radial coordinate θ (mas), shape (N,)
    u_grid : np.ndarray
        Spatial frequency grid (1/mas), shape (M,)
    s : float, optional
        Scale parameter (default 1.0)
    
    Returns
    -------
    np.ndarray
        ∂V/∂s evaluated at each point in u_grid, shape (M,)
    
    Notes
    -----
    The formula is:
        ∂V/∂s = -(2πu/s) * [∫ I(θ) J₁(2πuθ) θ² dθ] / [∫ I(θ) θ dθ]
    """
    # Denominator (independent of u)
    denom = np.trapezoid(intensity * scaled_rays, scaled_rays)
    
    # Container for the derivative
    dVds = np.empty_like(u_grid)
    
    for i, u in enumerate(u_grid):
        # J₁(2πuθ)
        J1 = scipy.special.j1(2.0 * np.pi * u * scaled_rays)
        
        # Numerator: ∫ I(θ) J₁(2πuθ) θ² dθ
        num = np.trapezoid(intensity * J1 * (scaled_rays ** 2), scaled_rays)
        
        # ∂V/∂s = -(2πu/s) * num / denom
        dVds[i] = -(2.0 * np.pi * u / s) * (num / denom)
    
    return dVds


def plot_satlas_derivative_all_bands(
    satlas_path: str | pathlib.Path,
    *,
    u_max: float = 3.0,
    n_u: int = 200,
    radius_m: float = radius_m,
    distance_m: float = distance_m,
    save_as: str | pathlib.Path = "satlas_derivative_all_bands.pdf",
    show: bool = True,
) -> plt.Figure:
    """
    Compute and plot ∂(V²)/∂s for all photometric bands in Satlas.
    
    Parameters
    ----------
    satlas_path
        Path to Satlas surface intensity file
    u_max, n_u
        Spatial frequency grid
    radius_m, distance_m
        Stellar geometry
    save_as
        Output filename
    show
        Display the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Load Satlas table
    satlas_df = pd.read_csv(
        pathlib.Path(satlas_path),
        sep=r"\s+",
        header=0,
        comment="#",
    )
    
    radius_col = "r(mas)"
    band_cols = [c for c in satlas_df.columns if c != radius_col]
    
    # Extend profile
    tr_max = 1.0000001 * radius_m / distance_m
    rmas_max = tr_max * (180.0 / np.pi) * 3600.0 * 1000.0
    zero_row = pd.Series(0.0, index=satlas_df.columns)
    zero_row[radius_col] = rmas_max
    satlas_df = pd.concat([satlas_df, zero_row.to_frame().T], ignore_index=True)
    
    scaled_rays = satlas_df[radius_col].values.astype(float)
    u_grid = np.linspace(0.0, u_max, n_u)
    
    # Compute V and ∂V/∂s for each band
    vis_dict = {}
    dVds_dict = {}
    dV2ds_dict = {}
    
    for band in band_cols:
        intensity = satlas_df[band].values
        
        # Compute V
        denom = np.trapezoid(intensity * scaled_rays, scaled_rays)
        vis = np.empty_like(u_grid)
        
        for i, u in enumerate(u_grid):
            J0 = scipy.special.j0(2.0 * np.pi * u * scaled_rays)
            num = np.trapezoid(intensity * J0 * scaled_rays, scaled_rays)
            vis[i] = num / denom
        
        # Compute ∂V/∂s
        dVds = compute_visibility_derivative(intensity, scaled_rays, u_grid, s=1.0)
        
        # Compute ∂(V²)/∂s
        dV2ds = 2.0 * vis * dVds
        
        vis_dict[band] = vis
        dVds_dict[band] = dVds
        dV2ds_dict[band] = dV2ds
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]
    
    for idx, band in enumerate(band_cols):
        col = colors[idx % len(colors)]
        ls = linestyles[(idx // len(colors)) % len(linestyles)]
        label = _strip_prefix(band)
        
        # Top panel: ∂V/∂s
        ax1.plot(u_grid, dVds_dict[band], label=label, color=col, linestyle=ls, linewidth=1.5)
        
        # Bottom panel: ∂(V²)/∂s
        ax2.plot(u_grid, dV2ds_dict[band], label=label, color=col, linestyle=ls, linewidth=1.5)
    
    # Top panel formatting
    ax1.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.7)
    ax1.set_ylabel(r"$\partial V / \partial s$", fontsize=12)
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.legend(title="Band", loc="best", fontsize="small", ncol=2)
    ax1.set_title("SATLAS Visibility Scale Derivative - All Bands", 
                  fontsize=14, fontweight='bold')
    
    # Bottom panel formatting
    ax2.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.7)
    ax2.set_xlabel(r"$u\;(\mathrm{mas}^{-1})$", fontsize=12)
    ax2.set_ylabel(r"$\partial |V|^2 / \partial s$", fontsize=12)
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend(title="Band", loc="best", fontsize="small", ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_as, bbox_inches="tight")
    if show:
        plt.show()
    
    return fig


def plot_fss_inverse_sqrt(
    satlas_path: str | pathlib.Path,
    df_with_noise: pd.DataFrame,
    filter_df: pd.DataFrame,
    *,
    star_name: str | None = None,
    filters: list[str] = ['V', 'R', 'I', 'H', 'K'],
    u_max: float = 3.0,
    n_u: int = 200,
    radius_m: float = radius_m,
    distance_m: float = distance_m,
    save_as: str | pathlib.Path = "fss_inverse_sqrt.pdf",
    show: bool = True,
) -> plt.Figure:
    """
    Compute and plot F_ss^{-1/2} for each filter as a function of baseline B.
    
    F_ss = (∂|V|²/∂s * inverse_noise)²
    F_ss^{-1/2} = 1/|∂|V|²/∂s * inverse_noise|
    
    The x-axis is baseline B (meters), where u = B/λ₀
    Therefore: B = u × λ₀
    
    Parameters
    ----------
    satlas_path
        Path to Satlas surface intensity file
    df_with_noise : pd.DataFrame
        Dataframe containing inverse noise values with columns {filter}_inv_noise
    filter_df : pd.DataFrame
        Filter properties dataframe containing effective frequencies
    star_name : str, optional
        Name of star to use. If None, uses first star in dataframe.
    filters : list[str]
        List of filters to plot
    u_max, n_u
        Spatial frequency grid (in 1/mas)
    radius_m, distance_m
        Stellar geometry
    save_as
        Output filename
    show
        Display the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Physical constants
    c = 2.99792458e8  # speed of light in m/s
    mas_to_rad = (np.pi / 180.0) * (1.0 / 3600.0) * (1.0 / 1000.0)  # mas to radians
    
    # Get star data
    if star_name is None:
        star_row = df_with_noise.iloc[0]
        star_name = star_row['Star']
    else:
        star_row = df_with_noise[df_with_noise['Star'] == star_name].iloc[0]
    
    print(f"Computing F_ss^{{-1/2}} for {star_name}")
    
    # Create filter frequency lookup
    filter_freq = {}
    for _, row in filter_df.iterrows():
        filter_freq[row['Filter']] = row['ν_eff_Hz']
    
    # Load Satlas table
    satlas_df = pd.read_csv(
        pathlib.Path(satlas_path),
        sep=r"\s+",
        header=0,
        comment="#",
    )
    
    radius_col = "r(mas)"
    
    # Extend profile
    tr_max = 1.0000001 * radius_m / distance_m
    rmas_max = tr_max * (180.0 / np.pi) * 3600.0 * 1000.0
    zero_row = pd.Series(0.0, index=satlas_df.columns)
    zero_row[radius_col] = rmas_max
    satlas_df = pd.concat([satlas_df, zero_row.to_frame().T], ignore_index=True)
    
    scaled_rays = satlas_df[radius_col].values.astype(float)
    u_grid = np.linspace(0.0, u_max, n_u)  # in 1/mas
    
    # Convert u from 1/mas to 1/radians
    # If u = 1 cycle/mas, then u = (1/mas_to_rad) cycles/radian
    # because 1 radian contains (1/mas_to_rad) milliarcseconds
    u_grid_rad = u_grid / mas_to_rad  # in 1/radians (divide, not multiply!)
    
    # Map filter names to Satlas column names
    filter_to_satlas = {
        'V': 'I/I0_V',
        'R': 'I/I0_R',
        'I': 'I/I0_I',
        'H': 'I/I0_H',
        'K': 'I/I0_K',
        'B': 'I/I0_B'
    }
    
    # Compute F_ss^{-1/2} for each filter
    fss_inv_sqrt_dict = {}
    baseline_dict = {}
    
    for filt in filters:
        # Get inverse noise for this filter
        inv_noise_col = f'{filt}_inv_noise'
        if inv_noise_col not in star_row.index:
            print(f"Warning: {inv_noise_col} not found for {star_name}, skipping {filt}")
            continue
        
        inverse_noise = star_row[inv_noise_col]
        
        if pd.isna(inverse_noise) or inverse_noise == 0:
            print(f"Warning: Invalid inverse_noise for {filt}, skipping")
            continue
        
        # Get effective frequency for this filter
        if filt not in filter_freq:
            print(f"Warning: Frequency for {filt} not found, skipping")
            continue
        
        nu_0 = filter_freq[filt]  # Hz
        lambda_0 = c / nu_0  # wavelength in meters
        
        # Calculate baseline: u = B/λ₀, so B = u × λ₀
        # u_grid_rad is in 1/radians, λ₀ is in meters, so B is in meters
        baseline = u_grid_rad * lambda_0
        baseline_dict[filt] = baseline
        
        # Get Satlas column for this filter
        satlas_col = filter_to_satlas.get(filt)
        if satlas_col is None or satlas_col not in satlas_df.columns:
            print(f"Warning: {filt} not found in Satlas data, skipping")
            continue
        
        intensity = satlas_df[satlas_col].values
        
        # Compute V
        denom = np.trapezoid(intensity * scaled_rays, scaled_rays)
        vis = np.empty_like(u_grid)
        
        for i, u in enumerate(u_grid):
            J0 = scipy.special.j0(2.0 * np.pi * u * scaled_rays)
            num = np.trapezoid(intensity * J0 * scaled_rays, scaled_rays)
            vis[i] = num / denom
        
        # Compute ∂V/∂s
        dVds = compute_visibility_derivative(intensity, scaled_rays, u_grid, s=1.0)
        
        # Compute ∂(V²)/∂s = 2V * ∂V/∂s
        dV2ds = 2.0 * vis * dVds
        
        # Compute F_ss = (∂|V|²/∂s * inverse_noise)²
        fss = (dV2ds * inverse_noise) ** 2
        
        # Compute F_ss^{-1/2} = 1/sqrt(F_ss) = 1/|∂|V|²/∂s * inverse_noise|
        # Handle potential zeros/negatives
        fss_inv_sqrt = np.zeros_like(fss)
        valid = fss > 0
        fss_inv_sqrt[valid] = 1.0 / np.sqrt(fss[valid])
        
        fss_inv_sqrt_dict[filt] = fss_inv_sqrt
        
        print(f"  {filt}: λ₀ = {lambda_0*1e9:.1f} nm, inverse_noise = {inverse_noise:.2e}")
        print(f"        Baseline range: {baseline.min():.2f} - {baseline.max():.2f} m")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]
    
    for idx, filt in enumerate(fss_inv_sqrt_dict.keys()):
        col = colors[idx % len(colors)]
        ls = linestyles[(idx // len(colors)) % len(linestyles)]
        ax.plot(baseline_dict[filt], fss_inv_sqrt_dict[filt], label=filt, 
                color=col, linestyle=ls, linewidth=2)
    
    ax.set_xlabel(r"$B$ (m)", fontsize=12)
    ax.set_ylabel(r"$\sigma_s$", fontsize=12)
    ax.set_title(f"{star_name}", 
                fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(title="Filter", loc="best", fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(5e-3, 1e3)
    # if star_name == "HD 17652":
    #     ax.set_ylim(1e-2, 1e3)
    ax.set_xscale('log')
    ax.set_xlim(10,1500)

    plt.tight_layout()
    plt.savefig(save_as, bbox_inches="tight")
    if show:
        plt.show()
    
    return fig

# Update the __main__ block
if __name__ == "__main__":
    # Original plots
    plot_intensity(
        filter_name="H",
        satlas_path="data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt",
        table12_path="data/phoenix/table12.dat",
        save_as="phoenix_vs_satlas_H.pdf",
        show=False,
    )
    print("Plot saved as phoenix_vs_satlas_H.pdf")

    plot_satlas_vis2_all_bands(
        satlas_path="data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt",
        u_max=3.0,
        n_u=300,
        save_as="satlas_vis2_all_bands.pdf",
        show=False,
    )
    print("All‑band Satlas visibility plot saved as satlas_vis2_all_bands.pdf")
    
    
    plot_satlas_derivative_all_bands(
        satlas_path="data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt",
        u_max=3.0,
        n_u=300,
        save_as="satlas_derivative_all_bands.pdf",
        show=False,
    )
    print("All‑band derivative plot saved as satlas_derivative_all_bands.pdf")

    # Get dataframe with inverse noise
    from rc_utils import master_df_with_inverse_noise
    df_with_noise = master_df_with_inverse_noise()



        # Plot for single star - NOW WITH filter_df
    from rc_utils import get_filter_properties_df
    filter_df = get_filter_properties_df()
    plot_fss_inverse_sqrt(
        satlas_path="data/output_ld-satlas_1762763642809/ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt",
        df_with_noise=df_with_noise,
        filter_df=filter_df,  # <-- Make sure this is included!
        star_name="HD 360",  # Use first star
        filters=['V', 'R', 'I', 'H', 'K'],
        u_max=10.0,
        n_u=1000,
        save_as="fss_hd360.pdf",
        show=False,
    )

    plot_fss_inverse_sqrt(
        satlas_path="data/output_ld-satlas_1764290684148/ld_satlas_surface.2t4800g275m10_Ir_all_bands.txt",
        df_with_noise=df_with_noise,
        filter_df=filter_df,  # <-- Make sure this is included!
        star_name="HD 17652",  # Use first star
        filters=['V', 'R', 'I', 'H', 'K'],
        u_max=10.0,
        n_u=1000,
        save_as="fss_hd17652.pdf",
        show=False,
    )
