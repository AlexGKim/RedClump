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
def _phoenix_intensity(mu: np.ndarray, coeffs: pd.Series) -> np.ndarray:
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
    a1, a2, a3, a4 = coeffs[["a1", "a2", "a3", "a4"]].values
    # μ^{k/2} for k = 1..4
    mu_pow = np.stack([mu ** (0.5 * k) for k in range(1, 5)], axis=0)
    term = np.dot(np.array([a1, a2, a3, a4]), (1.0 - mu_pow))
    return 1.0 - term


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
import scipy.special  #  <-- add this import at the top of the file

def plot_intensity(
    filter_name: str,
    satlas_path: str | pathlib.Path,
    table12_path: str | pathlib.Path,
    *,
    mu_grid: int = 200,
    u_max: float = 4,          # highest spatial frequency (in 1/mas)
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
        Maximum spatial frequency (units: 1/mas). Default 0.5 mas⁻¹.
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

    # ------------------------------------------------------------------
    # 3) Build μ grid and compute Phoenix profile
    # ------------------------------------------------------------------
    # Convert the angular radius (mas) → radians
    theta_rad = satlas_df["r(mas)"].values * (np.pi / 180 / 3600 / 1000)
    # μ = cos(θ) for a uniform‐disk geometry
    mu = np.sqrt(1.0 - (distance_m * theta_rad / radius_m) ** 2)

    phoenix_I = _phoenix_intensity(mu, coeff_row)                     # shape (N,)

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

    # Pre‑compute the denominator (the “zero‑baseline” integral) – common to both
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
    # 5) Plot (two sub‑plots)
    # ------------------------------------------------------------------
    if ax is None:                     # create a new figure with two axes
        fig, (ax_int, ax_vis) = plt.subplots(
            1, 2, figsize=(12, 4), constrained_layout=True
        )
    else:                               # user supplied a single Axes → make a twin
        fig = ax.figure
        ax_int = ax
        ax_vis = fig.add_axes([0.55, 0.15, 0.4, 0.75])   # manual placement

    # ---- intensity ----------------------------------------------------
    ax_int.plot(satlas_df["r(mas)"], phoenix_I,
                label="Phoenix", color="tab:blue")
    ax_int.plot(satlas_df["r(mas)"], satlas_df["I/I0_H"],
                label="Satlas (H)", color="tab:red", linestyle="--")
    ax_int.set_xlabel(r"$r\;(\mathrm{mas})$")
    ax_int.set_ylabel(r"$I(\mu)/I(1)$")
    ax_int.set_title(f"Intensity – filter {filter_name}")
    ax_int.grid(True, which="both", ls=":", alpha=0.6)
    ax_int.legend()

    # ---- visibility ----------------------------------------------------
    ax_vis.plot(u_grid, vis_phoenix**2,
                label="Phoenix", color="tab:blue")
    ax_vis.plot(u_grid, vis_satlas,
                label="Satlas (H)", color="tab:red", linestyle="--")
    ax_vis.set_xlabel(r"$u\;(\mathrm{mas}^{-1})$")
    ax_vis.set_ylabel(r"Visibility Squared")
    ax_vis.set_title("Visibility (Hankel transform)")
    ax_vis.grid(True, which="both", ls=":", alpha=0.6)
    ax_vis.legend()

    # ------------------------------------------------------------------
    # 6) Save / show
    # ------------------------------------------------------------------
    if save_as:
        plt.savefig(save_as, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    # Returning the intensity axis is kept for backward compatibility.
    return ax_int


# ----------------------------------------------------------------------
# Example usage (run the file directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    plot_intensity(
        filter_name="H",
        satlas_path="data/output_ld-satlas_1762763642809/ld_df.txt",
        table12_path="data/phoenix/table12.dat",
        save_as="phoenix_vs_satlas_H.png",
        show=False,
    )
    print("Plot saved as phoenix_vs_satlas_H.png")