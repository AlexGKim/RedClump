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
def plot_intensity(
    filter_name: str,
    satlas_path: str | pathlib.Path,
    table12_path: str | pathlib.Path,
    *,
    mu_grid: int = 200,
    ax: plt.Axes | None = None,
    save_as: str | pathlib.Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot the Phoenix intensity profile together with the Satlas reference.

    Parameters
    ----------
    filter_name : str
        One of ``u v b y U B V R I J H K`` (case‑sensitive). Determines which
        set of a‑coefficients to use.
    satlas_path : Path‑like
        Path to the Satlas intensity file (two‑column text file).
    table12_path : Path‑like
        Path to ``table12.dat``.
    mu_grid : int, optional
        Number of μ points for the Phoenix curve (default 200).
    ax : matplotlib Axes, optional
        Existing Axes to draw on; otherwise a new figure is created.
    save_as : Path‑like, optional
        If supplied, the figure is saved to this filename.
    show : bool, optional
        Call ``plt.show()`` at the end if True.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
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
    teff = 4800
    logg = 2.5
    feh = 1.0
    teff_phoenix, logg_phoenix, feh_phoenix = match_phoenix_parameters(teff, logg, feh)
    
    ans = coeffs_df[(coeffs_df["Teff"] == teff_phoenix) & (coeffs_df["logg"] == logg_phoenix)]

    # Take the first row (or you could select by logg/Teff etc.).
    coeff_row = ans[needed]
    # Rename to generic a1‑a4 so the intensity helper can use a fixed key set.
    coeff_row.columns = ["a1", "a2", "a3", "a4"]


    # ------------------------------------------------------------------
    # 3) Load Satlas data (first two columns: μ and I/I0_H)
    # ------------------------------------------------------------------
    satlas_df = pd.read_csv(
        pathlib.Path(satlas_path),
        delim_whitespace=True,
        header=0,
        comment="#",
    )

    angle = satlas_df["r(mas)"].values * (np.pi / 180 / 3600 / 1000)  # radians

    # Interpolate Satlas onto the same μ grid for a clean overlay.
    # satlas_I = np.interp(mu, satlas_df["r(mas)"].values, satlas_df["I/I0_H"].values)

    # ------------------------------------------------------------------
    # 2) Build μ grid and compute Phoenix profile
    # ------------------------------------------------------------------
    
    mu = np.sqrt(1- angle**2/radius_m**2*(distance_m**2))

    phoenix_I = _phoenix_intensity(mu, coeff_row.iloc[0])


    # ------------------------------------------------------------------
    # 4) Plot
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(satlas_df["r(mas)"], phoenix_I, label="Phoenix", color="tab:blue")
    ax.plot(satlas_df["r(mas)"], satlas_df["I/I0_H"], label="Satlas (H)", color="tab:red", linestyle="--")
    ax.set_xlabel(r"r(mas)")
    ax.set_ylabel(r"$I(\mu)/I(1)$")
    ax.set_title(f"Intensity profile – filter {filter_name}")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    if save_as:
        plt.savefig(save_as, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    return ax


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