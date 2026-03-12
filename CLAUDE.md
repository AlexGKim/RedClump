# CLAUDE.md — direct_plot.py

## Purpose

Computes and plots stellar limb-darkening intensity profiles (Phoenix and SATLAS/ATLAS9
grids), derives interferometric visibility curves (|V|^2) via a numerical Hankel transform,
and estimates angular-diameter uncertainty sigma_s (Fisher sensitivity F_ss^{-1/2}) as a
function of baseline. Science context: optical/near-IR intensity interferometry of Red Clump
stars to validate the surface-brightness-color (SBC) relation.

---

## Paper Connection

- Physics, formulae, and conventions follow the methodology in `paper/`.
- If a result or plot changes, update the corresponding figure or table in `paper/`.

---

## Repository Layout

```
.
├── direct_plot.py          # This module — all plotting and Hankel-transform logic
├── read_table12.py         # Parser for the fixed-width Phoenix coefficient file
├── gaia_phoenix.py         # match_phoenix_parameters(Teff, logg, FeH) helper
├── rc_utils.py             # master_df_with_inverse_noise(), get_filter_properties_df()
├── paper/                  # Associated paper (source of methodology, sink of results)
└── data/
    ├── phoenix/
    │   └── table12.dat     # Claret (2000) Phoenix LD coefficients
    ├── output_ld-satlas_1762763642809/
    │   └── ld_satlas_surface.2t4800g250m10_Ir_all_bands.txt   # HD 360 (logg=2.50)
    └── output_ld-satlas_1764290684148/
        └── ld_satlas_surface.2t4800g275m10_Ir_all_bands.txt   # HD 17652 (logg=2.75)
```

---

## Hardcoded Stellar Parameters

Star     | radius_rsun | distance_pc | Teff  | logg | [Fe/H] | SATLAS dir timestamp
---------|-------------|-------------|-------|------|--------|---------------------
HD 360   | 9.312       | 95.594      | 4800  | 2.50 | 1.0    | 1762763642809
HD 17652 | (in rc_utils) | (in rc_utils) | 4800 | 2.75 | 1.0 | 1764290684148

When adding a new star, update these values or refactor them into function parameters.

---

## Key Functions

Function                                                    | Purpose
------------------------------------------------------------|--------------------------------------------------
`_phoenix_intensity(mu2, coeffs)`                           | Evaluate Phoenix LD law on an array of mu^2 values
`plot_intensity(filter_names, satlas_path, ...)`            | Three-panel plot: intensity profile, |V|^2, delta|V|^2
`plot_satlas_vis2_all_bands(satlas_path, ...)`              | |V|^2 for every band column in a SATLAS file
`compute_visibility_derivative(intensity, rays, u, s)`      | dV/ds via J1 Hankel integral
`plot_satlas_derivative_all_bands(satlas_path, ...)`        | dV/ds and d|V|^2/ds for all bands
`plot_fss_inverse_sqrt(satlas_path, df, filter_df, ...)`    | sigma_s vs baseline B per filter
`create_inverse_noise_table(stars, filters, verbose, latex_format)` | Build sigma_{|V|^2}^{-1} table for a list of stars/filters
`_strip_prefix(col_name)`                                   | Remove I/I0_ prefix from SATLAS column names

---

## Common Gotchas

- **mu^2 guard**: mu^2 can go negative beyond the stellar limb. `_phoenix_intensity`
  masks with `w = mu2 >= 0.0` and returns zero there — do not remove this guard.
- **Unit consistency**: `u_grid` is in mas^{-1} throughout. Conversion to rad^{-1}
  (for B = u * lambda_0) uses `u_rad = u_mas / mas_to_rad` (divide, not multiply).
- **NumPy version**: `np.trapezoid` (NumPy ≥ 2.0); older versions use `np.trapz`.
- **Zero-row extension**: the synthetic zero-intensity row at r = 1.0000001 × R_star/d
  must be appended *before* any Hankel integral, not after.
- **Return value**: `plot_intensity` returns `ax_int` (the intensity axis) for
  backward compatibility, not the figure.
- **First null**: for some bands the first null requires impractically long baselines;
  primary-peak and secondary-maxima measurements are more robust.
- **Detector jitter/bandwidth**: verify sigma_t × delta_omega >> 1 in every
  multiplexed channel when adjusting instrument parameters.

---

## Physical / Mathematical Conventions

### Phoenix Limb-Darkening Law

```
I(mu) / I(1) = 1 - sum_{k=1}^{4} a_k * (1 - mu^{k/2})
```

mu = cos(angle from disc centre), a_k from table12.dat,
mu^2 = 1 - (theta / theta_star)^2.

### Hankel (Abel) Transform for Visibility

```
V(u) = integral I(theta) J0(2*pi*u*theta) theta dtheta
       / integral I(theta) theta dtheta
```

u is spatial frequency in mas^{-1} (u = B / lambda). Plotted quantity is |V|^2.
Numerical integration uses `np.trapezoid`.

### Visibility Scale Derivative

```
dV/ds   = -(2*pi*u/s) * [integral I(theta) J1(2*pi*u*theta) theta^2 dtheta]
                        / [integral I(theta) theta dtheta]
d|V|^2/ds = 2V * dV/ds
```

### Fisher Sensitivity (sigma_s)

```
F_ss      = sum (2V * dV/ds)^2 * sigma_{|V|^2}^{-2}
sigma_s   = F_ss^{-1/2}
```

### Noise Model (detector-jitter-dominated regime)

```
sigma_{|V|^2}^{-1} = (dGamma/dnu) * (T_obs / sigma_t)^{1/2} * (128*pi)^{-1/4}
```

dGamma/dnu = epsilon × A × F_nu / (h × nu_0); epsilon = throughput, A = collecting
area, F_nu = specific flux, T_obs = integration time, sigma_t = detector timing jitter.

---

## Data File Formats

**table12.dat** — fixed-width, parsed by `read_table12.read_table12()`. Columns include
Teff, logg, and a1<X>…a4<X> for each filter X (e.g., a1H, a2H, a3H, a4H).

**SATLAS surface-intensity file** — whitespace-separated with a header row. First column
is `r(mas)`; remaining columns are `I/I0_<Band>` (B, V, R, I, H, K). Comment lines begin
with `#`. A synthetic zero-intensity row is appended at r = 1.0000001 × R_star/d before
Hankel integration.

---

## Custom Module Contracts

```
read_table12.read_table12(path)             -> pd.DataFrame
gaia_phoenix.match_phoenix_parameters(Teff, logg, FeH) -> (Teff, logg, FeH)
rc_utils.master_df_with_inverse_noise()     -> DataFrame with columns: Star, <Filter>_inv_noise
rc_utils.get_filter_properties_df()         -> DataFrame with columns: Filter, nu_eff_Hz
```

---

## Output Files

File                             | Produced by
---------------------------------|--------------------------------------------------
`phoenix_vs_satlas_BH.pdf`       | `plot_intensity(['B','H'], ...)`
`satlas_vis2_all_bands.pdf`      | `plot_satlas_vis2_all_bands(...)`
`satlas_derivative_all_bands.pdf`| `plot_satlas_derivative_all_bands(...)`
`fss_hd360.pdf`                  | `plot_fss_inverse_sqrt(..., star_name="HD 360")`
`fss_hd17652.pdf`                | `plot_fss_inverse_sqrt(..., star_name="HD 17652")`

---

## Running the Module

```
python direct_plot.py
```

`show=False` saves plots to PDF without displaying interactively; set `show=True` to display.

---

## Quick Reference: Notional Targets

Star     | Angular diam | V mag | Notes
---------|-------------|-------|-------------------------------
HD 360   | 0.906 mas   | 5.986 | Smaller, fainter end of sample
HD 17652 | 1.835 mas   | 4.456 | Larger, brighter end of sample

H-band baselines of ~100 m reproduce PIONIER-like sensitivity; multi-baseline arrays
(e.g., CTAO, 37 SSTs) multiply effective exposure time by up to 666×.
