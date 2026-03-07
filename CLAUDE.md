# CLAUDE.md — direct_plot.py

## Project Overview

This module computes and plots **stellar limb-darkening intensity profiles** for
two atmospheric model grids — **Phoenix** and **SATLAS (ATLAS9)** — and derives
the corresponding **interferometric visibility curves** (|V|²) via a numerical
Hankel transform. A secondary capability computes the Fisher information
sensitivity F_ss^{-1/2} (angular-diameter uncertainty σ_s) as a function of
interferometric baseline length.

The science context is optical/near-IR stellar interferometry: given a model
intensity profile I(θ), the complex visibility is the zero-order Hankel
transform of I(θ), and its derivative with respect to the angular scale s drives
the precision with which the stellar angular diameter can be measured.

---

## Paper

This code is directly tied to the paper located in the `paper/` directory.

- The **implementation** (physics, formulae, and conventions used throughout
  this module) is based on the methodology described in that paper.
- The **figures and numerical results** produced by running this module are
  used directly in the paper.

When modifying this code, ensure that any changes to physical conventions,
units, or numerical methods remain consistent with the paper. If a result or
plot changes, the corresponding figure or table in `paper/` will need to be
updated accordingly.

---

## Repository Layout (relevant files)

```
.
├── direct_plot.py          # This module — all plotting and Hankel-transform logic
├── read_table12.py         # Parser for the fixed-width Phoenix coefficient file
├── gaia_phoenix.py         # match_phoenix_parameters(Teff, logg, FeH) helper
├── rc_utils.py             # master_df_with_inverse_noise(), get_filter_properties_df()
├── paper/                  # The associated paper (source of methodology and sink of results)
└── data/
    ├── phoenix/
    │   └── table12.dat                  # Claret (2000) Phoenix LD coefficients
    └── output_ld-satlas_<timestamp>/
        └── ld_satlas_surface.2t<Teff>g<logg>m<FeH>_Ir_all_bands.txt
```

---

## Physical / Mathematical Conventions

### Phoenix Limb-Darkening Law
```
I(μ) / I(1) = 1 − Σ_{k=1}^{4}  aₖ · (1 − μ^{k/2})
```
where μ = cos(angle from disc centre), aₖ are from `table12.dat`, and
`μ² = 1 − (d · θ / R)²` with d = distance, θ = angular radius, R = stellar
radius.

### Hankel (Abel) Transform for Visibility
```
V(u) = ∫ I(θ) J₀(2πu θ) θ dθ  /  ∫ I(θ) θ dθ
```
- u is the spatial frequency in **mas⁻¹**.
- The plotted quantity is |V|² (squared visibility amplitude).
- Numerical integration uses `np.trapezoid`.

### Visibility Scale Derivative
```
∂V/∂s = −(2πu/s) · [∫ I(θ) J₁(2πuθ) θ² dθ]  /  [∫ I(θ) θ dθ]
∂|V|²/∂s = 2V · ∂V/∂s
```

### Fisher Sensitivity (σ_s)
```
F_ss  = (∂|V|²/∂s · σ_noise⁻¹)²
σ_s   = F_ss^{−1/2}
```
Baseline B (metres) relates to spatial frequency via `u = B / λ₀`.

---

## Hardcoded Stellar Parameters

| Parameter        | Value              | Notes                            |
|------------------|--------------------|----------------------------------|
| `radius_rsun`    | 9.312 R☉           | HD 360                           |
| `distance_pc`    | 95.594 pc          | HD 360                           |
| Teff             | 4 800 K            | Used to select Phoenix row        |
| logg             | 2.5                | Used to select Phoenix row        |
| [Fe/H]           | 1.0 (input to `match_phoenix_parameters`) | |

When adding a new star, update these values or refactor them into function
parameters.

---

## Key Functions

| Function | Purpose |
|---|---|
| `_phoenix_intensity(mu2, coeffs)` | Evaluate Phoenix LD law on an array of μ² values |
| `plot_intensity(filter_names, satlas_path, table12_path, ...)` | Three-panel plot: intensity profile, |V|², Δ|V|² |
| `plot_satlas_vis2_all_bands(satlas_path, ...)` | |V|² for every band column in a Satlas file |
| `compute_visibility_derivative(intensity, scaled_rays, u_grid, s)` | ∂V/∂s via J₁ Hankel integral |
| `plot_satlas_derivative_all_bands(satlas_path, ...)` | ∂V/∂s and ∂|V|²/∂s for all bands |
| `plot_fss_inverse_sqrt(satlas_path, df_with_noise, filter_df, ...)` | σ_s vs baseline B per filter |
| `_strip_prefix(col_name)` | Remove `"I/I0_"` prefix from Satlas column names for legend labels |

---

## Filter ↔ Satlas Column Mapping

| Filter key | Satlas column  |
|------------|----------------|
| `'B'`      | `I/I0_B`       |
| `'V'`      | `I/I0_V`       |
| `'R'`      | `I/I0_R`       |
| `'I'`      | `I/I0_I`       |
| `'H'`      | `I/I0_H`       |
| `'K'`      | `I/I0_K`       |

The Phoenix coefficient columns follow the pattern `a1<filter>`, `a2<filter>`,
`a3<filter>`, `a4<filter>` (e.g., `a1H`, `a2H`, …).

---

## Data File Formats

### `table12.dat` (Phoenix coefficients)
Fixed-width format parsed by `read_table12.read_table12()`. Expected columns
include `Teff`, `logg`, and `a1<X>` … `a4<X>` for each filter X.

### Satlas surface-intensity file
Whitespace-separated, first row is a header. First column must be `r(mas)`;
remaining columns are named `I/I0_<Band>`. Comment lines begin with `#`.

A synthetic zero-intensity row is appended at `r = 1.0000001 × R★/d` (in mas)
before any Hankel integration to ensure the profile reaches zero cleanly at the
stellar limb.

---

## External Dependencies

```
numpy
pandas
matplotlib
scipy          # scipy.special.j0, j1
```

Custom modules (must be importable):
- `read_table12`  — `read_table12(path) → pd.DataFrame`
- `gaia_phoenix`  — `match_phoenix_parameters(Teff, logg, FeH) → (Teff, logg, FeH)`
- `rc_utils`      — `master_df_with_inverse_noise() → pd.DataFrame`,
                    `get_filter_properties_df() → pd.DataFrame`

`rc_utils.get_filter_properties_df()` must return a DataFrame with at least
columns `Filter` (string) and `ν_eff_Hz` (effective frequency in Hz).

`rc_utils.master_df_with_inverse_noise()` must return a DataFrame with columns
`Star` and `<Filter>_inv_noise` (one per filter).

---

## Output Files

| File | Produced by |
|---|---|
| `phoenix_vs_satlas_BH.pdf` | `plot_intensity(['B','H'], ...)` |
| `satlas_vis2_all_bands.pdf` | `plot_satlas_vis2_all_bands(...)` |
| `satlas_derivative_all_bands.pdf` | `plot_satlas_derivative_all_bands(...)` |
| `fss_hd360.pdf` | `plot_fss_inverse_sqrt(..., star_name="HD 360")` |
| `fss_hd17652.pdf` | `plot_fss_inverse_sqrt(..., star_name="HD 17652")` |

---

## Running the Module

```bash
python direct_plot.py
```

All `show=False` calls suppress interactive display; plots are saved directly to
PDF. Set `show=True` to display interactively.

---

## Common Gotchas

- **μ² can go negative** for θ beyond the stellar limb. `_phoenix_intensity`
  masks these with `w = mu2 >= 0.0` and returns zero there. Do not skip this
  guard when modifying the intensity evaluation.
- **Unit consistency**: `u_grid` is in **mas⁻¹** throughout the Hankel
  integrals. Conversion to radians⁻¹ (needed for `B = u × λ₀`) uses
  `u_rad = u_mas / mas_to_rad` (division, not multiplication).
- **`np.trapezoid`** is used (NumPy ≥ 2.0 name); earlier NumPy versions export
  this as `np.trapz`.
- The zero-row extension of the Satlas profile must be performed **before**
  computing any Hankel integrals, otherwise the profile is not properly
  truncated at the limb.
- `plot_intensity` returns the **intensity axis** (`ax_int`) for backward
  compatibility, not the figure.