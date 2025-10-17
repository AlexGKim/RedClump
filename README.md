# Gaia to UniformDisk Converter

This module provides functions to create `g2.models.sources.simple.UniformDisk` objects from Gaia photometric and astrometric data. It converts Gaia RP magnitudes, parallaxes, and angular diameters into the parameters needed for intensity interferometry modeling.

## Features

- **Magnitude to Flux Conversion**: Converts Gaia RP magnitudes to flux density using standard photometric zero-points
- **Angular Unit Conversion**: Converts angular diameters from milliarcseconds to radians
- **Robust Error Handling**: Validates input data and handles edge cases gracefully
- **Comprehensive Testing**: Full unit test suite ensuring accuracy and reliability
- **DataFrame Support**: Processes entire datasets efficiently using pandas

## Installation

Ensure you have the required dependencies:

```bash
pip install pandas numpy
```

The module also requires access to the `g2` library for the `UniformDisk` class.

## Quick Start

```python
import pandas as pd
from gaia_uniform_disk import create_uniform_disk_from_gaia

# Load your Gaia data
df = pd.read_csv('extended_data_table_2.csv')

# Create UniformDisk objects
star_disks = create_uniform_disk_from_gaia(df)

# Access a specific star
hd_360 = star_disks['HD 360']

# Calculate visibility for a 100m baseline
import numpy as np
baseline = np.array([100.0, 0.0, 0.0])
nu_0 = 3.76e14  # Gaia RP frequency
visibility = hd_360.V(nu_0, baseline)
print(f"Visibility: {abs(visibility):.4f}")
```

## Function Reference

### `create_uniform_disk_from_gaia(df, **kwargs)`

Main function that creates UniformDisk objects from Gaia data.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing stellar data
- `star_name_col` (str): Column name for star identifiers (default: 'Star')
- `rp_mag_col` (str): Column name for Gaia RP magnitudes (default: 'phot_rp_mean_mag')
- `parallax_col` (str): Column name for parallax in mas (default: 'parallax')
- `angular_diameter_col` (str): Column name for angular diameter in mas (default: 'LD')
- `reference_frequency` (float): Reference frequency in Hz (default: Gaia RP effective frequency)

**Returns:**
- `dict`: Dictionary mapping star names to UniformDisk objects

### `get_star_properties(star_disks, star_name)`

Get properties of a specific star's UniformDisk object.

**Parameters:**
- `star_disks` (dict): Dictionary from `create_uniform_disk_from_gaia`
- `star_name` (str): Name of the star to query

**Returns:**
- `dict`: Dictionary containing star properties (flux_density, angular_radius_rad, etc.)

## Physical Constants and Conversions

The module uses the following physical constants:

- **Gaia RP effective wavelength**: 797 nm
- **Gaia RP effective frequency**: 3.76 × 10¹⁴ Hz
- **Gaia RP zero-point magnitude**: 25.1161 mag
- **Gaia RP zero-point flux**: 1.29 × 10⁻⁹ W m⁻² Hz⁻¹

### Conversion Formulas

1. **Magnitude to Flux Density**:
   ```
   F_ν = F_ν,0 × 10^(-0.4 × (m_RP - m_0))
   ```

2. **Angular Diameter to Radius**:
   ```
   radius_rad = (diameter_mas / 2) × (π / (180 × 3600 × 1000))
   ```

3. **Parallax to Distance**:
   ```
   distance_pc = 1000 / parallax_mas
   ```

## Data Requirements

Your DataFrame must contain the following columns:

| Column | Description | Units |
|--------|-------------|-------|
| Star name | Stellar identifier | string |
| phot_rp_mean_mag | Gaia RP magnitude | magnitude |
| parallax | Parallax | milliarcseconds |
| LD | Angular diameter | milliarcseconds |

## Example Results

For the provided dataset (`extended_data_table_2.csv`):

| Star | Flux Density (W/m²/Hz) | Angular Radius (rad) | Angular Diameter (mas) | Visibility (100m) |
|------|------------------------|----------------------|------------------------|-------------------|
| HD 360 | 1.38e-01 | 2.20e-09 | 0.906 | 0.6693 |
| HD 9362 | 9.04e-01 | 5.58e-09 | 2.301 | 0.0919 |
| HD 16815 | 7.99e-01 | 5.45e-09 | 2.248 | 0.0794 |

## Testing

Run the unit tests to verify functionality:

```bash
python test_unit_conversions.py
```

Run the full example:

```bash
python full_test_example.py
```

## Error Handling

The function handles various error conditions:

- **Invalid parallax**: Negative or zero parallax values are skipped with warnings
- **Invalid angular diameter**: Zero or negative diameters are skipped
- **Missing data**: Rows with missing required data are skipped
- **Invalid DataFrame**: Proper validation of input structure

## Physical Validation

The module includes checks to ensure results are physically reasonable:

- Flux densities in the range 10⁻³⁰ to 1 W m⁻² Hz⁻¹
- Angular sizes appropriate for stellar objects
- Distances consistent with parallax measurements

## Files

- `gaia_uniform_disk.py`: Main module with conversion functions
- `test_unit_conversions.py`: Comprehensive unit tests
- `full_test_example.py`: Complete example with real data
- `README.md`: This documentation

## License

This code was generated for the RedClump project and is provided as-is for scientific use.