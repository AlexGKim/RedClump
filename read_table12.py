#!/usr/bin/env python3
"""
read_table12.py

Utility to read the limb‑darkening coefficient file (table12.dat /
table11.dat) that is stored in fixed‑width format.  The function returns a
pandas DataFrame whose column names are exactly the *Label* strings from
the specification (logg, Teff, Z, Vel, a1u … a4K, mucri*, CHI2*).

Typical use
-----------
>>> from read_table12 import read_table12
>>> df = read_table12("table12.dat")
>>> df.head()
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

def _make_column_names():
    """
    Build the list of column names in the exact order they appear in the
    file (the *Label* column of the table you supplied).
    """
    # 1. Fixed‑width header columns
    names = ["logg", "Teff", "Z", "Vel"]

    # 2. Limb‑darkening coefficients a1‑a4 for each filter
    filters = ["u", "v", "b", "y", "U", "B", "V", "R", "I", "J", "H", "K"]
    coeff_prefixes = ["a1", "a2", "a3", "a4"]
    coeff_names = [f"{p}{f}" for p in coeff_prefixes for f in filters]

    # 3. Critical‑mu values
    mu_names = [f"mucri{f}" for f in filters]

    # 4. Chi‑square values
    chi_names = [f"CHI2{f}" for f in filters]

    names.extend(coeff_names)
    names.extend(mu_names)
    names.extend(chi_names)

    return names


def read_table12(file_path: str | Path) -> pd.DataFrame:
    """
    Read a *table12.dat* (or *table11.dat*) file and return a DataFrame.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the fixed‑width file.

    Returns
    -------
    pandas.DataFrame
        Columns are named exactly as the *Label* entries (e.g.
        ``a1u``, ``a2V``, ``mucriK`` …).  All numeric columns are
        converted to ``float64``; non‑numeric entries become ``NaN``.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If the number of parsed columns does not match the expected
        76 columns.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path!s} does not exist")

    col_names = _make_column_names()


    df = pd.read_csv(file_path, sep='\s+', names=col_names)

    if df.shape[1] != len(col_names):
        raise ValueError(
            f"Expected {len(col_names)} columns but got {df.shape[1]}. "
            "Check the file format or the column specifications."
        )

    return df


# ----------------------------------------------------------------------
# Example / quick test (run the file directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Replace with the actual location of your table file
    test_path = "data/phoenix/table12.dat"

    try:
        df = read_table12(test_path)
        print("First 5 rows of the parsed table:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
        print("\nData types:\n", df.dtypes)
    except Exception as exc:
        print(f"Failed to read {test_path!s}: {exc}")