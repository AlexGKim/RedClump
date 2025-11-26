import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional

import pandas as pd

def read_table_b1(filepath):
    """
    Reads the TableB1.txt file and returns it as a pandas DataFrame.
    
    Parameters:
    -----------
    filepath : str
        Path to the TableB1.txt file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the astronomical data
    """
    # Read the file and skip the header to parse manually
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the header line and data lines
    header_idx = None
    for i, line in enumerate(lines):
        if 'Star' in line and 'K(mag)' in line:
            header_idx = i
            break
    
    if header_idx is None:
        header_idx = 0  # Assume first line is header
    
    # Parse data rows
    data = []
    for line in lines[header_idx + 1:]:
        if line.strip():  # Skip empty lines
            parts = line.split()
            if len(parts) >= 11:  # Valid data row
                # Star name is "HD" + number (first two parts)
                star = parts[0] + ' ' + parts[1]
                k_mag = parts[2]
                h_mag = parts[3]
                parallax = ' '.join(parts[4:7])  # e.g., "8.97 ± 0.13"
                ebv = parts[7]
                date = parts[8]
                baselines = parts[9]
                sp_channels = parts[10]
                bin_flag = parts[-1]
                calibrator = ' '.join(parts[11:-1])
                
                data.append([star, k_mag, h_mag, parallax, ebv, date, 
                           baselines, sp_channels, calibrator, bin_flag])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'Star', 'K(mag)', 'H(mag)', 'π(mas)', 'E(B–V)', 'Date',
        'Baselines', 'Sp. channels', 'Calibrator HD', 'Bin.'
    ])
    
    # Convert numeric columns
    df['K(mag)'] = pd.to_numeric(df['K(mag)'], errors='coerce')
    df['H(mag)'] = pd.to_numeric(df['H(mag)'], errors='coerce')
    df['E(B–V)'] = pd.to_numeric(df['E(B–V)'], errors='coerce')
    df['Sp. channels'] = pd.to_numeric(df['Sp. channels'], errors='coerce')
    
    # Replace em-dash with NaN in Bin. column
    df['Bin.'] = df['Bin.'].replace('–', None)
    
    return df



def get_table1_df() -> pd.DataFrame:
    """
    Load the Table1.txt file into a pandas DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing all columns from Table1.txt with values and uncertainties.
    """
    print("Loading stellar parameters from Table1.txt...")
    table1_data = []
    
    with open('data/Table1.txt', 'r') as f:
        lines = f.readlines()
        
        # Find header line
        header_idx = 0
        for i, line in enumerate(lines):
            if 'HD' in line and 'θLD' in line:
                header_idx = i
                break
        
        # Parse data rows
        for line in lines[header_idx + 1:]:
            if line.strip():  # Skip empty lines
                fields = line.split('\t')
                if len(fields) >= 8:
                    hd_num = fields[0].strip()
                    
                    # Helper function to parse value ± uncertainty
                    def parse_value_uncertainty(s):
                        s = s.strip().replace('−', '-')
                        if '±' in s:
                            parts = s.split('±')
                            val = float(parts[0].strip())
                            unc = float(parts[1].strip()) if len(parts) > 1 else None
                            return val, unc
                        else:
                            return float(s) if s else None, None
                    
                    theta_ld, theta_ld_unc = parse_value_uncertainty(fields[1])
                    log_teff, log_teff_unc = parse_value_uncertainty(fields[2])
                    logg, logg_unc = parse_value_uncertainty(fields[3])
                    feh, feh_unc = parse_value_uncertainty(fields[4])
                    ebv, ebv_unc = parse_value_uncertainty(fields[5])
                    log_r, log_r_unc = parse_value_uncertainty(fields[6])
                    log_l, log_l_unc = parse_value_uncertainty(fields[7])
                    
                    # Calculate Teff from log Teff
                    teff = 10**log_teff if log_teff is not None else None
                    
                    table1_data.append({
                        'Star': f"HD {hd_num}",
                        'theta_ld': theta_ld,
                        'theta_ld_unc': theta_ld_unc,
                        'log_teff': log_teff,
                        'log_teff_unc': log_teff_unc,
                        'teff': teff,
                        'logg': logg,
                        'logg_unc': logg_unc,
                        'feh': feh,
                        'feh_unc': feh_unc,
                        'ebv': ebv,
                        'ebv_unc': ebv_unc,
                        'log_r': log_r,
                        'log_r_unc': log_r_unc,
                        'log_l': log_l,
                        'log_l_unc': log_l_unc
                    })
    
    return pd.DataFrame(table1_data)

def master_df():
    # Load your data

    gaia_df = pd.read_csv('data/extended_data_table_2.csv')
    table1_df = get_table1_df()
    chara_df = read_table_b1('data/TableB1.txt')

    # # Get the star names from each dataframe
    # gaia_stars = set(gaia_df['Star'])
    # table1_stars = set(table1_df['Star'])
    # chara_stars = set(chara_df['Star'])
    # # Create a summary dataframe
    # all_stars = gaia_stars | table1_stars | chara_stars

    # summary = pd.DataFrame({
    #     'Star': sorted(all_stars),
    #     'in_gaia': [s in gaia_stars for s in sorted(all_stars)],
    #     'in_table1': [s in table1_stars for s in sorted(all_stars)],
    #     'in_chara': [s in chara_stars for s in sorted(all_stars)]
    # })

    # # Add a column showing how many dataframes contain each star
    # summary['count'] = summary[['in_gaia', 'in_table1', 'in_chara']].sum(axis=1)

    # # Filter to show stars not in all three
    # stars_not_in_all = summary[summary['count'] < 3]
    # print(stars_not_in_all)

    # # Summary statistics
    # print(f"\nStars in all 3 dataframes: {(summary['count'] == 3).sum()}")
    # print(f"Stars in exactly 2 dataframes: {(summary['count'] == 2).sum()}")
    # print(f"Stars in exactly 1 dataframe: {(summary['count'] == 1).sum()}")

    combined_df = gaia_df.merge(table1_df, on='Star', how='inner') \
                     .merge(chara_df, on='Star', how='inner')
    return combined_df

  

if __name__ == "__main__":
    df = master_df()
    # print(df.columns.tolist())
    print(df['ebv'])
