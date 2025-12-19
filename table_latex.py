import pandas as pd
import numpy as np

def format_value_with_uncertainty(value, uncertainty):
    r"""Format a value with its uncertainty using \pm notation."""
    return f"${value:.3f} \\pm {uncertainty:.3f}$"

def generate_latex_table(csv_file='data/extended_data_table_2.csv',
                        tableb1_file='data/TableB1.txt'):
    r"""
    Generate a LaTeX table from extended_data_table_2.csv.
    
    Columns: Star, LD, V, K, H, RA_Gaia, DEC_Gaia, phot_g_mean_mag,
             phot_bp_mean_mag, phot_rp_mean_mag
    
    Parameters with uncertainties (LD, V) are combined using \pm notation.
    H magnitude is read from TableB1.txt.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Read TableB1.txt (tab-separated)
    df_tableb1 = pd.read_csv(tableb1_file, sep='\t')
    
    # Merge the H magnitude column based on Star name
    df = df.merge(df_tableb1[['Star', 'H(mag)']], on='Star', how='left')
    
    # Start building the LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Extended Data Table 2}")
    latex.append("\\label{tab:extended_data_2}")
    latex.append("\\begin{tabular}{lcccccccc}")
    latex.append("\\hline")
    latex.append("\\hline")
    
    # Header row
    header = "Star & RA$_{\\rm Gaia}$ & DEC$_{\\rm Gaia}$ & LD & V & H & K & $G$ & $G_{\\rm BP}$ & $G_{\\rm RP}$ \\\\"
    latex.append(header)
    latex.append(" & (deg) & (deg) & (mas) & (mag) & (mag) & (mag) & (mag) & (mag) & (mag) \\\\")
    latex.append("\\hline")
    
    # Data rows
    for idx, row in df.iterrows():
        star = row['Star']
        
        # Gaia coordinates (moved to follow Star)
        ra_gaia = f"${row['RA_Gaia']:.6f}$"
        dec_gaia = f"${row['DEC_Gaia']:.6f}$"
        
        # Combine LD with its uncertainty
        ld = format_value_with_uncertainty(row['LD'], row['σLD'])
        
        # Combine V with its uncertainty
        v = format_value_with_uncertainty(row['V'], row['σV'])
        
        # H magnitude from TableB1
        h = f"${row['H(mag)']:.3f}$" if pd.notna(row['H(mag)']) else "$-$"
        
        # K magnitude (no uncertainty column in the specified output)
        k = f"${row['K']:.3f}$"
        
        # Gaia photometry
        g_mag = f"${row['phot_g_mean_mag']:.4f}$"
        bp_mag = f"${row['phot_bp_mean_mag']:.4f}$"
        rp_mag = f"${row['phot_rp_mean_mag']:.4f}$"
        
        # Build the row (reordered: Star, RA, DEC, LD, V, H, K, G, BP, RP)
        data_row = f"{star} & {ra_gaia} & {dec_gaia} & {ld} & {v} & {h} & {k} & {g_mag} & {bp_mag} & {rp_mag} \\\\"
        latex.append(data_row)
    
    # Close the table
    latex.append("\\hline")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Join all lines and print
    latex_table = "\n".join(latex)
    print(latex_table)
    
    return latex_table

if __name__ == "__main__":
    generate_latex_table()