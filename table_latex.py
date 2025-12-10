import pandas as pd
import numpy as np

def format_value_with_uncertainty(value, uncertainty):
    """Format a value with its uncertainty using \pm notation."""
    return f"${value:.3f} \\pm {uncertainty:.3f}$"

def generate_latex_table(csv_file='data/extended_data_table_2.csv'):
    """
    Generate a LaTeX table from extended_data_table_2.csv.
    
    Columns: Star, LD, V, K, RA_Gaia, DEC_Gaia, phot_g_mean_mag, 
             phot_bp_mean_mag, phot_rp_mean_mag
    
    Parameters with uncertainties (LD, V) are combined using \pm notation.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
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
    header = "Star & LD & V & K & RA$_{\\rm Gaia}$ & DEC$_{\\rm Gaia}$ & $G$ & $G_{\\rm BP}$ & $G_{\\rm RP}$ \\\\"
    latex.append(header)
    latex.append(" & (mas) & (mag) & (mag) & (deg) & (deg) & (mag) & (mag) & (mag) \\\\")
    latex.append("\\hline")
    
    # Data rows
    for idx, row in df.iterrows():
        star = row['Star']
        
        # Combine LD with its uncertainty
        ld = format_value_with_uncertainty(row['LD'], row['σLD'])
        
        # Combine V with its uncertainty
        v = format_value_with_uncertainty(row['V'], row['σV'])
        
        # K magnitude (no uncertainty column in the specified output)
        k = f"${row['K']:.3f}$"
        
        # Gaia coordinates
        ra_gaia = f"${row['RA_Gaia']:.6f}$"
        dec_gaia = f"${row['DEC_Gaia']:.6f}$"
        
        # Gaia photometry
        g_mag = f"${row['phot_g_mean_mag']:.4f}$"
        bp_mag = f"${row['phot_bp_mean_mag']:.4f}$"
        rp_mag = f"${row['phot_rp_mean_mag']:.4f}$"
        
        # Build the row
        data_row = f"{star} & {ld} & {v} & {k} & {ra_gaia} & {dec_gaia} & {g_mag} & {bp_mag} & {rp_mag} \\\\"
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