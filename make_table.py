import pandas as pd
from astroquery.vizier import Vizier

import matplotlib.pyplot as plt

import pandas as pd
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy
from gaia_zeropoint import (VEGA_FLUX_G, VEGA_FLUX_BP, VEGA_FLUX_RP,
                           GAIA_G_EFFECTIVE_WAVELENGTH, GAIA_BP_EFFECTIVE_WAVELENGTH, GAIA_RP_EFFECTIVE_WAVELENGTH,
                           GAIA_G_EFFECTIVE_FREQUENCY, GAIA_BP_EFFECTIVE_FREQUENCY, GAIA_RP_EFFECTIVE_FREQUENCY)


def create_and_save_dataframe_gaia(filename='extended_data_table_2.csv'):
    # Define the data
    data = [
        ['HD 360', 0.906, 0.014, 5.986, 0.005, 3.653, 0.009, 5.744, 0.036],
        ['HD 3750', 1.003, 0.019, 6.004, 0.005, 3.485, 0.002, 6.004, 0.043],
        ['HD 4211', 1.100, 0.009, 5.877, 0.028, 3.295, 0.004, 6.072, 0.035],
        ['HD 5722', 0.995, 0.018, 5.618, 0.012, 3.381, 0.010, 5.576, 0.042],
        ['HD 8651', 1.228, 0.011, 5.418, 0.006, 3.019, 0.002, 5.858, 0.023],
        ['HD 9362', 2.301, 0.017, 3.943, 0.006, 1.638, 0.000, 5.753, 0.020],
        ['HD 10142', 0.964, 0.004, 5.938, 0.009, 3.557, 0.007, 5.837, 0.017],
        ['HD 11977', 1.528, 0.010, 4.686, 0.011, 2.486, 0.002, 5.600, 0.021],
        ['HD 12438', 1.091, 0.015, 5.344, 0.007, 3.176, 0.004, 5.521, 0.033],
        ['HD 13468', 0.886, 0.009, 5.934, 0.013, 3.666, 0.009, 5.643, 0.028],
        ['HD 15220', 1.185, 0.015, 5.881, 0.006, 3.199, 0.007, 6.228, 0.030],
        ['HD 15248', 0.949, 0.018, 6.001, 0.003, 3.553, 0.010, 5.856, 0.043],
        ['HD 15779', 1.185, 0.013, 5.357, 0.013, 3.067, 0.006, 5.707, 0.029],
        ['HD 16815', 2.248, 0.009, 4.109, 0.003, 1.706, 0.000, 5.868, 0.014],
        ['HD 17652', 1.835, 0.010, 4.456, 0.005, 2.139, 0.001, 5.771, 0.017],
        ['HD 17824', 1.391, 0.013, 4.764, 0.010, 2.668, 0.002, 5.474, 0.025],
        ['HD 18784', 1.036, 0.013, 5.748, 0.004, 3.353, 0.014, 5.781, 0.030],
        ['HD 23319', 2.033, 0.010, 4.583, 0.005, 1.995, 0.001, 6.121, 0.016],
        ['HD 23526', 0.915, 0.020, 5.909, 0.008, 3.634, 0.017, 5.663, 0.049],
        ['HD 23940', 1.093, 0.020, 5.543, 0.007, 3.229, 0.002, 5.730, 0.042],
        ['HD 30814', 1.310, 0.008, 5.041, 0.010, 2.791, 0.006, 5.609, 0.020],
        ['HD 36874', 1.118, 0.010, 5.768, 0.010, 3.242, 0.002, 6.004, 0.024],
        ['HD 39523', 1.939, 0.013, 4.500, 0.006, 2.036, 0.001, 5.935, 0.019],
        ['HD 39640', 1.251, 0.016, 5.163, 0.005, 2.921, 0.006, 5.631, 0.030],
        ['HD 39910', 1.090, 0.006, 5.863, 0.005, 3.315, 0.015, 6.004, 0.017],
        ['HD 40020', 1.012, 0.022, 5.876, 0.003, 3.419, 0.013, 5.862, 0.049],
        ['HD 43899', 1.264, 0.016, 5.557, 0.015, 3.004, 0.010, 6.035, 0.033],
        ['HD 46116', 1.145, 0.030, 5.373, 0.005, 3.103, 0.009, 5.639, 0.058],
        ['HD 53629', 1.065, 0.023, 6.085, 0.006, 3.410, 0.017, 6.169, 0.049],
        ['HD 56160', 1.411, 0.010, 5.580, 0.000, 2.823, 0.010, 6.297, 0.019],
        ['HD 60060', 0.948, 0.009, 5.872, 0.002, 3.545, 0.018, 5.700, 0.023],
        ['HD 60341', 1.190, 0.021, 5.645, 0.015, 3.126, 0.010, 5.992, 0.043],
        ['HD 62713', 1.446, 0.010, 5.134, 0.017, 2.654, 0.005, 5.919, 0.025],
        ['HD 68312', 1.020, 0.022, 5.359, 0.008, 3.279, 0.011, 5.368, 0.049],
        ['HD 74622', 1.020, 0.014, 6.279, 0.010, 3.532, 0.013, 6.282, 0.033],
        ['HD 75916', 1.013, 0.020, 6.117, 0.009, 3.516, 0.008, 6.120, 0.045],
        ['HD 176704', 1.317, 0.010, 5.645, 0.005, 2.956, 0.007, 6.221, 0.020],
        ['HD 191584', 1.024, 0.021, 6.211, 0.012, 3.512, 0.009, 6.235, 0.047],
        ['HD 219784', 2.117, 0.023, 4.412, 0.008, 1.886, 0.001, 6.038, 0.025],
        ['HD 220572', 1.092, 0.012, 5.605, 0.020, 3.224, 0.003, 5.787, 0.027],
        ['HD 204381', 1.524, 0.015, 4.501, 0.008, 2.426, 0.001, 5.413, 0.033]
    ]
    
    # Define column names
    columns = ['Star', 'LD', 'σLD', 'V', 'σV', 'K', 'E(B-V)', 'Sv', 'σSv']
    
    # Create the DataFrame
    df = pd.DataFrame(data, columns=columns)
    df["Star"] = df["Star"].str.strip()

    # List of HD stars from your table
    hd_stars = df["Star"].tolist()

    # Configure SIMBAD to return Right Ascension and Declination.
    Simbad.add_votable_fields('ra', 'dec')

    # Query SIMBAD for all stars in the hd_stars list.
    simbad_result = Simbad.query_objects(hd_stars)

    # Convert the SIMBAD result to a pandas DataFrame.
    df_simbad = simbad_result.to_pandas()

    print("SIMBAD results (first few rows):")
    print(df_simbad.head())

    # Prepare a list to store Gaia photometry and coordinates results.
    gaia_results_list = []

    # Loop over each star from the SIMBAD query results.
    for index, row in df_simbad.iterrows():
        # Convert the SIMBAD RA and DEC into a SkyCoord object.
        coord = SkyCoord(ra=row['ra'], dec=row['dec'], unit=(u.deg, u.deg))
        print(row['user_specified_id'], coord)
        # Perform a cone search around this coordinate with a radius of 2 arcseconds.
        radius = 5 * u.arcsec
        job = Gaia.cone_search_async(coord, radius=radius)
        gaia_result = job.get_results()
        
        # If there is at least one match, choose the first one.
        if len(gaia_result) > 1:
            # Find the brightest object (lowest phot_g_mean_mag)
            phot_array = numpy.array(gaia_result['phot_g_mean_mag'])
            brightest_index = phot_array.argmin()
            source = gaia_result[brightest_index]
        elif len(gaia_result) > 0:
            source = gaia_result[0]
        else:
            raise Exception('No Gaia star found')
                    
        ra_gaia = source['ra']
        dec_gaia = source['dec']
        phot_g = source['phot_g_mean_mag']
        phot_bp = source['phot_bp_mean_mag']
        phot_rp = source['phot_rp_mean_mag']
        parallax = source['parallax']
        parallax_error = source['parallax_error']


        # Append the retrieved information to the results list.
        gaia_results_list.append({
            'Star': row['user_specified_id'],
            'RA_SIMBAD': row['ra'],
            'DEC_SIMBAD': row['dec'],
            'RA_Gaia': ra_gaia,
            'DEC_Gaia': dec_gaia,
            'phot_g_mean_mag': phot_g,
            'phot_bp_mean_mag': phot_bp,
            'phot_rp_mean_mag': phot_rp,
            'parallax': parallax,
            'parallax_error': parallax_error
        })

    # Create a pandas DataFrame from the Gaia results list.
    df_gaia = pd.DataFrame(gaia_results_list)

    df_gaia["Star"] = df_gaia["Star"].str.strip()

    # Display the final DataFrame showing both SIMBAD and Gaia coordinates and photometry.
    print("\nGaia Photometry and Coordinates for HD Stars:")
    print(df_gaia)

    # Merge the original DataFrame 'df' with 'df_gaia' on the "Star" column.
    df_merged = pd.merge(df, df_gaia, on="Star", how="left")
    print("\nMerged DataFrame (df merged with Gaia data):")
    print(df_merged)
    
    # Save the merged DataFrame to CSV.
    df_merged.to_csv(filename, index=False)
    print(f"DataFrame saved to '{filename}'.")
    return df_merged


def load_dataframe(filename='extended_data_table_2.csv'):
    """
    Reads the saved CSV file and returns the pandas DataFrame.
    
    Parameters:
        filename (str): The name of the CSV file to load.
    
    Returns:
        pd.DataFrame: The DataFrame read from the CSV file.
    """
    try:
        df = pd.read_csv(filename)
        print(f"DataFrame loaded from '{filename}'.")
        return df
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return None



def create_and_save_flux_density_table(filename='extended_data_table_2.csv', output_filename='flux_density_table.csv'):
    """
    Creates a table with Star names and F_nu flux densities for the three Gaia bands.
    
    Uses the formula: flux_density = VEGA_FLUX * 10**(-0.4 * magnitude)
    where VEGA_FLUX values are taken from gaia_zeropoint.py
    
    Parameters:
        filename (str): The name of the CSV file to load star data from.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Star', 'F_nu_G', 'F_nu_BP', 'F_nu_RP']
                     Flux densities are in units of W/(m^2 * Hz)
    """
    # Load the existing dataframe with Gaia photometry
    df = load_dataframe(filename)
    
    if df is None:
        print("Error: Could not load dataframe")
        return None
    
    # Calculate flux densities using the formula: flux_density = VEGA_FLUX * 10**(-0.4 * magnitude)
    # For G band
    df['F_nu_G'] = VEGA_FLUX_G * 10**(-0.4 * df['phot_g_mean_mag'])
    
    # For BP band
    df['F_nu_BP'] = VEGA_FLUX_BP * 10**(-0.4 * df['phot_bp_mean_mag'])
    
    # For RP band
    df['F_nu_RP'] = VEGA_FLUX_RP * 10**(-0.4 * df['phot_rp_mean_mag'])
    
    # Create a new dataframe with only the required columns
    flux_table = df[['Star', 'F_nu_G', 'F_nu_BP', 'F_nu_RP']].copy()
    
    print("Flux density table (units: W/(m^2 * Hz)):")
    print(flux_table.to_string(index=False))

    flux_table.to_csv(output_filename, index=False)
    print(f"Flux density table saved to '{output_filename}'.")
    
    return flux_table    
    return flux_table

# def main():
    # Test the new flux density table function

    
    # Load the DataFrame from the saved file
    # df = load_dataframe()
    # df['distance'] = 1000 / df['parallax'] # parsec

    # print(df[['Star', 'phot_rp_mean_mag', 'distance', 'LD']].to_string(index=False))
    # # fwe  # Removed this line as it seems to be a typo
    # # df['M_rp'] = df['phot_rp_mean_mag'] - 5*numpy.log10(df['distance']/10)
    # df['SB_rp'] = df['phot_rp_mean_mag'] + 2.5 * numpy.log10(numpy.pi * (df['LD']/2)**2)
    # # plt.scatter(df['distance'], df['SB_rp'] )
    # # plt.xlabel('distance (pc)')
    # # plt.ylabel('SB_rp')
    # # plt.show()

    # plt.scatter(df['distance'], df['distance'] * df['LD'])
    # plt.show()

    # plt.hist(df["LD"])
    # plt.xlabel("Angular Diameter (mas)")
    # plt.show()
# Example usage:
if __name__ == "__main__":

    # Create and save the DataFrame
    # extended_table = create_and_save_dataframe_gaia()

    # Example: Create flux density table
    # print("\n" + "="*50)
    # print("Creating flux density table...")
    # print("="*50)
    # flux_table = create_and_save_flux_density_table()
    
    # Create baseline tables
    print("\n" + "="*50)
    print("Creating baseline tables...")
    print("="*50)
    create_and_save_baseline_tables()


