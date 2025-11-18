import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from pathlib import Path

class LimbDarkening:
    """
    A class to compute limb darkening profiles using 4-parameter coefficients
    from Phoenix stellar atmosphere models.
    
    Based on the 4-parameter limb darkening law and coefficients from
    https://arxiv.org/pdf/2305.01704 (Equation 3).
    """
    
    def __init__(self, data_file_path="data/phoenix/table12.dat"):
        """
        Initialize the LimbDarkening class with coefficient data.
        
        Parameters:
        -----------
        data_file_path : str
            Path to the table12.dat file containing limb darkening coefficients
        """
        self.data_file_path = data_file_path
        self.filters = ['u', 'v', 'b', 'y', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
        self.coefficients = None
        self._load_data()
    
    def _load_data(self):
        """Load and parse the coefficient data from table12.dat"""
        try:
            # Read the fixed-width format file
            # Based on the byte-by-byte description from ReadMe.txt
            colspecs = [
                (0, 5),    # logg
                (6, 12),   # Teff
                (13, 17),  # Z
                (18, 22),  # Vel
            ]
            
            # Add column specifications for a1, a2, a3, a4 coefficients for each filter
            # Following the exact byte positions from the format description
            # a1 coefficients: bytes 24-35, 37-48, 50-61, 63-74, 76-87, 89-100, 102-113, 115-126, 128-139, 141-152, 154-165, 167-178
            a1_positions = [
                (23, 35),   # a1u
                (36, 48),   # a1v
                (49, 61),   # a1b
                (62, 74),   # a1y
                (75, 87),   # a1U
                (88, 100),  # a1B
                (101, 113), # a1V
                (114, 126), # a1R
                (127, 139), # a1I
                (140, 152), # a1J
                (153, 165), # a1H
                (166, 178), # a1K
            ]
            
            # a2 coefficients: bytes 180-191, 193-204, etc.
            a2_positions = [
                (179, 191),  # a2u
                (192, 204),  # a2v
                (205, 217),  # a2b
                (218, 230),  # a2y
                (231, 243),  # a2U
                (244, 256),  # a2B
                (257, 269),  # a2V
                (270, 282),  # a2R
                (283, 295),  # a2I
                (296, 308),  # a2J
                (309, 321),  # a2H
                (322, 334),  # a2K
            ]
            
            # a3 coefficients: bytes 336-347, 349-360, etc.
            a3_positions = [
                (335, 347),  # a3u
                (348, 360),  # a3v
                (361, 373),  # a3b
                (374, 386),  # a3y
                (387, 399),  # a3U
                (400, 412),  # a3B
                (413, 425),  # a3V
                (426, 438),  # a3R
                (439, 451),  # a3I
                (452, 464),  # a3J
                (465, 477),  # a3H
                (478, 490),  # a3K
            ]
            
            # a4 coefficients: bytes 492-503, 505-516, etc.
            a4_positions = [
                (491, 503),  # a4u
                (504, 516),  # a4v
                (517, 529),  # a4b
                (530, 542),  # a4y
                (543, 555),  # a4U
                (556, 568),  # a4B
                (569, 581),  # a4V
                (582, 594),  # a4R
                (595, 607),  # a4I
                (608, 620),  # a4J
                (621, 633),  # a4H
                (634, 646),  # a4K
            ]
            
            # Add all coefficient positions
            for pos in a1_positions:
                colspecs.append(pos)
            for pos in a2_positions:
                colspecs.append(pos)
            for pos in a3_positions:
                colspecs.append(pos)
            for pos in a4_positions:
                colspecs.append(pos)
            
            # Create column names
            col_names = ['logg', 'Teff', 'Z', 'Vel']
            for filter_name in self.filters:
                col_names.append(f'a1_{filter_name}')
            for filter_name in self.filters:
                col_names.append(f'a2_{filter_name}')
            for filter_name in self.filters:
                col_names.append(f'a3_{filter_name}')
            for filter_name in self.filters:
                col_names.append(f'a4_{filter_name}')
            
            # Read the file
            self.coefficients = pd.read_fwf(
                self.data_file_path,
                colspecs=colspecs,
                names=col_names,
                skiprows=0
            )
            
            # Convert coefficient columns to float, coercing errors to NaN
            for filter_name in self.filters:
                for coeff in ['a1', 'a2', 'a3', 'a4']:
                    col_name = f'{coeff}_{filter_name}'
                    self.coefficients[col_name] = pd.to_numeric(self.coefficients[col_name], errors='coerce')
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    
    def _interpolate_coefficients(self, logg, Teff, Z, Vel, filter_name):
        """
        Interpolate coefficients for given stellar parameters.
        
        Parameters:
        -----------
        logg : float
            Log surface gravity [cm/s²]
        Teff : float  
            Effective temperature [K]
        Z : float
            Metallicity [M/H]
        Vel : float
            Microturbulent velocity [km/s]
        filter_name : str
            Filter name (one of uvbyUBVRIJHK)
            
        Returns:
        --------
        tuple : (a1, a2, a3, a4) coefficients
        """
        if filter_name not in self.filters:
            raise ValueError(f"Filter '{filter_name}' not supported. Available: {self.filters}")
        
        # Prepare interpolation points and values
        points = self.coefficients[['logg', 'Teff', 'Z', 'Vel']].values
        target_point = np.array([[logg, Teff, Z, Vel]])
        
        coeffs = []
        for coeff in ['a1', 'a2', 'a3', 'a4']:
            col_name = f'{coeff}_{filter_name}'
            values = self.coefficients[col_name].values
            
            # Use linear interpolation
            try:
                interp_value = griddata(points, values, target_point, method='linear')[0]
                if np.isnan(interp_value):
                    # Fallback to nearest neighbor if linear fails
                    interp_value = griddata(points, values, target_point, method='nearest')[0]
                coeffs.append(interp_value)
            except:
                # If interpolation fails, find nearest neighbor manually
                distances = np.sqrt(np.sum((points - target_point)**2, axis=1))
                nearest_idx = np.argmin(distances)
                coeffs.append(values[nearest_idx])
        
        return tuple(coeffs)
    
    def get_limb_darkening_function(self, logg, Teff, Z, Vel, filter_name):
        """
        Get the limb darkening function I(μ)/I(0) for given stellar parameters.
        
        This implements the 4-parameter limb darkening law (Equation 3 from arXiv:2305.01704):
        I(μ)/I(0) = 1 - a₁(1-μ)^0.5 - a₂(1-μ)^1 - a₃(1-μ)^1.5 - a₄(1-μ)^2
        
        Parameters:
        -----------
        logg : float
            Log surface gravity [cm/s²]
        Teff : float  
            Effective temperature [K]
        Z : float
            Metallicity [M/H] 
        Vel : float
            Microturbulent velocity [km/s]
        filter_name : str
            Filter name (one of uvbyUBVRIJHK)
            
        Returns:
        --------
        function : callable
            Function that takes μ (cosine of angle) and returns I(μ)/I(0)
        """
        # Get interpolated coefficients
        a1, a2, a3, a4 = self._interpolate_coefficients(logg, Teff, Z, Vel, filter_name)
        
        def limb_darkening_profile(mu):
            """
            Compute the limb darkening profile I(μ)/I(0).
            
            Parameters:
            -----------
            mu : float or array-like
                Cosine of the angle between line of sight and surface normal (0 ≤ μ ≤ 1)
                
            Returns:
            --------
            float or array-like : I(μ)/I(0)
                Normalized intensity
            """
            mu = np.asarray(mu)
            
            # Ensure μ is in valid range [0, 1]
            mu = np.clip(mu, 0, 1)
            
            # 4-parameter limb darkening law with standard exponents
            term1 = a1 * (1 - mu)**0.5
            term2 = a2 * (1 - mu)**1.0  
            term3 = a3 * (1 - mu)**1.5
            term4 = a4 * (1 - mu)**2.0
            
            intensity = 1.0 - term1 - term2 - term3 - term4
            
            # Ensure intensity doesn't go negative
            return np.maximum(intensity, 0.0)
        
        return limb_darkening_profile
    
    def get_limb_darkening_function_angle(self, logg, Teff, Z, Vel, filter_name):
        """
        Get the limb darkening function I(angle)/I(0) in terms of angle.
        
        This provides the limb darkening as a function of angle from the surface normal,
        where μ = cos(angle). The angle is measured from the surface normal (0 at center,
        π/2 at limb).
        
        Parameters:
        -----------
        logg : float
            Log surface gravity [cm/s²]
        Teff : float
            Effective temperature [K]
        Z : float
            Metallicity [M/H]
        Vel : float
            Microturbulent velocity [km/s]
        filter_name : str
            Filter name (one of uvbyUBVRIJHK)
            
        Returns:
        --------
        function : callable
            Function that takes angle (in radians, 0 ≤ angle ≤ π/2) and returns I(angle)/I(0)
            where μ = cos(angle)
        """
        # Get interpolated coefficients
        a1, a2, a3, a4 = self._interpolate_coefficients(logg, Teff, Z, Vel, filter_name)
        
        def limb_darkening_profile_angle(angle):
            """
            Compute the limb darkening profile I(angle)/I(0) as function of angle.
            
            Parameters:
            -----------
            angle : float or array-like
                Angle from surface normal in radians (0 ≤ angle ≤ π/2)
                where μ = cos(angle)
                
            Returns:
            --------
            float or array-like : I(angle)/I(0)
                Normalized intensity
            """
            angle = np.asarray(angle)
            
            # Ensure angle is in valid range [0, π/2]
            angle = np.clip(angle, 0, np.pi/2)
            
            # Calculate μ = cos(angle)
            mu = np.cos(angle)
            
            # 4-parameter limb darkening law
            one_minus_mu = 1 - mu
            
            term1 = a1 * one_minus_mu**0.5
            term2 = a2 * one_minus_mu**1.0
            term3 = a3 * one_minus_mu**1.5
            term4 = a4 * one_minus_mu**2.0
            
            intensity = 1.0 - term1 - term2 - term3 - term4
            
            # Ensure intensity doesn't go negative
            return np.maximum(intensity, 0.0)
        
        return limb_darkening_profile_angle
    
    def __call__(self, logg, Teff, Z, Vel, filter_name):
        """
        Convenience method to get limb darkening function in terms of μ.
        Same as get_limb_darkening_function().
        """
        return self.get_limb_darkening_function(logg, Teff, Z, Vel, filter_name)

# Example usage:
if __name__ == "__main__":
    # Initialize the limb darkening class
    ld = LimbDarkening("data/phoenix/table12.dat")
    
    # Get limb darkening functions for solar-type star in V band
    stellar_params = {
        'logg': 4.4,    # Solar log g
        'Teff': 5778,   # Solar T_eff  
        'Z': 0.0,       # Solar metallicity
        'Vel': 1.0,     # Microturbulent velocity
        'filter_name': 'V'
    }
    
    # Function in terms of μ
    limb_func_mu = ld.get_limb_darkening_function(**stellar_params)
    
    # Function in terms of r
    limb_func_r = ld.get_limb_darkening_function_radial(**stellar_params)
    
    # Evaluate at different points
    mu_values = np.linspace(0.1, 1, 50)  # Avoid μ=0 at limb
    r_values = np.sqrt(1 - mu_values**2)  # Corresponding r values
    
    intensities_mu = limb_func_mu(mu_values)
    intensities_r = limb_func_r(r_values)
    
    # Verify they give the same results
    print("Maximum difference between μ and r formulations:", 
          np.max(np.abs(intensities_mu - intensities_r)))
    
    # Plot comparison
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot vs μ
        ax1.plot(mu_values, intensities_mu, 'b-', linewidth=2)
        ax1.set_xlabel('μ = cos(θ)')
        ax1.set_ylabel('I(μ)/I(0)')
        ax1.set_title('Limb Darkening vs μ')
        ax1.grid(True, alpha=0.3)
        
        # Plot vs r  
        ax2.plot(r_values, intensities_r, 'r-', linewidth=2)
        ax2.set_xlabel('r = √(1-μ²)')
        ax2.set_ylabel('I(r)/I(0)')
        ax2.set_title('Limb Darkening vs r')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Also show radial profile from center to limb
        plt.figure(figsize=(8, 6))
        r_profile = np.linspace(0, 1, 100)
        intensity_profile = limb_func_r(r_profile)
        plt.plot(r_profile, intensity_profile, 'g-', linewidth=2)
        plt.xlabel('Radial distance r (normalized)')
        plt.ylabel('I(r)/I(0)')
        plt.title('Stellar Limb Darkening Profile\n(Center to Limb)')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
        print("μ values:", mu_values[:5])
        print("Intensities (μ):", intensities_mu[:5])
        print("r values:", r_values[:5]) 
        print("Intensities (r):", intensities_r[:5])