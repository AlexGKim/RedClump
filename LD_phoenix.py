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
            colspecs = [
                (0, 5),    # logg
                (6, 12),   # Teff  
                (13, 17),  # Z
                (18, 22),  # Vel
            ]
            
            # Add column specifications for a1, a2, a3, a4 coefficients for each filter
            start_col = 23
            for i, filter_name in enumerate(self.filters):
                for coeff in ['a1', 'a2', 'a3', 'a4']:
                    colspecs.append((start_col, start_col + 12))
                    start_col += 12
            
            # Add mu_cri and CHI2 columns (we'll skip these for now)
            
            # Create column names
            col_names = ['logg', 'Teff', 'Z', 'Vel']
            for filter_name in self.filters:
                for coeff in ['a1', 'a2', 'a3', 'a4']:
                    col_names.append(f'{coeff}_{filter_name}')
            
            # Read the file
            self.coefficients = pd.read_fwf(
                self.data_file_path,
                colspecs=colspecs[:len(col_names)],
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
    
    def get_limb_darkening_function_radial(self, logg, Teff, Z, Vel, filter_name):
        """
        Get the limb darkening function I(r)/I(0) in terms of radial coordinate r.
        
        This provides the limb darkening as a function of r = √(1-μ²), where r is the
        normalized radial distance from disk center (0 ≤ r ≤ 1).
        
        The relationship is: μ = √(1-r²), so the 4-parameter law becomes:
        I(r)/I(0) = 1 - a₁r^0.5 - a₂r^1 - a₃r^1.5 - a₄r^2
        
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
            Function that takes r (radial distance, 0 ≤ r ≤ 1) and returns I(r)/I(0)
        """
        # Get interpolated coefficients
        a1, a2, a3, a4 = self._interpolate_coefficients(logg, Teff, Z, Vel, filter_name)
        
        def limb_darkening_profile_radial(r):
            """
            Compute the limb darkening profile I(r)/I(0) as function of radial distance.
            
            Parameters:
            -----------
            r : float or array-like
                Normalized radial distance from disk center (0 ≤ r ≤ 1)
                where r = √(1-μ²)
                
            Returns:
            --------
            float or array-like : I(r)/I(0)
                Normalized intensity
            """
            r = np.asarray(r)
            
            # Ensure r is in valid range [0, 1]
            r = np.clip(r, 0, 1)
            
            # Convert r to μ: μ = √(1-r²)
            mu = np.sqrt(1 - r**2)
            
            # 4-parameter limb darkening law
            # Since (1-μ) = (1-√(1-r²)) = 1-√(1-r²), we can express this directly in terms of r
            # But it's cleaner to first compute μ and then use (1-μ)
            one_minus_mu = 1 - mu
            
            term1 = a1 * one_minus_mu**0.5
            term2 = a2 * one_minus_mu**1.0  
            term3 = a3 * one_minus_mu**1.5
            term4 = a4 * one_minus_mu**2.0
            
            intensity = 1.0 - term1 - term2 - term3 - term4
            
            # Ensure intensity doesn't go negative
            return np.maximum(intensity, 0.0)
        
        return limb_darkening_profile_radial
    
    def get_limb_darkening_function_radial_direct(self, logg, Teff, Z, Vel, filter_name):
        """
        Get the limb darkening function I(r)/I(0) directly in terms of r.
        
        This is an alternative formulation that expresses the limb darkening law
        directly in terms of r = √(1-μ²). For the standard 4-parameter law,
        this becomes more complex algebraically, but can be useful for certain
        applications.
        
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
            Function that takes r and returns I(r)/I(0)
        """
        # Get interpolated coefficients  
        a1, a2, a3, a4 = self._interpolate_coefficients(logg, Teff, Z, Vel, filter_name)
        
        def limb_darkening_profile_radial_direct(r):
            """
            Compute limb darkening profile directly as function of r.
            
            Parameters:
            -----------
            r : float or array-like
                Normalized radial distance from disk center (0 ≤ r ≤ 1)
                
            Returns:
            --------
            float or array-like : I(r)/I(0)
                Normalized intensity
            """
            r = np.asarray(r)
            
            # Ensure r is in valid range [0, 1]  
            r = np.clip(r, 0, 1)
            
            # For the 4-parameter law: I(μ)/I(0) = 1 - a₁(1-μ)^0.5 - a₂(1-μ) - a₃(1-μ)^1.5 - a₄(1-μ)²
            # With μ = √(1-r²), we have (1-μ) = 1 - √(1-r²)
            # This can be simplified using the identity: 1 - √(1-r²) = r²/(1 + √(1-r²))
            # But for numerical stability, we'll use the direct computation
            
            mu = np.sqrt(1 - r**2)
            one_minus_mu = 1 - mu
            
            # Handle the case where r = 1 (μ = 0, limb)
            # When r = 1, one_minus_mu = 1
            
            term1 = a1 * one_minus_mu**0.5
            term2 = a2 * one_minus_mu**1.0
            term3 = a3 * one_minus_mu**1.5  
            term4 = a4 * one_minus_mu**2.0
            
            intensity = 1.0 - term1 - term2 - term3 - term4
            
            return np.maximum(intensity, 0.0)
        
        return limb_darkening_profile_radial_direct
    
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