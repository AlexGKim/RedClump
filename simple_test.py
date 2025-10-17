#!/usr/bin/env python3
"""
Simple test to debug the Gaia to UniformDisk converter.
"""

import pandas as pd
import numpy as np
import sys

print("Starting simple test...")

# Test basic imports
try:
    print("Testing pandas import...")
    import pandas as pd
    print("✓ pandas imported successfully")
    
    print("Testing numpy import...")
    import numpy as np
    print("✓ numpy imported successfully")
    
    print("Testing CSV loading...")
    df = pd.read_csv('extended_data_table_2.csv')
    print(f"✓ CSV loaded successfully with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    print("\nFirst row data:")
    print(df.iloc[0])
    
except Exception as e:
    print(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nTesting g2 import...")
try:
    sys.path.append('/Users/akim/Projects/g2')
    from g2.models.sources.simple import UniformDisk
    print("✓ g2.models.sources.simple.UniformDisk imported successfully")
    
    # Test creating a simple UniformDisk
    test_disk = UniformDisk(flux_density=1e-26, radius=1e-8)
    print(f"✓ UniformDisk created: flux={test_disk.flux_density}, radius={test_disk.radius}")
    
except Exception as e:
    print(f"✗ Error importing g2: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nTesting our module import...")
try:
    from gaia_uniform_disk import create_uniform_disk_from_gaia
    print("✓ gaia_uniform_disk module imported successfully")
except Exception as e:
    print(f"✗ Error importing our module: {str(e)}")
    import traceback
    traceback.print_exc()

print("Simple test completed.")