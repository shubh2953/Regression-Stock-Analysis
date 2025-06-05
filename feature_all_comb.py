import pandas as pd
import numpy as np
import sys
import os
from itertools import combinations
from math import comb

def calculate_output_columns(n_features):
    """
    Calculate the number of columns in the output dataset.
    
    Parameters:
    n_features (int): Number of input feature columns
    
    Returns:
    tuple: (total columns, number of combinations, breakdown dictionary)
    """
    n_date_cols = 1
    n_ticker_cols = 4
    n_target_cols = 1
    n_combinations = comb(n_features, 2)  # Calculate combinations C(n,2)
    
    total_columns = n_date_cols + n_ticker_cols + n_combinations + n_target_cols
    
    # Create breakdown of columns
    breakdown = {
        'Date columns': n_date_cols,
        'Ticker columns': n_ticker_cols,
        'Feature combinations': n_combinations,
        'Target columns': n_target_cols,
        'Total columns': total_columns
    }
    
    return total_columns, n_combinations, breakdown

def transform_features(data):
    """
    Transform features by creating geometric means for all possible feature combinations.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame with date, tickers, features, and target
    
    Returns:
    pd.DataFrame: DataFrame with date, tickers, transformed features, and target
    """
    try:
        # Separate different parts of the dataset
        date_col = data.iloc[:, 0]
        ticker_cols = data.iloc[:, 1:5]
        feature_cols = data.iloc[:, 5:-1]  # All columns except date, tickers, and target
        target_col = data.iloc[:, -1]
        
        # Convert feature columns to numeric, replacing any errors with NaN
        feature_cols = feature_cols.apply(pd.to_numeric, errors='coerce')
        
        # Create new transformed features
        transformed_features = pd.DataFrame()
        
        # Get all possible pairs of feature columns
        feature_indices = range(feature_cols.shape[1])
        feature_pairs = list(combinations(feature_indices, 2))
        
        # Generate geometric means for all feature pairs
        for i, j in feature_pairs:
            # Calculate geometric mean with handling for negative values
            product = feature_cols.iloc[:, i] * feature_cols.iloc[:, j]
            # Replace negative products with NaN to avoid sqrt of negative numbers
            product = product.where(product >= 0, np.nan)
            geom_mean = np.sqrt(product)
            
            # Create new feature name using the numeric format (var1_3 format)
            # Adding 1 to indices for 1-based naming convention
            new_feature_name = f"var{i+1}_{j+1}"
            
            # Add to transformed features
            transformed_features[new_feature_name] = geom_mean
        
        # Convert date column to string to prevent dtype issues
        date_col = date_col.astype(str)
        
        # Ensure ticker columns are strings
        ticker_cols = ticker_cols.astype(str)
        
        # Ensure target column is numeric
        target_col = pd.to_numeric(target_col, errors='coerce')
        
        # Combine all parts
        result = pd.concat([
            date_col,
            ticker_cols,
            transformed_features,
            target_col
        ], axis=1)
        
        # Remove rows where all transformed feature columns are NaN
        result = result.dropna(subset=transformed_features.columns, how='all')
        
        return result
    
    except Exception as e:
        print(f"Error in transformation: {str(e)}")
        raise

def ensure_csv_extension(filename):
    """Ensure the filename has a .csv extension"""
    base, ext = os.path.splitext(filename)
    if not ext:
        return f"{filename}.csv"
    elif ext.lower() != '.csv':
        return f"{base}.csv"
    return filename

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.csv output_file.csv")
        print("Example: python script.py input_data.csv transformed_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = ensure_csv_extension(sys.argv[2])
    
    try:
        # Read input CSV with specific dtypes to prevent warnings
        print(f"Reading data from {input_file}...")
        data = pd.read_csv(input_file, 
                          low_memory=False,
                          dtype={
                              'Date': str,
                              'Ticker A': str,
                              'Ticker B': str,
                              'Ticker C': str,
                              'Ticker D': str
                          })
        
        # Calculate number of input features and expected output columns
        n_features = data.shape[1] - 6  # Excluding date, tickers, and target
        total_cols, n_combinations, col_breakdown = calculate_output_columns(n_features)
        
        print("\nColumn Analysis:")
        print(f"Number of input features: {n_features}")
        print("\nOutput Dataset Structure:")
        for key, value in col_breakdown.items():
            print(f"- {key}: {value}")
        
        # Transform features
        print("\nCreating geometric mean combinations of features...")
        transformed_data = transform_features(data)
        
        # Save to CSV with specific parameters
        print(f"Saving transformed data to {output_file}...")
        transformed_data.to_csv(output_file, 
                              index=False,
                              sep=',',
                              encoding='utf-8',
                              lineterminator='\n',
                              quoting=1,  # QUOTE_ALL
                              float_format='%.6f')  # Format floating point numbers
        
        print("\nTransformation complete!")
        print(f"Original shape: {data.shape}")
        print(f"Transformed shape: {transformed_data.shape}")
        print(f"Rows removed due to invalid combinations: {data.shape[0] - transformed_data.shape[0]}")
        
        # Verify the output file
        print("\nVerifying output file...")
        test_read = pd.read_csv(output_file, low_memory=False)
        print(f"Output file verified - contains {test_read.shape[0]} rows and {test_read.shape[1]} columns")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The input file {input_file} is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease check that:")
        print("1. All feature columns contain numeric data")
        print("2. The input file has the correct format (date, tickers, features, target)")
        print("3. There are no missing values in the feature columns")
        sys.exit(1)

if __name__ == "__main__":
    main()