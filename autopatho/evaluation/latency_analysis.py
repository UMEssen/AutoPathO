import os
import ast
import glob
import numpy as np
import pandas as pd
from scipy import stats

# File pattern to match your CSV files (examples below)
FILE_PATTERN = "data/latency*.csv"

# Number of first samples to analyze (None for all samples)
N_SAMPLES = 100

OUTPUT_FILE = "results/latency_analysis_results.csv"  # Change to None if you don't want to save

def parse_latency_value(value):
    if pd.isna(value):
        return np.nan
    
    try:
        # If it's already a number, return it
        if isinstance(value, (int, float)):
            return float(value)
        
        # Convert to string and strip whitespace
        value_str = str(value).strip()
        
        # If it looks like a list string, try to parse it
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                # Use ast.literal_eval to safely parse the list
                parsed_list = ast.literal_eval(value_str)
                if isinstance(parsed_list, list) and len(parsed_list) > 0:
                    return float(parsed_list[0])
                else:
                    return np.nan
            except (ValueError, SyntaxError):
                # If ast parsing fails, try manual extraction
                # Remove brackets and quotes, split by comma, take first value
                clean_str = value_str.strip('[]').replace("'", "").replace('"', '')
                if ',' in clean_str:
                    clean_str = clean_str.split(',')[0]
                return float(clean_str.strip())
        else:
            # Try to convert directly to float
            return float(value_str)
    
    except (ValueError, TypeError, IndexError):
        return np.nan

def calculate_quartiles_and_iqr(data):
    """Calculate quartiles and interquartile range."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return q1, q3, iqr

def analyze_single_file(file_path, n_samples=None):
    """Analyze latency data from a single CSV file."""
    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"  Total rows: {len(df)}")
        
        # Find latency columns
        latency_columns = [col for col in df.columns if 'latency' in col.lower()]
        
        if not latency_columns:
            print("  No latency columns found!")
            return None
        
        print(f"  Found latency columns: {latency_columns}")
        
        results = {'file_name': os.path.basename(file_path)}
        
        for col in latency_columns:
            # Get latency values and parse them
            latency_raw = df[col].dropna()
            
            if len(latency_raw) == 0:
                print(f"    {col}: No valid data")
                continue
            
            # Parse the latency values from string format
            print(f"    {col}: Parsing latency values...")
            latency_data = []
            for value in latency_raw:
                parsed_value = parse_latency_value(value)
                if not np.isnan(parsed_value):
                    latency_data.append(parsed_value)
            
            latency_data = np.array(latency_data)
            
            if len(latency_data) == 0:
                print(f"    {col}: No valid numeric data after parsing")
                continue
            
            print(f"    {col}: Successfully parsed {len(latency_data)} values from {len(latency_raw)} entries")
            
            # Limit to first N samples if specified
            if n_samples is not None and n_samples > 0:
                latency_data = latency_data[:n_samples]
                print(f"    {col}: Using first {len(latency_data)} samples")
            else:
                print(f"    {col}: Using all {len(latency_data)} samples")
            
            # Calculate statistics
            mean_val = np.mean(latency_data)
            std_val = np.std(latency_data, ddof=1)
            median_val = np.median(latency_data)
            min_val = np.min(latency_data)
            max_val = np.max(latency_data)
            
            print(f"    {col} Statistics:")
            print(f"      Mean: {mean_val:.4f}s")
            print(f"      Std Dev: {std_val:.4f}s")
            print(f"      Median: {median_val:.4f}s")
            print(f"      Range: {min_val:.4f}s - {max_val:.4f}s")
            
            # Calculate quartiles and IQR
            if len(latency_data) > 1:
                q1, q3, iqr = calculate_quartiles_and_iqr(latency_data)
                print(f"      Q1 (25th percentile): {q1:.4f}s")
                print(f"      Q3 (75th percentile): {q3:.4f}s")
                print(f"      IQR (Interquartile Range): {iqr:.4f}s")
                print(f"      IQR Range: [{q1:.4f}s, {q3:.4f}s]")
                
                # Store results
                results.update({
                    f'{col}_count': len(latency_data),
                    f'{col}_mean': mean_val,
                    f'{col}_std': std_val,
                    f'{col}_median': median_val,
                    f'{col}_min': min_val,
                    f'{col}_max': max_val,
                    f'{col}_q1': q1,
                    f'{col}_q3': q3,
                    f'{col}_iqr': iqr
                })
            else:
                print(f"      IQR: Cannot calculate (need > 1 sample)")
        
        return results
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return None

def main():
    """Main analysis function."""
    print("=" * 60)
    print("LATENCY ANALYSIS")
    print("=" * 60)
    print(f"File pattern: {FILE_PATTERN}")
    print(f"Samples per file: {N_SAMPLES if N_SAMPLES else 'All'}")
    print(f"Output file: {OUTPUT_FILE if OUTPUT_FILE else 'None'}")
    
    # Find files
    files = glob.glob(FILE_PATTERN)
    
    if not files:
        print(f"\nNo files found matching pattern: {FILE_PATTERN}")
        return
    
    print(f"\nFound {len(files)} files:")
    for file in files:
        print(f"  - {file}")
    
    # Analyze each file
    all_results = []
    
    for file_path in files:
        result = analyze_single_file(file_path, N_SAMPLES)
        if result:
            all_results.append(result)
    
    # Create summary DataFrame and save if requested
    if all_results and OUTPUT_FILE:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(OUTPUT_FILE, index=False)
        print(f"\nDetailed results saved to: {OUTPUT_FILE}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Print summary comparison if multiple files
    if len(all_results) > 1:
        print("\nSUMMARY COMPARISON:")
        print("-" * 40)
        
        # Find common latency columns
        all_columns = set()
        for result in all_results:
            for key in result.keys():
                if key.endswith('_mean'):
                    all_columns.add(key.replace('_mean', ''))
        
        for col in sorted(all_columns):
            print(f"\n{col.upper()}:")
            for result in all_results:
                if f'{col}_mean' in result:
                    mean_val = result[f'{col}_mean']
                    iqr_val = result.get(f'{col}_iqr', 0)
                    q1_val = result.get(f'{col}_q1', 0)
                    q3_val = result.get(f'{col}_q3', 0)
                    print(f"  {result['file_name']}: {mean_val:.4f}s (IQR: {iqr_val:.4f}s, Q1-Q3: {q1_val:.4f}s-{q3_val:.4f}s)")

if __name__ == "__main__":
    main()