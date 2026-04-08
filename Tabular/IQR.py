import numpy as np

def detect_outliers_iqr(data):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    
    # Calculate the IQR
    iqr = q3 - q1
    
    # Define bounds (1.5 is the standard multiplier)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Identify indices of outliers
    outliers = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
    
    return outliers, (lower_bound, upper_bound)

# Example usage:
data = [10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 100, 12, 14, 11] # 100 is noise
indices, bounds = detect_outliers_iqr(data)

print(f"Lower Bound: {bounds[0]}, Upper Bound: {bounds[1]}")
print(f"Outlier values found: {[data[i] for i in indices]}")