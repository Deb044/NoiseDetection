import numpy as np

def detect_outliers_zscore(data, threshold=3):
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    
    # Calculate Z-scores for every point
    z_scores = [(x - mean) / std for x in data]
    
    # Identify indices of outliers
    outliers = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
    
    return outliers, z_scores

# Example usage:
data = [10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 100, 12, 14, 11] # 100 is noise
outlier_indices, all_z = detect_outliers_zscore(data)

print(f"Outlier indices found: {outlier_indices}")
print(f"Outlier values: {[data[i] for i in outlier_indices]}")