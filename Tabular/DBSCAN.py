from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def detect_noise_dbscan(df, eps=0.5, min_samples=5):
    # 1. Standardization is crucial (DBSCAN uses Euclidean distance)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # 2. Initialize DBSCAN
    # eps: Max distance between two samples to be considered neighbors
    # min_samples: Min samples in a neighborhood to define a "cluster"
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # 3. Fit and predict
    # DBSCAN labels noise as -1
    labels = dbscan.fit_predict(scaled_data)
    
    return labels

# Example with your chosen "Chronic Kidney Disease" style logic
# Assume 'age', 'bp', and 'glucose' are your numeric columns
data = {
    'age': [48, 7, 62, 48, 51, 60, 68, 11, 48, 200], # 200 is noise
    'bp': [80, 50, 80, 70, 80, 90, 70, 80, 80, 180], # 180 is noise
}
df = pd.DataFrame(data)

labels = detect_noise_dbscan(df, eps=1.0, min_samples=2)

# Identify noise
noise_indices = np.where(labels == -1)[0]
print(f"Noise detected at indices: {noise_indices}")
print(f"Noisy Data:\n {df.iloc[noise_indices]}")