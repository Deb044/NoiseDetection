from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

def detect_noise_isolation_forest(df, contamination=0.1):
    # 1. Initialize the model
    # contamination is the estimated % of noise in your dataset
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    # 2. Fit the model and predict
    # Returns 1 for normal, -1 for noise
    labels = iso_forest.fit_predict(df)
    
    # 3. Get the anomaly scores (lower scores = more noisy)
    scores = iso_forest.decision_function(df)
    
    return labels, scores

# Example usage (using a subset of your Stroke Prediction features)
data = {
    'age': [25, 30, 35, 40, 45, 50, 2, 95, 38, 42],
    'avg_glucose': [80, 85, 90, 95, 100, 105, 350, 400, 92, 88] 
}
df = pd.DataFrame(data)

labels, scores = detect_noise_isolation_forest(df)

# Identify rows flagged as noise
noise_indices = np.where(labels == -1)[0]

print(f"Noise detected at indices: {noise_indices}")
print(f"Rows flagged as noise:\n{df.iloc[noise_indices]}")