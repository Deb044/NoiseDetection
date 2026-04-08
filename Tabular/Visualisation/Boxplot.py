import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example Data (using Pima Diabetes BMI as inspiration)
data = np.concatenate([np.random.normal(30, 5, 200), np.random.normal(80, 10, 10), [0, 0, 0, 0]]) # 0s are noise
df = pd.DataFrame(data, columns=['BMI'])

# Create a boxplot to automatically flag noise
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['BMI'], color='lightblue')
plt.title('Boxplot for Noise Detection (BMI)', fontsize=14)
plt.xlabel('BMI Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()