import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a sample multivariate dataset (inspired by Stroke Prediction)
np.random.seed(42)
n_samples = 300
age = np.random.randint(18, 90, n_samples)
glucose = np.random.normal(100, 20, n_samples)
bmi = np.random.normal(28, 5, n_samples)

# Add complex noise: normal value, but impossible context
glucose[0], bmi[0] = 350, 15 # Multivariate noise (Glucose and BMI are inverted)
glucose[1] = 0.5 # Logical noise (effectively 0)
bmi[2] = 120 # Extreme outlier

df = pd.DataFrame({'Age': age, 'Glucose': glucose, 'BMI': bmi})

# Generate the pairplot to find "clusters of noise"
# corner=True removes the redundant top-right half of the plot matrix
sns.pairplot(df, diag_kind='kde', corner=True, plot_kws={'color': 'darkblue', 'alpha': 0.6})
plt.suptitle('Multivariate Pairplot for Complex Noise Detection', fontsize=16, y=1.02)
plt.show()