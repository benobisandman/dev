import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load an example dataset (you can replace this with your own data)
data = load_iris()
X = data.data


# Create a scatter matrix plot for all original dimensions
features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]
fig = px.scatter_matrix(data, dimensions=features, color="species")
fig.update_traces(diagonal_visible=False)
fig.show()


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA to the scaled data
pca = PCA().fit(X_scaled)

# Calculate cumulative explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot cumulative variance
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='black')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance (%)')
plt.title('Cumulative Variance Explained by Principal Components')
plt.axhline(y=0.95, color='grey', linestyle='--')
plt.text(1.1, 1, '95% cut-off threshold', color='black', fontsize=16)
plt.grid(axis='x')
plt.tight_layout()
plt.show()

