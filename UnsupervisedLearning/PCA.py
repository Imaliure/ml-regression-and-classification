# =====================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA) – EXPLANATION
# -----------------------------------------------------
# PCA is an unsupervised dimensionality reduction technique 
# that transforms the original data into a new coordinate system 
# where:
#   • The first principal component (PC1) captures the highest variance
#   • The second principal component (PC2) captures the second highest variance
#   • And so on...
#
# Key Advantages:
#   ✓ Reduces dimensionality while preserving most of the information (variance)
#   ✓ Removes multicollinearity
#   ✓ Visualizes high-dimensional data in 2D or 3D
#
# Important Concepts:
#   - Standardization is necessary before PCA (mean = 0, variance = 1)
#   - Explained Variance Ratio shows how much variance each component captures
#   - Cumulative variance helps determine the optimal number of components
# =====================================================

# ================================
# Import Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ================================
# Load Dataset
# ================================
df = pd.read_csv("./Data/Hitters.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Select only numerical columns (PCA works on numeric data)
df = df.select_dtypes(include=["float64", "int64"])
print(df.head())

# ================================
# Standardize the Data (Important for PCA)
# ================================
df_scaled = StandardScaler().fit_transform(df)

# ================================
# Apply PCA – Reduce to 2 Components for Visualization
# ================================
pca = PCA(n_components=2)
pca_fit = pca.fit_transform(df_scaled)

# Create a DataFrame for principal components
component_df = pd.DataFrame(data=pca_fit, columns=["PC1", "PC2"])
print(component_df.head())

# Explained variance ratio of each component
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Loadings (coefficients showing how features contribute to PC2)
print("PCA Component 2 Loadings:", pca.components_[1])

# ================================
# Determine Optimal Number of Components (Elbow Method)
# ================================
pca_full = PCA().fit(df_scaled)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs. Number of Components")
plt.grid(True)
plt.show()
