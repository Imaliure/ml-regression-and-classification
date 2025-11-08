# ================================================================
# K-MEANS CLUSTERING – EXPLANATION & IMPLEMENTATION
# ----------------------------------------------------------------
# K-Means is an unsupervised clustering algorithm used to partition
# data into K distinct non-overlapping clusters based on feature similarity.
#
# How it Works:
#   1. Choose the number of clusters K
#   2. Randomly initialize K cluster centroids
#   3. Assign each data point to the closest centroid (Euclidean distance)
#   4. Update centroids based on the mean of assigned points
#   5. Repeat Steps 3-4 until convergence (no change in centroids)
#
# Important Notes:
#   ✓ Only numeric variables can be used in K-Means
#   ✓ Scaling is recommended for better performance (not applied here intentionally to show raw clustering)
#   ✓ “Inertia” / “SSD” helps determine optimal K (Elbow Method)
# ================================================================

# ===========================
# 1. Import Required Libraries
# ===========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 2. Load Dataset
# ===========================
df = pd.read_csv("./Data/USArrests.csv")

# Remove non-numeric column (state names)
df_numeric = df.drop(columns=["Unnamed: 0"])

# ===========================
# 3. Basic Statistical Summary
# ===========================
print(df_numeric.describe().T)

# ===========================
# 4. Visualize Distributions (Histogram)
# ===========================
df_numeric.hist(figsize=(10, 8), bins=20)
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()

# ===========================
# 5. Train Initial K-Means Model (K=3)
# ===========================
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_numeric)

print("Number of Clusters:", kmeans.n_clusters)
print("Cluster Centers:\n", kmeans.cluster_centers_)

# ===========================
# 6. Plot Clusters using 2 Variables (Murder vs Assault)
# ===========================
k_means = KMeans(n_clusters=2, random_state=42).fit(df_numeric)
clusters = k_means.labels_
centers = k_means.cluster_centers_

plt.figure(figsize=(8, 5))
plt.scatter(df_numeric["Murder"], df_numeric["Assault"], c=clusters, cmap="viridis")
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.8, label="Centroids")
plt.xlabel("Murder Rate")
plt.ylabel("Assault Rate")
plt.title("K-Means Clustering (K=2)")
plt.legend()
plt.show()

# ===========================
# 7. Elbow Method (to determine optimal K)
# ===========================
ssd = []
K = range(1, 30)

for i in K:
    kmeans = KMeans(n_clusters=i, random_state=42).fit(df_numeric)
    ssd.append(kmeans.inertia_)   # Sum of Squared Distances (Inertia)

plt.figure(figsize=(8, 5))
plt.plot(K, ssd, "o-")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia / SSD")
plt.title("Elbow Method for Optimal K")
plt.show()

# ===========================
# 8. Automated Elbow Detection (Yellowbrick Library)
# ===========================
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2, 20), timings=False)
visualizer.fit(df_numeric)
visualizer.show()

# ===========================
# 9. Final Model (Example K = 4)
# ===========================
final_kmeans = KMeans(n_clusters=4, random_state=42).fit(df_numeric)
clusters = final_kmeans.labels_

# Create result table
result = pd.DataFrame({
    "State": df["Unnamed: 0"],
    "Cluster": clusters
})

print(result.head())
