# ================================================================
# HIERARCHICAL CLUSTERING – EXPLANATION & IMPLEMENTATION
# ----------------------------------------------------------------
# Hierarchical Clustering is an unsupervised learning technique
# that builds a hierarchy of clusters using a bottom-up approach.
#
# Key Concepts:
#   • Agglomerative Clustering: Each observation starts as its own cluster,
#     and clusters are merged step-by-step.
#   • Linkage Methods:
#       - "complete": maximum distance between clusters
#       - "average" : average distance between clusters
#   • Dendrogram: A tree-like diagram showing the merging process.
#
# Why use it?
#   ✓ No need to define number of clusters initially
#   ✓ Visual understanding of cluster grouping using dendrogram
# ================================================================

# ===========================
# 1. Import Required Libraries
# ===========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 2. Load Dataset
# ===========================
df = pd.read_csv("./Data/USArrests.csv")

# Keep only numeric variables (remove column with state names)
df_numeric = df.drop(columns=["Unnamed: 0"])

# ===========================
# 3. Compute Linkage Matrices
# ===========================
# 'complete' linkage → max distance between clusters
# 'average' linkage  → average distance between clusters
hc_complete = linkage(df_numeric, method="complete")
hc_average = linkage(df_numeric, method="average")

# ===========================
# 4. Plot Full Dendrogram (Complete Linkage)
# ===========================
plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete, leaf_font_size=10)
plt.show()

# ===========================
# 5. Truncated Dendrogram (Show Only Last 4 Merges)
# ===========================
plt.figure(figsize=(10, 5))
plt.title("Truncated Dendrogram (Last 4 Clusters)")
plt.xlabel("Cluster Groups")
plt.ylabel("Distance")
dendrogram(
    hc_complete,
    truncate_mode="lastp",  # show only last 'p' merges
    p=4,
    show_contracted=True,
    leaf_font_size=10
)
plt.show()
