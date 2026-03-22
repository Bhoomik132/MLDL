# ====================================================
# K-MEANS & HIERARCHICAL CLUSTERING (FIXED CODE)
# ====================================================

# -------- IMPORT LIBRARIES --------
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -------- LOAD DATASET --------
data = pd.read_csv("Mall_Customers.csv")

print("Original Data:\n", data.head())

# -------- HANDLE CATEGORICAL DATA --------
# Convert text (Male/Female etc.) to numeric
data = pd.get_dummies(data)

print("\nAfter Encoding:\n", data.head())

# -------- FEATURES --------
X = data

# -------- SCALING --------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====================================================
# K-MEANS CLUSTERING
# ====================================================
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster column
data['Cluster'] = kmeans_labels

print("\nK-Means Clusters Added:\n", data.head())

# -------- PLOT --------
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ====================================================
# HIERARCHICAL CLUSTERING
# ====================================================
linked = linkage(X_scaled, method='ward')

plt.figure()
dendrogram(linked)
plt.title("Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()