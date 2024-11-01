import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("diabetes1.csv")

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Initialize K-Means clustering
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)

# Simulate initial cluster centers as zeros (same shape as final centers)
# The shape of the final cluster centers is determined by (n_clusters, n_features)
print("\nInitial cluster centers:")
initial_centroids = [[0] * X_scaled.shape[1]] * 3  # Adjust for 3 clusters and number of features
print(initial_centroids)

# Apply K-Means clustering (fitting the model)
clusters = kmeans.fit_predict(X_scaled)

# Show final cluster centers (After fitting)
print("\nFinal cluster centers:")
print(kmeans.cluster_centers_)

# Show number of iterations (Epoch size)
print("\nNumber of iterations until convergence (Epoch size):", kmeans.n_iter_)

# Show final error rate (Inertia / Within-cluster sum of squares)
print("\nFinal error rate (Inertia):", kmeans.inertia_)

# Adding the cluster labels to the dataframe
df['Cluster'] = clusters

# Use PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=100)
plt.title('K-Means Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()
