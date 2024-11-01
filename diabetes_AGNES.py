import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Creating the dataset (from the image you shared)
data = pd.read_csv("diabetes1.csv")

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Perform Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = agg_cluster.fit_predict(X_scaled)

# Adding the cluster labels to the dataframe
df['Cluster'] = clusters

# Generate dendrogram
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Agglomerative Clustering")
dendrogram(Z)
plt.show()


# Visualize the clusters with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=100)
plt.title('Agglomerative Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()


