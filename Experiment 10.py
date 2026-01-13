# Implement K-means clustering to discover inherent patterns.

# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Iris.csv")

# Drop unnecessary column and handle missing values
df = df.drop(columns=['Id'])
df = df.dropna()

# Separate features and target
X = df.drop(columns=['Species'])
y = df['Species']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fit K-Means (3 clusters for 3 species)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Get cluster labels
clusters = kmeans.labels_

# Add cluster labels to dataframe
df['Cluster'] = clusters

# Evaluate clustering quality
sil_score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {sil_score:.3f}")
print("\nCluster Centers (Standardized Coordinates):\n", kmeans.cluster_centers_)

# Contingency table
print("\nContingency Table (True Species vs Cluster):")
print(pd.crosstab(df['Species'], df['Cluster']))
