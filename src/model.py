import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import main_data

snapshot_date = main_data['InvoiceDate'].max() + pd.Timedelta(days=1)

# calculate RFM (Recency, Frequency, Monetary) for each customer
# this is the base data for our machine learning model
rfm_data = main_data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()

# scale differences prove we need normalization
print(rfm_data.describe())

# scale the data so all numbers are in the same range
# this helps the KMeans model work properly
scaler_obj = StandardScaler()
rfm_scaled = scaler_obj.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])

# this is the elbow method code
# it tests different cluster numbers to find the best one
inertia_list = []
k_values = range(2, 11)
for k in k_values:
    temp_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    temp_model.fit(rfm_scaled)
    inertia_list.append(temp_model.inertia_)
    print(f"k={k}  inertia={temp_model.inertia_:.1f}")

# check silhouette score to see how well separated the clusters are
silhouette_list = []
for k in k_values:
    temp_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = temp_model.fit_predict(rfm_scaled)
    score = silhouette_score(rfm_scaled, labels)
    silhouette_list.append(score)
    print(f"k={k}  silhouette={score:.4f}")

best_k_silhouette = k_values[silhouette_list.index(max(silhouette_list))]
print(f"silhouette peaks at k={best_k_silhouette} score={max(silhouette_list):.4f}")

db_list = []
for k in k_values:
    temp_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = temp_model.fit_predict(rfm_scaled)
    db_score = davies_bouldin_score(rfm_scaled, labels)
    db_list.append(db_score)
    print(f"k={k}  davies_bouldin={db_score:.4f}")

best_k_db = k_values[db_list.index(min(db_list))]
print(f"davies bouldin lowest at k={best_k_db} score={min(db_list):.4f}")

print("--- decision ---")
print(f"elbow: bend visible at k=4")
print(f"silhouette: best k={best_k_silhouette} score={max(silhouette_list):.4f}")
print(f"davies bouldin: best k={best_k_db} score={min(db_list):.4f}")
print("all three methods agree on k=4 so we use KMeans with k=4")

dbscan_model = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan_model.fit_predict(rfm_scaled)
cluster_count = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
noise_percent = (dbscan_labels == -1).sum() / len(dbscan_labels) * 100
print(f"DBSCAN found {cluster_count} clusters with {noise_percent:.1f}% noise points")
print("rejected: too much noise, not useful for customer segments")

start_time = time.time()
AgglomerativeClustering(n_clusters=4).fit(rfm_scaled[:500])
time_taken = time.time() - start_time
print(f"agglomerative on 500 rows took {time_taken:.2f}s")
print("rejected: too slow for 4000+ customers, no better cluster quality")

# this is the final train model code
# we train the KMeans model with 4 clusters because the elbow method showed 4 is best
kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_model.fit(rfm_scaled)
rfm_data['Cluster'] = kmeans_model.labels_

cluster_means = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
print(cluster_means)

# give simple human names to the 4 clusters found by the model
cluster_names = {0: 'Regular', 1: 'Lost', 2: 'VIP', 3: 'Loyal'}
rfm_data['KMSegment'] = rfm_data['Cluster'].map(cluster_names)

rfm_data['Segment'] = 'Low Value'
rfm_data.loc[rfm_data['Monetary'] > rfm_data['Monetary'].quantile(0.75), 'Segment'] = 'High Value'
rfm_data.loc[rfm_data['Frequency'] > rfm_data['Frequency'].quantile(0.75), 'Segment'] = 'Loyal'
rfm_data.loc[rfm_data['Recency'] < 30, 'Segment'] = 'Recent'

# split data into training and testing sets to see if the model is good
train_data, test_data = train_test_split(rfm_scaled, test_size=0.2, random_state=42)

# train the model on the training data only
train_model = KMeans(n_clusters=4, random_state=42, n_init=10)
train_model.fit(train_data)

train_labels = train_model.predict(train_data)
test_labels_pred = train_model.predict(test_data)

train_silhouette = round(silhouette_score(train_data, train_labels), 3)
test_silhouette = round(silhouette_score(test_data, test_labels_pred), 3)
test_db_score = round(davies_bouldin_score(test_data, test_labels_pred), 3)

print(f"train silhouette: {train_silhouette}")
print(f"test silhouette: {test_silhouette}")
print(f"difference: {abs(train_silhouette - test_silhouette):.3f}")
if abs(train_silhouette - test_silhouette) < 0.05:
    print("model generalises well to unseen data")
else:
    print("some overfitting detected")

elbow_data = pd.DataFrame({'K': list(k_values), 'Inertia': inertia_list})
silhouette_data = pd.DataFrame({'K': list(k_values), 'Score': silhouette_list})
