import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering, KMeans
from typing import Tuple


def assign_clusters(
    som: MiniSom,
    data_df: pd.DataFrame,
    n_clusters: int = 4
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Perform k-means clustering on the trained SOM's node weights,
    then assign each data observation to the cluster of its Best Matching Unit.

    """
    # 1) Cluster the SOM's nodes
    weights = som.get_weights()  # shape (x_dim, y_dim, features)
    x_dim, y_dim, _ = weights.shape
    flat_weights = weights.reshape(x_dim * y_dim, -1)

    # ← changed: use KMeans instead of AgglomerativeClustering
    km          = KMeans(n_clusters=n_clusters, random_state=0)
    node_labels = km.fit_predict(flat_weights)

    # 2) For each observation, find its BMU
    features = data_df.drop(columns=["hex_x", "hex_y"], errors="ignore")
    values   = features.values
    bmus     = [som.winner(obs) for obs in values]
    bmu_x    = [pt[0] for pt in bmus]
    bmu_y    = [pt[1] for pt in bmus]
    flat_idx = [i * y_dim + j for i, j in bmus]

    # 3) Assign cluster label based on BMU's node label
    clusters = [int(node_labels[idx]) for idx in flat_idx]

    # 4) Return augmented DataFrame + the node_labels array
    df_out = data_df.copy()
    df_out['bmu_x']      = bmu_x
    df_out['bmu_y']      = bmu_y
    df_out['cluster'] = clusters

    return df_out, node_labels


def compute_cluster_means(
    hex_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute mean of each numeric variable for each cluster.

    Parameters:
        hex_df: DataFrame including 'cluster' and numeric feature columns.

    Returns:
        A DataFrame with 'cluster' and the mean of each numeric column.
    """
    numeric_cols = (
        hex_df
        .select_dtypes(include=[float, int])
        .columns
    )
    numeric_cols = [
        c for c in numeric_cols
        if c not in {"cluster", "bmu_x", "bmu_y", "hex_x", "hex_y"}
    ]
    cluster_means = (
        hex_df
        .groupby('cluster')[numeric_cols]
        .mean()
        .reset_index()
    )
    return cluster_means
