import MAIN as m

import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np

point_cloud=m.load_pc("bg_subtract_one_only.pcd")
array=m.pc_to_array(point_cloud)



points=array
from cuml.cluster import HDBSCAN

def perform_clustering(points, eps, min_samples):
    """
    Performs the clustering in a guiven array of points.

    Parameters:
    points (np array): Is the np.array with the point clouds to perform the clustering.
    eps (float): The maximum distance between points in the same cluster.
    min_samples (int): The minimum number os points to form a cluster.

    Returns:
    cluster_labels (np array): Numpy array represensting the identified clusters.
    num_clusters (int): The number of clusters found.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)


    #clustering = OPTICS(max_eps=eps, min_samples=min_samples).fit(points)
    cluster_labels = clustering.labels_
    num_clusters= len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    #clustering = GaussianMixture(n_components=n_components, covariance_type='spherical').fit(points)
    #cluster_labels = clustering.predict(points)
    #num_clusters = n_components

    return cluster_labels, num_clusters


import time
start_time = time.time()

cluster_labels, num_clusters = m.perform_clustering(points, eps=0.3, min_samples=10)



all_visualizations, bbox_info = m.centroid_and_box(points, cluster_labels, num_clusters)
end_time = time.time()
execution_time = end_time - start_time

print("TIME",execution_time)



pc_final=m.array_to_pc(array)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([pc_final] + all_visualizations)


# Print information about the bounding boxes
for cluster_id, info in bbox_info.items():
    center, extent, rotation_matrix = info
    print(f"Cluster {cluster_id} - Center: {center}, Extent: {extent}")