import MAIN as m
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture

seed = 45

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
    #clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)


    clustering = OPTICS(max_eps=eps, min_samples=min_samples).fit(points)
    cluster_labels = clustering.labels_
    num_clusters= len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    #clustering = GaussianMixture(n_components=n_components, covariance_type='spherical').fit(points)
    #cluster_labels = clustering.predict(points)
    #num_clusters = n_components

    return cluster_labels, num_clusters

# Load background point cloud
bg = m.load_pc("bg.pcd")
bg = m.pc_to_array(bg)

# Example usage of the function
width_cub = 0.4  # Width range
size_y = 0.3  # Adjust this value based on your requirements

# Define the starting and ending coordinates of the three straight lines
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)
line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)
line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)
spacing = 0.1
np.random.seed(seed)

np.random.seed(45)


def perform_clustering(points, eps, min_samples):
    """
    Performs the clustering in a guiven array of points.

    Parameters:
    points (np array): Is the np.array with the point clouds to perform the clustering.
    eps (float): The maximum distance between points in the same cluster.
    min_samples (int): The minimum number os points to form a cluster.

    Returns:
    cluster_labels (np array): Numpy array representing the identified clusters.
    num_clusters (int): The number of clusters found.
    """
    #clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    clustering = OPTICS(eps=eps, min_samples=min_samples).fit(points)
    cluster_labels = clustering.labels_
    #print(type(cluster_labels))
    num_clusters= len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    return cluster_labels, num_clusters
# Create the global path
path1 = m.create_straight_line(line1_start, line1_end, spacing)
path2 = m.create_straight_line(line2_start, line2_end, spacing)
path3 = m.create_straight_line(line3_start, line3_end, spacing)
global_path = m.merge_arrays(path1, path2)
global_path = m.merge_arrays(global_path, path3)

# Generate random noise points (adjust the number of points as needed)
num_noise_points = 100
noise_range_x = (-1.0, 7.0)  # Adjust the range based on your scene dimensions
noise_range_y = (-2.0, 6.0)
noise_points = m.create_random_points(num_noise_points, noise_range_x, noise_range_y)

# Calculate the range for generating a random x-coordinate for the cube
min_x = min(bg[:, 0]) + width_cub
max_x = max(bg[:, 0]) - width_cub

# Create the cube at a random position outside the walls
cub_random_x = np.random.uniform(min(bg[:, 0]), max(bg[:, 0]) - width_cub)
cub_random_y = np.random.uniform(min(bg[:, 0]), max(bg[:, 0]) - width_cub)
cub_x = m.create_cubic_object((cub_random_x, 0), width_cub, size_y, spacing)
cub_y = m.create_cubic_object((cub_random_y, 0), width_cub, size_y, spacing)

# Delete 20% of random points from cub_x
cub_x = m.delete_random_points(cub_x, delete_percentage=0)
#cub_x = m.delete_random_points(cub_x, delete_percentage=0.2)

# Combine the background, global path, noise points, cubic objects, and spherical object into a single point cloud
combined_cloud = m.merge_arrays(bg, noise_points, cub_x, cub_y)

# Subtract the objects from the background
result = m.subtract_array(bg, combined_cloud)

# Perform clustering on the result
cluster_labels, num_clusters = perform_clustering(result, eps=0.11, min_samples=5)

#cluster_labels, num_clusters = perform_clustering(result, n_components=2)

print("cluster",cluster_labels)
print("cluster",num_clusters)

# Call centroid_and_box function to get visualizations of centroids and bounding boxes
all_visualizations, bbox_info = m.centroid_and_box(result, cluster_labels, num_clusters)

# Convert the result array to a point cloud
result_point_cloud = o3d.geometry.PointCloud()
result_point_cloud.points = o3d.utility.Vector3dVector(result)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([result_point_cloud] + all_visualizations)

# Print information about the bounding boxes
for cluster_id, info in bbox_info.items():
    center, extent, rotation_matrix = info
    print(f"Cluster {cluster_id} - Center: {center}, Extent: {extent}")