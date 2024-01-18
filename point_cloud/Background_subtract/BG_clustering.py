import MAIN as m
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import MAIN as m
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN


point_cloud = m.load_pc("subtraction_result.pcd")

point_cloud_array = m.pc_to_array(point_cloud)

eps = 0.32
min_samples = 20 

# Perform clustering
cluster_labels, num_clusters = m.perform_clustering(point_cloud_array, eps, min_samples)

# Call centroid_and_box function to get visualizations of centroids and bounding boxes
all_visualizations, bbox_info = m.centroid_and_box(point_cloud_array, cluster_labels, num_clusters)

# Convert the result array to a point cloud
result_point_cloud = o3d.geometry.PointCloud()
result_point_cloud.points = o3d.utility.Vector3dVector(point_cloud_array)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([result_point_cloud] + all_visualizations)
