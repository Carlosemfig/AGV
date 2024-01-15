import MAIN as m
import open3d as o3d
import numpy as np
import random

# Load background point cloud
bg = m.load_pc("bg.pcd")
bg = m.pc_to_array(bg)

# Example usage of the function
width_cub = 0.4  # Width range
size_y = 0.3 # Adjust this value based on your requirements

# Define the starting and ending coordinates of the three straight lines
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)
line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)
line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)
spacing = 0.08

# Create the global path
path1 = m.create_straight_line(line1_start, line1_end, spacing)
path2 = m.create_straight_line(line2_start, line2_end, spacing)
path3 = m.create_straight_line(line3_start, line3_end, spacing)
global_path = m.merge_arrays(path1, path2)
global_path = m.merge_arrays(global_path, path3)

# Generate random noise points (adjust the number of points as needed)
num_noise_points = 150
noise_range_x = (-1.0, 7.0)  # Adjust the range based on your scene dimensions
noise_range_y = (-2.0, 6.0)
noise_points = m.create_random_points(num_noise_points, noise_range_x, noise_range_y)

# Calculate the range for generating a random x-coordinate for the cube
min_x = min(bg[:, 0]) + width_cub
max_x = max(bg[:, 0]) - width_cub

def generate_random_coordinates(x_range, y_range):
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_coordinate = random.uniform(x_min, x_max)
    y_coordinate = random.uniform(y_min, y_max)

    return x_coordinate, y_coordinate


cub_x = m.create_cubic_object(generate_random_coordinates(noise_range_x,noise_range_y), width_cub, size_y, spacing)
cub_y = m.create_cubic_object(generate_random_coordinates(noise_range_x,noise_range_y), width_cub, size_y, spacing)

# Delete 20% of random points from cub_x
cub_x = m.delete_random_points(cub_x, delete_percentage=0.2)

# Combine the background, global path, noise points, cubic objects, and spherical object into a single point cloud
combined_cloud = m.merge_arrays(bg, noise_points, cub_x, cub_y)

# Subtract the objects from the background
result = m.subtract_array(bg, combined_cloud)
cleaned_result=result
# Remove outliers from the combined point cloud
#cleaned_result = m.remove_outliers(result, eps=0.11, min_samples=5)

# Perform clustering on the cleaned result
cluster_labels, num_clusters = m.perform_clustering(cleaned_result, eps=0.11, min_samples=5)

# Call centroid_and_box function to get visualizations of centroids and bounding boxes
all_visualizations, bbox_info = m.centroid_and_box(cleaned_result, cluster_labels, num_clusters)

# Convert the cleaned result array to a point cloud
cleaned_result_point_cloud = o3d.geometry.PointCloud()
cleaned_result_point_cloud.points = o3d.utility.Vector3dVector(cleaned_result)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([cleaned_result_point_cloud] + all_visualizations)

# Print information about the bounding boxes
for cluster_id, info in bbox_info.items():
    center, extent, rotation_matrix = info
    print(f"Cluster {cluster_id} - Center: {center}, Extent: {extent}")