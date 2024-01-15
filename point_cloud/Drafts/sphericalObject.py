import MAIN as m
import open3d as o3d
import numpy as np

# Load background point cloud
bg = m.load_pc("bg.pcd")
bg = m.pc_to_array(bg)

# Example usage of the function
width_cub = 0.4  # Width range
size_y = 0.3  # Adjust this value based on your requirements
spacing = 0.1  # Adjust this value based on your requirements

# Define the starting and ending coordinates of the three straight lines
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)
line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)
line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)
spacing = 0.1
rounding_radius=size_y / 2 

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

# Create the cube at a random position outside the walls
cub_random_x = np.random.uniform(min(bg[:, 0]), max(bg[:, 0]) - width_cub)
cub_random_y = np.random.uniform(min(bg[:, 0]), max(bg[:, 0]) - width_cub)
cub_x = m.create_cubic_object((cub_random_x, 0), width_cub, size_y, spacing)
cub_y = m.create_cubic_object((cub_random_y, 0), width_cub, size_y, spacing)

# Create an approximate spherical object
spherical_radius = 0.3
spherical_center = (1.0, 2.0, 0.0)
num_points_theta = 20
num_points_phi = 10

spherical_object = m.create_approximate_spherical_object(
    spherical_center, spherical_radius, spacing, num_points_theta, num_points_phi
)

# Delete random points
spherical_object = m.delete_random_points(spherical_object, delete_percentage=0.1)

# Delete of random points
cub_x = m.delete_random_points(cub_x, delete_percentage=0.1)

# Create a circle with the specified rounding radius
circle_radius = rounding_radius
circle_center = (cub_random_x, 0.0, 0.0)
num_points_circle = 50

circle_coordinates = m.create_circle(circle_center, circle_radius, num_points_circle)

# Convert circle_coordinates to a numpy array for visualization
circle_coordinates_np = np.array(circle_coordinates)

# Convert the circle_coordinates to an Open3D PointCloud
circle_point_cloud = o3d.geometry.PointCloud()
circle_point_cloud.points = o3d.utility.Vector3dVector(circle_coordinates_np)

# Combine the background, global path, noise points, cubic objects, and spherical object into a single point cloud
combined_cloud = m.merge_arrays(bg, noise_points, cub_x, cub_y, spherical_object, circle_coordinates)

# Convert the combined_cloud to an Open3D PointCloud
combined_point_cloud = o3d.geometry.PointCloud()
combined_point_cloud.points = o3d.utility.Vector3dVector(combined_cloud)

# Subtract the objects from the background
result = m.subtract_array(bg, combined_cloud)

# Perform clustering on the result
cluster_labels, num_clusters = m.perform_clustering(result, eps=0.11, min_samples=5)

# Call centroid_and_box function to get visualizations of centroids and bounding boxes
all_visualizations, bbox_info = m.centroid_and_box(result, cluster_labels, num_clusters)

# Convert the result array to an Open3D PointCloud
result_point_cloud = o3d.geometry.PointCloud()
result_point_cloud.points = o3d.utility.Vector3dVector(result)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([result_point_cloud, circle_point_cloud] + all_visualizations)

# Print information about the bounding boxes
for cluster_id, info in bbox_info.items():
    center, extent, rotation_matrix = info
    print(f"Cluster {cluster_id} - Center: {center}, Extent: {extent}")