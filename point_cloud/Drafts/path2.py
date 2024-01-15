import MAIN as m
import open3d as o3d
import numpy as np

# Load background point cloud
bg = m.load_pc("bg.pcd")
bg = m.pc_to_array(bg)

spacing = 0.1  # Adjust this value based on your requirements

# Define the starting and ending coordinates of the three straight lines
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)
line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)
line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)
path1 = m.create_straight_line(line1_start, line1_end, spacing)
path2 = m.create_straight_line(line2_start, line2_end, spacing)
path3 = m.create_straight_line(line3_start, line3_end, spacing)
global_path = m.merge_arrays(path1, path2)
global_path = m.merge_arrays(global_path, path3)

# Create a new diagonal path
diagonal_line_start = (5.0, 0.0)
diagonal_line_end = (1.0, 7.0)
diagonal_path = m.create_straight_line(diagonal_line_start, diagonal_line_end, spacing)

# Convert NumPy array to Open3D PointCloud
bg_cloud = o3d.geometry.PointCloud()
bg_cloud.points = o3d.utility.Vector3dVector(bg)

# Convert paths to Open3D PointCloud
global_path_cloud = o3d.geometry.PointCloud()
global_path_cloud.points = o3d.utility.Vector3dVector(global_path)

diagonal_path_cloud = o3d.geometry.PointCloud()
diagonal_path_cloud.points = o3d.utility.Vector3dVector(diagonal_path)

# Visualize the global path and diagonal path
o3d.visualization.draw_geometries([bg_cloud, global_path_cloud, diagonal_path_cloud])