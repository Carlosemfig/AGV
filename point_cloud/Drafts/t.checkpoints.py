import MAIN as m
import open3d as o3d
import numpy as np

# Load background point cloud
bg = m.load_pc("bg.pcd")
bg = m.pc_to_array(bg)

# Define the starting and ending coordinates of the three straight lines
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)
line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)
line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)
spacing = 0.1

# Create the global path
path1 = m.create_straight_line(line1_start, line1_end, spacing)
path2 = m.create_straight_line(line2_start, line2_end, spacing)
path3 = m.create_straight_line(line3_start, line3_end, spacing)
global_path = m.merge_arrays(path1, path2)
global_path = m.merge_arrays(global_path, path3)

# create checkpoints in path
checkpoint_spacing = 8
checkpoints = m.create_checkpoints(global_path, checkpoint_spacing)

print("Checkpoints:", checkpoints)

# Combine the background, global path, and checkpoints into a single point cloud
combined_cloud = m.merge_arrays(bg, checkpoints)

# Convert the combined_cloud array to a point cloud
combined_point_cloud = o3d.geometry.PointCloud()
combined_point_cloud.points = o3d.utility.Vector3dVector(combined_cloud)

# Visualize the combined point cloud
o3d.visualization.draw_geometries([combined_point_cloud])
