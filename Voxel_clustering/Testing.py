import os
import open3d as o3d

# Folder path containing the PLY files
folder_path = r"C:\Users\inesolopes\Documents\CARLA_0.9.14\WindowsNoEditor\PythonAPI\util\tutorial\sensor_1_output"

# Create a list to store point clouds
point_clouds = []

# Iterate through PLY files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".ply"):
        file_path = os.path.join(folder_path, filename)
        pcd = o3d.io.read_point_cloud(file_path)
        point_clouds.append(pcd)

# Visualize all point clouds together
o3d.visualization.draw_geometries(point_clouds)