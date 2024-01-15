import os
import open3d as o3d

# List of folder paths containing PLY files
folder_paths = [
    r"C:\Users\inesolopes\Documents\CARLA_0.9.14\WindowsNoEditor\PythonAPI\util\tutorial\sensor_1_output",
    r"C:\Users\inesolopes\Documents\CARLA_0.9.14\WindowsNoEditor\PythonAPI\util\tutorial\sensor_2_output",
    r"C:\Users\inesolopes\Documents\CARLA_0.9.14\WindowsNoEditor\PythonAPI\util\tutorial\sensor_3_output",
    r"C:\Users\inesolopes\Documents\CARLA_0.9.14\WindowsNoEditor\PythonAPI\util\tutorial\sensor_4_output"
]

# Create a list to store point clouds
point_clouds = []

# Iterate through folders
for folder_path in folder_paths:
    # Iterate through PLY files in each folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".ply"):
            file_path = os.path.join(folder_path, filename)
            pcd = o3d.io.read_point_cloud(file_path)
            point_clouds.append(pcd)

# Visualize all point clouds together
o3d.visualization.draw_geometries(point_clouds)