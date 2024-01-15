import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt

"""________________Background__________"""
# Specify the path to the Pickle file
file_path = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_new.pkl'


# Load the data from the Pickle file
with open(file_path, 'rb') as file:
    loaded_data_dict = pickle.load(file)

print("Comprimento",len(loaded_data_dict))


num_arrays_to_merge =99
i=0
merged_array = []

for key, value in loaded_data_dict.items():
    if i < num_arrays_to_merge:
        merged_array.append(value)
        #print("merged array",merged_array)
        i=i+1

merged_array = np.concatenate(merged_array, axis=0)

# Convert the array to an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(merged_array)

# Voxel size (adjust as needed)
voxel_size = 0.02

# Voxelization
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
# Get voxel centers
voxel_centers = np.asarray(voxel_grid.get_voxels())


grid_indices = np.array([voxel.grid_index for voxel in voxel_centers])
# Reshape the array to have separate columns for x, y, and z coordinates
voxel_centers_reshaped = grid_indices.reshape(-1, 3)
np.set_printoptions(threshold=np.inf)




"""________________OBJECT__________"""
# Specify the path to the Pickle file
file_path_obj = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_moving.pkl'

# Load the data from the Pickle file
with open(file_path_obj, 'rb') as file:
    loaded_data_dict_obj = pickle.load(file)

print("Comprimento",len(loaded_data_dict))


num_arrays_to_merge_obj =10
i=0
merged_array_obj = []

for key, value in loaded_data_dict_obj.items():
    if i < num_arrays_to_merge_obj:
        merged_array_obj.append(value)
        #print("merged array",merged_array)
        i=i+1

merged_array_obj = np.concatenate(merged_array_obj, axis=0)
# Convert the array to an Open3D point cloud
pcd_object = o3d.geometry.PointCloud()
pcd_object.points = o3d.utility.Vector3dVector(merged_array_obj)

# Voxelization of object
voxel_grid_object = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_object, voxel_size=voxel_size)


"""__________diference____________"""


# Background subtraction

# Get voxel indices of background and object
voxel_indices_background = np.asarray(voxel_grid.get_voxels())
voxel_indices_object = np.asarray(voxel_grid_object.get_voxels())

# Identify indices of object voxels that are not present in the background
foreground_voxel_indices = np.setdiff1d(voxel_indices_object, voxel_indices_background)

# Create a new VoxelGrid for foreground using the identified voxel indices
foreground_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_object, voxel_size=voxel_size, indices=foreground_voxel_indices)


# Visualize the voxelized point cloud
o3d.visualization.draw_geometries([foreground_voxel_grid])

# Save the voxelized point cloud to a file
o3d.io.write_voxel_grid("voxelized_point_cloud.vox", foreground_voxel_grid)
