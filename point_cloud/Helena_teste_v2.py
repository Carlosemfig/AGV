import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt

# Specify the path to the Pickle file
file_path = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_new.pkl'


# Load the data from the Pickle file
with open(file_path, 'rb') as file:
    loaded_data_dict = pickle.load(file)

print("Comprimento",len(loaded_data_dict))
"""
first_timestamp = list(loaded_data_dict.keys())[0]
first_array = loaded_data_dict[first_timestamp]
pc=m.array_to_pc(first_array)



"""


# Merge all arrays in the dictionary
#merged_array = np.concatenate(list(loaded_data_dict.values()), axis=0)




num_arrays_to_merge =1
i=0
merged_array = []

for key, value in loaded_data_dict.items():
    if i < num_arrays_to_merge:
        merged_array.append(value)
        #print("merged array",merged_array)
        i=i+1

"""_________ARRAY_______"""
merged_array = np.concatenate(merged_array, axis=0)
# Find the minimum and maximum values for each column
np.set_printoptions(threshold=np.inf)
# Write the results to a text file
with open("voxel_info_before.txt", "w") as f:
    print("Scaled Voxel Centers:", file=f)
    print(merged_array, file=f)



# Calculate the range along each axis
x_range = np.ptp(merged_array[:, 0])
y_range = np.ptp(merged_array[:, 1])
z_range = np.ptp(merged_array[:, 2])

# Calculate the scale factors along each axis
x_scale_arr = x_range / 2.0  # Assuming the data is centered, adjust if necessary
y_scale_arr = y_range / 2.0
z_scale_arr = z_range / 2.0

# Print or use the scale factors
print("X Scale Factor:", x_scale_arr)
print("Y Scale Factor:", y_scale_arr)
print("Z Scale Factor:", z_scale_arr)

"""_____POINT CLOUD_____"""
# Convert the array to an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(merged_array)


"""_______VOXEL______"""
# Voxel size (adjust as needed)
voxel_size = 0.02

# Voxelization
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
# Get voxel centers
voxel_centers = np.asarray(voxel_grid.get_voxels())


grid_indices = np.array([voxel.grid_index for voxel in voxel_centers])
# Reshape the array to have separate columns for x, y, and z coordinates
voxel_centers_reshaped = grid_indices.reshape(-1, 3)

# Calculate the range along each axis
x_range = np.ptp(voxel_centers_reshaped[:, 0])
y_range = np.ptp(voxel_centers_reshaped[:, 1])
z_range = np.ptp(voxel_centers_reshaped[:, 2])

# Calculate the scale factors along each axis
x_scale_vox = x_range / 2.0
y_scale_vox = y_range / 2.0
z_scale_vox = z_range / 2.0
print("X Scale Factor for Voxel Grid:", x_scale_vox)
print("Y Scale Factor for Voxel Grid:", y_scale_vox)
print("Z Scale Factor for Voxel Grid:", z_scale_vox)



"""_____VOXEL RESCALED ____-____"""

x_mult=x_scale_vox/x_scale_arr
y_mult=y_scale_vox/y_scale_arr
z_mult=z_scale_vox/z_scale_arr

# Scale adjustment for the voxel grid
scaled_np_arr = merged_array * np.array([x_mult, y_mult, z_mult])



# Calculate the range along each axis
x_range = np.ptp(scaled_np_arr[:, 0])
y_range = np.ptp(scaled_np_arr[:, 1])
z_range = np.ptp(scaled_np_arr[:, 2])

# Calculate the scale factors along each axis
x_scale_arr = x_range / 2.0  # Assuming the data is centered, adjust if necessary
y_scale_arr = y_range / 2.0
z_scale_arr = z_range / 2.0

# Print or use the scale factors
print("X Scale Factor_reshape:", x_scale_arr)
print("Y Scale Factor_reshape:", y_scale_arr)
print("Z Scale Factor_reshape:", z_scale_arr)

np.set_printoptions(threshold=np.inf)
# Write the results to a text file
with open("voxel_info_after.txt", "w") as f:
    print("Scaled Voxel Centers:", file=f)
    print(scaled_np_arr, file=f)
# Print or use the scale factors



"""
np.set_printoptions(threshold=np.inf)
with open("voxel_info.txt", "w") as f:
    print("Voxel Grid Information:", file=f)
    print(voxel_grid, file=f)
    
    print("\nVoxel Centers:", file=f)
    print(voxel_centers, file=f)"""

#print([voxel_grid])
# Visualize the voxelized point cloud
o3d.visualization.draw_geometries([voxel_grid])

# Save the voxelized point cloud to a file
o3d.io.write_voxel_grid("voxelized_point_cloud.vox", voxel_grid)
