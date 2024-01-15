#carregar o fundo
import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt

voxel_size = 0.1

# Load voxel grid from the saved file
loaded_voxel_grid = o3d.io.read_voxel_grid("saved_voxel_grid.ply")

bg=loaded_voxel_grid


def voxel_centers(voxel_grid):
    voxel_centers = np.asarray(voxel_grid.get_voxels())
    grid_indices = np.array([voxel.grid_index for voxel in voxel_centers])
    # Reshape the array to have separate columns for x, y, and z coordinates
    voxel_centers_reshaped = grid_indices.reshape(-1, 3)
    return voxel_centers_reshaped


"""____________________obj_________________________"""
#file_path_obj = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_moving.pkl'
file_path_obj = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_track_1.pkl'

# Load the data from the Pickle file
with open(file_path_obj, 'rb') as file:
    loaded_data_dict_obj = pickle.load(file)



num_arrays_to_merge_obj =50
i=0
merged_array_obj = []

for key, value in loaded_data_dict_obj.items():
    if i < num_arrays_to_merge_obj:
        merged_array_obj.append(value)
        #print("merged array",merged_array)
        i=i+1

obj = np.concatenate(merged_array_obj, axis=0)


def remove_duplicate_points(points):
    """
    Remove duplicate points from an array.

    Parameters:
    - points (np.ndarray): Array of 3D points, each row representing [x, y, z]

    Returns:
    - np.ndarray: Array of unique points
    """
    unique_points = np.unique(points, axis=0)
    return unique_points


obj=remove_duplicate_points(obj)


"""-------------data processing_________________________"""
def voxel_range(voxel_grid):
    voxel_centers = np.asarray(voxel_grid.get_voxels())
    grid_indices = np.array([voxel.grid_index for voxel in voxel_centers])
    voxel_centers_reshaped = grid_indices.reshape(-1, 3)
    centroid_original = np.mean(voxel_centers_reshaped, axis=0)
    # Calculate the range along each axis
    x_range = np.ptp(voxel_centers_reshaped[:, 0])
    y_range = np.ptp(voxel_centers_reshaped[:, 1])
    z_range = np.ptp(voxel_centers_reshaped[:, 2])

    x_scale_vox = x_range / 2.0
    y_scale_vox = y_range / 2.0
    z_scale_vox = z_range / 2.0
    return(x_scale_vox,y_scale_vox,z_scale_vox)

vox_scale=voxel_range(bg)

def array_range(array):

    # Calculate the range along each axis
    x_range = np.ptp(array[:, 0])
    y_range = np.ptp(array[:, 1])
    z_range = np.ptp(array[:, 2])

    x_scale_arr = x_range / 2.0  # Assuming the data is centered, adjust if necessary
    y_scale_arr = y_range / 2.0
    z_scale_arr = z_range / 2.0
    return(x_scale_arr,y_scale_arr,z_scale_arr)

arr_scale=array_range(obj)

def rescale_array(voxel_grid,array):

    voxel_centers = np.asarray(voxel_grid.get_voxels())
    grid_indices = np.array([voxel.grid_index for voxel in voxel_centers])
    voxel_centers_reshaped = grid_indices.reshape(-1, 3)
    
    centroid_voxel = np.mean(voxel_centers_reshaped, axis=0)
    # Calculate the range along each axis
    x_range = np.ptp(voxel_centers_reshaped[:, 0])
    y_range = np.ptp(voxel_centers_reshaped[:, 1])
    z_range = np.ptp(voxel_centers_reshaped[:, 2])
    x_scale_vox = x_range / 2.0
    y_scale_vox = y_range / 2.0
    z_scale_vox = z_range / 2.0
    vox_scale=(x_scale_vox,y_scale_vox,z_scale_vox)

    
    # Calculate the range along each axis
    
    x_range = np.ptp(array[:, 0])
    y_range = np.ptp(array[:, 1])
    z_range = np.ptp(array[:, 2])
    x_scale_arr = x_range / 2.0  # Assuming the data is centered, adjust if necessary
    y_scale_arr = y_range / 2.0
    z_scale_arr = z_range / 2.0
    arr_scale=(x_scale_arr,y_scale_arr,z_scale_arr)



    x_mult=vox_scale[0]/arr_scale[0]
    y_mult=vox_scale[1]/arr_scale[1]
    z_mult=vox_scale[2]/arr_scale[2]
    scaled_arr = array * np.array([x_mult, y_mult, z_mult])
    centroid_array = np.mean(scaled_arr, axis=0)
    

    # Compute the translation vector
    translation = centroid_voxel - centroid_array

    # Apply the translation to align the scaled array with the original
    scaled_arr += translation

    return scaled_arr

resc_obj=rescale_array(bg,obj)


def is_point_inside_voxel(point, voxel_center, voxel_extent):
    """
    Check if a 3D point is inside a voxel.

    Parameters:
    - point (np.ndarray): 3D coordinates of the point [x, y, z]
    - voxel_center (np.ndarray): 3D coordinates of the voxel center [x, y, z]
    - voxel_extent (float): Size of the voxel in each dimension

    Returns:
    - bool: True if the point is inside the voxel, False otherwise
    """
    # Calculate the minimum and maximum bounds of the voxel
    voxel_min_bound = voxel_center - voxel_extent / 2
    voxel_max_bound = voxel_center + voxel_extent / 2

    # Check if the point is inside the voxel
    inside_voxel = all(voxel_min_bound <= point) and all(point <= voxel_max_bound)
    
    return inside_voxel


def points_outside_all_voxels(points, voxel_centers, voxel_extent):
    """
    Check if each point in an array is outside all voxels in a set.

    Parameters:
    - points (np.ndarray): Array of 3D points, each row representing [x, y, z]
    - voxel_centers (np.ndarray): Array of voxel centers, each row representing [x, y, z]
    - voxel_extent (float): Size of the voxel in each dimension

    Returns:
    - np.ndarray: Array of points that are not inside any voxel
    """
    points_outside_voxels = []

    for point in points:
        # Check if the point is inside any voxel
        inside_any_voxel = any(is_point_inside_voxel(point, voxel_center, voxel_extent) for voxel_center in voxel_centers)

        # If the point is not inside any voxel, add it to the result array
        if not inside_any_voxel:
            points_outside_voxels.append(point)

    return np.array(points_outside_voxels)


def points_outside_all_voxels(points, voxel_grid):
    """
    Check if each point in an array is outside all occupied voxels in a voxel grid.

    Parameters:
    - points (np.ndarray): Array of 3D points, each row representing [x, y, z]
    - voxel_grid (o3d.geometry.VoxelGrid): Open3D VoxelGrid object representing the occupied voxels

    Returns:
    - np.ndarray: Array of points that are not inside any voxel
    """
    queries = o3d.utility.Vector3dVector(points)
    included_points = voxel_grid.check_if_included(queries)

    included_points_array = np.array(included_points)
    
    points_outside_voxels = points[~included_points_array]
    
    return points_outside_voxels


def voxel_extents_to_pc(voxel_centers, voxel_size):
    voxel_extents = []
    for center in voxel_centers:
        # Assuming the voxel is a cube, you can modify this based on your voxel shape
        extent = np.array([voxel_size, voxel_size, voxel_size]) / 2.0
        voxel_extents.append(center - extent)
        voxel_extents.append(center + extent)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(voxel_extents)))


pc_obj=m.array_to_pc(resc_obj)
pc_bg=m.array_to_pc(voxel_centers(bg))


pc_obj.paint_uniform_color([1, 0, 0])  # Red

pc_bg.paint_uniform_color([0, 0, 1])   # Blue
# Convert voxel centers to a point cloud
#pc_voxel_extents = voxel_extents_to_pc(voxel_centers(bg), voxel_size)

# Combine the three point clouds
combined_pc = o3d.geometry.PointCloud()
combined_pc += pc_obj
combined_pc += pc_bg
#ombined_pc += pc_voxel_extents


# Visualize the overlapping
o3d.visualization.draw_geometries([combined_pc])


#visualize the voxel 
o3d.visualization.draw_geometries([bg])

#visualize the result.
print("antes",len(obj))

result = points_outside_all_voxels(obj, bg)
pc_result=m.array_to_pc(result)
m.visualize(pc_result)
print("DONE")
print("depois",len(result))

path_file= r"C:\Users\hvendas\Desktop\GIT\app\point_cloud\bg_subtract_one_only.pcd"
m.save_pc(pc_result,path_file)

