#carregar o fundo
import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt


# Load voxel grid from the saved file
loaded_voxel_grid = o3d.io.read_voxel_grid("bg_voxel_grid.ply")

bg=loaded_voxel_grid




"""____________________obj_________________________"""
#file_path_obj = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_moving.pkl'
file_path_obj = r'data_dict_foreground.pkl'

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



result = points_outside_all_voxels(obj, bg)
pc_result=m.array_to_pc(result)
m.visualize(pc_result)
print("DONE")
print("depois",len(result))

path_file= r"subtraction_result.pcd"
m.save_pc(pc_result,path_file)
