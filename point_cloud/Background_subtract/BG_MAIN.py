#carregar o fundo
import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt
from pygroundsegmentation import GroundPlaneFitting

def loadFileToArray(file_path):
    """
    Load background model file to a ndarray .
 
    Parameters:
    - file_path : path to background model file
 
    Returns:
    - merged_array: Array of background model points
    """

    num_arrays_to_merge =99
    i=0
    merged_array = []

    with open(file_path, 'rb') as file:
        loaded_data_dict = pickle.load(file)

    #passamos para array  
    for key, value in loaded_data_dict.items():
        if i < num_arrays_to_merge:
            merged_array.append(value)
            #print("merged array",merged_array)
            i=i+1
    merged_array = np.concatenate(merged_array, axis=0)

    return merged_array


def remove_duplicate_points(ndarray):
    """
    Remove duplicate points from an array.
 
    Parameters:
    - np.ndarray : Array of 3D points, each row representing [x, y, z]
 
    Returns:
    - unique_points : Array of unique points
    """
    unique_points = np.unique(ndarray, axis=0)
    return unique_points

def pointSegmentation(ndarray):
    """
    Process of classifying point clouds into multiple homogeneous regions, the points in the same region will have the same properties
 
    Parameters:
    - np.ndarray : Array of 3D points, each row representing [x, y, z]
 
    Returns:
    - segmentationArray: Array of segmented points
    """

    ground_estimator = GroundPlaneFitting() #Instantiate one of the Estimators

    ground_idxs = ground_estimator.estimate_ground(ndarray)
    segmentationArray = ndarray[ground_idxs]

    return segmentationArray


def calculateMinAndMaxPoints(ndarray):

    """
    Calculate the minimum and maximum value point for each 3D axis.
 
    Parameters:
    - ndarray : Array of 3D points, each row representing [x, y, z]
 
    Returns:
    - minAndMaxList : List of the minimum and maximum pont values for the X,Y,Z dimensions
    """
    minAndMaxList = []  
    cloud_x = ndarray[:, 0]
    cloud_y = ndarray[:, 1]
    cloud_z = ndarray[:, 2]

    x_max, x_min = np.max(cloud_x), np.min(cloud_x)
    y_max, y_min = np.max(cloud_y), np.min(cloud_y)
    z_max, z_min = np.max(cloud_z), np.min(cloud_z)

    minAndMaxList.append(x_max)
    minAndMaxList.append(x_min)
    minAndMaxList.append(y_max)                     
    minAndMaxList.append(y_min)
    minAndMaxList.append(z_max)
    minAndMaxList.append(z_min )

    #print('x_max: ', x_max,  ', x_min: ', x_min)
    #print('y_max: ', y_max, ', y_min: ', y_min)
    #print('z_max: ', z_max, ', z_min: ', z_min)
    #print('Number of points: ', ndarray.size)

    return minAndMaxList


def remove_ground_plane(ndarray,groundThresholder):
    """
    Remove ground plane points.
 
    Parameters:
    - ndarray: Array of 3D points, each row representing [x, y, z]
    - groundThresholder: value in the Z dimension in wich all points below will be deleted
 
    Returns:
    - ground_points: Array of unique points, without the points bellow the threshold
    """
    
    # Find indices of points below the ground threshold
    ground_indices = np.where(ndarray[:, 2] >= groundThresholder)[0] 
    # Extract ground points using the indices
    ground_points = ndarray[ground_indices]

    return ground_points

def voxelization(array,size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=size)
    return (voxel_grid)





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



def voxel_extents_to_pc(voxel_centers, voxel_size):
    voxel_extents = []
    for center in voxel_centers:
        # Assuming the voxel is a cube, you can modify this based on your voxel shape
        extent = np.array([voxel_size, voxel_size, voxel_size]) / 2.0
        voxel_extents.append(center - extent)
        voxel_extents.append(center + extent)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(voxel_extents)))


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


def array_range(array):

    # Calculate the range along each axis
    x_range = np.ptp(array[:, 0])
    y_range = np.ptp(array[:, 1])
    z_range = np.ptp(array[:, 2])

    x_scale_arr = x_range / 2.0  # Assuming the data is centered, adjust if necessary
    y_scale_arr = y_range / 2.0
    z_scale_arr = z_range / 2.0
    return(x_scale_arr,y_scale_arr,z_scale_arr)


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


def voxel_centers(voxel_grid):
    voxel_centers = np.asarray(voxel_grid.get_voxels())
    grid_indices = np.array([voxel.grid_index for voxel in voxel_centers])
    # Reshape the array to have separate columns for x, y, and z coordinates
    voxel_centers_reshaped = grid_indices.reshape(-1, 3)
    return voxel_centers_reshaped
