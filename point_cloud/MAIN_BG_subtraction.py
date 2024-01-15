import numpy as np
import open3d as o3d


def voxel_centers(voxel_grid):
    voxel_centers = np.asarray(voxel_grid.get_voxels())
    grid_indices = np.array([voxel.grid_index for voxel in voxel_centers])
    # Reshape the array to have separate columns for x, y, and z coordinates
    voxel_centers_reshaped = grid_indices.reshape(-1, 3)
    return voxel_centers_reshaped


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



def array_range(array):

    # Calculate the range along each axis
    x_range = np.ptp(array[:, 0])
    y_range = np.ptp(array[:, 1])
    z_range = np.ptp(array[:, 2])

    x_scale_arr = x_range / 2.0  # Assuming the data is centered, adjust if necessary
    y_scale_arr = y_range / 2.0
    z_scale_arr = z_range / 2.0
    return(x_scale_arr,y_scale_arr,z_scale_arr)



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

def array_to_pc(array):
    """
    Transforms an np.format into a point_cloud format.

    Parameters:
    array(np.array): The array of points meant to be transformed.

    Returns:
    result_cloud (point_cloud): The array given but in point_cloud format.
    """
    result_cloud = o3d.geometry.PointCloud()
    result_cloud.points = o3d.utility.Vector3dVector(array)
    return result_cloud



def visualize(point_cloud):
    """
    Allows the visualization of one point_cloud.

    Parameters:
    point_cloud (point_cloud): The final point cloud to visualize. 
    """

    # The Input is a pointcloud structure
    o3d.visualization.draw_geometries([point_cloud])
