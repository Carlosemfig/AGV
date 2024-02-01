#carregar o fundo
import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt
from pygroundsegmentation import GroundPlaneFitting
import json
import statistics



def get_2dboxes_from_json(json_name):
    with open(json_name) as json_file:
        data = json.load(json_file)

    boxes = {}
    box_id_counter = 0

    for feature in data['features']:
        coordinates = feature['geometry']['coordinates'][0]
        # Assuming the coordinates are in the format [x, y], extract x and y values
        x_values = [coord[0] for coord in coordinates]
        y_values = [coord[1] for coord in coordinates]
        
        corners= list(zip(x_values, y_values))
        corners.pop()
        boxes[box_id_counter] = corners

        # Increment box_id for the next box
        box_id_counter += 1
    return boxes



def transform_2d_to_3d(boxes_dict,base=-2,height=2):
    new_boxes={}
    for box_id, corners in boxes_dict.items():
        # Convert 2D corners to 3D corners

        corners_scaled = [(coord[0], coord[1]) for coord in corners]

        corners_3d = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1))*base))
        corners_top = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1)) * height))
        final_corners=np.vstack((corners_3d, corners_top))
        new_boxes[box_id] = final_corners
    return new_boxes


def create_3d_bboxes(new_boxes_dict, line_color=[0.0, 1.0, 0.0]):
    geometries = []
    multiplier=1
    for box_id, corners in new_boxes_dict.items():
        # Convert 2D corners to 3D corners
        print("corners",corners)
        corners_scaled = [(coord[0] * multiplier, coord[1] * multiplier) for coord in corners]

        corners_3d = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1))))
        corners_top = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1))))

        # Create 3D line sets for the bounding box edges
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Set color and thickness
        line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array(line_color), (len(lines), 1)))
        
        geometries.append(line_set)

    return geometries


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

def most_common_z_value(points):
    """
    Find the most common Z value in an array of 3D points.

    Parameters:
    - points (np.ndarray): Array of 3D points, each row representing [x, y, z]

    Returns:
    - float: Most common Z value
    """
    #z_values = points[:, 2]  # Extract the Z values from the array
    z_values = np.round(points[:, 2], decimals=1)
    print(z_values)
    most_common_z = statistics.mode(z_values)
    print("mais commum",most_common_z)
    return most_common_z

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
    ground_indices = np.where(ndarray[:, 2] > groundThresholder)[0] 
    # Extract ground points using the indices
    ground_points = ndarray[ground_indices]

    return ground_points



def remove_celling_plane(ndarray,cellingThresholder):
    """
    Remove ground plane points.
 
    Parameters:
    - ndarray: Array of 3D points, each row representing [x, y, z]
    - groundThresholder: value in the Z dimension in wich all points below will be deleted
 
    Returns:
    - ground_points: Array of unique points, without the points bellow the threshold
    """
    
    # Find indices of points below the ground threshold
    ground_indices = np.where(ndarray[:, 2] < cellingThresholder)[0] 
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

def points_outside_all_boxes(points, boxes):
    """
    Check if each point in an array is outside all specified boxes.

    Parameters:
    - points (np.ndarray): Array of 3D points, each row representing [x, y, z]
    - boxes (dict): Dictionary of boxes, where keys are box indices and values are arrays representing the corners of the boxes

    Returns:
    - np.ndarray: Array of points that are not inside any box
    """
    queries = np.array(points)
    points_inside_boxes = np.zeros(len(queries), dtype=bool)

    for box_index, box_corners in boxes.items():
        # Convert box_corners to a numpy array for easier manipulation
        box_corners = np.array(box_corners)

        # Extract min and max coordinates for each dimension
        min_coords = np.min(box_corners, axis=0)
        max_coords = np.max(box_corners, axis=0)

        # Check if each point is inside the current box
        inside_box_mask = np.all((queries >= min_coords) & (queries <= max_coords), axis=1)

        # Update points_inside_boxes to mark points inside any box as True
        points_inside_boxes |= inside_box_mask

    # Find points that are not inside any box
    points_outside_boxes = queries[~points_inside_boxes]

    return points_outside_boxes






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



def subtract_fences(obj,bboxes,treshold):
    obj=remove_duplicate_points(obj)

    min_and_max_point_list=calculateMinAndMaxPoints(obj)

    groundThresholder = min_and_max_point_list[5] + treshold

    obj = remove_ground_plane(obj,groundThresholder)
    result = points_outside_all_boxes(obj, bboxes)
    return result

def subtract_bg(obj,bg_voxel,treshold):
    """
    A function that given a array and a voxel model of the background,
    Returns the array resulting from the subtraction.

    Parameters:
    obj(np.array)
    bg.voxel(voxel_grid)
    treshold (float) - the margin to erase the flor, the bigger the higher the floor is erased.

    returns:
    Result (np.array)
    """

    obj=remove_duplicate_points(obj)

    most_common_z=most_common_z_value(obj)

    groundThresholder = most_common_z+ treshold

    obj = remove_ground_plane(obj,groundThresholder)
    result = points_outside_all_voxels(obj, bg_voxel)
    return result

def final_subtraction(obj,bboxes,bg_voxel,treshold):
    obj=remove_duplicate_points(obj)

    most_common_z=most_common_z_value(obj)
    groundThresholder = most_common_z+ treshold

    obj = remove_ground_plane(obj,groundThresholder)

    result = points_outside_all_boxes(obj, bboxes)

    result = points_outside_all_voxels(result, bg_voxel)

    return result




def get_fences(json_name,base=-2,top=2):
    with open(json_name) as json_file:
        data = json.load(json_file)

    boxes = {}
    box_id_counter = 0

    for feature in data['features']:
        coordinates = feature['geometry']['coordinates'][0]
        # Assuming the coordinates are in the format [x, y], extract x and y values
        x_values = [coord[0] for coord in coordinates]
        y_values = [coord[1] for coord in coordinates]
        
        corners= list(zip(x_values, y_values))
        corners.pop()
        corners_scaled = [(coord[0], coord[1]) for coord in corners]

        corners_3d = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1))*base))
        corners_top = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1)) * top))
        final_corners=np.vstack((corners_3d, corners_top))

        boxes[box_id_counter] = final_corners

        # Increment box_id for the next box
        box_id_counter += 1
    return boxes



def transform_2d_to_3d(boxes_dict,base=-2,height=2):
    new_boxes={}
    for box_id, corners in boxes_dict.items():
        # Convert 2D corners to 3D corners

        corners_scaled = [(coord[0], coord[1]) for coord in corners]

        corners_3d = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1))*base))
        corners_top = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1)) * height))
        final_corners=np.vstack((corners_3d, corners_top))
        new_boxes[box_id] = final_corners
    return new_boxes
