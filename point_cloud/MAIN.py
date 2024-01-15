import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
import numpy as np
import math
from scipy.spatial.transform import Rotation

########## GENERAL ##########

def load_pc(file_path):
    """
    Loads a point cloud in .pcd format.
    
    Parameters:
    file path (str): Is the name of the file ex: "result.pcd"
    
    Returns:
    pc (point_cloud): The content of the file in point_cloud format.
    """
    # Input is the file path
    # Output is the point cloud in a array shape
    pc = o3d.io.read_point_cloud(file_path)
    return pc

def save_pc(point_cloud, file_path):
    """
    Saves a point_cloud file in .pcd
    
    Parameters:
    point_cloud(point_cloud): Is the point cloud meant to be saved
    file path (str): Is the name of the file ex: "result.pcd"
    
    """
    # Input is the point cloud to be saved and the path to save it
    o3d.io.write_point_cloud(file_path, point_cloud)

def subtract_array(bg_points, object_points):
    """
    Subtract two np.arrays.

    Parameters:
    bg_points(np.array): Is the background meant to be subtracted.
    object_points(np.array): Is the map with the objects from where the bg is meant to be subtracted.

    Returns:
    result_points(np.array): The resulting points from the subtraction in a np.array.
    """
    # Input is the arrays for both the bg and the map with objects
    # Output is the resulting point cloud in array format
    result_points = np.array([point for point in object_points if not np.any(np.all(point == bg_points, axis=1))])
    return result_points

def pc_to_array(point_cloud):
    """
    Transforms an point_cloud in a np.array format.

    Parameters:
    point_cloud(point_cloud): The point_cloud meant to be transformed.

    Returns:
    array(np.array): The point_cloud given but in a np.array format.
    """
    array=np.asarray(point_cloud.points)
    return array

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

def create_approximate_spherical_object(center, radius, spacing, num_points_theta=50, num_points_phi=25):
    """
    Create points representing an approximate spherical object on one side (hemisphere).

    Parameters:
    center (tuple): Center coordinates (x, y, z) of the sphere.
    radius (float): Radius of the sphere.
    spacing (float): Spacing between points on the sphere.
    num_points_theta (int): Number of points in the azimuthal direction (around the z-axis).
    num_points_phi (int): Number of points in the polar direction (from the positive z-axis).

    Returns:
    sphere_points (np array): Points representing the approximate spherical side.
    """
    sphere_points = []

    for i in range(num_points_phi):
        phi = i * (np.pi / (2 * num_points_phi))  # Ranges from 0 to pi/2

        for j in range(num_points_theta):
            theta = j * (2 * np.pi / num_points_theta)  # Ranges from 0 to 2*pi

            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)

            sphere_points.append((x, y, z))

    return np.array(sphere_points)

#def create_cube_and_rounded_side(center, side_length, rounding_radius, num_points_theta=50):
    """
    Create coordinates for a cube with one rounded side (hemisphere) in 3D.

    Parameters:
    center (tuple): Center coordinates (x, y, z) of the cube.
    side_length (float): Length of the cube sides.
    rounding_radius (float): Radius of the rounding hemisphere.
    num_points_theta (int): Number of points in the azimuthal direction (around the z-axis).

    Returns:
    cube_coordinates (list of tuples): Coordinates representing the cube with a rounded side in 3D.
    """
    cube_coordinates = []

    # Cube coordinates
    cube_side = side_length / 2
    cube_coordinates.extend([
        (center[0] - cube_side, center[1] - cube_side, center[2] - cube_side),
        (center[0] + cube_side, center[1] - cube_side, center[2] - cube_side),
        (center[0] + cube_side, center[1] + cube_side, center[2] - cube_side),
        (center[0] - cube_side, center[1] + cube_side, center[2] - cube_side),
        (center[0] - cube_side, center[1] - cube_side, center[2] + cube_side),
        (center[0] + cube_side, center[1] - cube_side, center[2] + cube_side),
        (center[0] + cube_side, center[1] + cube_side, center[2] + cube_side),
        (center[0] - cube_side, center[1] + cube_side, center[2] + cube_side)
    ])

    # Rounded side coordinates (hemisphere)
    for i in range(num_points_theta):
        theta = i * (2 * np.pi / num_points_theta)  # Ranges from 0 to 2*pi

        x = center[0] + cube_side * np.cos(theta)
        y = center[1] + cube_side * np.sin(theta)
        z = center[2] + cube_side + rounding_radius * np.sin(np.pi / 2 * np.cos(theta))

        cube_coordinates.append((x, y, z))

    return cube_coordinates

def merge_arrays(*arrays):
    """
    Merges multiple arrays into a single array.

    Parameters:
    *arrays (np array): Variable number of arrays to be merged.

    Returns:
    merged_array (np array): The merged array.
    """
    merged_array = np.vstack(arrays)
    return merged_array

def random_coordinate_from_array(array):
    """
    Function that gets a random pair of coordinates from a given array. 

    Parameters:
    array (np array): Can be a path or a plane for example.

    Returns:
    x,y (tuple): Random coordinates belonging to the given array.
    """

    # Get the number of rows in the global path
    num_rows = array.shape[0]

    # Generate a random index within the range of available rows
    random_index = np.random.randint(0, num_rows)

    # Get the x and y coordinates from the randomly selected row
    x, y, _ = array[random_index]
    #print(random_index)

    return x, y

def create_straight_line(line_start, line_end, spacing):
    """
    Creates a np_array of a strait line. 

    Parameters:
    line_start (tuple): Is the (x,y) cordinates where the line starts.
    line_end (tuple): Is the (x,y) cordinates where the line ends.
    spacing (float): Is the space between the represented points of the line.

    Returns:
    path (np array): A np_array representation of a line. 
    """
    x1=line_start[0]
    y1=line_start[1]

    x2=line_end[0]
    y2=line_end[1]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Calculate the number of points needed based on the spacing
    num_points = int(distance / spacing)
    
    # Generate the path points
    x_path = np.linspace(x1, x2, num_points,endpoint=False)
    
    y_path = np.linspace(y1, y2, num_points,endpoint=False)
    z_path = np.zeros(num_points)  # Set the z-coordinates to be at floor level
    
    path = np.column_stack((x_path, y_path, z_path))
    return path

def create_cubic_object(center, size_x,size_y, spacing):
    """
    Creates a np_array of a cubic object. 

    Parameters:
    center (tuple): Is the (x,y) cordinates for the center of the cube.
    size_x (float): Is the value of the width (x-direction) of the cube.
    size_y (float): Is the value of the lenght (y-direction), and height (z_direction) of the cube.
    spacing (float): Is the space between the represented points of the cube.

    Returns:
    object (np array): A np_array representation of a cube. 
    """

    # Calculate the dimensions of the cubic object
    h_width= size_x/2
    h_lenght=size_y/2
    #h_lenght=length/2
    #print(round(center[0],1),round(center[1], 1))


    width_range = np.arange(round(center[0],1) - h_width, round(center[0],1) + h_width +0.0000000001, spacing)
    length_range = np.arange(round(center[1], 1) - h_lenght, round(center[1], 1) + h_lenght +0.0000000001, spacing)
    height_range = np.arange(0, size_y, spacing)

    #print("width and lenght",width_range,length_range)
    # Create a grid of points to represent the cubic object
    x, y, z = np.meshgrid(width_range, length_range, height_range, indexing='ij')

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    object=np.column_stack((x, y, z))

    return object

def subtract_main(bg_path,obj_path,result_path="result.pcd"):
    
    all=[]
    # Input the path for the bg, objects and where to save the resulting file
    # Output saves the resulting file and allows to visualize it
    bg=load_pc(bg_path)
    obj=load_pc(obj_path)

    bg=bg.points
    obj=obj.points

    result=subtract_array(bg,obj)

    result=array_to_pc(result)

    save_pc(result, result_path)
    all.append(result)
    run_visualizer(all)

def constant_x(length, height, plane, spacing):
    """
    Creates a np_array of a x-plane. 

    Parameters:
    lenght (tuple): (a,b), where a is the first point and b is the last in y-direction.
    height (tuple): (a,b), where a is the first point and b is the last in z-direction.
    plane (float): Is the value of x where the plane is to be placed.
    spacing (float): Is the space between the represented points of the plane.

    Returns:
    object (np array): A np_array representation of a plane. 
    """
    # Create a grid of points to represent the floor
    y = np.arange(length[0], length[1], spacing)
    z= np.arange(height[0], height[1], spacing)
    y, z = np.meshgrid(y, z)
    x = np.zeros_like(y)  # Set the Z coordinate to zero for the floor
    if plane != 0:
        x[x == 0] = plane

    
    # Flatten the grid to create a point cloud
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    plane=np.column_stack((x, y, z))

    return plane

def constant_y(width, height, plane, spacing):
    """
    Creates a np_array of a y-plane. 

    Parameters:
    width (tuple): (a,b), where a is the first point and b is the last in x-direction.
    height (tuple): (a,b), where a is the first point and b is the last in z-direction.
    plane (float): Is the value of y where the plane is to be placed.
    spacing (float): Is the space between the represented points of the plane.

    Returns:
    object (np array): A np_array representation of a plane. 
    """
    # Create a grid of points to represent the floor
    x = np.arange(width[0], width[1], spacing)
    z= np.arange(height[0], height[1], spacing)
    x, z = np.meshgrid(x, z)
    y = np.zeros_like(x)  # Set the Z coordinate to zero for the floor
    if plane != 0:
        y[y == 0] = plane

    
    # Flatten the grid to create a point cloud
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    plane=np.column_stack((x, y, z))

    return plane

def constant_z(width, length, plane, spacing):
    """
    Creates a np_array of a z-plane. 

    Parameters:
    width (tuple): (a,b), where a is the first point and b is the last in x-direction.
    lenght (tuple): (a,b), where a is the first point and b is the last in y-direction.
    plane (float): Is the value of z where the plane is to be placed.
    spacing (float): Is the space between the represented points of the plane.

    Returns:
    object (np array): A np_array representation of a plane. 
    """
    # Create a grid of points to represent the floor
    x = np.arange(width[0], width[1], spacing)
    y = np.arange(length[0], length[1], spacing)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)  # Set the Z coordinate to zero for the floor
    if plane != 0:
        z[z == 0] = plane

    
    # Flatten the grid to create a point cloud
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    plane=np.column_stack((x, y, z))

    return plane

def create_circle(center, radius, num_points):
    """
    Create 3D coordinates for a circle.

    Parameters:
    - center: Tuple (x, y, z) representing the center of the circle.
    - radius: Radius of the circle.
    - num_points: Number of points to sample along the circle.

    Returns:
    - List of tuples representing the 3D coordinates of the circle.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = center[2] * np.ones_like(theta)  # Assuming the circle is in the xy-plane

    circle_coordinates = list(zip(x, y, z))
    return circle_coordinates





########## CLUSTERING ##########

def remove_outliers(points, eps, min_samples):
    """
    Uses DBSCAN to find and remove outliers from a given array of points.

    Parameters:
    points (np array): Numpy array with the point clouds to remove outliers.
    eps (float): The maximum distance between points in the same cluster.
    min_samples (int): The minimum number of points to form a cluster.

    Returns:
    cleaned_points (np array): Numpy array with outliers removed.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(points)
    
    # Identify noise points (outliers)
    noise_indices = (cluster_labels == -1)
    
    # Remove noise points
    cleaned_points = points[~noise_indices]
    
    return cleaned_points

#def perform_clustering_new(points, eps, min_samples):
    """
    Performs clustering on a given array of points after removing outliers.

    Parameters:
    points (np array): Numpy array with the point clouds to perform clustering.
    eps (float): The maximum distance between points in the same cluster.
    min_samples (int): The minimum number of points to form a cluster.

    Returns:
    cluster_labels (np array): Numpy array representing the identified clusters.
    num_clusters (int): The number of clusters found.
    """
    # Remove outliers
    points = remove_outliers(points, eps, min_samples)
    
    # Perform clustering on the cleaned points
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    cluster_labels = clustering.labels_
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    return cluster_labels, num_clusters
 
def perform_clustering(points, eps, min_samples):
    """
    Performs the clustering in a guiven array of points.

    Parameters:
    points (np array): Is the np.array with the point clouds to perform the clustering.
    eps (float): The maximum distance between points in the same cluster.
    min_samples (int): The minimum number os points to form a cluster.

    Returns:
    cluster_labels (np array): Numpy array represensting the identified clusters.
    num_clusters (int): The number of clusters found.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)


    #clustering = OPTICS(max_eps=eps, min_samples=min_samples).fit(points)
    cluster_labels = clustering.labels_
    num_clusters= len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    #clustering = GaussianMixture(n_components=n_components, covariance_type='spherical').fit(points)
    #cluster_labels = clustering.predict(points)
    #num_clusters = n_components

    return cluster_labels, num_clusters

def delete_random_points(point_cloud, delete_percentage):
    """
    Delete a percentage of random points from a point cloud.

    Parameters:
    point_cloud (np array): Input point cloud array.
    delete_percentage (float): Percentage of points to delete randomly.

    Returns:
    modified_point_cloud (np array): Point cloud array with random points deleted.
    """
    num_points = point_cloud.shape[0]
    num_points_to_delete = int(delete_percentage * num_points)

    if num_points_to_delete > 0:
        indices_to_delete = np.random.choice(num_points, num_points_to_delete, replace=False)
        point_cloud[indices_to_delete, :] = np.nan  # You can use any value to indicate deleted points

    # Remove NaN values (deleted points) from the array
    modified_point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1)]

    return modified_point_cloud

def centroid_and_box(points,cluster_labels,num_clusters):
    """
    Uses the clustering previous step to return the bbox.

    Parameters:
    points (np.array): The map where the clustering was performed.
    cluster_labels (np.array): The result of the clustering function with the points agregated.
    num_clusters (int): The number of clusters found in the clustering function.

    Returns:
    Error: NO CLUSTERS FOUND. (If no clusters have been found in the previous clustering step)
    all (list): With the point cloud to visualize (use run_visualizer).
    bbox (dict): Key is the bbox id and for each key we got: center, extent and rotation matrix.
    """
    # Input is points (the point cloud that is being analysed)
    # Input is the cluster_labels and num_clusters resulting from the perform clustering function
    # Output is the list with all the point clouds to add to the visualizer
    all=[]
    bbox={}
    if num_clusters==0:
        return ("NO CLUSTERS FOUND.")
    
    for cluster_id in range(num_clusters):
        if cluster_id == -1:
            # Skip noise points (cluster label -1)
            continue
        #print(cluster_id)
        #print(cluster_labels)
        object_points = points[cluster_labels == cluster_id]
        
        # Calculate the centroid of the object
        centroid = np.mean(object_points, axis=0)
        
        


        # Calculate the bounding box of the object
        min_bound = np.min(object_points, axis=0)
        max_bound = np.max(object_points, axis=0)

        # Create a colored point cloud for the object's centroid
        centroid_color = [0, 0, 1]  # Blue color
        centroid_cloud = o3d.geometry.PointCloud()
        centroid_cloud.points = o3d.utility.Vector3dVector([centroid])
        centroid_cloud.colors = o3d.utility.Vector3dVector([centroid_color])

        # Create a colored bounding box for the object
        object_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(object_points))
        
        object_bbox.color = [0, 1, 0]  # Green color
        center=object_bbox.center
        #print("centro",center)
        extend=object_bbox.extent
        rotation_matrix = object_bbox.R
        np.set_printoptions(precision=2, suppress=True)
        #print("this is the detected rot mat",rotation_matrix)

        # Extract rotation angle from the rotation matrix
        angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)

        #print("Rotation angle (degrees):", round(angle_deg,1))

        # Add the objects to the visualizer
        all.append(centroid_cloud)
        all.append(object_bbox)
        bbox [cluster_id]= (center,extend,rotation_matrix)
    
    return all, bbox

def visualize(point_cloud):
    """
    Allows the visualization of one point_cloud.

    Parameters:
    point_cloud (point_cloud): The final point cloud to visualize. 
    """

    # The Input is a pointcloud structure
    o3d.visualization.draw_geometries([point_cloud])

def run_visualizer(point_cloud_list):
    """
    Allows the visualization of oa list of point_clouds.

    Parameters:
    point_cloud (list): A List containing all the objects to visualize in a point_cloud format.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for point_cloud in point_cloud_list:
        vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window() 

def get_bbox(map):
    """
    Function that retrives the bbox.

    Parameters:
    map (np array): Is the map where we want to retrieve the bbox.

    Returns:
    bbox (dict): The keys are the box id and for each key (center,extent).
    """
    
    Eps=0.2
    Min_samples=10


    Labels, Number=perform_clustering(map,Eps,Min_samples)

    all,bbox = centroid_and_box(map,Labels,Number)

    return bbox

def rotate_cubic_object(obj, degrees):
    """
    Rotates a cubic object around its own center.

    Parameters:
    obj (np array): Input cubic object in array format.
    degrees (float): Number of degrees to rotate.

    Returns:
    rotated_obj (np array): Rotated cubic object in array format.
    """

    # Calculate the center of the object
    center = np.mean(obj, axis=0)

    # Translate the object to the origin
    translated_obj = obj - center

    # Assuming the rotation is around the z-axis
    rotation_matrix = Rotation.from_euler('z', degrees, degrees=True).as_matrix()

    # Apply the rotation matrix to the translated object
    rotated_obj = np.dot(translated_obj, rotation_matrix.T)

    # Translate the object back to its original position
    rotated_obj += center

    return rotated_obj

def object_detection_main(result_path):
    # Input is the path for the point cloud that is meant to be analysed
    # Runs the visualizer and shows the bounding boxes and centroids of the objects
    
    cloud= load_pc(result_path)
    point_cloud = pc_to_array(cloud)

    Eps=0.2
    Min_samples=10


    Labels, Number=perform_clustering(point_cloud,Eps,Min_samples)

    all, bbox= centroid_and_box(point_cloud,Labels,Number)

    all.append(cloud)
    run_visualizer(all)

def create_random_points(num_points, x_range, y_range):
    """
    Adds np.array points to the map scattered in the map.

    Parameters:
    num_points (int): The Number of points we aim to add to the map.
    x_range (float): (x1,x2) The inicial position and final position of the points in the x axis.
    y_range (float): (y1,y2) The inicial position and final position of the points in the y axis.


    Returns:
    random_points (np.array): The np_array containing the coordinates for the scattered points.
    """
    
    random_x = np.random.uniform(x_range[0], x_range[1], num_points)
    random_y = np.random.uniform(y_range[0], y_range[1], num_points)
    random_z = np.zeros(num_points)  # Assuming z-coordinate is 0 for noise points
    random_points = np.column_stack((random_x, random_y, random_z))
    return random_points



########## TRACKING ##########

def create_checkpoints(path, spacing, z_offset=0.0):
    """
    Creates checkpoints along the given path.

    Parameters:
    path (np array): The path represented as a numpy array.
    spacing (int): The spacing between checkpoints.
    z_offset (float, optional): The offset in the z-direction. Defaults to 0.0.

    Returns:
    checkpoints (np array): Numpy array representing checkpoints along the path.
    """
    # Ensure that the path is a numpy array
    path = np.array(path)
    
    # Ensure that the spacing is a positive value
    spacing = abs(spacing)

    checkpoints = []
    
    # Iterate over the path with the specified spacing
    for i in range(0, len(path), int(spacing)):
        checkpoint = path[i].copy()  # Create a copy to avoid modifying the original path
        checkpoint[2] += z_offset    # Apply the z-offset
        checkpoints.append(checkpoint)
    
    # Add the last point of the path as the final checkpoint
    final_checkpoint = path[-1].copy()
    final_checkpoint[2] += z_offset
    checkpoints.append(final_checkpoint)


    return np.array(checkpoints)

def index_of_curves_in_path(path, threshold=1e-10):
    vectors = np.diff(path, axis=0)
    data=np.diff(vectors,axis=0)
    non_zero_indices = []

    for i, vector in enumerate(data):
        # Use numpy to check if the vector is different from the zero vector
        if np.linalg.norm(vector) > threshold:
            non_zero_indices.append(i+2)

    return non_zero_indices

def create_checkpoints_with_variable_spacing(path, spacing_away=4, spacing_near=2, z_offset=0.4, threshold=1):
    """
    Creates checkpoints along the given path with variable spacing near provided indexes.

    Parameters:
    path (np array): The path represented as a numpy array.
    indexes (list): List of indexes where the spacing should be different.
    spacing_away (int, optional): Spacing away from the indexes. Defaults to 1.0.
    spacing_near (int, optional): Spacing near the indexes. Defaults to 0.1.
    z_offset (float, optional): The offset in the z-direction. Defaults to 0.0.
    threshold (float, optional): Threshold to determine what is considered "near" the indexes. Defaults to 1e-10.

    Returns:
    checkpoints (np array): Numpy array representing checkpoints along the path.
    """
    # Ensure that the path is a numpy array
    
    path = np.array(path)
    indexes=index_of_curves_in_path(path)
    checkpoints = []
    
    # Iterate over the path
    for i in range(len(path)):
        # Check if the current index is near any provided indexes
        near_index = any(np.linalg.norm(path[i] - path[idx]) < threshold for idx in indexes)

        # Choose the appropriate spacing based on whether the current index is near or away from provided indexes
        current_spacing = spacing_near if near_index else spacing_away

        # Add checkpoint with the chosen spacing
        if i == 0 or i == len(path) - 1 or i % int(current_spacing) == 0:
            checkpoint = path[i].copy()
            checkpoint[2] += z_offset
            checkpoints.append(checkpoint)

    return np.array(checkpoints)

def is_normalized(vector):
    return math.isclose(np.linalg.norm(vector), 1.0)

def find_closest_checkpoint_new(position, checkpoints):
    """
    Finds the next checkpoint in the given list of checkpoints.

    Parameters:
    position (np.array): The (x, y, z) current position of the robot center.
    checkpoints (np.array): An array with all the checkpoints.

    Returns:
    next_checkpoint (np array): The (x, y, z) position of the next checkpoint.
    """

    closest_point = None
    min_distance = float('inf')

    for point in checkpoints:
        distance = np.linalg.norm(np.array(position[:2]) - np.array(point[:2]))

        if distance < min_distance:
            min_distance = distance
            closest_point = point

    # Find the index of the closest point
    index_closest_point = np.argmin(np.linalg.norm(checkpoints[:, :2] - np.array(position[:2]), axis=1))

    # Ensure that the index is not the last index to avoid index out of range
    if index_closest_point < len(checkpoints) - 1:
        # Return the next checkpoint in the list
        next_checkpoint = checkpoints[index_closest_point + 1]
    else:
        # Return the last checkpoint if the current one is the last in the list
        next_checkpoint = checkpoints[index_closest_point]

    return next_checkpoint

def find_closest_checkpoint(position, checkpoints):
    """
    Finds the closest checkpoint to the current position.

    Parameters:
    position (np.array): The (x,y,z) current position of the robot center.
    checkpoints (np.array): An array with all the checkpoints.

    Returns:
    path[index_closest_point] (np array): The (x,y,z) position of the next checkpoint.
    """

    closest_point = None
    min_distance = float('inf')

    for point in checkpoints:
        distance = np.linalg.norm(np.array(position[:2]) - np.array(point[:2]))

        if distance < min_distance:
            min_distance = distance
            closest_point = point

    # Find the index of the closest point
    index_closest_point = np.argmin(np.linalg.norm(checkpoints[:, :2] - np.array(position[:2]), axis=1))

    # Ensure that the index is not the last index to avoid index out of range
    if index_closest_point < len(checkpoints) - 1:
        # Check if the next point is on the direction to the end of the path
        vector_to_point = np.array(closest_point[:2]) - np.array(position[:2])
        vector_to_end = np.array(checkpoints[-1][:2]) - np.array(position[:2])

        dot_product = np.dot(vector_to_point, vector_to_end)

        if dot_product > 0:
            return checkpoints[index_closest_point]

        # If not, return the next point on the path
        return checkpoints[index_closest_point + 1]

    return checkpoints[index_closest_point]

def find_smaller_angle(vetor_x, vetor_target):
    """
    Finds if the vector x is the vector with smaller angle with the vetor target or if is the oposite vector to vector x.

    Parameters:
    vetor_x (list): Is the x axis from the robot that we are acessing the change.
    vetor_target (list): Is the vector that points to the next checkpoint.

    Returns:
    (bool): True if the x vector and the target vector have the smaller angle and 
            False if the oposite vector and the target vector have the smaller angle.
    """
    #print("vetor target",vetor_target)
    #print("vetor_x",vetor_x)
    
    oposite_vetor=[-vetor_x[0],- vetor_x[1],vetor_x[2]]
    #print("oposite_vetor",oposite_vetor)
    teta=math.acos(np.dot(vetor_target,vetor_x))
    #print("teta vetor x",teta)
    teta_oposite=math.acos(np.dot(vetor_target,oposite_vetor))
    #print("teta vetor oposite",teta_oposite)
    if teta<=teta_oposite:
        #print("entrou no loop do true")
        return True
    return False

def create_vector(checkpoint, current_position):
    """
    Given 2 positions it returns the normalized vector between them two.

    Parameters:
    checkpoint(np.array): Is the point of detination of the vector.
    current_position (np.array): Is the origin of the vector.

    Returns:
    vector (np.array): Normalized vector with origin in current_position and destination in checkpoint.
    """
    vector=np.array(checkpoint) - np.array(current_position)
    magnitude = np.linalg.norm(vector)
    if magnitude != 0:
        normalized_vector = vector / magnitude
        return normalized_vector
    else:
        return vector 

def normalize_angle(angle):
    # Use modulo to bring the angle within the range [0, 360)
    normalized_angle = angle % 360.0
    
    # Ensure the result is positive
    if normalized_angle < 0:
        normalized_angle += 360.0
    
    return normalized_angle

def change_mat_bbox(rot_mat, target_vector):
    #print("this is the mat before", rot_mat)
    vector_x=np.dot(rot_mat,[1,0,0])
    #print("vector_x",vector_x)
    #print(find_smaller_angle(vector_x,target_vector))
    #print("before",rot_mat)

    #print("is true or false:", find_smaller_angle(vector_x,target_vector))
    if find_smaller_angle(vector_x,target_vector)==False:
        #print("entra no loop do false?")
        teta= math.pi
        change_mat=np.array([[np.cos(teta), -np.sin(teta), 0],
                [-np.sin(teta), np.cos(teta), 0],
                [0, 0, 1]])
        new_rotation_matrix = np.dot(change_mat,rot_mat)
        #print("after",new_rotation_matrix)
        return new_rotation_matrix
    else:
        return rot_mat

def matrix_to_angle(matrix):
    """
    Transforms a rotation matrix in an angle in degrees.

    Parameters:
    matrix (np.array): The rotation matrix meant to be tranformed.

    Returns:
    angle_deg (float): Returns the angle in degrees that corresponds to the matrix.
    """
    angle_rad = np.arctan2(matrix[1, 0], matrix[0, 0])
    angle_deg = np.degrees(angle_rad)
    angle_deg=round(angle_deg,2)
    return angle_deg

def red_dot(red_point):
    """
    Given the coordinates of the red_point. Draws a red dot.

    Parameters:
    red_point (np.array): The (x,y,z) coordinates where to draw the red dot.

    Returns:
    red_point_cloud (point_cloud): Returns the point_cloud format of the red dot.
    """
    red_point_cloud = o3d.geometry.PointCloud()
    red_point_cloud.points = o3d.utility.Vector3dVector(np.array([red_point]))

    # Color the red point cloud in red
    red_color = [1, 0, 0]
    red_point_cloud.paint_uniform_color(red_color)
    # Scale the point cloud to the specified size
    #red_point_cloud.scale(1, center=red_point)
    # Scale the point cloud to the specified size
    return red_point_cloud

def update_path(current_position, next_checkpoint, original_path, spacing):
    """
    Updates the path based on the current position, next checkpoint, and original path.

    Parameters:
    current_position (tuple): Current position (x, y, z) of the robot.
    next_checkpoint (tuple): Next checkpoint (x, y, z) on the original path.
    original_path (np array): Original path as a numpy array.
    spacing (float): Spacing between points in the path.

    Returns:
    updated_path (np array): Updated path after the current position and next checkpoint.
    """
    # Create a straight line between the current position and the next checkpoint
    straight_line = create_straight_line(current_position[:2], next_checkpoint[:2], spacing)

    # Find the index of the next checkpoint in the original path
    checkpoint_index = np.where(np.all(original_path[:, :2] == np.array(next_checkpoint)[:2], axis=1))[0][0]

    # Slice the original path from the current checkpoint to the end
    remaining_path = original_path[checkpoint_index:]

    # Combine the straight line and the remaining path
    updated_path = np.vstack((straight_line, remaining_path))

    return updated_path

class EuclideanDistTracker3D:

    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_dict):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for index, (center, dimensions) in objects_dict.items():
            cx, cy, cz = center
            # For simplicity, use the average of dimensions for radius
            radius = sum(dimensions) / 3.0

            # Find out if that object was detected already
            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                #dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2 + (cz - pt[2])**2)
                dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2)
                #print(dist)
                if dist < 0.2:  # Adjust the threshold as needed
                    self.center_points[obj_id] = (cx, cy, cz)
                    objects_bbs_ids.append([cx, cy, cz, radius, obj_id])
                    same_object_detected = True
                    break

            # New object is detected; assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy, cz)
                objects_bbs_ids.append([cx, cy, cz, radius, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    
class EuclideanDistTracker3D_new:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_dict,distancia):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for index, (center, dimensions,rot_mat) in objects_dict.items():
            cx, cy, cz = center
            dx, dy, dz = dimensions


            cx=round(cx,1)
            cy=round(cy,1)
            cz=round(cz,1)

            dx=round(dx,1)
            dy=round(dy,1)
            dz=round(dz,1)
        

            # Find out if that object was detected already
            same_object_detected = False
            for obj_id, (pt,dim,rot_mat,trajectory) in self.center_points.items():
                #dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2 + (cz - pt[2])**2)
                dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2)
                #print(dist)
                if dist < distancia:  # Adjust the threshold as needed
                    self.center_points[obj_id] = ((cx, cy, cz), (dx, dy, dz), rot_mat, trajectory + [(cx, cy, cz)])
                    objects_bbs_ids.append([cx, cy, cz,  dx, dy, dz, rot_mat, obj_id])
                    same_object_detected = True
                    break

            # New object is detected; assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = ((cx, cy, cz), (dx, dy, dz),rot_mat,[(cx, cy, cz)])
                objects_bbs_ids.append([cx, cy, cz, dx, dy, dz, rot_mat, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    
class EuclideanDistTracker3D_new_new:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_dict,distancia):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for index, (center, dimensions,rot_mat) in objects_dict.items():
            cx, cy, cz = center
            dx, dy, dz = dimensions


            cx=round(cx,1)
            cy=round(cy,1)
            cz=round(cz,1)

            dx=round(dx,1)
            dy=round(dy,1)
            dz=round(dz,1)
        

            # Find out if that object was detected already
            same_object_detected = False
            for obj_id, (pt,dim,trajectory) in self.center_points.items():
                #dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2 + (cz - pt[2])**2)
                dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2)
                #print(dist)
                if dist < distancia:  # Adjust the threshold as needed
                    self.center_points[obj_id] = ((cx, cy, cz), (dx, dy, dz), trajectory + [(cx, cy, cz)])
                    objects_bbs_ids.append([cx, cy, cz,  dx, dy, dz,  obj_id])
                    same_object_detected = True
                    break

            # New object is detected; assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = ((cx, cy, cz), (dx, dy, dz),[(cx, cy, cz)])
                objects_bbs_ids.append([cx, cy, cz, dx, dy, dz, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    
def is_box_in_path(bbox, path):
    """
    Checks if box is in path.

    Parameters:
    bbox (o3d.geometry.OrientedBoundingBox()): Representation of an object in the space.
    path (np array): Representation of the path in a np array format

    Returns:
    bool: True if the any of the path (x,y) cordinates are the same as the box, False otherwise.
    """
    center = bbox.center
    extent = bbox.extent

    # Calculate the bounding box coordinates
    bbox_x_min = center[0] - extent[0]
    bbox_x_max = center[0] + extent[0]
    bbox_y_min = center[1] - extent[1]
    bbox_y_max = center[1] + extent[1]

    # Check if any coordinate of the bounding box is inside the path
    for point in path:
        x, y, _ = point
        if (bbox_x_min <= x <= bbox_x_max and bbox_y_min <= y <= bbox_y_max):
            
            return True

    return False

def is_center_in_path(center, path):
    """
    Checks if the center of the box coordinates are in path.

    Parameters:
    center (tuple): (x,y) are the center cordinates of the box.
    path (np array): Representation of the path in a np array format

    Returns:
    bool: True if the any of the path (x,y) cordinates are the same as the box, False otherwise.
    """
    x_box,y_box=center

    # Check if any coordinate of the bounding box is inside the path
    for point in path:
        x, y, _ = point
        if (x==x_box and y==y_box):
            return True

    return False










########## DRIVING ##########

def velocity_value(teta_e,max_value):
    """
    Gives a value of the velocity of the robot based on the ajustment angle.
    Parameters:
    teta_e (float): The ajustment angle value in degrees.
    max_value (float): The maximum velocity, used when the robot is moving forward.

    Returns:
    velocity (float): The value of velocity used in that timestep, to calculate the distance to move. 
    """
    teta_e=math.radians(teta_e)
    velocity=max_value*math.cos(teta_e)
    return velocity

def ajust_angle(beta, teta_e, delta_t):
    """
    Gives the ajustment angle that the robot need to rotate in a time step.
    Parameters:
    beta (float): Is the constant of porporcionality for the angular velocity to be calculated.
    teta_e (float): The ajustment angle, required to achive the checkpoint in degrees.
    delta_t (float): The time corresponding to a time_step

    Returns:
    teta_ajust (float): Is the angle in degrees that the robot needs to rotate in the next iteration.
    """
    ang_vel=beta*teta_e
    teta_ajust=ang_vel*delta_t
    return(teta_ajust)

def get_teta_e(trajectory,center,rot_mat,checkpoints):
    """
    Given, the checkpoints and the rot mat from the bbox, returns the teta_e.
    Parameters:
    trajectory (list): List that stores the last positions of the object.
    center(np.array): The center of the bbox (x,y,z).
    rot_mat (np.array): Rotation matrix associated with the bbox detected. 
    checkpoints (np.array): Equally spaced checkpoints of the path.

    Returns:
    teta_e (np.array): The angle (in degrees) that the robot needs to rotate to face the next checkpoint (t).
    updated_rot_mat(np.array): The updated matrix associated with the bbox detected.
    """
    #Gives the direction the car was moving, in the last iteration.
    vector_front=create_vector(trajectory[-1],trajectory[-2])
    
    #Finds the closest checkpoint to the center of the bbox detected.
    next_checkpoint=find_closest_checkpoint_new(center,checkpoints)
    print("next_checkpoint",next_checkpoint)
    
    #The rot_mat of the bbox may need to be updated based on the vector_front.
    updated_rot_mat=change_mat_bbox(rot_mat,vector_front)
    
    #The angle between the x_axis of the bbox and the x_inertial. 
    teta_p=matrix_to_angle(updated_rot_mat)
    print("teta_p",teta_p)
    #The angle betwen the center_to_next_checkpoint and the x_inertial.
    teta_t=points_to_angle(center,next_checkpoint)
    print("teta_t",teta_t)
    
    #The rotation that the robot needs to perform to be facing the next_checkpoint.
    teta_e=teta_t-teta_p
    return teta_e, updated_rot_mat

def next_point(teta_e,delta_t,v_max,beta,rot_mat,center):
  """
  Given the ajustment angle and the velocity that the car moves returns the (x,y,z) of the exact position in the next time step. 
  Parameters:
  teta_e (np.array): The angle that the robot needs to rotate to face the next checkpoint (t).
  delta_t (float): The time step.
  v_max (float): The maximum value that the velocity can have.
  beta(float): Constant used to calculate the ajustment angle for each iteration.
  rot_mat(np.array): The rotation matrix associated with the bbox.
  center (np.array): The center of the detected bbox detected.


  Returns:
  next_pos(np.array): (x,y,z) position of the robot center in the next iteration.
  teta_ajust(float): The angle (in degrees) that the robot is rotating in the next iteration.
  """
  #Calculate the angles that the robot is rotating in the next iteration.
  teta_ajust=ajust_angle(beta,teta_e,delta_t)
  vector_p=np.dot(rot_mat, np.array([1,0,0])) [:2]
  adj= np.radians(teta_ajust)
  adjustment_matrix = np.array([[np.cos(adj), -np.sin(adj)],
                           [np.sin(adj), np.cos(adj)]])
  #Apply the value of this rotation to the x_axis of the boinding box to understand the moving direction.
  pointing_direction = np.dot(adjustment_matrix, vector_p)
  #Calculation of the velocity value based on the teta_e value.
  v=velocity_value(teta_e,v_max)
  #Calculates the next position based on the center, the velocity and the pointing direction.
  next_pos=dist_to_pos(center,v*delta_t,pointing_direction)
  return next_pos,teta_ajust

def points_to_angle(point_p,point_t):
    """
    Calculates de angle between the x axis and the p_t segment.

    Parameters:
    point_p (np.array): The center of the robot.
    point_t (np.array): The checkpoint where the robot is going.

    Returns:
    angle_deg (float): Returns the angle in degrees that corresponds to x axis and the p_t segment angle.
    """

    x_t=point_t[0]
    y_t=point_t[1]
    x_p=point_p[0]
    y_p=point_p[1]
    angle_radians = math.atan2((y_t - y_p),(x_t - x_p))
    angle_degrees = math.degrees(angle_radians)
    angle_deg=round(angle_degrees,2)
    return angle_deg

def dist_to_pos(current_position, distance, pointing_direction):
    """
    Given the distance of the next position, it returns the coordinates of the next position.

    Parameters:
    current_position (np.array): Coordinates (x, y, z) of the current position where the robot is placed.
    distance (float): Distance that the next position is from the current_position.
    pointing_direction (np.array): Normalized direction vector (x,y) in which the robot is pointing.

    Returns:
    next_position (np.array): Coordinates (x, y, z) of the next position where to place the robot.
    """
    # Calculate the next position based on the pointing direction and distance
    next_position = current_position[:2] + distance * pointing_direction
    next_position=np.array([next_position[0],next_position[1],current_position[2]])
    
    
    return next_position



if __name__ == "__main__":
    #subtract_main("bg.pcd","object_2.pcd")
    object_detection_main("result.pcd")
