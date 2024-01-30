import MAIN as m
import numpy as np
import open3d as o3d

#cube=m.create_cubic_object((1,2),0.5,0.3,0.1)
cube=m.create_cubic_object((1,2),0.5,1,0.1)

def project_and_remove_duplicates(points):
    # Project all points onto the z-axis
    projected_points = points[:, :2]

    # Find unique points in the projected 2D points
    unique_projected_points, indices = np.unique(projected_points, axis=0, return_index=True)

    return unique_projected_points

def distance_along_vector(points, vector):
    # Normalize the vector
    normalized_vector = vector / np.linalg.norm(vector)

    # Project the points onto the normalized vector
    projections = np.dot(points, normalized_vector)

    # Find the two points with the minimum and maximum projections
    min_index = np.argmin(projections)
    max_index = np.argmax(projections)

    # Calculate the distance between the two furthest points
    distance = np.abs(projections[max_index] - projections[min_index])

    return distance

def distance_along_vector(points, vector):
    # Normalize the vector
    normalized_vector = vector / np.linalg.norm(vector)

    # Project the points onto the normalized vector
    projections = np.dot(points, normalized_vector)

    # Find the two points with the minimum and maximum projections
    min_index = np.argmin(projections)
    max_index = np.argmax(projections)

    # Calculate the distance between the two furthest points
    distance = np.abs(projections[max_index] - projections[min_index])

    return distance

def min_max_z_coordinates(data_array):
    z_coordinates = data_array[:, 2]
    min_z = np.min(z_coordinates)
    max_z = np.max(z_coordinates)
    distance=max_z-min_z
    middle_point = (min_z + max_z) / 2.0
    return distance,middle_point


def find_rectangle_coordinates(points):
    # Use the provided function to get unique 2D points
    unique_projected_points = project_and_remove_duplicates(points)

    # Find minimum and maximum x, y coordinates from the unique projected points
    x_min = np.min(unique_projected_points[:, 0])
    x_max = np.max(unique_projected_points[:, 0])
    y_min = np.min(unique_projected_points[:, 1])
    y_max = np.max(unique_projected_points[:, 1])

    # Construct the rectangle coordinates
    rectangle_coordinates = [(x_min, y_min, 0), (x_max, y_max, 0)]

    return rectangle_coordinates

def find_rectangle_center(rectangle_coordinates):
    # Extract x, y coordinates of the two corners
    x1, y1, _ = rectangle_coordinates[0]
    x2, y2, _ = rectangle_coordinates[1]

    # Calculate the center coordinates
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_z = 0  # Since the rectangle is in the z=0 plane

    # Return the center coordinates
    return (center_x, center_y, center_z)

def find_rectangle_directions(rectangle_coordinates):
    # Extract x, y coordinates of the two corners
    x1, y1, _ = rectangle_coordinates[0]
    x2, y2, _ = rectangle_coordinates[1]

    # Calculate the lengths of the sides of the rectangle
    length_x = abs(x2 - x1)
    length_y = abs(y2 - y1)

    # Determine the longer and shorter sides
    if length_x >= length_y:
        # X direction is longer or equal
        x_direction = np.array([(x2 - x1) / length_x, (y2 - y1) / length_x, 0])
        y_direction = np.array([-(y2 - y1) / length_x, (x2 - x1) / length_x, 0])
    else:
        # Y direction is longer
        x_direction = np.array([(y2 - y1) / length_y, -(x2 - x1) / length_y, 0])
        y_direction = np.array([(x2 - x1) / length_y, (y2 - y1) / length_y, 0])

    return x_direction, y_direction

def bbox_implementation(cube):

    rectangle=project_and_remove_duplicates(cube)
    #este é o centro em 2D
    mean_vector = np.mean(rectangle, axis=0)
    centered_points = rectangle - mean_vector
    centered_points_T = centered_points.T
    #ver a matrix de covariância
    covariance_matrix = np.dot(centered_points_T, centered_points) / (len(rectangle) - 1)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    vector_3d = np.append(eigenvectors[0], 0.0)
    projection_x = distance_along_vector(rectangle, eigenvectors[0])
    projection_y = distance_along_vector(rectangle, eigenvectors[1])
    z_lenght,middle_point= min_max_z_coordinates(cube)

    #matriz de rotação
    eigenvector = eigenvectors[0]
    inertial_reference = np.array([[1, 0]])
    dot_product = np.dot(eigenvector,  inertial_reference.T)
    magnitude_eigenvector = np.linalg.norm(eigenvector)
    magnitude_inertial_reference = np.linalg.norm(inertial_reference)
    cos_theta = dot_product / (magnitude_eigenvector * magnitude_inertial_reference)
    teta = np.arccos(cos_theta)[0]
    #teta = np.degrees(teta)
    rot_mat=np.array([[np.cos(teta), -np.sin(teta), 0],
                     [-np.sin(teta), np.cos(teta), 0],
                     [0, 0, 1]])
    
    center=(mean_vector[0],mean_vector[1],middle_point)
    extent=[projection_x,projection_y,z_lenght]

    return center,extent,rot_mat

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
        #print(cluster_id)
        #print(cluster_labels)
        object_points = points[cluster_labels == cluster_id]
        centroid,extent,rot_mat=bbox_implementation(object_points)
        print("centroid",centroid)
        print("extent",extent)
        print("rot_mat",rot_mat)
       
    

        # Create a colored point cloud for the object's centroid
        centroid_color = [0, 0, 1]  # Blue color
        centroid_cloud = o3d.geometry.PointCloud()
        centroid_cloud.points = o3d.utility.Vector3dVector([centroid])
        centroid_cloud.colors = o3d.utility.Vector3dVector([centroid_color])

        # Create a colored bounding box for the object
        object_bbox = o3d.geometry.OrientedBoundingBox(centroid, rot_mat, extent)
        
        object_bbox.color = [0, 1, 0]  # Green color
        

        #print("Rotation angle (degrees):", round(angle_deg,1))

        # Add the objects to the visualizer
        all.append(object_bbox)
        all.append(centroid_cloud)
        #all.append(new_bbox)
        bbox [cluster_id]= (centroid,extent,rot_mat)
    
    return all, bbox


import MAIN as m
import open3d as o3d
import numpy as np
import random

# Load background point cloud
bg = m.load_pc("bg.pcd")
bg = m.pc_to_array(bg)

num_noise_points = 100
num_cubes=3
x_range=(-5,5)
y_range=(-5,5)
spacing = 0.08
cube_size_range=(0.3,0.6)

# Example usage of the function
width_cub = 0.4  # Width range
lenght_cub = 0.3  # Adjust this value based on your requirements

#creates noise points in the range defined
noise_points = m.create_random_points(num_noise_points, x_range, y_range)

def generate_random_coordinates(x_range, y_range):
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_coordinate = random.uniform(x_min, x_max)
    y_coordinate = random.uniform(y_min, y_max)

    return x_coordinate, y_coordinate

def generate_random_cube_size(cube_size_range, min_difference=0.1):
    size_x_min, size_x_max = cube_size_range
    
    # Ensure size_x and size_y are different by at least min_difference
    while True:
        size_x = random.uniform(size_x_min, size_x_max)
        size_y = random.uniform(size_x_min, size_x_max)
        if abs(size_x - size_y) >= min_difference:
            break

    return size_x, size_y

def draw3DRectangle(ax, x1, y1, z1, x2, y2, z2):
    # the Translate the datatwo sets of coordinates form the apposite diagonal points of a cuboid
    ax.plot([x1, x2], [y1, y1], [z1, z1], color='b') # | (up)
    ax.plot([x2, x2], [y1, y2], [z1, z1], color='b') # -->
    ax.plot([x2, x1], [y2, y2], [z1, z1], color='b') # | (down)
    ax.plot([x1, x1], [y2, y1], [z1, z1], color='b') # <--

    ax.plot([x1, x2], [y1, y1], [z2, z2], color='b') # | (up)
    ax.plot([x2, x2], [y1, y2], [z2, z2], color='b') # -->
    ax.plot([x2, x1], [y2, y2], [z2, z2], color='b') # | (down)
    ax.plot([x1, x1], [y2, y1], [z2, z2], color='b') # <--
    
    ax.plot([x1, x1], [y1, y1], [z1, z2], color='b') # | (up)
    ax.plot([x2, x2], [y2, y2], [z1, z2], color='b') # -->
    ax.plot([x1, x1], [y2, y2], [z1, z2], color='b') # | (down)
    ax.plot([x2, x2], [y1, y1], [z1, z2], color='b') # <--

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
    height_range = np.arange(0, min(size_x,size_y), spacing)

    #print("width and lenght",width_range,length_range)
    # Create a grid of points to represent the cubic object
    x, y, z = np.meshgrid(width_range, length_range, height_range, indexing='ij')

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    object=np.column_stack((x, y, z))

    return object

def generate_random_cubes(num_cubes, x_range, y_range, cube_size_range):
    accumulated_cloud = None

    for _ in range(num_cubes):
        # Generate random coordinates and sizes
        center = generate_random_coordinates(x_range, y_range)
        size_x, size_y = generate_random_cube_size(cube_size_range)
        #size_x=0.4
        #size_y=0.3
        # Create a cubic object
        cube = create_cubic_object(center, size_x, size_y, spacing)

        # Merge the new cube with the accumulated cloud
        if accumulated_cloud is None:
            accumulated_cloud = cube
        else:
            accumulated_cloud = m.merge_arrays(accumulated_cloud, cube)

    return accumulated_cloud


resulting_cloud = generate_random_cubes(num_cubes, x_range, y_range, cube_size_range)
# Combine the background, global path, noise points, cubic objects, and spherical object into a single point cloud
resulting_cloud = m.delete_random_points(resulting_cloud, delete_percentage=0.4)


combined_cloud = m.merge_arrays(noise_points,resulting_cloud)


cluster_labels, num_clusters = m.perform_clustering(combined_cloud, eps=0.1, min_samples=2)
print (cluster_labels)




# Call centroid_and_box function to get visualizations of centroids and bounding boxes
all_visualizations, bbox_info = centroid_and_box(combined_cloud, cluster_labels, num_clusters)
print(bbox_info)






combined_cloud_point_cloud=m.array_to_pc(combined_cloud)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([combined_cloud_point_cloud] + all_visualizations)

# Print information about the bounding boxes
for cluster_id, info in bbox_info.items():
    center, extent, rotation_matrix = info
    print(f"Cluster {cluster_id} - Center: {center}, Extent: {extent}")