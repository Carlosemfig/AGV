import MAIN as m
import open3d as o3d
import numpy as np
import random

# Load background point cloud
bg = m.load_pc("bg.pcd")
bg = m.pc_to_array(bg)

num_noise_points = 0
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
resulting_cloud = m.delete_random_points(resulting_cloud, delete_percentage=0.2)

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

# Create an approximate spherical object
spherical_radius = 0.3
spherical_center = (1.0, 2.0, 0.0)
num_points_theta = 20
num_points_phi = 10

spherical_object = create_approximate_spherical_object(
    spherical_center, spherical_radius, spacing, num_points_theta, num_points_phi
)

combined_cloud = m.merge_arrays(noise_points, spherical_object, resulting_cloud)

cluster_labels, num_clusters = m.perform_clustering(combined_cloud, eps=0.1, min_samples=2)
print (cluster_labels)

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
        # Vetor desejado para a última coluna
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        # Substituir a última coluna
        rotation_matrix=np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [-np.sin(yaw), np.cos(yaw), 0],
                     [0, 0, 1]])
        
        new_bbox = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extend)

        #new_bbox.rotate(rotation_matrix)
        np.set_printoptions(precision=2, suppress=True)
        #print("this is the detected rot mat",rotation_matrix)

        # Extract rotation angle from the rotation matrix
        angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)

        #print("Rotation angle (degrees):", round(angle_deg,1))

        # Add the objects to the visualizer
        all.append(object_bbox)
        all.append(centroid_cloud)
        #all.append(new_bbox)
        bbox [cluster_id]= (center,extend,rotation_matrix)
    
    return all, bbox


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