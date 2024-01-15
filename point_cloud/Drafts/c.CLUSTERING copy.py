import MAIN as m
import open3d as o3d
import numpy as np
import random
import numpy as np
import numpy.linalg as LA
# Load background point cloud
bg = m.load_pc("bg.pcd")
bg = m.pc_to_array(bg)

num_noise_points = 0
num_cubes=1
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
        #size_x, size_y = generate_random_cube_size(cube_size_range)
        size_x=0.4
        size_y=0.3
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


combined_cloud = m.merge_arrays(noise_points,resulting_cloud)


cluster_labels, num_clusters = m.perform_clustering(combined_cloud, eps=0.1, min_samples=2)
print (cluster_labels)
cube=m.create_cubic_object((1,2),0.7,0.5,0.1)
def centroid_and_box_new(points,cluster_labels,num_clusters):
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
        #print("object points",object_points)
       
        center = np.mean(object_points, axis=0)
        means= np.mean(object_points, axis=1)

        # Center the data by subtracting means
        #centered_points = object_points - means[:, np.newaxis]
        #print("centered points",centered_points)
        # Calculate the covariance matrix
        cov = np.cov(object_points,rowvar=False)
        print("cov mat",cov)
        eval, evec = LA.eig(cov)
        print(eval, evec)
        # Sort eigenvalues and eigenvectors in descending order
        order = np.argsort(eval)[::-1]
        eval = eval[order]
        evec = evec[:, order]
        
        # Calculate the extent (half of the side lengths along each principal component)
        extent = np.sqrt(eval)

        # Calculate the rotation matrix
        rotation_matrix = evec

        # Calculate the extent (half of the side lengths along each principal component)
        #extent = np.sqrt(eval)
        
    return center, extent, rotation_matrix




cube=m.create_cubic_object((1,2),0.7,0.5,0.1)

# Call centroid_and_box function to get visualizations of centroids and bounding boxes
all_visualizations, bbox_info = m.centroid_and_box(combined_cloud, cluster_labels, num_clusters)
print("original",bbox_info)




#center, extent, rotation_matrix = centroid_and_box_new(combined_cloud, cluster_labels, num_clusters)

center = np.mean(cube, axis=0)
means= np.mean(cube, axis=1)

# Center the data by subtracting means
centered_points = cube - means[:, np.newaxis]

# Calculate the covariance matrix
cov = np.cov(centered_points,rowvar=False)

eval, evec = LA.eig(cov)

# Sort eigenvalues and eigenvectors in descending order
order = np.argsort(eval)[::-1]
eval = eval[order]
evec = evec[:, order]

# Calculate the extent (half of the side lengths along each principal component)
extent = 2* np.sqrt(eval.real)

# Calculate the rotation matrix
rotation_matrix = evec
















combined_cloud_point_cloud=m.array_to_pc(combined_cloud)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([combined_cloud_point_cloud] + all_visualizations)

# Print information about the bounding boxes
for cluster_id, info in bbox_info.items():
    center, extent, rotation_matrix = info
    print(f"Cluster {cluster_id} - Center: {center}, Extent: {extent}")