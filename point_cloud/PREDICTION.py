import MAIN as m
import numpy as np
import open3d as o3d
import MAIN as m
import open3d as o3d
import numpy as np
import random


#_____________________________BBOX IMPLEMENTATION____________________#

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
    rectangle = project_and_remove_duplicates(cube)
    # Check if there are enough points to compute covariance matrix
    if len(rectangle) <= 1:
        print("Error: Insufficient points to compute covariance matrix.")

    mean_vector = np.mean(rectangle, axis=0)
    centered_points = rectangle - mean_vector
    centered_points_T = centered_points.T

    # Covariance matrix
    covariance_matrix = np.dot(centered_points_T, centered_points) / (
        len(rectangle) - 1
    )

    # Principal components (eigenvectors) and values
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Order eigenvectors by eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # x-direction vector
    x_direction = eigenvectors[:, 0]

    # Ensure y is perpendicular to x
    z_direction = np.array([0, 0, 1])  # assuming z as the up direction
    y_direction = np.cross(z_direction, x_direction)[:2]

    projection_x = distance_along_vector(rectangle, x_direction)
    projection_y = distance_along_vector(rectangle, y_direction)

    # Calculate extent along z
    z_length, middle_point = min_max_z_coordinates(cube)

    # Rotation matrix
    x_direction = (x_direction[0], x_direction[1], 0)
    y_direction = (y_direction[0], y_direction[1], 0)

    rot_mat = np.column_stack((x_direction, y_direction, z_direction))

    center = np.asarray([mean_vector[0], mean_vector[1], middle_point])
    extent = np.asarray([projection_x, projection_y, z_length])

    return center, extent, rot_mat

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
        #print("centroid",centroid)
        #print("extent",extent)
        #print("rot_mat",rot_mat)
       
    

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










#_______________________________DATA FOR VISUALIZATION______________________#

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

cluster_labels, num_clusters = m.perform_clustering(combined_cloud, eps=0.3, min_samples=10)
#print (cluster_labels)


# Call centroid_and_box function to get visualizations of centroids and bounding boxes
all_visualizations, bbox_info = centroid_and_box(combined_cloud, cluster_labels, num_clusters)
#print("BBOX_INFO_1",bbox_info)









#____________________________________PREDICTION CODE_______________________________________#

print("bbox_info",bbox_info)

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
    next_position=(next_position[0],next_position[1],current_position[2])
    
    
    return next_position


def predicted_bbox(bbox_info,velocity=2,angle=5,delta_t=0.3,box_color=[1,0,0]):
    all=[]
    bbox_updated={}
    for key, bbox in bbox_info.items():
        #print("key",key)
            
        center,extent,rotation_matrix=bbox
        print("PREVIOUS ROTATION MATRIX",rotation_matrix)
        # Extracting the angle
        prev_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Convert the angle to degrees
        prev_angle_degrees = np.degrees(prev_angle)
        print("PREVIOUS ANGLES IN DEGREES",prev_angle_degrees)
        new_angle=angle+prev_angle_degrees
        new_angle= np.radians(new_angle)
        x_direction=[np.cos(new_angle),-np.sin(new_angle),0]
        z_direction = np.array([0, 0, 1]) 
        y_direction = np.cross(z_direction, x_direction)[:2]

        # Rotation matrix
        x_direction = (x_direction[0], x_direction[1], 0)
        y_direction = (y_direction[0], y_direction[1], 0)

        next_rot_mat = np.column_stack((x_direction, y_direction, z_direction))

        """#matrix with the adicional angle
        next_rot_mat=np.array([[np.cos(new_angle), -np.sin(new_angle), 0],
                    [-np.sin(new_angle), np.cos(new_angle), 0],
                    [0, 0, 1]])
        print("NEXT ROTATION MATRIX",next_rot_mat)
"""
        next_angle = np.arctan2(next_rot_mat[1, 0], next_rot_mat[0, 0])

        # Convert the angle to degrees
        next_angle = np.degrees(next_angle)
        print("NEXT ANGLE IN DEGREES",next_angle)
        #new rotation matrix, resulting from adding the angle.

        vector_p=np.dot(next_rot_mat, np.array([1,0,0])) [:2]
        next_center=dist_to_pos(center,velocity*delta_t,vector_p)

        #print("center",next_center)
        #print("extent",extent)
        #print("rot_mat",next_rot_mat)
        
        

        # Create a colored point cloud for the object's centroid
        centroid_color = box_color  # Not Blue color
        centroid_cloud = o3d.geometry.PointCloud()
        centroid_cloud.points = o3d.utility.Vector3dVector([next_center])
        centroid_cloud.colors = o3d.utility.Vector3dVector([centroid_color])

        # Create a colored bounding box for the object
        object_bbox = o3d.geometry.OrientedBoundingBox(next_center, next_rot_mat, extent)
        object_bbox.color = box_color  # Not Green color
        

        #print("Rotation angle (degrees):", round(angle_deg,1))

        # Add the objects to the visualizer
        all.append(object_bbox)
        all.append(centroid_cloud)

        #all.append(new_bbox)
        bbox_updated[key]= (next_center,extent,next_rot_mat)
    
    return all,bbox_updated

all_visualizations_2,bbox_info_2=predicted_bbox(bbox_info)
print("BBOX_INFO_2",bbox_info_2)
print("Original bbox",bbox_info[0])

all_visualizations_3,bbox_info_3=predicted_bbox(bbox_info,delta_t=0.6,box_color=[0.7,0.3,0])
#print("BBOX_INFO_2",bbox_info_2)

combined_cloud_point_cloud=m.array_to_pc(combined_cloud)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([combined_cloud_point_cloud] + all_visualizations+all_visualizations_2+all_visualizations_3)

# Print information about the bounding boxes
for cluster_id, info in bbox_info.items():
    center, extent, rotation_matrix = info
    #print(f"Cluster {cluster_id} - Center: {center}, Extent: {extent}")



#___________________BBOXES AVERAGE___________________#
# This should receive 2 sets of boundig boxes:
#   -The detected
#   -The predicted
#In both sets the corresponding bboxes should be identified with the same id in the dictionary.
#The 2 bboxes should then be feded to a function that calculates the average for both the bounding boxes
#Resulting in a new bounding box.

bbox_detected=bbox_info[0]
bbox_predicted=bbox_info_2[0]


def average_bbox(bbox_detected,bbox_predicted):
    all=[]

    d_center,d_extent,d_rot_mat=bbox_detected
    p_center,p_extent,p_rot_mat=bbox_predicted
    
    x_center=(d_center[0]+p_center[0])/2
    y_center=(d_center[1]+p_center[1])/2
    z_center=(d_center[2]+p_center[2])/2
    new_center=(x_center,y_center,z_center)

    x_extent=(d_extent[0]+p_extent[0])/2
    y_extent=(d_extent[1]+p_extent[1])/2
    z_extent=(d_extent[2]+p_extent[2])/2
    new_extent=np.array([x_extent,y_extent,z_extent])


    d_angle = np.arctan2(d_rot_mat[1, 0], d_rot_mat[0, 0])
    p_angle = np.arctan2(p_rot_mat[1, 0], p_rot_mat[0, 0])
    d_angle=abs(d_angle)
    p_angle=abs(p_angle)
    print("detected angle",np.degrees(d_angle))
    print("predicted angle",np.degrees(p_angle))
    new_angle=(d_angle+p_angle)/2
    print("new angle",np.degrees(new_angle))
    x_direction=[np.cos(new_angle),-np.sin(new_angle),0]
    z_direction = np.array([0, 0, 1]) 
    y_direction = np.cross(z_direction, x_direction)[:2]
    x_direction = (x_direction[0], x_direction[1], 0)
    y_direction = (y_direction[0], y_direction[1], 0)
    new_rot_mat = np.column_stack((x_direction, y_direction, z_direction))


    # Create a colored point cloud for the object's centroid
    centroid_color = [0,0,1]  # Blue color
    centroid_cloud = o3d.geometry.PointCloud()
    centroid_cloud.points = o3d.utility.Vector3dVector([new_center])
    centroid_cloud.colors = o3d.utility.Vector3dVector([centroid_color])

    # Create a colored bounding box for the object
    object_bbox = o3d.geometry.OrientedBoundingBox(new_center, new_rot_mat, new_extent)
    object_bbox.color = [0,0,1]  # Blue color
    

    #print("Rotation angle (degrees):", round(angle_deg,1))

    # Add the objects to the visualizer
    all.append(object_bbox)
    all.append(centroid_cloud)
    
    return (new_center,new_extent,new_rot_mat), all

print("detected bbox",bbox_detected)
print("predicted bbox",bbox_predicted)

info,visualization_data=average_bbox(bbox_detected,bbox_predicted)

print("average bbox",info)




# Visualize the result along with centroids and bounding boxes
#o3d.visualization.draw_geometries([combined_cloud_point_cloud] + all_visualizations+all_visualizations_2+visualization_data)



    


