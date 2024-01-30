import numpy as np
import math
import open3d as o3d
import MAIN as m
center=(-4.873843669425922, -2.4277728241002636, 0.0)
extent=[0.09357910186006047, 0.0, 0.0]
rotation_matrix=np.array([[-0.03876318, -0.99924843, 
 0.        ],
       [-0.99924843, -0.03876318,  0.        ],
       [ 0.        ,  0.        ,  1.        ]])
original_bbox=(center,extent,rotation_matrix)
v=1
deltat=0.5
angle=10

def centroid_and_box(original_bbox):
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

    
    centroid,extent,rot_mat=original_bbox
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

    
    return all

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


def predicted_bbox(bbox,v,angle,delta_t):
    center,extent,rotation_matrix=bbox
    
    angle= np.radians(angle)
    #matrix with the adicional angle
    angle_matrix=np.array([[np.cos(angle), -np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]])
    next_rot_mat = np.dot(angle_matrix, rotation_matrix)
    #new rotation matrix, resulting from adding the angle.

    vector_p=np.dot(next_rot_mat, np.array([1,0,0])) [:2]
    next_center=dist_to_pos(center,v*delta_t,vector_p)
    final_bbox=(next_center,extent,next_rot_mat)

    return final_bbox


next_pos=predicted_bbox(original_bbox,v,angle,deltat)
print(next_pos)

"""
bbox_list=[]

center1,extent1,rot_mat1=original_bbox

center2,extent2,rot_mat2=next_pos
# Create a colored bounding box for the object
centroid_color = [0, 0, 1]  # Blue color
centroid_cloud = o3d.geometry.PointCloud()
centroid_cloud.points = o3d.utility.Vector3dVector([center1])
centroid_cloud.colors = o3d.utility.Vector3dVector([centroid_color])
object_bbox1 = o3d.geometry.OrientedBoundingBox(center1, rot_mat1, extent1)
object_bbox1.color = [0, 1, 0]  # Green color
bbox_list.append(object_bbox1)
bbox_list.append(centroid_cloud)

# Create a colored bounding box for the object
object_bbox2 = o3d.geometry.OrientedBoundingBox(center2, rot_mat2, extent2)
object_bbox2.color = [1, 0, 0]  # Not Green color
        
bbox_list.append(object_bbox1)"""
#bbox_list.append(object_bbox2)
#creates noise points in the range defined
"""x_range=(-5,5)
y_range=(-5,5)
noise_points = m.create_random_points(100,x_range,y_range)
noise_points=m.array_to_pc(noise_points)"""
# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries(centroid_and_box(original_bbox))
