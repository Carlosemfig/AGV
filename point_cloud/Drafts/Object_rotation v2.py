import math

import MAIN as m
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import time
import random
import math
from scipy.spatial.transform import Rotation

# Dar load do backgorund
bg=m.load_pc("bg.pcd")
bg=m.pc_to_array(bg)


# Example usage of the function
center_cub=(1,1)
width_cub = 0.6  # Width range
length_cub = 0.6  # Length range
height_cub = 0.6  # Height of the cubic object
space=0.1

# Define the starting and ending coordinates of the three straight lines
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)
line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)
line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)
path1 =m.create_straight_line(line1_start, line1_end, space)
path2=m.create_straight_line(line2_start,line2_end,space)
path3=m.create_straight_line(line3_start,line3_end,space)
global_path= m.merge_arrays(path1,path2)
global_path=m.merge_arrays(global_path,path3)

final_map=m.merge_arrays(global_path,bg)

y_axis=m.create_straight_line((1,1),(1,15),0.1)
x_axis=m.create_straight_line((1,1),(15,1),0.1)

axis=m.merge_arrays(y_axis,x_axis)

cube=m.create_cubic_object((1,2),0.4,0.6,0.1)

rot_cube=m.rotate_cubic_object(cube,-20)

object=m.merge_arrays(rot_cube,bg)
#object=m.merge_arrays(cube,bg)

result= m.subtract_array(bg,object)


Eps=0.2
Min_samples=10

#encontrar a bbox e a matriz de rotação para depois calcular o teta_p

Labels, Number=m.perform_clustering(result,Eps,Min_samples)

def rotate_points(points, rotation_matrix, center):
    # Translate points to the origin
    translated_points = points - center
    # Rotate points
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    # Translate points back to the original position
    rotated_points += center
    return rotated_points

def rotate_points_around_center(points, rotation_matrix, center):
    # Translate points to the origin
    translated_points = points - center
    # Rotate points
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    # Translate points back to the original position
    rotated_points += center
    return rotated_points

def calculate_bounding_box_vertices(center, extend, rotation_matrix):
    # Calculate half extents along each axis
    half_extents = np.abs(extend) / 2.0

    # Define the eight vertices of the bounding box
    vertices = np.array([
        [-half_extents[0], -half_extents[1], -half_extents[2]],
        [ half_extents[0], -half_extents[1], -half_extents[2]],
        [ half_extents[0],  half_extents[1], -half_extents[2]],
        [-half_extents[0],  half_extents[1], -half_extents[2]],
        [-half_extents[0], -half_extents[1],  half_extents[2]],
        [ half_extents[0], -half_extents[1],  half_extents[2]],
        [ half_extents[0],  half_extents[1],  half_extents[2]],
        [-half_extents[0],  half_extents[1],  half_extents[2]],
    ])

    # Rotate the vertices around the center
    rotated_vertices = rotate_points_around_center(vertices, rotation_matrix, center)

    return rotated_vertices



def centroid_and_box(points, cluster_labels, num_clusters):
    all = []
    bbox = {}

    for cluster_id in range(num_clusters):
        object_points = points[cluster_labels == cluster_id]

        # Calculate the bounding box of the object
        object_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(object_points))

        
        target_vector=[0,1,0]


        object_bbox.color = [0, 1, 0]  # Green color
        center=object_bbox.center
        #print("centro",center)
        extend=object_bbox.extent
        rotation_matrix = object_bbox.R
        np.set_printoptions(precision=2, suppress=True)
        vector_x=np.dot(rotation_matrix,[1,0,0])
        cos_teta=np.dot(target_vector,vector_x)
        teta=math.acos(cos_teta)
        #print(teta)
        rot_mat=np.array([[np.cos(teta), -np.sin(teta), 0],
                     [np.sin(teta), np.cos(teta), 0],
                     [0, 0, 1]])
        print("rotation mat",rot_mat)

        object_bbox.rotate(rot_mat,center)
        rotation_matrix = object_bbox.R

        # Calculate rotated bounding box vertices
        rotated_vertices = calculate_bounding_box_vertices(center, extend, rot_mat)

        # Create a line set to visualize the bounding box
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(rotated_vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Add the line set to the visualizer
        all.append(line_set)

        bbox[cluster_id] = (center, extend,rotation_matrix )

    return all, bbox


all, bbox= centroid_and_box(result,Labels,Number)
centro=bbox[0][0]
extend=bbox[0][1]
rotation=bbox[0][2]

teta_p=m.matrix_to_angle(rotation)


point1_list = [10, 1, 0]
point2_list = [3, 3, 0]

# Convert the lists to NumPy arrays
point1_np = np.array(point1_list)
point2_np = np.array(point2_list)

# Combine the two arrays into a single 2D array
points_array = np.vstack((point1_np, point2_np))

# Convert the 2D array to a point cloud
point_cloud_T = m.array_to_pc(points_array)

point_T=m.find_closest_checkpoint(centro,points_array)

teta_t=m.points_to_angle(centro,point_T)

teta_e=teta_t-teta_p

print("Teta P =",teta_p)
print("Teta_T =",teta_t)
print("Teta_E =",teta_e)


final_map=m.array_to_pc(final_map)
axis=m.array_to_pc(axis)
point_cloud_T = m.array_to_pc(points_array)
all.append(point_cloud_T)
all.append(axis)
m.run_visualizer(all)


