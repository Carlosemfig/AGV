import MAIN as m
import open3d as o3d
import numpy as np
import random
# Load background point cloud
bg = m.load_pc("bg.pcd")
bg = m.pc_to_array(bg)

# Example usage of the function
center_cub = (1, 1)
width_cub = 0.4  # Width range
size_y = 0.3  # Adjust this value based on your requirements
spacing = 0.1  # Adjust this value based on your requirements

# Define the starting and ending coordinates of the three straight lines
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)
line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)
line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)
path1 = m.create_straight_line(line1_start, line1_end, spacing)
path2 = m.create_straight_line(line2_start, line2_end, spacing)
path3 = m.create_straight_line(line3_start, line3_end, spacing)
global_path = m.merge_arrays(path1, path2)
global_path = m.merge_arrays(global_path, path3)




# Create the cube at a random position outside the path
cub_random_x = np.random.uniform(min(global_path[:, 0]), max(global_path[:, 0]))
cub_random_y = np.random.uniform(min(global_path[:, 1]), max(global_path[:, 1]))

# Ensure that the cube is placed far enough from the path
min_distance = 1.0  # Adjust this value as needed
while np.min(np.sqrt((cub_random_x - global_path[:, 0]) ** 2 +
                     (cub_random_y - global_path[:, 1]) ** 2)) < min_distance:
    cub_random_x = np.random.uniform(min(global_path[:, 0]), max(global_path[:, 0]))
    cub_random_y = np.random.uniform(min(global_path[:, 1]), max(global_path[:, 1]))

# Provide the correct arguments to create_cubic_object
cub = m.create_cubic_object((cub_random_x, cub_random_y), 0.4, 0.6, spacing)
random_value = random.uniform(-90, +90)
print("object rotation angle:",random_value)

cub= m.rotate_cubic_object(cub,random_value)






# create checkpoints in path
checkpoint_spacing = 8
checkpoints = m.create_checkpoints(global_path, checkpoint_spacing)

# Find the next checkpoint
current_position = (cub_random_x, cub_random_y, 0.0)  # Assuming Z is 0 for simplicity
next_checkpoint = m.find_closest_checkpoint(current_position, checkpoints)


vector=m.create_vector(next_checkpoint,current_position)

Eps=0.2
Min_samples=10

Labels, Number=m.perform_clustering(cub,Eps,Min_samples)

all, bbox= m.centroid_and_box_new(cub,Labels,Number,vector)


bbox_center=bbox[0][0]
bbox_matrix=bbox[0][2]
teta_p=m.matrix_to_angle(bbox_matrix)




teta_t=m.points_to_angle(bbox_center,next_checkpoint)
teta_e=teta_t-teta_p
print("Teta P =",teta_p)
print("Teta_T =",teta_t)
print("Teta_E =",teta_e)

# Create a straight line between the current position and the next checkpoint
straight_line = m.create_straight_line(current_position[:2], next_checkpoint[:2], spacing)

# Update the path based on the current position and next checkpoint
updated_path = m.update_path(current_position, next_checkpoint, global_path, spacing)

# Combine the cube, walls, and updated path into a single point cloud
#combined_cloud = m.merge_arrays(bg, cub, straight_line, updated_path)
combined_cloud = m.merge_arrays( cub, straight_line, updated_path)
# Convert the combined_cloud array to a point cloud
combined_point_cloud = o3d.geometry.PointCloud()
combined_point_cloud.points = o3d.utility.Vector3dVector(combined_cloud)

# Visualize the combined point cloud
o3d.visualization.draw_geometries([combined_point_cloud])

